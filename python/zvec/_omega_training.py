# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OMEGA model training module."""

import argparse
import os
import sys
import numpy as np

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.isotonic import IsotonicRegression
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def train_omega_model(csv_path: str, output_dir: str, verbose: bool = False, topk: int = 100):
    """Train OMEGA model from CSV training data.

    Args:
        csv_path: Path to CSV file with training data
        output_dir: Directory to save trained model and tables
        verbose: Enable verbose logging
        topk: Top-K value used during training data collection (default: 100)

    Returns:
        str: Path to the trained model directory
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError(
            "LightGBM is required for OMEGA training. "
            "Install it with: pip install lightgbm"
        )

    if verbose:
        print(f"Loading training data from: {csv_path}")

    # Load CSV data
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Extract features and labels
    # CSV format: query_id,hops_visited,cmps_visited,dist_1st,dist_start,stat_0,...,stat_6,label
    query_ids = df['query_id'].values.astype(np.int32)
    X = df[['hops_visited', 'cmps_visited', 'dist_1st', 'dist_start',
            'stat_0', 'stat_1', 'stat_2', 'stat_3', 'stat_4', 'stat_5', 'stat_6']].values
    y = df['label'].values

    if verbose:
        print(f"Loaded {len(df)} training records from {len(np.unique(query_ids))} queries")
        print(f"Feature shape: {X.shape}")
        print(f"Label distribution: {np.sum(y==0)} negative, {np.sum(y==1)} positive")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Train LightGBM binary classifier
    model_path = os.path.join(output_dir, "model.txt")
    threshold_table_path = os.path.join(output_dir, "threshold_table.txt")

    if verbose:
        print("Training LightGBM model...")

    # Split data
    query_ids_train, query_ids_test, X_train, X_test, y_train, y_test = train_test_split(
        query_ids, X, y, test_size=0.2, shuffle=False
    )

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

    # Training parameters
    # Calculate scale_pos_weight safely
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)

    if n_positive == 0:
        raise ValueError(f"No positive samples in training data! All labels are 0.")
    if n_negative == 0:
        raise ValueError(f"No negative samples in training data! All labels are 1.")

    scale_pos_weight = n_negative / n_positive

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss'],
        'num_leaves': 31,
        'boost_from_average': False,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'verbose': 0 if not verbose else 1,
        'num_threads': 8,
        'scale_pos_weight': scale_pos_weight,
    }

    if verbose:
        print(f"Training samples: {len(y_train)} ({n_positive} positive, {n_negative} negative)")
        print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # Train model
    num_round = 100
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=num_round,
        callbacks=[lgb.record_evaluation(evals_result)]
    )

    # Save model
    model.save_model(model_path)
    if verbose:
        print(f"Model saved to: {model_path}")

    # Generate threshold table using isotonic regression
    if verbose:
        print("Generating threshold table...")

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # Calibrate using isotonic regression
    isotonic_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
    y_pred_calibrated = isotonic_reg.fit_transform(y_pred, y_test)

    # Generate threshold table
    sorted_indices = np.argsort(y_pred)
    sorted_confidences = y_pred[sorted_indices]
    sorted_probabilities = y_pred_calibrated[sorted_indices]
    sorted_confidences_10000x = np.round(sorted_confidences * 10000)

    # Remove duplicates
    _, unique_indices = np.unique(sorted_confidences_10000x, return_index=True)
    unique_confidences = sorted_confidences[unique_indices]
    unique_probabilities = sorted_probabilities[unique_indices]

    with open(threshold_table_path, "w") as f:
        for conf, prob in zip(unique_confidences, unique_probabilities):
            f.write(f"{conf:.4f},{prob:.6f}\n")

    if verbose:
        print(f"Threshold table saved to: {threshold_table_path}")

    # Generate placeholder interval table (not used in zvec's current implementation)
    interval_table_path = os.path.join(output_dir, "interval_table.txt")
    with open(interval_table_path, "w") as f:
        for recall_pct in range(0, 101, 1):
            recall = recall_pct / 100.0
            initial_interval = max(int(100 * (1 - recall)), 1)
            min_interval = max(int(10 * (1 - recall)), 1)
            f.write(f"{recall:.2f},{initial_interval},{min_interval}\n")

    if verbose:
        print(f"Interval table saved to: {interval_table_path}")

    # Generate placeholder gt_collected_table and gt_cmps_all_table
    # These tables require access to ground truth data during search, which is not
    # available from the CSV export. They should be generated during the training
    # data collection phase in C++.

    # Create empty placeholder files with the correct format
    gt_collected_table_path = os.path.join(output_dir, "gt_collected_table.txt")
    with open(gt_collected_table_path, "w") as f:
        # Format: row_index:value1,value2,...,valueK
        # Each row represents a "collected" count, columns are ranks
        for collected in range(topk + 1):
            row_values = ["1.0" if i < collected else "0.0" for i in range(topk)]
            f.write(f"{collected}:{','.join(row_values)}\n")

    if verbose:
        print(f"GT collected table (placeholder) saved to: {gt_collected_table_path}")

    gt_cmps_all_table_path = os.path.join(output_dir, "gt_cmps_all_table.txt")
    with open(gt_cmps_all_table_path, "w") as f:
        # Format: row_index:value1,value2,...,value100
        # Each row represents a rank, columns are percentiles (1-100)
        for rank in range(topk + 1):
            percentiles = [str(rank * 10 + p) for p in range(100)]  # Placeholder values
            f.write(f"{rank}:{','.join(percentiles)}\n")

    if verbose:
        print(f"GT cmps all table (placeholder) saved to: {gt_cmps_all_table_path}")

    # Print final statistics
    if verbose:
        print("\nTraining complete!")
        print(f"Model directory: {output_dir}")
        print("Generated files:")
        print(f"  - model.txt")
        print(f"  - threshold_table.txt")
        print(f"  - interval_table.txt")
        print(f"  - gt_collected_table.txt (placeholder)")
        print(f"  - gt_cmps_all_table.txt (placeholder)")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train OMEGA model from collected training data"
    )
    parser.add_argument(
        "command",
        choices=["train"],
        help="Command to execute"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file with training data"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Top-K value used during training (default: 100)"
    )

    args = parser.parse_args()

    if args.command == "train":
        try:
            train_omega_model(
                csv_path=args.input,
                output_dir=args.output,
                verbose=args.verbose,
                topk=args.topk
            )
            print("✓ Training completed successfully")
            sys.exit(0)
        except Exception as e:
            print(f"✗ Training failed: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
