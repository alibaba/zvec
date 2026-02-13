from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2
from loguru import logger

COMMIT_ID = os.environ["COMMIT_ID"]
DATASET = os.environ["DATASET"]
DATABASE_URL = os.environ["DATABASE_URL"]


def get_latest_result() -> Path:
    # VectorDBBench saves results in a nested structure
    results_dir = Path("/opt/VectorDBBench/vectordb_bench/results/Zvec")
    list_of_files = list(results_dir.glob("*.json"))
    return max(list_of_files, key=lambda p: p.stat().st_ctime)


def upload_to_postgres():
    logger.info(f"Uploading metrics to PostgreSQL: {DATABASE_URL}")
    result_file = get_latest_result()
    with result_file.open() as f:
        data = json.load(f)
    logger.info(f"Loaded data from {result_file}: {data}")

    # VectorDBBench result schema: run_id, task_label, results[{ metrics, task_config, label }]
    result = data["results"][0]
    metrics = result["metrics"]
    logger.info(f"Metrics: {metrics}")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    sql = """
          INSERT INTO bench_results (commit_sha, dataset, qps, recall, p99, p95, ndcg, load_time)
          VALUES (%s, %s, %s, %s, %s, %s, %s, %s) \
          """

    values = (
        COMMIT_ID,
        DATASET,
        metrics["qps"],
        metrics["recall"],
        metrics["serial_latency_p99"],
        metrics["serial_latency_p95"],
        metrics["ndcg"],
        metrics["load_duration"],
    )

    cur.execute(sql, values)
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Successfully uploaded metrics from {result_file}")


if __name__ == "__main__":
    upload_to_postgres()

# CREATE TABLE bench_results (
#                                id SERIAL PRIMARY KEY,
#                                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
#                                commit_sha VARCHAR(40),
#                                dataset VARCHAR(100),
#                                qps FLOAT,
#                                recall FLOAT,
#                                p99 FLOAT,
#                                p95 FLOAT,
#                                ndcg FLOAT,
#                                load_time FLOAT
# );`
