from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2


def get_latest_result() -> Path:
    # VectorDBBench saves results in a nested structure
    results_dir = Path("/opt/VectorDBBench/vectordb_bench/results/Zvec")
    list_of_files = list(results_dir.glob("*.json"))
    return max(list_of_files, key=lambda p: p.stat().st_ctime)


def upload_to_postgres():
    result_file = get_latest_result()
    with result_file.open() as f:
        data = json.load(f)

    # Based on your log, we extract the core metrics
    # Note: Structure might vary slightly based on VectorDBBench version
    metrics = data["results"][0]["result"]

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    sql = """
          INSERT INTO bench_results (commit_sha, dataset, db_label, qps, recall, p99, p95, ndcg, load_time)
          VALUES (%s, %s, %s, %s, %s, %s, %s, %s) \
          """

    values = (
        os.environ["COMMIT_ID"],
        data["results"][0]["case_config"]["dataset"]["data"]["name"],  # dataset name
        os.environ["DB_LABEL"],
        metrics["qps"],
        metrics["recall"],
        metrics["serial_latency_p99"],
        metrics["serial_latency_p95"],
        metrics["ndcg"],
        metrics["load_dur"],
    )

    cur.execute(sql, values)
    conn.commit()
    cur.close()
    conn.close()
    # print(f"Successfully uploaded metrics from {result_file}")


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
