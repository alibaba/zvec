"""End-to-end compatibility smoke test for the macOS x86_64 wheel."""

from __future__ import annotations

import faulthandler
import logging
import platform
import tempfile
from pathlib import Path

import zvec


def main() -> None:
    faulthandler.enable(all_threads=True)
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise RuntimeError(f"expected an x86_64 runtime, got {machine}")
    logging.info("macOS x64 smoke: imported zvec on %s", machine)

    with tempfile.TemporaryDirectory(prefix="zvec-macos-x64-") as temp_dir:
        collection_path = Path(temp_dir) / "smoke_collection"
        logging.info("macOS x64 smoke: creating schema")
        schema = zvec.CollectionSchema(
            name="macos_x64_smoke",
            vectors=[
                zvec.VectorSchema(
                    "dense",
                    zvec.DataType.VECTOR_FP32,
                    4,
                    index_param=zvec.HnswIndexParam(),
                ),
                zvec.VectorSchema(
                    "sparse",
                    zvec.DataType.SPARSE_VECTOR_FP32,
                    index_param=zvec.HnswIndexParam(),
                ),
            ],
        )

        logging.info("macOS x64 smoke: creating collection")
        collection = zvec.create_and_open(str(collection_path), schema)
        documents = [
            zvec.Doc(
                id="doc_1",
                vectors={
                    "dense": [0.1, 0.2, 0.3, 0.4],
                    "sparse": {1: 1.0, 3: 0.5},
                },
            ),
            zvec.Doc(
                id="doc_2",
                vectors={
                    "dense": [0.4, 0.3, 0.2, 0.1],
                    "sparse": {1: 0.5, 2: 1.0},
                },
            ),
            zvec.Doc(
                id="doc_3",
                vectors={
                    "dense": [0.2, 0.2, 0.2, 0.2],
                    "sparse": {2: 0.25, 3: 1.0},
                },
            ),
        ]

        logging.info("macOS x64 smoke: inserting documents")
        write_results = collection.insert(documents)
        if len(write_results) != len(documents) or not all(
            result.ok() for result in write_results
        ):
            raise RuntimeError(f"insert failed: {write_results}")

        logging.info("macOS x64 smoke: flushing collection")
        collection.flush()
        logging.info("macOS x64 smoke: optimizing collection")
        collection.optimize()

        logging.info("macOS x64 smoke: querying dense and sparse vectors")
        dense_results = collection.query(
            zvec.Query(field_name="dense", vector=[0.1, 0.2, 0.3, 0.4]),
            topk=3,
        )
        sparse_results = collection.query(
            zvec.Query(field_name="sparse", vector={1: 1.0, 3: 0.5}),
            topk=3,
        )
        if not dense_results or not sparse_results:
            raise RuntimeError("dense or sparse query returned no documents")
        if collection.stats.doc_count != len(documents):
            raise RuntimeError(
                f"expected {len(documents)} docs, got {collection.stats.doc_count}"
            )

        logging.info("macOS x64 smoke: closing and reopening collection")
        collection.close()
        reopened = zvec.open(str(collection_path))
        try:
            reopened_results = reopened.query(
                zvec.Query(field_name="dense", vector=[0.1, 0.2, 0.3, 0.4]),
                topk=3,
            )
            if not reopened_results:
                raise RuntimeError("query after reopen returned no documents")
        finally:
            reopened.close()

    logging.info(
        "macOS x64 Python smoke: imported, inserted, optimized, queried dense/sparse, "
        "and reopened successfully"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
