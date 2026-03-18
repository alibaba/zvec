# -*- coding: utf-8 -*-
"""
test_collection_crash_recovery_deletedoc.py

This script is used to test Zvec's recovery capability after simulating a "power failure" (forced process termination) during bulk document deletion (insertdoc).
It first successfully creates a collection in the main process, then starts a subprocess to open the collection and perform bulk document deletion operations.
During the deletion operation, the subprocess is forcibly terminated to simulate a scenario where the Zvec process crashes during document deletion.
Finally, the main process attempts to reopen the collection and verify its state and functionality.

Note: This script assumes that Zvec is a Python extension library. Directly killing the Python subprocess running Zvec operations
may not perfectly simulate the impact of system-level power failure on the C++ layer, but it can test the file state of the Zvec Python extension after abnormal process termination.
"""

import zvec
import time
import tempfile
import subprocess
import signal
import sys
import os
import pytest
import json  # Used to pass operation parameters and results
import threading

try:
    import psutil  # Used for more reliable process management
except ImportError:
    psutil = None  # If psutil is not installed, set it to None
from fixture_helper import *
from doc_helper import generate_doc

from distance_helper import *
from fixture_helper import *
from doc_helper import *


def singledoc_and_check(
    collection: Collection, insert_doc, operator="insert", is_delete=1
):
    if operator == "insert":
        result = collection.insert(insert_doc)
    elif operator == "upsert":
        result = collection.upsert(insert_doc)
    elif operator == "update":
        result = collection.update(insert_doc)
    else:
        logging.error("operator value is error!")

    assert bool(result)
    assert result.ok()

    stats = collection.stats
    assert stats is not None
    # assert stats.doc_count == 1

    fetched_docs = collection.fetch([insert_doc.id])
    assert len(fetched_docs) == 1
    assert insert_doc.id in fetched_docs

    fetched_doc = fetched_docs[insert_doc.id]

    assert is_doc_equal(fetched_doc, insert_doc, collection.schema)
    assert hasattr(fetched_doc, "score"), "Document should have a score attribute"
    assert fetched_doc.score == 0.0, (
        "Fetch operation should return default score of 0.0"
    )

    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        if v != {}:
            query_result = collection.query(
                VectorQuery(field_name=v, vector=insert_doc.vectors[v]),
                topk=1024,
            )
            assert len(query_result) > 0, (
                f"Expected at least 1 query result, but got {len(query_result)}"
            )

            found_doc = None
            for doc in query_result:
                if doc.id == insert_doc.id:
                    found_doc = doc
                    break
            assert found_doc is not None, (
                f"deleted document {insert_doc.id} not found in query results"
            )
            assert is_doc_equal(found_doc, insert_doc, collection.schema, True, False)
    if is_delete == 1:
        collection.delete(insert_doc.id)
        assert collection.stats.doc_count == 0, "Document should be deleted"


class TestCollectionCrashRecoveryDeleteDoc:
    """
    Test Zvec collection recovery capability after simulating power failure/process crash during document deletion.
    Focus on verifying whether the file remains consistent after interruption of document deletion operations,
    and whether it can be reopened and used normally.
    """

    # Script content for subprocess to execute Zvec document deletion operations
    # Write this script content to a temporary file and execute it in the subprocess.
    ZVEC_SUBPROCESS_SCRIPT_DELETEDOC = """
import zvec
import time
import json
import sys
import os
import math
import random
import string
from typing import Literal
import pytest


def generate_constant_vector(
    i: int, dimension: int, dtype: Literal["int8", "float16", "float32"] = "float32"
):
    if dtype == "int8":
        vec = [(i % 127)] * dimension
        vec[i % dimension] = ((i + 1) % 127)
    else:
        base_val = (i % 1000) / 256.0
        special_val = ((i + 1) % 1000) / 256.0
        vec = [base_val] * dimension
        vec[i % dimension] = special_val

    return vec


def generate_sparse_vector(i: int):
    return {i: i + 0.1}


def generate_vectordict(i: int, schema: zvec.CollectionSchema):
    doc_fields = {}
    doc_vectors = {}
    for field in schema.fields:
        if field.data_type == zvec.DataType.BOOL:
            doc_fields[field.name] = i % 2 == 0
        elif field.data_type == zvec.DataType.INT32:
            doc_fields[field.name] = i
        elif field.data_type == zvec.DataType.UINT32:
            doc_fields[field.name] = i
        elif field.data_type == zvec.DataType.INT64:
            doc_fields[field.name] = i
        elif field.data_type == zvec.DataType.UINT64:
            doc_fields[field.name] = i
        elif field.data_type == zvec.DataType.FLOAT:
            doc_fields[field.name] = float(i) + 0.1
        elif field.data_type == zvec.DataType.DOUBLE:
            doc_fields[field.name] = float(i) + 0.11
        elif field.data_type == zvec.DataType.STRING:
            doc_fields[field.name] = f"test_{i}"
        elif field.data_type == zvec.DataType.ARRAY_BOOL:
            doc_fields[field.name] = [i % 2 == 0, i % 3 == 0]
        elif field.data_type == zvec.DataType.ARRAY_INT32:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == zvec.DataType.ARRAY_UINT32:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == zvec.DataType.ARRAY_INT64:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == zvec.DataType.ARRAY_UINT64:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == zvec.DataType.ARRAY_FLOAT:
            doc_fields[field.name] = [float(i + 0.1), float(i + 1.1), float(i + 2.1)]
        elif field.data_type == zvec.DataType.ARRAY_DOUBLE:
            doc_fields[field.name] = [float(i + 0.11), float(i + 1.11), float(i + 2.11)]
        elif field.data_type == zvec.DataType.ARRAY_STRING:
            doc_fields[field.name] = [f"test_{i}", f"test_{i + 1}", f"test_{i + 2}"]
        else:
            raise ValueError(f"Unsupported field type: {field.data_type}")

    for vector in schema.vectors:
        if vector.data_type == zvec.DataType.VECTOR_FP16:
            doc_vectors[vector.name] = generate_constant_vector(
                i, vector.dimension, "float16"
            )
        elif vector.data_type == zvec.DataType.VECTOR_FP32:
            doc_vectors[vector.name] = generate_constant_vector(
                i, vector.dimension, "float32"
            )
        elif vector.data_type == zvec.DataType.VECTOR_INT8:
            doc_vectors[vector.name] = generate_constant_vector(
                i,
                vector.dimension,
                "int8",
            )
        elif vector.data_type == zvec.DataType.SPARSE_VECTOR_FP32:
            doc_vectors[vector.name] = generate_sparse_vector(i)
        elif vector.data_type == zvec.DataType.SPARSE_VECTOR_FP16:
            doc_vectors[vector.name] = generate_sparse_vector(i)
        else:
            raise ValueError(f"Unsupported vector type: {vector.data_type}")

    return doc_fields, doc_vectors


def generate_doc(i: int, schema: zvec.CollectionSchema) -> zvec.Doc:
    doc_fields = {}
    doc_vectors = {}
    doc_fields, doc_vectors = generate_vectordict(i, schema)
    doc = zvec.Doc(id=str(i), fields=doc_fields, vectors=doc_vectors)
    return doc


def run_zvec_deletedoc_operations(args_json_str):
    args = json.loads(args_json_str)
    collection_path = args["collection_path"]
    num_docs_to_delete = args.get("num_docs_to_delete", 100)  # Number of documents to insert
    batch_size = args.get("batch_size", 10)  # Batch size for each deletion
    delay_between_batches = args.get("delay_between_batches", 0.1)  # Delay between batches

    print(f"[Subprocess] Starting Zvec insert document operations on {collection_path} at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Subprocess] Will insert {num_docs_to_delete} documents in batches of {batch_size}")

    try:
        # Open existing collection
        collection = zvec.open(collection_path)
        print(f"[Subprocess] Successfully opened collection.")

        deleted_count = 0
        for i in range(0, num_docs_to_delete, batch_size):
            # Calculate the number of documents in the current batch
            current_batch_size = min(batch_size, num_docs_to_delete - i)

            if current_batch_size==batch_size:
                      
                doc_ids= [str(i) for i in range(i*batch_size,(i+1)*batch_size)]
            else:
                doc_ids= [str(i) for i in range(i*batch_size, num_docs_to_delete)]
            
            result = collection.delete(doc_ids)
            
            # Check return value - insert returns a list of document IDs
            assert len(result) == len(doc_ids)
            for i in range(len(result)):
                if i < len(doc_ids):
                    assert result[i].ok()
            deleted_count += len(doc_ids)
            print(f"[Subprocess] Batch deletion successful, deleted {len(doc_ids)} documents, total deleted: {deleted_count}")



            # Add small delay to allow interruption opportunity
            time.sleep(delay_between_batches)

        print(f"[Subprocess] Completed inserting {deleted_count} documents.")

        if hasattr(collection, "close"):
            collection.close()
        else:
            del collection  # Use del as fallback
        print(f"[Subprocess] Closed collection after deletion operations.")

    except Exception as e:
        print(f"[Subprocess] Error during document deletion operations: {e}")
        import traceback
        traceback.print_exc()
        # Optionally re-raise or handle differently
        raise  # Re-raising may be useful depending on how parent process responds

    print(f"[Subprocess] Document deletion operations completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    args_json_str = sys.argv[1]
    run_zvec_deletedoc_operations(args_json_str)
"""

    def test_insertdoc_simulate_crash_during_bulk_insert(
        self, full_schema_1024, collection_option, basic_schema
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process.
                  Then start a subprocess to open the collection and perform bulk document deletion operations.
                  During the bulk deletion operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_path = f"{temp_dir}/test_collection_deletedoc_crash_recovery"

            # Step 1: Successfully create collection in main process
            print(
                f"[Test] Step 1: Creating collection in main process, path: {collection_path}..."
            )
            coll = zvec.create_and_open(
                path=collection_path, schema=full_schema_1024, option=collection_option
            )
            assert coll is not None
            print(f"[Test] Step 1.1: Collection created successfully.")
            single_doc = generate_doc(2001, coll.schema)
            singledoc_and_check(coll, single_doc, is_delete=0)
            print(f"[Test] Step 1.2: Verified collection data write successful.")

            # Insert initial documents that will be deleted later
            initial_docs = []
            for i in range(0, 1000):  # Insert 200 documents for updating
                doc = generate_doc(i, coll.schema)
                initial_docs.append(doc)

            insert_results = coll.insert(initial_docs)
            print(
                f"[Test] Step 1.3: deleted {len(initial_docs)} initial documents for updating."
            )

            del coll
            print(f"[Test] Step 1.3: Closed collection.")

            # Step 2: Prepare and run subprocess for bulk deletion operations
            # Write subprocess script to temporary file
            subprocess_script_path = f"{temp_dir}/zvec_subprocess_deletedoc.py"
            with open(subprocess_script_path, "w", encoding="utf-8") as f:
                f.write(self.ZVEC_SUBPROCESS_SCRIPT_DELETEDOC)

            # Prepare subprocess parameters
            subprocess_args = {
                "collection_path": collection_path,
                "num_docs_to_delete": 200,  # Insert 200 documents to allow for interruption
                "batch_size": 10,  # Insert 10 documents per batch
                "delay_between_batches": 0.2,  # 0.2 second delay between batches to increase interruption timing
            }
            args_json_str = json.dumps(subprocess_args)

            print(
                f"[Test] Step 2: Starting bulk deletion operations in subprocess, path: {collection_path}"
            )
            # Start subprocess to execute bulk deletion operations
            proc = subprocess.Popen(
                [sys.executable, subprocess_script_path, args_json_str]
            )

            # Wait briefly to allow subprocess to begin deletion operations
            time.sleep(2)  # Wait 2 seconds to allow deletion loop to start

            print(
                f"[Test] Step 2: Simulating crash/power failure by terminating subprocess PID {proc.pid}..."
            )
            # Suddenly kill subprocess (simulate power failure or crash during deletion operations)
            if psutil:
                try:
                    # Use psutil to reliably terminate process and all its children
                    parent = psutil.Process(proc.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        child.kill()
                    parent.kill()
                    proc.wait(timeout=5)
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    subprocess.TimeoutExpired,
                ):
                    # If psutil is unavailable or process has been terminated, fall back to original method
                    proc.send_signal(signal.SIGKILL)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(
                            f"[Test] Subprocess {proc.pid} could not be terminated with SIGKILL, force killing..."
                        )
                        proc.kill()
                        proc.wait()
            else:
                # If no psutil, use standard method to terminate process
                proc.send_signal(signal.SIGKILL)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(
                        f"[Test] Subprocess {proc.pid} could not be terminated with SIGKILL, force killing..."
                    )
                    proc.kill()
                    proc.wait()
            print(f"[Test] Subprocess {proc.pid} has been terminated.")

            # Clean up temporary script file
            os.remove(subprocess_script_path)

            # Step 3: Verify recovery situation in main process
            print(
                f"[Test] Step 3: Attempting to open collection after simulating crash during document deletion operations..."
            )
            # Verification 3.1: Check if collection can be successfully opened after crash
            recovered_collection = zvec.open(collection_path)
            assert recovered_collection is not None, (
                "Cannot open collection after crash"
            )
            print(f"[Test] Step 3.1: Verified collection can be opened after crash...")

            # Verification 3.2: Check data integrity (document count and content)
            print(f"[Test] Step 3.2: Verifying data integrity...")
            query_result = recovered_collection.query(topk=1024)
            # We expect some documents to have been successfully deleted before crash
            # The exact number depends on when the crash occurred during the bulk deletion process
            print(
                f"[Test] Step 3.2: Found {len(query_result)} documents after crash (expected 0-{subprocess_args['num_docs_to_delete']})"
            )

            current_count = recovered_collection.stats.doc_count
            assert recovered_collection.stats.doc_count >= 1
            assert len(query_result) <= recovered_collection.stats.doc_count, (
                f"query_result count = {len(query_result)},stats.doc_count = {recovered_collection.stats.doc_count}"
            )

            # Verify existing documents have correct structure
            if len(query_result) > 0:
                for doc in query_result[:1024]:
                    if doc.id == "2001":
                        print("Found 2001 data!")
                        fetched_docs = recovered_collection.fetch([doc.id])
                        print("doc.id:\n")
                        print(doc.id)
                        print("fetched_docs:\n")
                        print(fetched_docs)
                        assert len(fetched_docs) == 1
                        assert doc.id in fetched_docs
                        assert is_doc_equal(
                            fetched_docs["2001"],
                            single_doc,
                            recovered_collection.schema,
                        ), f"result doc={fetched_doc},doc_exp={single_doc}"
                        break
                    else:
                        fetched_docs = recovered_collection.fetch([doc.id])
                        print("doc.id,fetched_docs:\n")
                        print(doc.id, fetched_docs)
                        exp_doc = generate_doc(int(doc.id), recovered_collection.schema)
                        assert len(fetched_docs) == 1
                        assert doc.id in fetched_docs
                        assert is_doc_equal(
                            fetched_docs["1"], exp_doc, recovered_collection.schema
                        ), f"result doc={fetched_docs},doc_exp={exp_doc}"

            # 3.4: Check if index is complete and query function works properly
            print(f"[Test] Step 3.4: Verifying index integrity and query function...")
            filtered_query = recovered_collection.query(filter=f"int32_field >=-100")
            print(
                f"[Test] Step 3.4.2: Field-filtered query returned {len(filtered_query)} documents"
            )
            assert len(filtered_query) > 0
            for doc in query_result:
                if doc.id == "2001":
                    print("Found 2001 data!")
                    fetched_docs = recovered_collection.fetch([doc.id])
                    print("doc.id:\n")
                    print(doc.id)
                    print("fetched_docs:\n")
                    print(fetched_docs)
                    assert len(fetched_docs) == 1
                    assert doc.id in fetched_docs
                    assert is_doc_equal(
                        fetched_docs["2001"], single_doc, recovered_collection.schema
                    ), f"result doc={fetched_doc},doc_exp={single_doc}"
                    break
                else:
                    fetched_docs = recovered_collection.fetch([doc.id])
                    print("doc.id,fetched_docs:\n")
                    print(doc.id, fetched_docs)
                    exp_doc = generate_doc(int(doc.id), recovered_collection.schema)
                    assert len(fetched_docs) == 1
                    assert doc.id in fetched_docs
                    assert is_doc_equal(
                        fetched_docs["1"], exp_doc, recovered_collection.schema
                    ), f"result doc={fetched_docs},doc_exp={exp_doc}"

            # Verification 3.5: Test insertion functionality after recovery
            print(f"[Test] Step 3.5.1: Testing insertion functionality after recovery")
            test_insert_doc = generate_doc(
                9999, full_schema_1024
            )  # Use original schema from fixture
            singledoc_and_check(
                recovered_collection, test_insert_doc, operator="insert", is_delete=0
            )

            # Verification 3.6: Test update functionality after recovery
            print(f"[Test] Step 3.6: Testing update functionality after recovery...")
            updated_doc = generate_update_doc(2001, recovered_collection.schema)
            singledoc_and_check(
                recovered_collection, updated_doc, operator="update", is_delete=0
            )

            # 3.7: Test deletion  after recovery
            print(f"[Test] Step 3.7: Testing deletion functionality after recovery...")
            doc_ids = ["9999"]
            result = recovered_collection.delete(doc_ids)
            assert len(result) == len(doc_ids)
            for item in result:
                assert item.ok()
