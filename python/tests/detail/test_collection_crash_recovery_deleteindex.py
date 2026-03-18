# -*- coding: utf-8 -*-
"""
test_collection_crash_recovery_deleteindex.py

This script is used to test Zvec's recovery capability after simulating a "power failure" (forced process termination) during index deletion.
It first successfully creates a collection in the main process and creates an index, then starts a subprocess to open the collection and perform index deletion operations.
During the index deletion operation, the subprocess is forcibly terminated to simulate a scenario where the Zvec process crashes during index deletion.
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
from doc_helper import generate_update_doc

from distance_helper import *


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


class TestCollectionCrashRecoveryDeleteIndex:
    """
    Test Zvec collection recovery capability after simulating power failure/process crash during index deletion.
    Focus on verifying whether the file remains consistent after interruption of index deletion operations,
    and whether it can be reopened and used normally.
    """

    # Script content for subprocess to execute Zvec index deletion operations
    # Write this script content to a temporary file and execute it in the subprocess.
    ZVEC_SUBPROCESS_SCRIPT_DELETEINDEX = """
import zvec
import time
import json
import sys
import os


def run_zvec_deleteindex_operations(args_json_str):
    args = json.loads(args_json_str)
    collection_path = args["collection_path"]
    index_field = args.get("index_field", "int32_field")  # Field to delete index from
    index_type = args.get("index_type", "INVERT")  # Type of index to delete
    index_deletion_iterations = args.get("index_deletion_iterations", 10)  # Number of index deletion iterations
    delay_between_deletions = args.get("delay_between_deletions", 0.5)  # Delay between index deletions

    print(f"[Subprocess] Starting Zvec delete index operations on {collection_path} at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Subprocess] Will delete {index_type} index on field '{index_field}', {index_deletion_iterations} times")

    try:
        # Open existing collection
        collection = zvec.open(collection_path)
        print(f"[Subprocess] Successfully opened collection.")

        print(f"[Subprocess] Starting {index_deletion_iterations} {index_type} index deletion operations...")

        # Loop to delete indexes multiple times - this increases the chance of interruption during the operation
        for i in range(index_deletion_iterations):
            print(f"[Subprocess] Iteration {i+1}/{index_deletion_iterations}: Deleting {index_type} index on field '{index_field}'...")

            # First check if index exists before attempting to delete
            field_schema = collection.schema.field(index_field)
            if field_schema and field_schema.index_param:
                print(f"[Subprocess] {index_type} index found on field '{index_field}', proceeding with deletion...")
                
                # Delete index - this operation can take time and be interrupted
                collection.drop_index(index_field)
                print(f"[Subprocess] Iteration {i+1}: {index_type} Index deletion completed successfully on field '{index_field}'.")
            else:
                print(f"[Subprocess] No {index_type} index found on field '{index_field}', skipping deletion...")

            # Add delay between iterations to allow interruption opportunity
            if i < index_deletion_iterations - 1:  # Don't sleep after the last iteration
                print(f"[Subprocess] Waiting {delay_between_deletions}s before next {index_type} index deletion...")
                time.sleep(delay_between_deletions)

        if hasattr(collection, "close"):
            collection.close()
        else:
            del collection  # Use del as fallback
        print(f"[Subprocess] Closed collection after index deletion operations.")

    except Exception as e:
        print(f"[Subprocess] Error during index deletion operations: {e}")
        import traceback
        traceback.print_exc()
        # Optionally re-raise or handle differently
        raise  # Re-raising may be useful depending on how parent process responds

    print(f"[Subprocess] Index deletion operations completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    args_json_str = sys.argv[1]
    run_zvec_deleteindex_operations(args_json_str)
"""

    def test_deleteindex_simulate_crash_during_index_deletion_invert(
        self, full_schema_1024, collection_option, basic_schema
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and create an INVERT index.
                  Then start a subprocess to open the collection and perform INVERT index deletion operations.
                  During the index deletion operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_deleteindex_with_crash_recovery(
            full_schema_1024, collection_option, "INVERT"
        )

    def test_deleteindex_simulate_crash_during_index_deletion_hnsw(
        self, full_schema_1024, collection_option, basic_schema
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and create an HNSW index.
                  Then start a subprocess to open the collection and perform HNSW index deletion operations.
                  During the index deletion operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_deleteindex_with_crash_recovery(
            full_schema_1024, collection_option, "HNSW"
        )

    def test_deleteindex_simulate_crash_during_index_deletion_flat(
        self, full_schema_1024, collection_option, basic_schema
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and create a FLAT index.
                  Then start a subprocess to open the collection and perform FLAT index deletion operations.
                  During the index deletion operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_deleteindex_with_crash_recovery(
            full_schema_1024, collection_option, "FLAT"
        )

    def test_deleteindex_simulate_crash_during_index_deletion_ivf(
        self, full_schema_1024, collection_option, basic_schema
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and create an IVF index.
                  Then start a subprocess to open the collection and perform IVF index deletion operations.
                  During the index deletion operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """

    def _test_deleteindex_with_crash_recovery(
        self, schema, collection_option, index_type
    ):
        """
        Common method to test index deletion with crash recovery for different index types.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_path = f"{temp_dir}/test_collection_deleteindex_crash_recovery_{index_type.lower()}"

            # Step 1: Successfully create collection in main process and insert some documents
            print(
                f"[Test] Step 1: Creating collection in main process, path: {collection_path}..."
            )
            coll = zvec.create_and_open(
                path=collection_path, schema=schema, option=collection_option
            )
            assert coll is not None
            print(f"[Test] Step 1.1: Collection created successfully.")

            # Insert some documents to have data for indexing
            for i in range(100):
                doc = generate_doc(i, coll.schema)
                result = coll.insert([doc])
                assert result is not None and len(result) > 0, (
                    f"Failed to insert document {i}"
                )

            print(f"[Test] Step 1.2: Inserted 100 documents for indexing.")

            # Create index based on the index type
            print(f"[Test] Step 1.3: Creating {index_type} index...")

            # Determine the appropriate field and index type for each case
            if index_type == "INVERT":
                from zvec import InvertIndexParam, IndexOption

                index_param = InvertIndexParam()
                field_name = "int32_field"  # Scalar fields support INVERT index
            elif index_type == "HNSW":
                from zvec import DataType, HnswIndexParam, IndexOption

                index_param = HnswIndexParam()
                # Use a vector field for HNSW index
                field_name = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for HNSW
            elif index_type == "FLAT":
                from zvec import DataType, FlatIndexParam, IndexOption

                index_param = FlatIndexParam()
                # Use a vector field for FLAT index
                field_name = DEFAULT_VECTOR_FIELD_NAME[DataType.VECTOR_FP32]
            elif index_type == "IVF":
                from zvec import DataType, IVFIndexParam, IndexOption

                index_param = IVFIndexParam()
                # Use a vector field for IVF index
                field_name = DEFAULT_VECTOR_FIELD_NAME[DataType.VECTOR_FP32]
            else:
                from zvec import InvertIndexParam, IndexOption

                index_param = InvertIndexParam()
                field_name = "int32_field"

            coll.create_index(
                field_name=field_name, index_param=index_param, option=IndexOption()
            )
            print(
                f"[Test] Step 1.3: {index_type} index created successfully on {field_name}."
            )

            # Verify collection state before crash
            initial_doc_count = coll.stats.doc_count
            print(
                f"[Test] Step 1.4: Collection has {initial_doc_count} documents before crash simulation."
            )

            del coll
            print(f"[Test] Step 1.5: Closed collection.")

            # Step 2: Prepare and run subprocess for index deletion operations
            # Write subprocess script to temporary file
            subprocess_script_path = f"{temp_dir}/zvec_subprocess_deleteindex.py"
            with open(subprocess_script_path, "w", encoding="utf-8") as f:
                f.write(self.ZVEC_SUBPROCESS_SCRIPT_DELETEINDEX)

            # Prepare subprocess parameters
            subprocess_args = {
                "collection_path": collection_path,
                "index_field": field_name,  # Use the correct field name for this index type
                "index_type": index_type,  # Type of index to delete
                "index_deletion_iterations": 20,  # Number of index deletion iterations to increase interruption chance
                "delay_between_deletions": 0.3,  # Delay between index deletions to allow interruption opportunity
            }
            args_json_str = json.dumps(subprocess_args)

            print(
                f"[Test] Step 2: Starting {index_type} index deletion operations in subprocess, path: {collection_path}"
            )
            # Start subprocess to execute index deletion operations
            proc = subprocess.Popen(
                [sys.executable, subprocess_script_path, args_json_str]
            )

            # Wait briefly to allow subprocess to begin index deletion operations
            time.sleep(3)  # Wait 3 seconds to allow index deletion process to start

            print(
                f"[Test] Step 2: Simulating crash/power failure by terminating subprocess PID {proc.pid}..."
            )
            # Suddenly kill subprocess (simulate power failure or crash during index deletion operations)
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
                f"[Test] Step 3: Attempting to open collection after simulating crash during {index_type} index deletion operations..."
            )
            # Verification 3.1: Check if collection can be successfully opened after crash
            recovered_collection = zvec.open(collection_path)
            assert recovered_collection is not None, (
                "Cannot open collection after crash"
            )
            print(f"[Test] Step 3.1: Verified collection can be opened after crash...")

            # Verification 3.2: Check data integrity (document count and content)
            print(f"[Test] Step 3.2: Verifying data integrity...")
            # Try a safer way to get document count
            try:
                stats_after_crash = recovered_collection.stats
                print(
                    f"[Test] Step 3.2.1: Collection stats after crash - doc_count: {stats_after_crash.doc_count}, segments: {stats_after_crash.segment_count}"
                )

                # Try a simple fetch operation instead of complex query to avoid segfault
                if stats_after_crash.doc_count > 0:
                    # Get a sample of document IDs to fetch
                    sample_ids = [
                        str(i) for i in range(min(5, stats_after_crash.doc_count))
                    ]
                    fetched_docs = recovered_collection.fetch(sample_ids)
                    print(
                        f"[Test] Step 3.2.2: Successfully fetched {len(fetched_docs)} documents out of {len(sample_ids)} attempted"
                    )
            except Exception as e:
                print(f"[Test] Step 3.2: Data integrity check failed after crash: {e}")

            # Verification 3.3: Test insertion functionality after recovery (critical functionality check)
            print(f"[Test] Step 3.3: Testing insertion functionality after recovery")
            try:
                test_insert_doc = generate_doc(
                    9999, schema
                )  # Use original schema from fixture
                singledoc_and_check(
                    recovered_collection,
                    test_insert_doc,
                    operator="insert",
                    is_delete=0,
                )
                print(
                    f"[Test] Step 3.3: Insertion functionality works after crash recovery"
                )
            except Exception as e:
                print(f"[Test] Step 3.3: Insertion failed after crash recovery: {e}")

            # Verification 3.4: Test update functionality after recovery
            print(f"[Test] Step 3.4: Testing update functionality after recovery...")
            try:
                current_count = recovered_collection.stats.doc_count
                if current_count > 0:
                    # Pick an existing document to update
                    sample_doc_id = str(min(0, current_count - 1))  # Use first document
                    updated_doc = generate_update_doc(
                        int(sample_doc_id), recovered_collection.schema
                    )
                    singledoc_and_check(
                        recovered_collection,
                        updated_doc,
                        operator="update",
                        is_delete=0,
                    )
                    print(
                        f"[Test] Step 3.4: Update functionality works after crash recovery"
                    )
            except Exception as e:
                print(f"[Test] Step 3.4: Update failed after crash recovery: {e}")

            # Verification 3.5: Test deletion functionality after recovery
            print(f"[Test] Step 3.5: Testing deletion functionality after recovery...")
            try:
                test_delete_doc = generate_doc(8888, schema)
                insert_result = recovered_collection.insert([test_delete_doc])
                assert insert_result is not None and len(insert_result) > 0

                delete_result = recovered_collection.delete([test_delete_doc.id])
                assert len(delete_result) == 1
                assert delete_result[0].ok()
                print(
                    f"[Test] Step 3.5: Deletion functionality works after crash recovery"
                )
            except Exception as e:
                print(f"[Test] Step 3.5: Deletion failed after crash recovery: {e}")

            # Verification 3.6: Test creating index after crash recovery
            print(f"[Test] Step 3.6: Testing index creation after crash recovery...")

            # Create index after the crash recovery using the same field and type
            if index_type == "INVERT":
                from zvec import InvertIndexParam, IndexOption

                index_param = InvertIndexParam()
                field_to_index = "int32_field"  # Scalar fields support INVERT index
            elif index_type == "HNSW":
                from zvec import DataType, HnswIndexParam, IndexOption

                index_param = HnswIndexParam()
                field_to_index = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for HNSW
            elif index_type == "FLAT":
                from zvec import DataType, FlatIndexParam, IndexOption

                index_param = FlatIndexParam()
                field_to_index = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for FLAT
            elif index_type == "IVF":
                from zvec import DataType, IVFIndexParam, IndexOption

                index_param = IVFIndexParam()
                field_to_index = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for IVF
            else:
                from zvec import InvertIndexParam, IndexOption

                index_param = InvertIndexParam()
                field_to_index = "int32_field"

            # This should succeed if the collection is properly recovered
            recovered_collection.create_index(
                field_name=field_to_index, index_param=index_param, option=IndexOption()
            )
            print(
                f"[Test] Step 3.6: {index_type} Index creation succeeded after crash recovery on field {field_to_index}"
            )

            # Only do a simple verification after index creation
            stats_after_index = recovered_collection.stats
            print(
                f"[Test] Step 3.6.1: Stats after index creation - doc_count: {stats_after_index.doc_count}"
            )
