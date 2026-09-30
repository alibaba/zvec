# -*- coding: utf-8 -*-
"""
test_collection_crash_recovery_createindex.py

This script is used to test Zvec's recovery capability after simulating a "power failure" (forced process termination) during index creation.
It first successfully creates a collection in the main process and inserts some documents, then starts a subprocess to open the collection and perform index creation operations.
During the index creation operation, the subprocess is forcibly terminated to simulate a scenario where the Zvec process crashes during index building.
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


# @pytest.mark.skip("Known issue")
class TestCollectionCrashRecoveryCreateIndex:
    """
    Test Zvec collection recovery capability after simulating power failure/process crash during index creation.
    Focus on verifying whether the file remains consistent after interruption of index creation operations,
    and whether it can be reopened and used normally.
    """

    # Script content for subprocess to execute Zvec index creation operations
    # Write this script content to a temporary file and execute it in the subprocess.
    ZVEC_SUBPROCESS_SCRIPT_CREATEINDEX = """
import zvec
import time
import json
import sys
import os


def run_zvec_createindex_operations(args_json_str):
    args = json.loads(args_json_str)
    collection_path = args["collection_path"]
    index_field = args.get("index_field", "int32_field")  # Field to create index on
    index_type = args.get("index_type", "INVERT")  # Type of index to create
    index_creation_iterations = args.get("index_creation_iterations", 10)  # Number of index creation iterations
    delay_between_creations = args.get("delay_between_creations", 0.5)  # Delay between index creations

    print(f"[Subprocess] Starting Zvec create index operations on {collection_path} at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Subprocess] Will create {index_type} index on field '{index_field}', {index_creation_iterations} times")

    try:
        # Open existing collection
        collection = zvec.open(collection_path)
        print(f"[Subprocess] Successfully opened collection.")

        print(f"[Subprocess] Starting {index_creation_iterations} {index_type} index creation operations...")

        # Loop to create indexes multiple times - this increases the chance of interruption during the operation
        for i in range(index_creation_iterations):
            print(f"[Subprocess] Iteration {i+1}/{index_creation_iterations}: Creating {index_type} index on field '{index_field}'...")

            # Create index - this operation can take time and be interrupted
            # Import the required index parameter classes
            if index_type == "INVERT":
                from zvec import InvertIndexParam, IndexOption
                collection.create_index(
                    field_name=index_field, 
                    index_param=InvertIndexParam(), 
                    option=IndexOption()
                )
            elif index_type == "HNSW":
                from zvec import HnswIndexParam, IndexOption
                collection.create_index(
                    field_name=index_field, 
                    index_param=HnswIndexParam(), 
                    option=IndexOption()
                )
            elif index_type == "FLAT":
                from zvec import FlatIndexParam, IndexOption
                collection.create_index(
                    field_name=index_field, 
                    index_param=FlatIndexParam(), 
                    option=IndexOption()
                )
            elif index_type == "IVF":
                from zvec import IVFIndexParam, IndexOption
                collection.create_index(
                    field_name=index_field, 
                    index_param=IVFIndexParam(), 
                    option=IndexOption()
                )
            else:
                print(f"[Subprocess] Unknown index type: {index_type}")
                raise ValueError(f"Unknown index type: {index_type}")

            print(f"[Subprocess] Iteration {i+1}: {index_type} Index creation completed successfully on field '{index_field}'.")

            # Add delay between iterations to allow interruption opportunity
            if i < index_creation_iterations - 1:  # Don't sleep after the last iteration
                print(f"[Subprocess] Waiting {delay_between_creations}s before next index creation...")
                time.sleep(delay_between_creations)

        if hasattr(collection, "close"):
            collection.close()
        else:
            del collection  # Use del as fallback
        print(f"[Subprocess] Closed collection after index creation operations.")

    except Exception as e:
        print(f"[Subprocess] Error during index creation operations: {e}")
        import traceback
        traceback.print_exc()
        # Optionally re-raise or handle differently
        raise  # Re-raising may be useful depending on how parent process responds

    print(f"[Subprocess] Index creation operations completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    args_json_str = sys.argv[1]
    run_zvec_createindex_operations(args_json_str)
"""

    def test_createindex_simulate_crash_during_index_creation_invert(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform INVERT index creation operations.
                  During the index creation operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_createindex_with_crash_recovery(
            full_schema_1024, collection_option, "INVERT"
        )

    def test_createindex_simulate_crash_during_index_creation_hnsw(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform HNSW index creation operations.
                  During the index creation operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_createindex_with_crash_recovery(
            full_schema_1024, collection_option, "HNSW"
        )

    def test_createindex_simulate_crash_during_index_creation_flat(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform FLAT index creation operations.
                  During the index creation operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_createindex_with_crash_recovery(
            full_schema_1024, collection_option, "FLAT"
        )

    def test_createindex_simulate_crash_during_index_creation_ivf(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform IVF index creation operations.
                  During the index creation operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_createindex_with_crash_recovery(
            full_schema_1024, collection_option, "IVF"
        )

    def _test_createindex_with_crash_recovery(
        self, schema, collection_option, index_type
    ):
        """
        Common method to test index creation with crash recovery for different index types.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_path = f"{temp_dir}/test_collection_createindex_crash_recovery_{index_type.lower()}"

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

            # Verify collection state before crash
            initial_doc_count = coll.stats.doc_count
            print(
                f"[Test] Step 1.3: Collection has {initial_doc_count} documents before crash simulation."
            )

            del coll
            print(f"[Test] Step 1.4: Closed collection.")

            # Step 2: Prepare and run subprocess for index creation operations
            # Write subprocess script to temporary file
            subprocess_script_path = f"{temp_dir}/zvec_subprocess_createindex.py"
            with open(subprocess_script_path, "w", encoding="utf-8") as f:
                f.write(self.ZVEC_SUBPROCESS_SCRIPT_CREATEINDEX)

            # Determine the appropriate field for each index type
            if index_type == "INVERT":
                field_for_index = "int32_field"  # Scalar fields support INVERT index
            elif index_type == "HNSW":
                from zvec import DataType

                field_for_index = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for HNSW
            elif index_type == "FLAT":
                from zvec import DataType

                field_for_index = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for FLAT
            elif index_type == "IVF":
                from zvec import DataType

                field_for_index = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for IVF
            else:
                print("index_type is error!")

            # Prepare subprocess parameters
            subprocess_args = {
                "collection_path": collection_path,
                "index_field": field_for_index,  # Use appropriate field for this index type
                "index_type": index_type,  # Type of index to create
                "index_creation_iterations": 20,  # Number of index creation iterations to increase interruption chance
                "delay_between_creations": 0.3,  # Delay between index creations to allow interruption opportunity
            }
            args_json_str = json.dumps(subprocess_args)

            print(
                f"[Test] Step 2: Starting {index_type} index creation operations in subprocess, path: {collection_path}"
            )
            # Start subprocess to execute index creation operations
            proc = subprocess.Popen(
                [sys.executable, subprocess_script_path, args_json_str]
            )

            # Wait briefly to allow subprocess to begin index creation operations
            time.sleep(3)  # Wait 3 seconds to allow indexing process to start

            print(
                f"[Test] Step 2: Simulating crash/power failure by terminating subprocess PID {proc.pid}..."
            )
            # Suddenly kill subprocess (simulate power failure or crash during index creation operations)
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
                f"[Test] Step 3: Attempting to open collection after simulating crash during document insertion operations..."
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
            # We expect some documents to have been successfully inserted before crash
            # The exact number depends on when the crash occurred during the bulk insertion process
            print(f"[Test] Step 3.2: Found {len(query_result)} documents after crash")

            current_count = recovered_collection.stats.doc_count
            assert recovered_collection.stats.doc_count >= 1
            assert len(query_result) <= current_count, (
                f"query_result count = {len(query_result)},stats.doc_count = {recovered_collection.stats.doc_count}"
            )

            # Verify existing documents have correct structure
            if len(query_result) > 0:
                for doc in query_result[:1024]:
                    fetched_docs = recovered_collection.fetch([doc.id])
                    # print("doc.id,fetched_docs:\n")
                    # print(doc.id, fetched_docs)
                    exp_doc = generate_doc(int(doc.id), recovered_collection.schema)
                    assert len(fetched_docs) == 1
                    assert doc.id in fetched_docs
                    assert is_doc_equal(
                        fetched_docs[doc.id], exp_doc, recovered_collection.schema
                    ), f"result doc={fetched_docs},doc_exp={exp_doc}"

            # 3.4: Check if index is complete and query function works properly
            print(f"[Test] Step 3.4: Verifying index integrity and query function...")
            filtered_query = recovered_collection.query(filter=f"int32_field >=-100")
            print(
                f"[Test] Step 3.4.2: Field-filtered query returned {len(filtered_query)} documents"
            )
            assert len(filtered_query) > 0
            for doc in query_result:
                fetched_docs = recovered_collection.fetch([doc.id])
                # print("doc.id,fetched_docs:\n")
                # print(doc.id, fetched_docs)
                exp_doc = generate_doc(int(doc.id), recovered_collection.schema)
                assert len(fetched_docs) == 1
                assert doc.id in fetched_docs
                assert is_doc_equal(
                    fetched_docs[doc.id], exp_doc, recovered_collection.schema
                ), f"result doc={fetched_docs},doc_exp={exp_doc}"

            # Verification 3.5: Test insertion functionality after recovery
            print(f"[Test] Step 3.5.1: Testing insertion functionality after recovery")
            test_insert_doc = generate_doc(
                9999, schema
            )  # Use original schema from fixture
            singledoc_and_check(
                recovered_collection, test_insert_doc, operator="insert", is_delete=0
            )

            # Verification 3.6: Test update functionality after recovery
            print(f"[Test] Step 3.6: Testing update functionality after recovery...")
            updated_doc = generate_update_doc(9999, recovered_collection.schema)
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

            # Verification 3.8: Test creating index after crash recovery
            print(f"[Test] Step 3.8: Testing index creation after crash recovery...")

            # Now try to create an index after the crash recovery
            if index_type == "INVERT":
                from zvec import InvertIndexParam, IndexOption

                index_param = InvertIndexParam()
            elif index_type == "HNSW":
                from zvec import HnswIndexParam, IndexOption

                index_param = HnswIndexParam()
            elif index_type == "FLAT":
                from zvec import FlatIndexParam, IndexOption

                index_param = FlatIndexParam()
            elif index_type == "IVF":
                from zvec import IVFIndexParam, IndexOption

                index_param = IVFIndexParam()
            else:
                from zvec import InvertIndexParam, IndexOption

                index_param = InvertIndexParam()

            # Determine the appropriate field for each index type
            if index_type == "INVERT":
                field_to_recreate = "int32_field"  # Scalar fields support INVERT index
            elif index_type == "HNSW":
                from zvec import DataType

                field_to_recreate = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for HNSW
            elif index_type == "FLAT":
                from zvec import DataType

                field_to_recreate = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for FLAT
            elif index_type == "IVF":
                from zvec import DataType

                field_to_recreate = DEFAULT_VECTOR_FIELD_NAME[
                    DataType.VECTOR_FP32
                ]  # Use vector field for IVF
            else:
                field_to_recreate = "int32_field"  # Default to scalar field

            # This should succeed if the collection is properly recovered
            recovered_collection.create_index(
                field_name=field_to_recreate,
                index_param=index_param,
                option=IndexOption(),
            )
            print(
                f"[Test] Step 3.8: {index_type} Index creation succeeded after crash recovery on field {field_to_recreate}"
            )

            # Only do a simple verification after index creation
            stats_after_index = recovered_collection.stats
            print(
                f"[Test] Step 3.8.1: Stats after index creation - doc_count: {stats_after_index.doc_count}"
            )

            # 3.9: Check if index is complete and query function works properly
            print(f"[Test] Step 3.9: Verifying index integrity and query function...")
            # Use a simpler query that matches the field type
            if index_type == "INVERT":
                # Query on scalar field
                filtered_query = recovered_collection.query(
                    filter=f"int32_field >= 0", topk=10
                )
                print(
                    f"[Test] Step 3.9.1: Field-filtered query returned {len(filtered_query)} documents"
                )
                assert len(filtered_query) > 0
            elif index_type in ["HNSW", "FLAT", "IVF"]:
                # Query on vector field using vector search
                import random

                test_vector = [
                    random.random() for _ in range(VECTOR_DIMENSION_1024)
                ]  # Assuming 1024-dim vector
                vector_query_result = recovered_collection.query(
                    VectorQuery(field_name=field_to_recreate, vector=test_vector),
                    topk=5,
                )
                print(
                    f"[Test] Step 3.9.1: Vector query returned {len(vector_query_result)} documents"
                )
                assert len(vector_query_result) > 0
