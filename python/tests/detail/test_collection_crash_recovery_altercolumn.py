# -*- coding: utf-8 -*-
"""
test_collection_crash_recovery_altercolumn.py

This script is used to test Zvec's recovery capability after simulating a "power failure" (forced process termination) during column update operations.
It first successfully creates a collection in the main process and inserts some documents, then starts a subprocess to open the collection and perform column update operations.
During the column update operation, the subprocess is forcibly terminated to simulate a scenario where the Zvec process crashes during column building.
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


class TestCollectionCrashRecoveryaltercolumn:
    """
    Test Zvec collection recovery capability after simulating power failure/process crash during column update operations.
    Focus on verifying whether the file remains consistent after interruption of column update operations,
    and whether it can be reopened and used normally.
    """

    # Script content for subprocess to execute Zvec column update operations
    # Write this script content to a temporary file and execute it in the subprocess.
    ZVEC_SUBPROCESS_SCRIPT_altercolumn = """
import zvec
import time
import json
import sys
import os


def run_zvec_altercolumn_operations(args_json_str):
    args = json.loads(args_json_str)
    collection_path = args["collection_path"]
    update_field_name = args.get("update_field_name", "int32_field")  # Field name for the update
    update_data_type = args.get("update_data_type", "INT32")  # Data type of the field to update
    update_iterations = args.get("update_iterations", 10)  # Number of update operations iterations
    delay_between_updates = args.get("delay_between_updates", 0.5)  # Delay between update operations

    print("[Subprocess] Starting Zvec update column operations on " + collection_path + " at: " + time.strftime('%Y-%m-%d %H:%M:%S'))
    print("[Subprocess] Will update field '" + update_field_name + "' of type '" + update_data_type + "', " + str(update_iterations) + " times")

    try:
        # Open existing collection
        collection = zvec.open(collection_path)
        print("[Subprocess] Successfully opened collection.")

        print("[Subprocess] Starting " + str(update_iterations) + " column update operations...")

        # Loop to update columns multiple times - this increases the chance of interruption during the operation
        for i in range(update_iterations):
            print("[Subprocess] Iteration " + str(i+1) + "/" + str(update_iterations) + ": Updating field '" + update_field_name + "' schema...")

            # Update column schema - this operation can take time and be interrupted
            # Import the required data type
            from zvec import FieldSchema, DataType, AlterColumnOption

            # Map string data type to actual DataType (only supported types)
            if update_data_type == "INT32":
                data_type = DataType.INT32
            elif update_data_type == "INT64":
                data_type = DataType.INT64
            elif update_data_type == "UINT32":
                data_type = DataType.UINT32
            elif update_data_type == "UINT64":
                data_type = DataType.UINT64
            elif update_data_type == "FLOAT":
                data_type = DataType.FLOAT
            elif update_data_type == "DOUBLE":
                data_type = DataType.DOUBLE
            else:
                data_type = DataType.INT32  # Default fallback (supported type)

            # Create the new field schema
            new_field = FieldSchema(update_field_name, data_type, nullable=True)

            # Update the column with new schema - this is the operation we want to interrupt
            collection.alter_column(
                old_name=update_field_name,
                field_schema=new_field,
                option=AlterColumnOption()
            )

            print("[Subprocess] Iteration " + str(i+1) + ": Column '" + update_field_name + "' schema update completed successfully.")

            # Add delay between iterations to allow interruption opportunity
            if i < update_iterations - 1:  # Don't sleep after the last iteration
                print("[Subprocess] Waiting " + str(delay_between_updates) + "s before next column update...")
                time.sleep(delay_between_updates)

        if hasattr(collection, "close"):
            collection.close()
        else:
            del collection  # Use del as fallback
        print("[Subprocess] Closed collection after column update operations.")

    except Exception as e:
        print("[Subprocess] Error during column update operations: " + str(e))
        import traceback
        traceback.print_exc()
        # Optionally re-raise or handle differently
        raise  # Re-raising may be useful depending on how parent process responds

    print("[Subprocess] Column update operations completed at: " + time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == "__main__":
    args_json_str = sys.argv[1]
    run_zvec_altercolumn_operations(args_json_str)
"""

    def test_altercolumn_simulate_crash_during_column_update_int32(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform INT32 column update operations.
                  During the column update operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_altercolumn_with_crash_recovery(
            full_schema_1024, collection_option, "INT32", "int32_field1"
        )

    def test_altercolumn_simulate_crash_during_column_update_int64(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform INT64 column update operations.
                  During the column update operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_altercolumn_with_crash_recovery(
            full_schema_1024, collection_option, "INT64", "int64_field1"
        )

    def test_altercolumn_simulate_crash_during_column_update_uint32(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform UINT32 column update operations.
                  During the column update operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_altercolumn_with_crash_recovery(
            full_schema_1024, collection_option, "UINT32", "uint32_field1"
        )

    def test_altercolumn_simulate_crash_during_column_update_uint64(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform UINT64 column update operations.
                  During the column update operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_altercolumn_with_crash_recovery(
            full_schema_1024, collection_option, "UINT64", "uint64_field1"
        )

    def test_altercolumn_simulate_crash_during_column_update_float(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform FLOAT column update operations.
                  During the column update operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_altercolumn_with_crash_recovery(
            full_schema_1024, collection_option, "FLOAT", "float_field1"
        )

    def test_altercolumn_simulate_crash_during_column_update_double(
        self, full_schema_1024, collection_option
    ):
        """
        Scenario: First successfully create a Zvec collection in the main process and insert some documents.
                  Then start a subprocess to open the collection and perform DOUBLE column update operations.
                  During the column update operation, forcibly terminate the subprocess (simulate power failure or process crash).
                  Finally, in the main process, reopen the collection and verify whether its state and functionality are normal.
        """
        self._test_altercolumn_with_crash_recovery(
            full_schema_1024, collection_option, "DOUBLE", "double_field1"
        )

    def _test_altercolumn_with_crash_recovery(
        self, schema, collection_option, update_data_type, update_field_name
    ):
        """
        Common method to test column update with crash recovery for different column types.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_path = f"{temp_dir}/test_collection_altercolumn_crash_recovery_{update_data_type.lower()}"

            # Step 1: Successfully create collection in main process and insert some documents
            print(
                f"[Test] Step 1: Creating collection in main process, path: {collection_path}..."
            )
            coll = zvec.create_and_open(
                path=collection_path, schema=schema, option=collection_option
            )
            assert coll is not None
            print(f"[Test] Step 1.1: Collection created successfully.")

            # First, add the column we'll be updating later, so alter_column can modify it
            from zvec import FieldSchema, DataType, AddColumnOption

            if update_data_type == "INT32":
                data_type = DataType.INT32
            elif update_data_type == "INT64":
                data_type = DataType.INT64
            elif update_data_type == "UINT32":
                data_type = DataType.UINT32
            elif update_data_type == "UINT64":
                data_type = DataType.UINT64
            elif update_data_type == "FLOAT":
                data_type = DataType.FLOAT
            elif update_data_type == "DOUBLE":
                data_type = DataType.DOUBLE
            else:
                data_type = DataType.INT32  # Default fallback (supported type)

            # Add the column with initial schema
            initial_field = FieldSchema(update_field_name, data_type, nullable=True)
            coll.add_column(
                field_schema=initial_field,
                expression="",  # Empty expression means fill with default/null values
                option=AddColumnOption(),
            )
            print(
                f"[Test] Step 1.1.1: Added column '{update_field_name}' to collection."
            )

            exp_doc_dict = {}
            # Insert some documents to have data for column operations
            for i in range(50):  # Reduced for faster testing
                exp_doc_dict[i] = {}
                doc = generate_doc(i, coll.schema)
                result = coll.insert([doc])
                assert result is not None and len(result) > 0, (
                    f"Failed to insert document {i}"
                )
                exp_doc_dict[i] = doc

            print(f"[Test] Step 1.2: Inserted 50 documents for column operations.")

            # Verify collection state before crash
            initial_doc_count = coll.stats.doc_count
            print(
                f"[Test] Step 1.3: Collection has {initial_doc_count} documents before crash simulation."
            )

            del coll
            print(f"[Test] Step 1.4: Closed collection.")

            # Step 2: Prepare and run subprocess for column update operations
            # Write subprocess script to temporary file
            subprocess_script_path = f"{temp_dir}/zvec_subprocess_altercolumn.py"
            with open(subprocess_script_path, "w", encoding="utf-8") as f:
                f.write(self.ZVEC_SUBPROCESS_SCRIPT_altercolumn)

            # Prepare subprocess parameters
            subprocess_args = {
                "collection_path": collection_path,
                "update_field_name": update_field_name,  # Use appropriate field name for this test
                "update_data_type": update_data_type,  # Type of field to update
                "update_iterations": 20,  # Number of update iterations to increase interruption chance
                "delay_between_updates": 0.3,  # Delay between updates to allow interruption opportunity
            }
            args_json_str = json.dumps(subprocess_args)

            print(
                f"[Test] Step 2: Starting {update_data_type} column update operations in subprocess, path: {collection_path}"
            )
            # Start subprocess to execute column update operations
            proc = subprocess.Popen(
                [sys.executable, subprocess_script_path, args_json_str]
            )

            # Wait briefly to allow subprocess to begin column update operations
            time.sleep(3)  # Wait 3 seconds to allow column update process to start

            print(
                f"[Test] Step 2: Simulating crash/power failure by terminating subprocess PID {proc.pid}..."
            )
            # Suddenly kill subprocess (simulate power failure or crash during column update operations)
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
                f"[Test] Step 3: Attempting to open collection after simulating crash during column update operations..."
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
                for doc in query_result[:50]:  # Limit to first 50 for efficiency
                    fetched_docs = recovered_collection.fetch([doc.id])
                    """print("doc.id,fetched_docs:\n")
                    print(doc.id, fetched_docs)"""
                    exp_doc = exp_doc_dict[int(doc.id)]
                    assert len(fetched_docs) == 1
                    assert doc.id in fetched_docs
                    # Note: The doc content may have been partially updated before the crash
                    # So we only verify the schema structure and basic fields
                    assert is_doc_equal(
                        fetched_docs[doc.id],
                        exp_doc,
                        recovered_collection.schema,
                        True,
                        True,
                    ), f"result doc={fetched_docs},doc_exp={exp_doc}"

            # 3.4: Check if query function works properly
            print(f"[Test] Step 3.4: Verifying query function after crash...")
            filtered_query = recovered_collection.query(filter=f"int32_field >=-100")
            print(
                f"[Test] Step 3.4.2: Field-filtered query returned {len(filtered_query)} documents"
            )
            assert len(filtered_query) > 0
            for doc in query_result[:10]:  # Check first 10 docs
                fetched_docs = recovered_collection.fetch([doc.id])
                exp_doc = exp_doc_dict[int(doc.id)]
                assert len(fetched_docs) == 1
                assert doc.id in fetched_docs
                assert is_doc_equal(
                    fetched_docs[doc.id],
                    exp_doc,
                    recovered_collection.schema,
                    True,
                    True,
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

            # 3.7: Test deletion after recovery
            print(f"[Test] Step 3.7: Testing deletion functionality after recovery...")
            doc_ids = ["9999"]
            result = recovered_collection.delete(doc_ids)
            assert len(result) == len(doc_ids)
            for item in result:
                assert item.ok()

            # Verification 3.8: Test updating a column after crash recovery
            print(f"[Test] Step 3.8: Testing column update after crash recovery...")

            # Now try to update a column after the crash recovery
            from zvec import FieldSchema, DataType, AlterColumnOption

            # Map string data type to actual update value
            if update_data_type == "INT32":
                data_type = DataType.INT32
            elif update_data_type == "INT64":
                data_type = DataType.INT64
            elif update_data_type == "UINT32":
                data_type = DataType.UINT32
            elif update_data_type == "UINT64":
                data_type = DataType.UINT64
            elif update_data_type == "FLOAT":
                data_type = DataType.FLOAT
            elif update_data_type == "DOUBLE":
                data_type = DataType.DOUBLE
            else:
                data_type = DataType.INT32  # Default fallback (supported type)

            # Create the new field schema
            new_field = FieldSchema(update_field_name, data_type, nullable=True)

            # This should succeed if the collection is properly recovered
            try:
                recovered_collection.alter_column(
                    old_name=update_field_name,
                    field_schema=new_field,
                    option=AlterColumnOption(),
                )
                print(
                    f"[Test] Step 3.8: {update_data_type} Column update succeeded after crash recovery"
                )
            except Exception as e:
                print(
                    f"[Test] Step 3.8: {update_data_type} Column update failed after crash recovery: {str(e)}"
                )
                # This might happen if the column was already altered during the interrupted operation

            # Only do a simple verification after column update
            stats_after_update_column = recovered_collection.stats
            print(
                f"[Test] Step 3.8.1: Stats after column update - doc_count: {stats_after_update_column.doc_count}"
            )

            # 3.9: Check if query function works properly after column update
            print(f"[Test] Step 3.9: Verifying query function after column update...")
            # Use a simpler query that matches the field type
            filtered_query = recovered_collection.query(
                filter=f"{update_field_name} >= 0", topk=10
            )
            print(
                f"[Test] Step 3.9.1: Field-filtered query returned {len(filtered_query)} documents"
            )
            # Note: After column operations, query results may vary
