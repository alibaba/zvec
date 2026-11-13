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
from __future__ import annotations

import operator
import os
import pickle
import subprocess
import sys

import pytest
import zvec


LAYOUT = zvec.InvertIndexLayout


def matching_ids(collection, expression):
    return {doc.id for doc in collection.query(filter=expression, topk=1000)}


def verify_values(collection, values):
    for bound in [0, 1, 7, 15]:
        for symbol, compare in [
            ("=", operator.eq),
            ("!=", operator.ne),
            ("<", operator.lt),
            ("<=", operator.le),
            (">", operator.gt),
            (">=", operator.ge),
        ]:
            expected = {
                key
                for key, value in values.items()
                if value is not None and compare(value, bound)
            }
            assert matching_ids(collection, f"value {symbol} {bound}") == expected
    assert matching_ids(collection, "value IS NULL") == {
        key for key, value in values.items() if value is None
    }
    assert matching_ids(collection, "value IS NOT NULL") == {
        key for key, value in values.items() if value is not None
    }
    assert matching_ids(collection, "value = 1 OR value > 7") == {
        key
        for key, value in values.items()
        if value is not None and (value == 1 or value > 7)
    }
    assert matching_ids(collection, "value >= 1 AND value <= 7") == {
        key for key, value in values.items() if value is not None and 1 <= value <= 7
    }
    assert matching_ids(collection, "value = 1 AND value > 1") == set()
    assert matching_ids(collection, "value = 7 AND value > 1") == {
        key for key, value in values.items() if value == 7
    }
    assert matching_ids(collection, "value IN (0, 1, 7)") == {
        key for key, value in values.items() if value in (0, 1, 7)
    }
    assert matching_ids(collection, "value NOT IN (0, 1, 7)") == {
        key
        for key, value in values.items()
        if value is not None and value not in (0, 1, 7)
    }


def test_layout_parameters():
    assert zvec.InvertIndexParam().layout == LAYOUT.TERM
    assert zvec.InvertIndexParam(True).layout == LAYOUT.TERM_RANGE
    params = zvec.InvertIndexParam(layout=LAYOUT.TERM_BSI)
    assert params.layout == LAYOUT.TERM_BSI
    assert params.to_dict()["layout"] == "TERM_BSI"
    assert pickle.loads(pickle.dumps(params)).layout == LAYOUT.TERM_BSI
    for flag in (True, False):
        with pytest.raises(ValueError, match="either layout"):
            zvec.InvertIndexParam(flag, layout=LAYOUT.TERM_BSI)
    # Older pickles contain only the two boolean options.
    restored = zvec.InvertIndexParam.__new__(zvec.InvertIndexParam)
    restored.__setstate__((True, False))
    assert restored.layout == LAYOUT.TERM_RANGE


@pytest.mark.parametrize(
    "data_type",
    [
        zvec.DataType.INT32,
        zvec.DataType.INT64,
        zvec.DataType.UINT32,
        zvec.DataType.UINT64,
    ],
)
def test_switch_update_delete_merge_and_reopen(tmp_path, data_type):
    path = str(tmp_path / "collection")
    schema = zvec.CollectionSchema(
        "bsi",
        fields=[
            zvec.FieldSchema(
                "value",
                data_type,
                nullable=True,
                index_param=zvec.InvertIndexParam(True),
            ),
            zvec.FieldSchema(
                "other", zvec.DataType.INT32, index_param=zvec.InvertIndexParam()
            ),
        ],
    )
    collection = zvec.create_and_open(path, schema)
    values = {str(i): None if i % 9 == 0 else i % 16 for i in range(80)}
    statuses = collection.insert(
        [
            zvec.Doc(id=key, fields={"value": value, "other": int(key) % 2})
            for key, value in values.items()
        ]
    )
    assert all(status.ok() for status in statuses)
    verify_values(collection, values)
    for layout in (LAYOUT.TERM_BSI, LAYOUT.TERM, LAYOUT.TERM_RANGE, LAYOUT.TERM_BSI):
        collection.create_index("value", zvec.InvertIndexParam(layout=layout))
        assert collection.schema.field("value").index_param.layout == layout
        verify_values(collection, values)
        collection.close()
        collection = zvec.open(path)
        assert collection.schema.field("value").index_param.layout == layout
        verify_values(collection, values)
    collection.create_index("value", zvec.InvertIndexParam(layout=LAYOUT.TERM_BSI))
    assert collection.upsert(zvec.Doc(id="1", fields={"value": 15, "other": 1})).ok()
    values["1"] = 15
    assert collection.update(zvec.Doc(id="2", fields={"value": None})).ok()
    values["2"] = None
    assert collection.delete("3").ok()
    del values["3"]
    assert collection.insert(zvec.Doc(id="new", fields={"value": 7, "other": 1})).ok()
    values["new"] = 7
    verify_values(collection, values)
    assert matching_ids(collection, "value > 7 AND other = 1") == {
        key
        for key, value in values.items()
        if key != "new" and value is not None and value > 7 and int(key) % 2 == 1
    }
    collection.optimize()
    verify_values(collection, values)
    collection.close()
    collection = zvec.open(path)
    verify_values(collection, values)
    collection.drop_index("value")
    verify_values(collection, values)
    collection.create_index("value", zvec.InvertIndexParam(layout=LAYOUT.TERM_BSI))
    verify_values(collection, values)
    collection.close()


@pytest.mark.parametrize(
    "data_type",
    [
        zvec.DataType.FLOAT,
        zvec.DataType.DOUBLE,
        zvec.DataType.BOOL,
        zvec.DataType.STRING,
        zvec.DataType.ARRAY_INT32,
    ],
)
def test_reject_unsupported_fields(tmp_path, data_type):
    with pytest.raises((ValueError, RuntimeError), match="TERM_BSI"):
        schema = zvec.CollectionSchema(
            "invalid",
            fields=[
                zvec.FieldSchema(
                    "value",
                    data_type,
                    index_param=zvec.InvertIndexParam(layout=LAYOUT.TERM_BSI),
                ),
            ],
        )
        zvec.create_and_open(str(tmp_path / "invalid"), schema)


def test_wal_replay_after_process_exit(tmp_path):
    path = str(tmp_path / "recovery")
    script = r"""
import os
import sys
import zvec
schema = zvec.CollectionSchema("recovery", fields=[
    zvec.FieldSchema("value", zvec.DataType.INT64, nullable=True,
        index_param=zvec.InvertIndexParam(layout=zvec.InvertIndexLayout.TERM_BSI))
])
collection = zvec.create_and_open(sys.argv[1], schema)
assert all(status.ok() for status in collection.insert(
    [zvec.Doc(id=str(i), fields={"value": i}) for i in range(40)]))
collection.flush()
assert all(status.ok() for status in collection.insert(
    [zvec.Doc(id=str(i), fields={"value": i}) for i in range(40, 80)]))
assert collection.upsert(zvec.Doc(id="1", fields={"value": 15})).ok()
assert collection.update(zvec.Doc(id="2", fields={"value": None})).ok()
assert collection.delete("3").ok()
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", script, path],
        check=True,
        timeout=60,
        env=os.environ.copy(),
    )
    values = {str(i): i for i in range(80)}
    values["1"] = 15
    values["2"] = None
    del values["3"]
    for _ in range(2):
        collection = zvec.open(path)
        verify_values(collection, values)
        collection.close()
