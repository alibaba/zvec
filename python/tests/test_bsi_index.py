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
CACHE_MODES = [
    zvec.BsiCacheMode.RESIDENT,
    zvec.BsiCacheMode.ON_DEMAND,
    zvec.BsiCacheMode.LRU,
]


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


@pytest.mark.parametrize("mode", CACHE_MODES)
def test_bsi_cache_parameters(mode):
    params = zvec.InvertIndexParam(
        layout=LAYOUT.TERM_BSI, bsi_cache_mode=mode, bsi_cache_bytes=123456
    )
    assert params.to_dict()["bsi_cache_mode"] == mode.name
    restored = pickle.loads(pickle.dumps(params))
    assert restored.bsi_cache_mode == mode
    assert restored.bsi_cache_bytes == 123456
    with pytest.raises(ValueError, match="capacity"):
        zvec.InvertIndexParam(
            layout=LAYOUT.TERM_BSI,
            bsi_cache_mode=zvec.BsiCacheMode.LRU,
            bsi_cache_bytes=0,
        )
    with pytest.raises(ValueError, match="TERM_BSI"):
        zvec.InvertIndexParam(
            layout=LAYOUT.TERM_RANGE, bsi_cache_mode=zvec.BsiCacheMode.ON_DEMAND
        )


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


@pytest.mark.parametrize("cache_mode", CACHE_MODES)
@pytest.mark.parametrize("boundary", ["none", "flush", "optimize"])
def test_wal_replay_after_process_exit(tmp_path, cache_mode, boundary):
    path = str(tmp_path / "recovery")
    script = r"""
import os
import sys
import zvec
schema = zvec.CollectionSchema("recovery", fields=[
    zvec.FieldSchema("value", zvec.DataType.INT64, nullable=True,
        index_param=zvec.InvertIndexParam(layout=zvec.InvertIndexLayout.TERM_BSI,
            bsi_cache_mode=getattr(zvec.BsiCacheMode, sys.argv[2])))
])
collection = zvec.create_and_open(sys.argv[1], schema)
assert all(status.ok() for status in collection.insert(
    [zvec.Doc(id=str(i), fields={"value": i}) for i in range(40)]))
if sys.argv[3] == "flush":
    collection.flush()
elif sys.argv[3] == "optimize":
    collection.optimize()
assert all(status.ok() for status in collection.insert(
    [zvec.Doc(id=str(i), fields={"value": i}) for i in range(40, 80)]))
assert collection.upsert(zvec.Doc(id="1", fields={"value": 15})).ok()
assert collection.update(zvec.Doc(id="2", fields={"value": None})).ok()
assert collection.delete("3").ok()
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", script, path, cache_mode.name, boundary],
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


@pytest.mark.parametrize(
    "layout, cache_mode",
    [(LAYOUT.TERM_RANGE, zvec.BsiCacheMode.RESIDENT)]
    + [(LAYOUT.TERM_BSI, mode) for mode in CACHE_MODES],
)
def test_vector_filter_estimate_and_term_recall(tmp_path, layout, cache_mode):
    cache_bytes = 1024 if layout == LAYOUT.TERM_BSI else 64 << 20
    schema = zvec.CollectionSchema(
        "bsi_vector_routes",
        fields=[
            zvec.FieldSchema(
                "value",
                zvec.DataType.INT32,
                nullable=True,
                index_param=zvec.InvertIndexParam(
                    layout=layout,
                    bsi_cache_mode=cache_mode,
                    bsi_cache_bytes=cache_bytes,
                ),
            ),
            zvec.FieldSchema(
                "mirror",
                zvec.DataType.INT32,
                nullable=True,
                index_param=zvec.InvertIndexParam(
                    layout=layout,
                    bsi_cache_mode=cache_mode,
                    bsi_cache_bytes=cache_bytes,
                ),
            ),
        ],
        vectors=[
            zvec.VectorSchema(
                "vec",
                zvec.DataType.VECTOR_FP32,
                8,
                index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.L2),
            )
        ],
    )
    path = str(tmp_path / "collection")
    collection = zvec.create_and_open(path, schema)
    values = {i: None if i % 17 == 0 else i for i in range(1000)}
    assert all(
        status.ok()
        for status in collection.insert(
            [
                zvec.Doc(
                    id=str(i),
                    fields={"value": value, "mirror": value},
                    vectors={"vec": [float(i)] * 8},
                )
                for i, value in values.items()
            ]
        )
    )
    conditions = [
        ("value < 0", lambda x: x < 0),
        ("value < 30", lambda x: x < 30),
        ("value > 250", lambda x: x > 250),
        ("value >= 350 AND value <= 600", lambda x: 350 <= x <= 600),
        ("value = 1 AND value > 1", lambda x: False),
        ("value < 100 AND mirror >= 100", lambda x: False),
        ("value < 100 OR mirror < 200", lambda x: x < 200),
        ("value IN (1, 7, 500) AND mirror >= 7", lambda x: x in (7, 500)),
        (
            "value NOT IN (1, 7, 500) AND mirror < 30",
            lambda x: x < 30 and x not in (1, 7, 500),
        ),
        ("value IS NULL", lambda x: False),
    ]

    def verify():
        if layout == LAYOUT.TERM_BSI:
            assert collection.stats.bsi_memory_bytes["value"] > 0
            assert collection.stats.bsi_memory_bytes["mirror"] > 0
        for expression, predicate in conditions:
            expected = [
                i
                for i, value in values.items()
                if (
                    value is None
                    if expression == "value IS NULL"
                    else value is not None and predicate(value)
                )
            ]
            assert matching_ids(collection, expression) == {str(i) for i in expected}
            docs = collection.query(
                zvec.Query("vec", vector=[0.0] * 8),
                filter=expression,
                topk=21,
            )
            assert [doc.id for doc in docs] == [str(i) for i in expected[:21]]

    # Writing BSI uses samples; sealed fields also have CDF statistics. Small
    # estimates use term IDs, and broad ranges use membership filters.
    verify()
    collection.flush()
    collection.close()
    collection = zvec.open(path)
    verify()
    collection.optimize()
    verify()
    collection.close()
    collection = zvec.open(path, option=zvec.CollectionOption(read_only=True))
    assert collection.schema.field("value").index_param.bsi_cache_mode == cache_mode
    assert collection.schema.field("value").index_param.bsi_cache_bytes == cache_bytes
    verify()
    collection.close()
