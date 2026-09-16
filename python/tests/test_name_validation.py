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

import re

import pytest
import zvec


@pytest.fixture
def collection(tmp_path):
    schema = zvec.CollectionSchema(
        "validation", fields=zvec.FieldSchema("text", zvec.DataType.STRING)
    )
    coll = zvec.create_and_open(str(tmp_path / "collection"), schema)
    try:
        yield coll
    finally:
        coll.destroy()


@pytest.mark.parametrize("name", ["x", "ab", "集合 / 2026 🔎", "集" * 85 + "a"])
def test_collection_names_round_trip(tmp_path, name):
    path = str(tmp_path / "collection")
    schema = zvec.CollectionSchema(
        name, fields=zvec.FieldSchema("text", zvec.DataType.STRING)
    )
    coll = zvec.create_and_open(path, schema)
    try:
        assert coll.schema.name == name
        coll.close()
        coll = zvec.open(path)
        assert coll.schema.name == name
    finally:
        coll.close()


def test_document_ids_round_trip_without_normalization(collection):
    ids = [
        "user:123",
        "https://example.com/articles/42?a=b",
        "订单-2026-🙂",
        "界" * 341 + "a",  # 1024 UTF-8 bytes.
        "doc",
        " doc",
        "doc ",
        " ",
        "é",
        "e\u0301",
    ]
    text = "正文\nsecond line\tvalue"
    statuses = collection.insert(
        [zvec.Doc(id=doc_id, fields={"text": text}) for doc_id in ids]
    )
    assert all(status.ok() for status in statuses)
    collection.flush()
    fetched = collection.fetch(ids)
    assert set(fetched) == set(ids)
    for doc_id in ids:
        assert fetched[doc_id].id == doc_id
        assert fetched[doc_id].field("text") == text


@pytest.mark.parametrize("operation", ["insert", "update", "upsert"])
@pytest.mark.parametrize(
    "doc_id,reason",
    [
        ("", "must not be empty"),
        ("doc\0id", "null character"),
        ("doc\nid", "newline"),
        ("界" * 341 + "ab", "exceeds 1024 bytes (got 1025)"),
        (b"\xff", "not valid UTF-8"),
        ("\ud800", "not valid UTF-8"),
    ],
)
def test_invalid_id_rejects_batch_before_writing(collection, operation, doc_id, reason):
    if operation == "update":
        assert collection.insert(zvec.Doc("valid", fields={"text": "before"})).ok()

    docs = [
        zvec.Doc("valid", fields={"text": "after"}),
        zvec.Doc(doc_id, fields={"text": "invalid"}),
    ]
    with pytest.raises(ValueError) as exc_info:
        getattr(collection, operation)(docs)

    message = str(exc_info.value)
    assert message.startswith("Invalid doc:")
    assert reason in message
    if doc_id == "\ud800":
        # Conversion fails in Python before native validation runs.
        assert "document at index 1" in message
    else:
        assert "document at index" not in message
    assert "offset" not in message
    if doc_id:
        assert "id[" in message
    assert "\n" not in message
    assert "\0" not in message
    fetched = collection.fetch("valid")
    if operation == "update":
        assert fetched["valid"].field("text") == "before"
    else:
        assert fetched == {}


@pytest.mark.parametrize("operation", ["insert", "update", "upsert"])
def test_conversion_type_error_identifies_document_before_writing(
    collection, operation
):
    if operation == "update":
        assert collection.insert(zvec.Doc("valid", fields={"text": "before"})).ok()

    docs = [
        zvec.Doc("valid", fields={"text": "after"}),
        zvec.Doc("invalid", fields={"text": 42}),
    ]
    with pytest.raises(TypeError) as exc_info:
        getattr(collection, operation)(docs)

    message = str(exc_info.value)
    assert message.endswith(" (document at index 1)")
    assert message.count("document at index") == 1
    fetched = collection.fetch("valid")
    if operation == "update":
        assert fetched["valid"].field("text") == "before"
    else:
        assert fetched == {}


@pytest.mark.parametrize(
    "name,reason",
    [
        ("", "must not be empty"),
        ("collection\0name", "null character"),
        ("collection\nname", "newline"),
        ("集" * 86, "exceeds 256 bytes (got 258)"),
    ],
)
def test_invalid_collection_names_report_the_reason(tmp_path, name, reason):
    schema = zvec.CollectionSchema(
        name, fields=zvec.FieldSchema("text", zvec.DataType.STRING)
    )
    with pytest.raises(ValueError) as exc_info:
        zvec.create_and_open(str(tmp_path / "collection"), schema)
    message = str(exc_info.value)
    assert message.startswith("Invalid schema:")
    assert "collection name" in message
    assert reason in message
    assert "offset" not in message
    if name:
        assert "collection name[" in message
    assert "\n" not in message
    assert "\0" not in message


@pytest.mark.parametrize(
    "doc_id,preview",
    [
        ("order\n[123]", r"order\n\[123\]"),
        (b"\xff", r"\xFF"),
        ("x" * 1025, "x" * 32 + "..."),
    ],
)
def test_id_error_includes_safe_preview(collection, doc_id, preview):
    with pytest.raises(ValueError) as exc_info:
        collection.insert(zvec.Doc(doc_id, fields={"text": "value"}))
    assert f"Invalid doc: id[{preview}]" in str(exc_info.value)


def test_long_field_name_and_rejected_rename_preserve_data(tmp_path):
    field_name = "f" * 64
    schema = zvec.CollectionSchema(
        "fields", fields=zvec.FieldSchema(field_name, zvec.DataType.INT32)
    )
    coll = zvec.create_and_open(str(tmp_path / "collection"), schema)
    try:
        assert coll.insert(zvec.Doc("doc", fields={field_name: 42})).ok()
        with pytest.raises(ValueError) as exc_info:
            coll.alter_column(field_name, new_name="f" * 65)
        message = str(exc_info.value)
        assert message.startswith("Invalid schema:")
        assert "exceeds 64 bytes (got 65)" in message
        assert "offset" not in message
        assert coll.schema.field(field_name) is not None
        assert coll.fetch("doc")["doc"].field(field_name) == 42
    finally:
        coll.destroy()


@pytest.mark.parametrize("operation", ["insert", "update", "upsert"])
def test_surrogate_id_has_a_readable_encoding_error(collection, operation):
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            r"Invalid doc: id['\ud800'] is not valid UTF-8 (document at index 0)"
        )
        + "$",
    ):
        getattr(collection, operation)(zvec.Doc("\ud800", fields={"text": "value"}))
    assert collection.stats.doc_count == 0


@pytest.mark.parametrize("kind", ["collection", "field", "vector"])
def test_surrogate_schema_name_has_a_readable_encoding_error(kind):
    with pytest.raises(
        ValueError, match="^Invalid schema: .* is not valid UTF-8$"
    ) as exc_info:
        if kind == "collection":
            zvec.CollectionSchema("\ud800")
        elif kind == "field":
            zvec.FieldSchema("\ud800", zvec.DataType.INT32)
        else:
            zvec.VectorSchema("\ud800", zvec.DataType.VECTOR_FP32, dimension=2)
    assert r"['\ud800']" in str(exc_info.value)


@pytest.mark.parametrize("invalid", [0, False, [], {}, b""])
@pytest.mark.parametrize("argument", ["new_name", "field_schema"])
def test_falsey_alter_arguments_are_not_silently_ignored(tmp_path, invalid, argument):
    schema = zvec.CollectionSchema(
        "fields", fields=zvec.FieldSchema("value", zvec.DataType.INT32)
    )
    coll = zvec.create_and_open(str(tmp_path / "collection"), schema)
    try:
        assert coll.insert(zvec.Doc("doc", fields={"value": 42})).ok()
        kwargs = (
            {
                "new_name": invalid,
                "field_schema": zvec.FieldSchema("renamed", zvec.DataType.INT32),
            }
            if argument == "new_name"
            else {"new_name": "renamed", "field_schema": invalid}
        )
        with pytest.raises(TypeError, match="^Invalid schema:"):
            coll.alter_column("value", **kwargs)
        assert coll.schema.field("value") is not None
        assert coll.schema.field("renamed") is None
        assert coll.fetch("doc")["doc"].field("value") == 42
    finally:
        coll.destroy()


@pytest.mark.parametrize("kind", ["field", "vector"])
@pytest.mark.parametrize("name", ["bad\nname", "x" * 10000])
def test_duplicate_name_errors_are_escaped_and_bounded(kind, name):
    item = (
        zvec.FieldSchema(name, zvec.DataType.INT32)
        if kind == "field"
        else zvec.VectorSchema(name, zvec.DataType.VECTOR_FP32, dimension=2)
    )
    with pytest.raises(ValueError) as exc_info:
        zvec.CollectionSchema("duplicates", **{kind + "s": [item, item]})
    message = str(exc_info.value)
    assert message.startswith("Invalid schema: duplicate")
    assert "\n" not in message
    assert len(message) < 256
    if "\n" in name:
        assert "\\n" in message
    else:
        assert "..." in message


def test_native_schema_rejects_null_field_pointer():
    from zvec._zvec.schema import _CollectionSchema

    with pytest.raises(ValueError, match="^Invalid schema:"):
        _CollectionSchema("fields", [None])
