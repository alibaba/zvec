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
"""End-to-end: jieba FTS works without any user configuration.

`import zvec` is supposed to register the wheel-bundled jieba dict
directory via `set_default_jieba_dict_dir`. With that in place a user can
declare an FTS field with `tokenizer_name="jieba"`, leave `extra_params`
empty, and Chinese full-text search just works.

Falls back to GTEST_SKIP-equivalent when running against a build that did
not bundle the dict (e.g., source-tree dev install without the install
step). In that case CI will rely on the C++ unit tests instead.
"""

from __future__ import annotations

import os
import sys

import pytest
import zvec
from zvec import (
    Collection,
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    FtsIndexParam,
)
from zvec.model.param.query import Fts, Query


def _bundled_dict_dir() -> str:
    """Path zvec.__init__ would have registered; empty when not bundled."""
    return zvec.get_default_jieba_dict_dir()


def _bundled_dict_files_exist() -> bool:
    """Whether the registered default actually contains the dict files.

    `importlib.resources` happily returns a path even when the data dir was
    not installed (e.g. source-tree dev runs); only an installed wheel has
    the files on disk.
    """
    import os

    base = _bundled_dict_dir()
    if not base:
        return False
    return os.path.isfile(os.path.join(base, "jieba.dict.utf8")) and os.path.isfile(
        os.path.join(base, "hmm_model.utf8")
    )


@pytest.fixture(scope="module", autouse=True)
def _require_bundled_dict():
    if not _bundled_dict_files_exist():
        pytest.skip(
            "Bundled jieba dict not found at zvec/data/jieba_dict/ — "
            "this test requires an installed wheel (not a source-tree dev "
            "build without the install step).",
        )


@pytest.fixture(scope="function")
def jieba_collection(tmp_path_factory) -> Collection:
    """FTS-only collection using jieba tokenizer and no explicit dict path."""
    # env-var shadows GlobalConfig in the priority chain.
    if os.environ.get("ZVEC_JIEBA_DICT_DIR"):
        pytest.skip("ZVEC_JIEBA_DICT_DIR shadows the bundled default")
    temp_dir = tmp_path_factory.mktemp("zvec_jieba_default")
    collection_path = temp_dir / "fts_jieba"

    schema = zvec.CollectionSchema(
        name="fts_jieba_default",
        fields=[
            FieldSchema("title", DataType.STRING, nullable=False),
            FieldSchema(
                "content",
                DataType.STRING,
                nullable=False,
                # Deliberately omit extra_params — the bundled default must
                # be picked up via GlobalConfig.jieba_dict_dir.
                index_param=FtsIndexParam(
                    tokenizer_name="jieba",
                    filters=["lowercase"],
                ),
            ),
        ],
    )

    coll = zvec.create_and_open(
        path=str(collection_path),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    assert coll is not None
    try:
        yield coll
    finally:
        try:
            coll.destroy()
        except Exception as e:
            print(f"Warning: failed to destroy collection: {e}")


def test_jieba_works_without_explicit_dict_path(jieba_collection: Collection):
    """User opens collection, inserts CJK doc, searches — no init() / no
    extra_params / no env var / no manual setter call. Just `import zvec`."""
    docs = [
        Doc(id="pk_1", fields={"title": "t1", "content": "中华人民共和国成立"}),
        Doc(id="pk_2", fields={"title": "t2", "content": "无关文档"}),
    ]
    insert_results = jieba_collection.insert(docs)
    assert all(r.ok() for r in insert_results)

    hits = jieba_collection.query(
        queries=Query(field_name="content", fts=Fts(match_string="中华")),
        topk=10,
    )
    ids = {doc.id for doc in hits}
    assert "pk_1" in ids
    assert "pk_2" not in ids


def test_default_dict_dir_is_registered_on_import():
    """Sanity check: zvec.__init__ registered a non-empty default."""
    assert _bundled_dict_dir() != ""


def test_user_can_override_default_at_runtime():
    """zvec.set_default_jieba_dict_dir can be called any time to override."""
    saved = zvec.get_default_jieba_dict_dir()
    try:
        zvec.set_default_jieba_dict_dir("/tmp/zvec/jieba-override")
        assert zvec.get_default_jieba_dict_dir() == "/tmp/zvec/jieba-override"
    finally:
        zvec.set_default_jieba_dict_dir(saved)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.environ writes may not propagate across CRT to zvec.pyd",
)
def test_env_var_overrides_global_config(monkeypatch, tmp_path_factory):
    """ZVEC_JIEBA_DICT_DIR beats GlobalConfig in jieba's resolution chain."""
    bundled = _bundled_dict_dir()
    monkeypatch.setenv("ZVEC_JIEBA_DICT_DIR", bundled)
    saved_global = zvec.get_default_jieba_dict_dir()
    try:
        zvec.set_default_jieba_dict_dir("/zvec/intentionally/missing/global")

        temp_dir = tmp_path_factory.mktemp("zvec_jieba_env")
        schema = zvec.CollectionSchema(
            name="fts_jieba_env",
            fields=[
                FieldSchema("title", DataType.STRING, nullable=False),
                FieldSchema(
                    "content",
                    DataType.STRING,
                    nullable=False,
                    index_param=FtsIndexParam(
                        tokenizer_name="jieba",
                        filters=["lowercase"],
                    ),
                ),
            ],
        )
        coll = zvec.create_and_open(
            path=str(temp_dir / "fts_jieba_env"),
            schema=schema,
            option=CollectionOption(read_only=False, enable_mmap=True),
        )
        assert coll is not None
        try:
            results = coll.insert(
                [
                    Doc(id="pk_1", fields={"title": "t", "content": "搜索引擎技术"}),
                ]
            )
            assert all(r.ok() for r in results)
            hits = coll.query(
                queries=Query(field_name="content", fts=Fts(match_string="搜索")),
                topk=10,
            )
            assert {d.id for d in hits} == {"pk_1"}
        finally:
            coll.destroy()
    finally:
        zvec.set_default_jieba_dict_dir(saved_global)
