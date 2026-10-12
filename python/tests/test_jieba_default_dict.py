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
"""End-to-end coverage for embedded jieba resources and external overrides.

A user can declare an FTS field with tokenizer_name="jieba", leave
extra_params empty, and search Chinese text without configuring or deploying
any dictionary files. Explicit external directories retain their precedence.
"""

from __future__ import annotations

import json
import subprocess
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


@pytest.fixture(autouse=True)
def isolate_jieba_config(monkeypatch):
    """Keep process-wide dictionary overrides independent between tests."""
    saved = zvec.get_default_jieba_dict_dir()
    # The environment variable takes precedence over the global override.
    monkeypatch.delenv("ZVEC_JIEBA_DICT_DIR", raising=False)
    zvec.set_default_jieba_dict_dir("")
    try:
        yield
    finally:
        zvec.set_default_jieba_dict_dir(saved)


def _create_jieba_collection(path, extra_params: str = "") -> Collection:
    """Create an FTS-only collection with an optional dictionary override."""
    schema = zvec.CollectionSchema(
        name="fts_jieba_default",
        fields=[
            FieldSchema("title", DataType.STRING, nullable=False),
            FieldSchema(
                "content",
                DataType.STRING,
                nullable=False,
                # An empty extra_params must select the embedded default
                # without a directory registered by the Python package.
                index_param=FtsIndexParam(
                    tokenizer_name="jieba",
                    filters=["lowercase"],
                    extra_params=extra_params,
                ),
            ),
        ],
    )
    return zvec.create_and_open(
        path=str(path),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )


@pytest.fixture
def jieba_collection(tmp_path) -> Collection:
    """FTS-only collection using jieba tokenizer and no explicit dict path."""
    coll = _create_jieba_collection(tmp_path / "fts_jieba")
    assert coll is not None
    try:
        yield coll
    finally:
        coll.close()


@pytest.fixture
def external_dict_dir(tmp_path):
    """A small dictionary whose segmentation differs from the embedded one."""
    dict_dir = tmp_path / "external_jieba"
    dict_dir.mkdir()
    (dict_dir / "jieba.dict.utf8").write_text(
        "搜索引擎技术 100 n\n无关文档 100 n\n", encoding="utf-8"
    )
    # A minimal valid model keeps this directory usable by every cut mode.
    emit_probs = "搜:0,索:0,引:0,擎:0,技:0,术:0,无:0,关:0,文:0,档:0\n"
    (dict_dir / "hmm_model.utf8").write_text(
        "0 0 0 0\n" * 5 + emit_probs * 4, encoding="utf-8"
    )
    return dict_dir


def _matching_ids(collection: Collection, text: str) -> set[str]:
    return {
        doc.id
        for doc in collection.query(
            queries=Query(field_name="content", fts=Fts(match_string=text)),
            topk=10,
        )
    }


def _assert_external_dictionary(collection: Collection):
    results = collection.insert(
        [
            Doc(id="pk_1", fields={"title": "t1", "content": "搜索引擎技术"}),
            Doc(id="pk_2", fields={"title": "t2", "content": "无关文档"}),
        ]
    )
    assert all(r.ok() for r in results)
    assert _matching_ids(collection, "搜索引擎技术") == {"pk_1"}
    # Only the whole phrase is in the external dictionary. The embedded
    # dictionary would also index the shorter word in full mode.
    assert _matching_ids(collection, "技术") == set()


def test_jieba_works_without_explicit_dict_path(jieba_collection: Collection):
    """Insert CJK docs and search with no init(), paths, or environment var."""
    docs = [
        Doc(id="pk_1", fields={"title": "t1", "content": "中华人民共和国成立"}),
        Doc(id="pk_2", fields={"title": "t2", "content": "无关文档"}),
    ]
    insert_results = jieba_collection.insert(docs)
    assert all(r.ok() for r in insert_results)
    assert _matching_ids(jieba_collection, "中华") == {"pk_1"}


def test_embedded_dictionary_works_after_reopen(jieba_collection: Collection):
    results = jieba_collection.insert(
        [Doc(id="pk_1", fields={"title": "t", "content": "搜索引擎技术"})]
    )
    assert all(r.ok() for r in results)
    path = jieba_collection.path
    jieba_collection.flush()
    jieba_collection.close()

    reopened = zvec.open(path)
    try:
        assert _matching_ids(reopened, "搜索") == {"pk_1"}
    finally:
        reopened.close()


def test_import_does_not_register_external_dict_directory():
    """A fresh import leaves the external override empty."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import zvec; assert zvec.get_default_jieba_dict_dir() == ''",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_user_can_override_default_at_runtime():
    """The public setter remains available for an external directory."""
    zvec.set_default_jieba_dict_dir("/tmp/zvec/jieba-override")
    assert zvec.get_default_jieba_dict_dir() == "/tmp/zvec/jieba-override"
    zvec.set_default_jieba_dict_dir("")
    assert zvec.get_default_jieba_dict_dir() == ""


def test_global_dictionary_override(tmp_path, external_dict_dir):
    zvec.set_default_jieba_dict_dir(str(external_dict_dir))
    coll = _create_jieba_collection(
        tmp_path / "fts_jieba_global", json.dumps({"cut_mode": "full"})
    )
    try:
        _assert_external_dictionary(coll)
    finally:
        coll.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.environ writes may not propagate across CRT to zvec.pyd",
)
def test_env_var_overrides_global_config(monkeypatch, tmp_path, external_dict_dir):
    """ZVEC_JIEBA_DICT_DIR beats GlobalConfig in jieba's resolution chain."""
    monkeypatch.setenv("ZVEC_JIEBA_DICT_DIR", str(external_dict_dir))
    zvec.set_default_jieba_dict_dir(str(tmp_path / "missing_global_dict"))
    coll = _create_jieba_collection(
        tmp_path / "fts_jieba_env", json.dumps({"cut_mode": "full"})
    )
    try:
        _assert_external_dictionary(coll)
    finally:
        coll.close()


def test_field_dictionary_overrides_global_and_environment(
    monkeypatch, tmp_path, external_dict_dir
):
    zvec.set_default_jieba_dict_dir(str(tmp_path / "missing_global_dict"))
    monkeypatch.setenv("ZVEC_JIEBA_DICT_DIR", str(tmp_path / "missing_env_dict"))
    coll = _create_jieba_collection(
        tmp_path / "fts_jieba_field",
        json.dumps({"jieba_dict_dir": str(external_dict_dir), "cut_mode": "full"}),
    )
    try:
        _assert_external_dictionary(coll)
    finally:
        coll.close()


@pytest.mark.parametrize("config_source", ["global", "environment", "field"])
def test_invalid_dictionary_override_does_not_use_embedded_default(
    monkeypatch, tmp_path, config_source
):
    if config_source == "environment" and sys.platform == "win32":
        pytest.skip("os.environ writes may not propagate across CRT to zvec.pyd")

    missing_dir = str(tmp_path / "missing_jieba_dict")
    extra_params = ""
    if config_source == "global":
        zvec.set_default_jieba_dict_dir(missing_dir)
    elif config_source == "environment":
        monkeypatch.setenv("ZVEC_JIEBA_DICT_DIR", missing_dir)
    else:
        extra_params = json.dumps({"jieba_dict_dir": missing_dir})

    with pytest.raises(ValueError, match="[Jj]ieba"):
        _create_jieba_collection(tmp_path / "fts_jieba_invalid", extra_params)
