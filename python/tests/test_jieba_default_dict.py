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
import os
import subprocess
import sys
import textwrap

import pytest

# Only the whole phrase is in the external dictionary. The embedded
# dictionary would also index the shorter word in full mode.
_EXTERNAL_MATCHES = {"搜索引擎技术": ["pk_1"], "技术": []}

_JIEBA_CHECK = textwrap.dedent(
    """\
    import json
    import sys

    import zvec
    from zvec.model.param.query import Fts, Query

    config = json.loads(sys.argv[1])
    schema = zvec.CollectionSchema(
        name="fts_jieba_default",
        fields=[
            zvec.FieldSchema("title", zvec.DataType.STRING, nullable=False),
            zvec.FieldSchema(
                "content",
                zvec.DataType.STRING,
                nullable=False,
                # Empty parameters exercise embedded resources without any
                # directory registered by the Python package.
                index_param=zvec.FtsIndexParam(
                    tokenizer_name="jieba",
                    filters=["lowercase"],
                    extra_params=config["extra_params"],
                ),
            ),
        ],
    )
    try:
        collection = zvec.create_and_open(
            path=config["path"],
            schema=schema,
            option=zvec.CollectionOption(read_only=False, enable_mmap=True),
        )
    except ValueError as error:
        if not config["invalid_override"]:
            raise
        assert "jieba" in str(error).lower(), str(error)
        sys.exit(0)
    if config["invalid_override"]:
        collection.close()
        raise AssertionError("An invalid override must not use embedded resources")

    def matching_ids(text):
        return {
            doc.id
            for doc in collection.query(
                queries=Query(field_name="content", fts=Fts(match_string=text)),
                topk=10,
            )
        }

    def check_matches():
        for text, expected in config["expected_matches"].items():
            assert matching_ids(text) == set(expected), text

    try:
        results = collection.insert(
            [
                zvec.Doc(id="pk_1", fields={"title": "t1", "content": "搜索引擎技术"}),
                zvec.Doc(id="pk_2", fields={"title": "t2", "content": "无关文档"}),
            ]
        )
        assert all(result.ok() for result in results)
        check_matches()
        if config["reopen"]:
            collection.flush()
            collection.close()
            collection = zvec.open(config["path"])
            check_matches()
    finally:
        collection.close()
    """
)


def _run_jieba_check(
    tmp_path,
    *,
    env_dict_dir=None,
    extra_params="",
    expected_matches=None,
    invalid_override=False,
    reopen=False,
):
    """Pass dictionary overrides in the environment before loading native code."""
    env = os.environ.copy()
    env.pop("ZVEC_JIEBA_DICT_DIR", None)
    if env_dict_dir is not None:
        env["ZVEC_JIEBA_DICT_DIR"] = str(env_dict_dir)
    config = {
        "path": str(tmp_path / "fts_jieba"),
        "extra_params": extra_params,
        "expected_matches": (
            expected_matches
            if expected_matches is not None
            else {"搜索引擎技术": ["pk_1"], "技术": ["pk_1"]}
        ),
        "invalid_override": invalid_override,
        "reopen": reopen,
    }
    result = subprocess.run(
        [sys.executable, "-c", _JIEBA_CHECK, json.dumps(config)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


def test_jieba_works_without_explicit_dict_path(tmp_path):
    """A fresh import can index and search CJK text without init or dict paths."""
    _run_jieba_check(tmp_path)


def test_embedded_dictionary_works_after_reopen(tmp_path):
    _run_jieba_check(tmp_path, reopen=True)


def test_user_dictionary_supplements_embedded_resources_after_reopen(tmp_path):
    """User words augment the embedded dictionary without replacing it."""
    user_dict_path = tmp_path / "user.dict.utf8"
    user_dict_path.write_text("搜索引擎技术 100000000 n\n", encoding="utf-8")
    _run_jieba_check(
        tmp_path,
        extra_params=json.dumps(
            {"cut_mode": "mix", "user_dict_path": str(user_dict_path)}
        ),
        # The user phrase is indexed as one token, while the other document
        # still uses words from the embedded main dictionary.
        expected_matches={
            "搜索引擎技术": ["pk_1"],
            "技术": [],
            "文档": ["pk_2"],
        },
        reopen=True,
    )


def test_environment_dictionary_override(tmp_path, external_dict_dir):
    """ZVEC_JIEBA_DICT_DIR selects an external default dictionary."""
    _run_jieba_check(
        tmp_path,
        env_dict_dir=external_dict_dir,
        extra_params=json.dumps({"cut_mode": "full"}),
        expected_matches=_EXTERNAL_MATCHES,
    )


def test_field_dictionary_overrides_environment(tmp_path, external_dict_dir):
    _run_jieba_check(
        tmp_path,
        env_dict_dir=tmp_path / "missing_env_dict",
        extra_params=json.dumps(
            {"jieba_dict_dir": str(external_dict_dir), "cut_mode": "full"}
        ),
        expected_matches=_EXTERNAL_MATCHES,
    )


@pytest.mark.parametrize("config_source", ["environment", "field"])
def test_invalid_dictionary_override_does_not_use_embedded_default(
    tmp_path, config_source
):
    missing_dir = str(tmp_path / "missing_jieba_dict")
    _run_jieba_check(
        tmp_path,
        env_dict_dir=missing_dir if config_source == "environment" else None,
        extra_params=(
            json.dumps({"jieba_dict_dir": missing_dir})
            if config_source == "field"
            else ""
        ),
        invalid_override=True,
    )
