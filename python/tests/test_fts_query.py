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
"""Tests for FTS (Full-Text Search) query support in the Python SDK."""

import pickle

import pytest

from zvec.model.param.query import Fts, Query


class TestFtsQueryValidation:
    """Test FTS parameter validation in Query dataclass."""

    def test_fts_query_string_only(self):
        """Query with only query_string in Fts should be valid."""
        q = Query(
            field_name="content", fts=Fts(query_string='+hello -world "exact phrase"')
        )
        q._validate()
        assert q.fts.query_string == '+hello -world "exact phrase"'
        assert q.fts.match_string is None
        assert q.has_fts() is True

    def test_fts_match_string_only(self):
        """Query with only match_string in Fts should be valid."""
        q = Query(field_name="content", fts=Fts(match_string="machine learning"))
        q._validate()
        assert q.fts.match_string == "machine learning"
        assert q.fts.query_string is None
        assert q.has_fts() is True

    def test_fts_query_string_and_match_string_mutually_exclusive(self):
        """Cannot provide both query_string and match_string in Fts."""
        q = Query(
            field_name="content",
            fts=Fts(query_string="+hello", match_string="hello world"),
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            q._validate()

    def test_no_fts(self):
        """Query without FTS fields should have has_fts() == False."""
        q = Query(field_name="embedding", vector=[0.1, 0.2, 0.3])
        assert q.has_fts() is False

    def test_vector_and_fts_mutually_exclusive(self):
        """Cannot combine vector search with FTS in a single Query."""
        q = Query(
            field_name="embedding",
            vector=[0.1, 0.2, 0.3],
            fts=Fts(match_string="deep learning"),
        )
        with pytest.raises(ValueError, match="Cannot combine fts with vector search"):
            q._validate()

    def test_fts_without_vector_or_id(self):
        """Query with only FTS (no vector, no id) should be valid."""
        q = Query(field_name="content", fts=Fts(query_string="hello"))
        q._validate()
        assert q.has_vector() is False
        assert q.has_id() is False
        assert q.has_fts() is True


class TestFtsQueryBinding:
    """Test FTS binding layer (_Fts)."""

    def test_import_fts_query(self):
        """_Fts should be importable from _zvec.param."""
        from _zvec.param import _Fts

        fts = _Fts()
        assert fts.query_string == ""
        assert fts.match_string == ""

    def test_fts_query_set_fields(self):
        """Setting fields on _Fts should work."""
        from _zvec.param import _Fts

        fts = _Fts()
        fts.query_string = "+hello -world"
        assert fts.query_string == "+hello -world"

        fts2 = _Fts()
        fts2.match_string = "machine learning"
        assert fts2.match_string == "machine learning"

    def test_fts_query_pickle(self):
        """_Fts should support pickling."""
        from _zvec.param import _Fts

        fts = _Fts()
        fts.query_string = "+vector search"
        fts.match_string = ""

        data = pickle.dumps(fts)
        restored = pickle.loads(data)
        assert restored.query_string == "+vector search"
        assert restored.match_string == ""

    def test_search_query_fts_field(self):
        """_SearchQuery should have fts field."""
        from _zvec.param import _Fts, _SearchQuery

        vq = _SearchQuery()
        # fts should be None by default (optional)
        assert vq.fts is None

        # set fts
        fts = _Fts()
        fts.query_string = "hello"
        vq.fts = fts
        assert vq.fts is not None
        assert vq.fts.query_string == "hello"

    def test_search_query_pickle_with_fts(self):
        """_SearchQuery with fts should survive pickling."""
        from _zvec.param import _Fts, _SearchQuery

        vq = _SearchQuery()
        vq.topk = 10
        vq.field_name = "embedding"
        fts = _Fts()
        fts.match_string = "test query"
        vq.fts = fts

        data = pickle.dumps(vq)
        restored = pickle.loads(data)
        assert restored.topk == 10
        assert restored.field_name == "embedding"
        assert restored.fts is not None
        assert restored.fts.match_string == "test query"

    def test_search_query_pickle_without_fts(self):
        """_SearchQuery without fts should survive pickling."""
        from _zvec.param import _SearchQuery

        vq = _SearchQuery()
        vq.topk = 5
        vq.field_name = "vec"

        data = pickle.dumps(vq)
        restored = pickle.loads(data)
        assert restored.topk == 5
        assert restored.field_name == "vec"
        assert restored.fts is None
