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

import os
import pickle
import sys
import time


import numpy as np
import pytest
from zvec import (
    AddColumnOption,
    AlterColumnOption,
    CollectionOption,
    FlatIndexParam,
    HnswIndexParam,
    OmegaIndexParam,
    IndexOption,
    InvertIndexParam,
    IVFIndexParam,
    OptimizeOption,
    HnswQueryParam,
    OmegaQueryParam,
    IVFQueryParam,
    VectorQuery,
    IndexType,
    MetricType,
    QuantizeType,
    DataType,
    VectorSchema,
)

from _zvec.param import _VectorQuery

IS_ANDROID = hasattr(sys, "getandroidapilevel") or "ANDROID_ROOT" in os.environ
OMEGA_ENABLED = os.environ.get("ZVEC_ENABLE_OMEGA", "1").lower() not in {
    "0",
    "off",
    "false",
    "no",
}
OMEGA_AVAILABLE = OMEGA_ENABLED and not IS_ANDROID
OMEGA_ANDROID_SKIP = pytest.mark.skipif(
    not OMEGA_AVAILABLE, reason="OMEGA is disabled on this build/platform"
)

# ----------------------------
# Invert Index Param Test Case
# ----------------------------


class TestInvertIndexParam:
    def test_default(self):
        param = InvertIndexParam()
        assert param.enable_range_optimization is False
        assert param.enable_extended_wildcard is False
        assert param.type == IndexType.INVERT

    def test_custom(self):
        param = InvertIndexParam(
            enable_range_optimization=True, enable_extended_wildcard=True
        )
        assert param.enable_range_optimization is True
        assert param.enable_extended_wildcard is True

    def test_readonly(self):
        param = InvertIndexParam()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            param.enable_range_optimization = False
            param.enable_extended_wildcard = False


# ----------------------------
# Hnsw Index Param Test Case
# ----------------------------


class TestHnswIndexParam:
    def test_default(self):
        param = HnswIndexParam()
        assert param.metric_type == MetricType.IP
        assert param.m == 50
        assert param.ef_construction == 500
        assert param.quantize_type == QuantizeType.UNDEFINED
        assert param.type == IndexType.HNSW

    def test_custom(self):
        param = HnswIndexParam(
            metric_type=MetricType.L2,
            m=10,
            ef_construction=1000,
            quantize_type=QuantizeType.FP16,
        )
        assert param.metric_type == MetricType.L2
        assert param.m == 10
        assert param.ef_construction == 1000
        assert param.quantize_type == QuantizeType.FP16

    @pytest.mark.parametrize(
        "attr", ["metric_type", "m", "ef_construction", "quantize_type"]
    )
    def test_readonly_attributes(self, attr):
        param = HnswIndexParam()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, attr, getattr(param, attr))


# ----------------------------
# Flat Index Param Test Case
# ----------------------------
class TestFlatIndexParam:
    def test_default(self):
        param = FlatIndexParam()
        assert param.type == IndexType.FLAT
        assert param.quantize_type == QuantizeType.UNDEFINED
        assert param.metric_type == MetricType.IP

    def test_custom(self):
        param = FlatIndexParam(
            metric_type=MetricType.L2, quantize_type=QuantizeType.INT8
        )
        assert param.metric_type == MetricType.L2
        assert param.quantize_type == QuantizeType.INT8

    @pytest.mark.parametrize("attr", ["metric_type", "quantize_type"])
    def test_readonly_attributes(self, attr):
        param = FlatIndexParam()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, attr, getattr(param, attr))


# ----------------------------
# Ivf Index Param Test Case
# ----------------------------
class TestIVFIndexParam:
    def test_default(self):
        param = IVFIndexParam()
        assert param.metric_type == MetricType.IP
        assert param.n_list == 0
        assert param.quantize_type == QuantizeType.UNDEFINED
        assert param.type == IndexType.IVF

    def test_custom(self):
        param = IVFIndexParam(
            metric_type=MetricType.L2, n_list=1000, quantize_type=QuantizeType.FP16
        )
        assert param.metric_type == MetricType.L2
        assert param.n_list == 1000
        assert param.quantize_type == QuantizeType.FP16
        assert param.type == IndexType.IVF

    @pytest.mark.parametrize("attr", ["metric_type", "n_list", "quantize_type"])
    def test_readonly_attributes(self, attr):
        param = IVFIndexParam()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, attr, getattr(param, attr))


# ----------------------------
# OMEGA Index Param Test Case
# ----------------------------
@OMEGA_ANDROID_SKIP
class TestOmegaIndexParam:
    def test_default(self):
        param = OmegaIndexParam()
        assert param.type == IndexType.OMEGA
        assert param.metric_type == MetricType.IP
        assert param.m == 50
        assert param.ef_construction == 500
        assert param.quantize_type == QuantizeType.UNDEFINED
        assert param.min_vector_threshold == 100000
        assert param.num_training_queries == 1000
        assert param.ef_training == 1000
        assert param.window_size == 100
        assert param.ef_groundtruth == 0
        assert param.k_train == 1

    def test_custom(self):
        param = OmegaIndexParam(
            metric_type=MetricType.COSINE,
            m=32,
            ef_construction=700,
            quantize_type=QuantizeType.INT8,
            min_vector_threshold=1234,
            num_training_queries=567,
            ef_training=890,
            window_size=42,
            ef_groundtruth=321,
            k_train=3,
        )
        assert param.metric_type == MetricType.COSINE
        assert param.m == 32
        assert param.ef_construction == 700
        assert param.quantize_type == QuantizeType.INT8
        assert param.min_vector_threshold == 1234
        assert param.num_training_queries == 567
        assert param.ef_training == 890
        assert param.window_size == 42
        assert param.ef_groundtruth == 321
        assert param.k_train == 3

    def test_pickle_round_trip(self):
        param = OmegaIndexParam(
            metric_type=MetricType.COSINE,
            m=24,
            ef_construction=320,
            quantize_type=QuantizeType.INT8,
            min_vector_threshold=2048,
            num_training_queries=256,
            ef_training=640,
            window_size=48,
            ef_groundtruth=96,
            k_train=2,
        )
        restored = pickle.loads(pickle.dumps(param))
        assert restored.type == IndexType.OMEGA
        assert restored.metric_type == MetricType.COSINE
        assert restored.m == 24
        assert restored.ef_training == 640
        assert restored.k_train == 2


# ----------------------------
# CollectionOption Test Case
# ----------------------------
class TestCollectionOption:
    def test_default(self):
        option = CollectionOption()
        assert option is not None
        assert option.read_only == False
        assert option.enable_mmap == True

    def test_custom(self):
        option = CollectionOption(read_only=True, enable_mmap=False)
        assert option.read_only == True
        assert option.enable_mmap == False

        option = CollectionOption(read_only=False, enable_mmap=True)
        assert option.read_only == False
        assert option.enable_mmap == True

    @pytest.mark.parametrize("attr", ["read_only", "enable_mmap"])
    def test_readonly_attributes(self, attr):
        param = CollectionOption()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, attr, getattr(param, attr))


# ----------------------------
# IndexOption Test Case
# ----------------------------
class TestIndexOption:
    def test_default(self):
        option = IndexOption()
        assert option is not None
        assert option.concurrency == 0

    def test_custom(self):
        option = IndexOption(concurrency=10)
        assert option.concurrency == 10

    @pytest.mark.parametrize("attr", ["concurrency"])
    def test_readonly_attributes(self, attr):
        param = IndexOption()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, attr, getattr(param, attr))


# ----------------------------
# AddColumnOption Test Case
# ----------------------------
class TestAddColumnOption:
    def test_default(self):
        option = AddColumnOption()
        assert option is not None
        assert option.concurrency == 0

    def test_custom(self):
        option = AddColumnOption(concurrency=10)
        assert option.concurrency == 10

    @pytest.mark.parametrize("attr", ["concurrency"])
    def test_readonly_attributes(self, attr):
        param = AddColumnOption()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, attr, getattr(param, attr))


# ----------------------------
# AlterColumnOption Test Case
# ----------------------------
class TestAlterColumnOption:
    def test_default(self):
        option = AlterColumnOption()
        assert option is not None
        assert option.concurrency == 0

    def test_custom(self):
        option = AlterColumnOption(concurrency=10)
        assert option.concurrency == 10

    @pytest.mark.parametrize("attr", ["concurrency"])
    def test_readonly_attributes(self, attr):
        param = AlterColumnOption()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, attr, getattr(param, attr))


# ----------------------------
# OptimizeOption Test Case
# ----------------------------
class TestOptimizeOption:
    def test_default(self):
        option = OptimizeOption()
        assert option is not None
        assert option.concurrency == 0

    def test_custom(self):
        option = OptimizeOption(concurrency=10)
        assert option.concurrency == 10

    @pytest.mark.parametrize("attr", ["concurrency"])
    def test_readonly_attributes(self, attr):
        param = OptimizeOption()
        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, attr, getattr(param, attr))


# ----------------------------
# HnswQueryParam Test Case
# ----------------------------
class TestHnswQueryParam:
    def test_default(self):
        param = HnswQueryParam()
        assert param is not None
        assert param.ef == 300
        assert param.is_using_refiner == False
        assert param.radius == 0
        assert param.is_linear == False

    def test_custom(self):
        param = HnswQueryParam(ef=10, is_using_refiner=True, radius=30, is_linear=True)
        assert param.ef == 10
        assert param.is_using_refiner == True
        assert param.radius == 30
        assert param.is_linear == True

    def test_readonly_attributes(self):
        param = HnswQueryParam()
        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
            with pytest.raises(AttributeError, match=match_pattern):
                param.ef = 10
                param.is_using_refiner = True
                param.radius = 30
                param.is_linear = True


# ----------------------------
# OMEGA Query Param Test Case
# ----------------------------
@OMEGA_ANDROID_SKIP
class TestOmegaQueryParam:
    def test_default(self):
        param = OmegaQueryParam()
        assert param.type == IndexType.OMEGA
        assert param.ef == 300
        assert param.target_recall == pytest.approx(0.95)
        assert param.radius == pytest.approx(0.0)
        assert param.is_linear is False
        assert param.is_using_refiner is False

    def test_custom(self):
        param = OmegaQueryParam(
            ef=480,
            target_recall=0.92,
            radius=1.5,
            is_linear=True,
            is_using_refiner=True,
        )
        assert param.type == IndexType.OMEGA
        assert param.ef == 480
        assert param.target_recall == pytest.approx(0.92)
        assert param.radius == pytest.approx(1.5)
        assert param.is_linear is True
        assert param.is_using_refiner is True

    def test_pickle_round_trip(self):
        param = OmegaQueryParam(
            ef=384,
            target_recall=0.91,
            radius=0.25,
            is_linear=True,
            is_using_refiner=True,
        )
        restored = pickle.loads(pickle.dumps(param))
        assert restored.type == IndexType.OMEGA
        assert restored.ef == 384
        assert restored.target_recall == pytest.approx(0.91)
        assert restored.radius == pytest.approx(0.25)
        assert restored.is_linear is True
        assert restored.is_using_refiner is True


# # ----------------------------
# # IVFQueryParam Test Case
# # ----------------------------
# class TestIVFQueryParam:
#     def test_default(self):
#         param = IVFQueryParam()
#         assert param is not None
#         assert param.nprobe == 10
#         assert param.is_using_refiner == False
#         assert param.radius == 0
#         assert param.is_linear == False
#         assert param.scale_factor == 10
#
#     def test_custom(self):
#         param = IVFQueryParam(
#             nprobe=20,
#             is_using_refiner=True,
#             radius=30,
#             is_linear=True,
#             scale_factor=40
#         )
#         assert param.nprobe == 20
#         assert param.is_using_refiner == True
#         assert param.radius == 30
#         assert param.is_linear == True
#         assert param.scale_factor == 40


class TestVectorQuery:
    def test_init_with_valid_id(self):
        vq = VectorQuery(field_name="embedding", id="doc123")
        assert vq.field_name == "embedding"
        assert vq.id == "doc123"
        assert vq.vector is None
        assert vq.param is None

    def test_init_with_valid_vector(self):
        vec = [0.1, 0.2, 0.3]
        param = HnswQueryParam(ef=300)
        vq = VectorQuery(field_name="embedding", vector=vec, param=param)
        assert vq.field_name == "embedding"
        assert vq.vector == vec
        assert vq.param == param

    @OMEGA_ANDROID_SKIP
    def test_init_with_valid_omega_param(self):
        vec = [0.1, 0.2, 0.3]
        param = OmegaQueryParam(ef=256, target_recall=0.91)
        vq = VectorQuery(field_name="embedding", vector=vec, param=param)
        assert vq.field_name == "embedding"
        assert vq.vector == vec
        assert vq.param == param
        assert vq.param.target_recall == pytest.approx(0.91)

    def test_init_both_id_and_vector_raises_error(self):
        with pytest.raises(ValueError):
            VectorQuery(field_name="embedding", id="doc123", vector=[0.1])._validate()

    def test_init_without_field_name_raises_error(self):
        with pytest.raises(ValueError):
            VectorQuery(field_name=None)._validate()

    def test_has_id_returns_true_when_id_set(self):
        vq = VectorQuery(field_name="embedding", id="doc123")
        assert vq.has_id()

    def test_has_id_returns_false_when_no_id(self):
        vq = VectorQuery(field_name="embedding", vector=[0.1])
        assert not vq.has_id()

    def test_has_vector_returns_true_with_non_empty_vector(self):
        vq = VectorQuery(field_name="embedding", vector=[0.1])
        assert vq.has_vector()

    def test_validate_fails_on_both_id_and_vector(self):
        vq = VectorQuery(field_name="test", id="doc123", vector=[0.1])
        with pytest.raises(ValueError):
            vq._validate()


@OMEGA_ANDROID_SKIP
class TestVectorSchemaWithOmega:
    def test_accepts_omega_index_param(self):
        schema = VectorSchema(
            name="dense",
            data_type=DataType.VECTOR_FP32,
            dimension=8,
            index_param=OmegaIndexParam(
                metric_type=MetricType.COSINE,
                m=16,
                ef_construction=300,
                window_size=64,
            ),
        )
        assert schema.index_param.type == IndexType.OMEGA
        assert schema.index_param.metric_type == MetricType.COSINE
        assert schema.index_param.window_size == 64
