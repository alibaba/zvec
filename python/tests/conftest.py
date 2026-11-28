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

import platform
import struct

import pytest


@pytest.fixture(scope="session")
def require_diskann():
    if (
        struct.calcsize("P") != 8
        or platform.system() not in ("Linux", "Darwin", "Windows")
        or platform.machine().lower() not in ("x86_64", "amd64", "aarch64", "arm64")
    ):
        pytest.skip("DiskAnn is supported on 64-bit Linux/macOS/Windows (x86_64/ARM64)")


@pytest.fixture(scope="session")
def require_rabitq(tmp_path_factory):
    import zvec

    schema = zvec.CollectionSchema(
        name="rabitq_probe",
        vectors=[
            zvec.VectorSchema(
                "vec",
                zvec.DataType.VECTOR_FP32,
                64,
                index_param=zvec.IVFIndexParam(
                    quantize_type=zvec.QuantizeType.RABITQ, n_list=1
                ),
            )
        ],
    )
    try:
        collection = zvec.create_and_open(
            path=str(tmp_path_factory.mktemp("rabitq_probe") / "collection"),
            schema=schema,
        )
    except RuntimeError as exc:
        if any(
            reason in str(exc)
            for reason in (
                "RaBitQ is not supported on this platform",
                "RabitQ requires AVX2/FMA, AVX512F/BW/DQ, or NEON to be supported",
            )
        ):
            pytest.skip(str(exc))
        raise
    collection.close()
