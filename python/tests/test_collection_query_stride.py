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
"""Positive-stride public query regression with isolated runtime settings."""

import subprocess
import sys
import textwrap


def test_public_flat_query_positive_stride_matches_contiguous_copy(tmp_path):
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import zvec
        import zvec._zvec as native

        print(f"zvec_path={zvec.__file__}; native_path={native.__file__}", flush=True)

        # Initialize only this fresh child, not the shared pytest worker.
        zvec.init(memory_limit_mb=512, query_threads=1, optimize_threads=1)
        schema = zvec.CollectionSchema(
            name="stride_query",
            vectors=[
                zvec.VectorSchema(
                    "vector",
                    zvec.DataType.VECTOR_FP32,
                    dimension=2,
                    index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.L2),
                )
            ],
        )
        collection = zvec.create_and_open(sys.argv[1], schema)
        try:
            statuses = collection.insert(
                [
                    zvec.Doc(id="logical_match", vectors={"vector": [1.0, 3.0]}),
                    zvec.Doc(
                        id="linear_storage_match", vectors={"vector": [1.0, 99.0]}
                    ),
                ]
            )
            assert len(statuses) == 2
            assert all(status.ok() for status in statuses)
            base = np.array([1, 99, 3, 99], dtype=np.float32)
            view = base[::2]
            assert not view.flags.c_contiguous
            contiguous = view.copy()
            expected = collection.query(zvec.Query("vector", vector=contiguous), topk=1)
            actual = collection.query(zvec.Query("vector", vector=view), topk=1)
            expected_ids = [doc.id for doc in expected]
            actual_ids = [doc.id for doc in actual]
            print(f"contiguous_top1={expected_ids}; strided_top1={actual_ids}", flush=True)
            assert expected_ids == ["logical_match"], expected_ids
            assert actual_ids == expected_ids, (
                f"strided_top1={actual_ids}; contiguous_top1={expected_ids}"
            )
            np.testing.assert_allclose(
                [doc.score for doc in actual], [doc.score for doc in expected]
            )
        finally:
            collection.close()
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "collection")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
