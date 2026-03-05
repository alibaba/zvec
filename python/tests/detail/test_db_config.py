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

import logging
import pytest
import tempfile
import os
import sys
import subprocess

import zvec
import zvec
from zvec import LogType, LogLevel

# Error messages
INITIALIZATION_ERROR_MSG = "initialization failed"
RUNTIME_ERROR_MSG = "RuntimeError"
VALUE_ERROR_MSG = "ValueError"
TYPE_ERROR_MSG = "TypeError"


# ==================== helper ====================
def run_in_subprocess(func):
    def wrapper(*args, **kwargs):
        if os.getenv("RUNNING_IN_SUBPROCESS"):
            return func(*args, **kwargs)

        env = os.environ.copy()
        env["RUNNING_IN_SUBPROCESS"] = "1"
        env["PYTEST_CURRENT_TEST"] = func.__name__

        import inspect

        filepath = inspect.getfile(func)
        qualname = func.__qualname__.replace(".", "::")
        test_id = f"{filepath}::{qualname}"

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        cmd = [sys.executable, "-m", "pytest", "-v", "-s", test_id]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.fail(
                f"Subprocess test {func.__name__} failed with code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    return wrapper


# ==================== Fixtures ====================
@pytest.fixture(scope="function")
def temp_log_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("logs")


# ==================== Tests ====================
class TestDbConfigInitialization:
    @run_in_subprocess
    def test_init_default(self):
        # default config
        # log_type: Optional[LogType] = LogType.CONSOLE,
        # log_level: Optional[LogLevel] = LogLevel.WARN,
        # log_dir: Optional[str] = "./logs",
        # log_basename: Optional[str] = "zvec.log",
        # log_file_size: Optional[int] = 2048,
        # log_overdue_days: Optional[int] = 7,
        zvec.init()

    @run_in_subprocess
    def test_init_file_logger(self):
        from pathlib import Path
        import shutil

        zvec.init(
            log_level=LogLevel.DEBUG,
            log_type=LogType.FILE,
        )
        # assert logdir exist
        log_dir = Path("./logs")
        assert log_dir.exists()

        # validate write log
        col = zvec.create_and_open(
            "/tmp/test/1",
            zvec.CollectionSchema(
                name="test",
                vectors=zvec.VectorSchema(
                    dimension=4,
                    data_type=zvec.DataType.VECTOR_FP32,
                    name="image",
                ),
            ),
        )
        col.insert(docs=[zvec.Doc(id="1", vectors={"image": [1.0, 2.0, 3.0, 4.0]})])
        assert any(log_dir.glob("zvec.log.*"))

        # clear
        col.destroy()
        shutil.rmtree(log_dir, ignore_errors=True)

    @run_in_subprocess
    def test_init_with_mixed_config(self):
        zvec.init(
            memory_limit_mb=128,
            log_type=LogType.FILE,
            query_threads=1,
            log_level=LogLevel.WARN,
        )

    @run_in_subprocess
    def test_repeated_initialization(self):
        # Calling init() repeatedly is allowed:
        # it succeeds but becomes a no-op after the first successful init()
        zvec.init()


class TestDbConfigMemoryLimitValidation:
    @run_in_subprocess
    def test_memory_limit_min_valid(self):
        # MIN_MEMORY_LIMIT_BYTES is 100M
        with pytest.raises(RuntimeError):
            zvec.init(memory_limit_mb=99)

    @run_in_subprocess
    def test_memory_limit_invalid_value(self):
        # memory_limit_mb must >= 0 and must be int and if None, set default value
        with pytest.raises(ValueError):
            zvec.init(memory_limit_mb=0)
        with pytest.raises(ValueError):
            zvec.init(memory_limit_mb=-1)
        with pytest.raises(TypeError):
            zvec.init(memory_limit_mb="512")
        with pytest.raises(TypeError):
            zvec.init(memory_limit_mb=512.5)


class TestDbConfigThreadValidation:
    @run_in_subprocess
    def test_query_threads(self):
        zvec.init(query_threads=1)

    @run_in_subprocess
    def test_query_threads_invalid(self):
        # query_threads must >= 0 and must be int and if None, set default value
        with pytest.raises(ValueError):
            zvec.init(query_threads=0)
        with pytest.raises(ValueError):
            zvec.init(query_threads=-1)
        with pytest.raises(TypeError):
            zvec.init(query_threads="value")
        with pytest.raises(TypeError):
            zvec.init(query_threads=512.5)
        with pytest.raises(TypeError):
            zvec.init(query_threads="512")

    @run_in_subprocess
    def test_optimize_threads(self):
        zvec.init(optimize_threads=1)

    @run_in_subprocess
    def test_optimize_threads_invalid(self):
        # optimize_threads must >= 0 and must be int and if None, set default value
        with pytest.raises(ValueError):
            zvec.init(optimize_threads=0)
        with pytest.raises(ValueError):
            zvec.init(optimize_threads=-1)
        with pytest.raises(TypeError):
            zvec.init(optimize_threads="value")
        with pytest.raises(TypeError):
            zvec.init(optimize_threads=512.5)
        with pytest.raises(TypeError):
            zvec.init(optimize_threads="512")


class TestDbConfigRatioValidation:
    @run_in_subprocess
    def test_init_invert_to_forward_scan_ratio(self):
        # must be in [0,1]
        zvec.init(invert_to_forward_scan_ratio=0.8)

    @run_in_subprocess
    def test_init_invert_to_forward_scan_ratio_invalid(self):
        with pytest.raises(ValueError):
            zvec.init(invert_to_forward_scan_ratio=1.1)
        with pytest.raises(ValueError):
            zvec.init(invert_to_forward_scan_ratio=-0.1)
        with pytest.raises(TypeError):
            zvec.init(invert_to_forward_scan_ratio="0.8")

    @run_in_subprocess
    def test_init_brute_force_by_keys_ratio(self):
        zvec.init(brute_force_by_keys_ratio=0.8)

    @run_in_subprocess
    def test_init_brute_force_by_keys_ratio_invalid(self):
        with pytest.raises(ValueError):
            zvec.init(brute_force_by_keys_ratio=1.1)
        with pytest.raises(ValueError):
            zvec.init(brute_force_by_keys_ratio=-0.1)
        with pytest.raises(TypeError):
            zvec.init(brute_force_by_keys_ratio="0.8")


class TestDbConfigLogValidation:
    @run_in_subprocess
    def test_log_type_valid(self):
        zvec.init(log_type=LogType.CONSOLE)

    @run_in_subprocess
    def test_log_type_invalid(self):
        with pytest.raises(TypeError):
            zvec.init(log_type="FILE")
        with pytest.raises(TypeError):
            zvec.init(log_type="")
        with pytest.raises(TypeError):
            zvec.init(log_type="invalid")
        with pytest.raises(TypeError):
            zvec.init(log_type=123)

    @run_in_subprocess
    def test_log_level_valid(self):
        zvec.init(log_level=LogLevel.ERROR)

    @run_in_subprocess
    def test_log_level_invalid(self):
        with pytest.raises(TypeError):
            zvec.init(log_level="WARN")
        with pytest.raises(TypeError):
            zvec.init(log_level="")
        with pytest.raises(TypeError):
            zvec.init(log_level="invalid")
        with pytest.raises(TypeError):
            zvec.init(log_level=123)

    @run_in_subprocess
    def test_init_file_logger(self):
        from pathlib import Path
        import shutil

        temp_dir = tempfile.mkdtemp(prefix="log_test_")
        abs_temp_dir = os.path.abspath(temp_dir)

        zvec.init(
            log_level=LogLevel.DEBUG,
            log_type=LogType.FILE,
            log_dir=abs_temp_dir,
            log_basename="test",
        )

        # assert logdir exist
        log_dir = Path(abs_temp_dir)
        assert log_dir.exists()

        # validate write log
        col = zvec.create_and_open(
            "/tmp/test/1",
            zvec.CollectionSchema(
                name="test",
                vectors=zvec.VectorSchema(
                    dimension=4,
                    data_type=zvec.DataType.VECTOR_FP32,
                    name="image",
                ),
            ),
        )
        col.insert(docs=[zvec.Doc(id="1", vectors={"image": [1.0, 2.0, 3.0, 4.0]})])
        assert any(log_dir.glob("test.*"))

        # clear
        col.destroy()
        shutil.rmtree(log_dir, ignore_errors=True)

    @run_in_subprocess
    def test_log_file_size_invalid(self):
        with pytest.raises(TypeError):
            zvec.init(log_type=LogType.FILE, log_file_size="df")

        with pytest.raises(ValueError):
            zvec.init(log_type=LogType.FILE, log_file_size=0)

        with pytest.raises(ValueError):
            zvec.init(log_type=LogType.FILE, log_file_size=-1)

    @run_in_subprocess
    def test_log_overdue_days_invalid(self):
        with pytest.raises(TypeError):
            zvec.init(log_type=LogType.FILE, log_overdue_days="df")

        with pytest.raises(ValueError):
            zvec.init(log_type=LogType.FILE, log_overdue_days=0)

        with pytest.raises(ValueError):
            zvec.init(log_type=LogType.FILE, log_overdue_days=-1)
