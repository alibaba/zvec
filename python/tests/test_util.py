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

from unittest.mock import MagicMock, patch

import pytest
from zvec import require_module


# ----------------------------
# require_module func Test Case
# ----------------------------
def test_require_module_success():
    module = require_module("os")
    assert module is not None
    assert hasattr(module, "path")


def test_require_module_with_submodule_success():
    module = require_module("os.path")
    assert module is not None
    assert hasattr(module, "join")


def test_require_module_import_error():
    with pytest.raises(ImportError) as exc_info:
        require_module("nonexistent_module")

    exception_msg = str(exc_info.value)
    assert "Required package 'nonexistent_module' is not installed." in exception_msg


def test_require_module_with_mitigation_import_error():
    with pytest.raises(ImportError) as exc_info:
        require_module("nonexistent_module.submodule", mitigation="custom_package")

    exception_msg = str(exc_info.value)
    assert "Required package 'custom_package' is not installed." in exception_msg
    assert (
        "Module 'nonexistent_module.submodule' is part of 'nonexistent_module'"
        in exception_msg
    )
    assert "please pip install 'custom_package'." in exception_msg


def test_require_module_submodule_import_error():
    with pytest.raises(ImportError) as exc_info:
        require_module("os.nonexistent_submodule")

    exception_msg = str(exc_info.value)
    assert (
        "Required package 'os.nonexistent_submodule' is not installed." in exception_msg
    )
    assert "Module 'os.nonexistent_submodule' is part of 'os'" in exception_msg
    assert "please pip install 'os'." in exception_msg


@patch("importlib.import_module")
def test_require_module_wraps_original_exception(mock_import_module):
    original_exception = ImportError("Original error")
    mock_import_module.side_effect = original_exception

    with pytest.raises(ImportError) as exc_info:
        require_module("some_module")

    assert exc_info.value.__cause__ is original_exception


@patch("importlib.import_module")
def test_require_module_calls_importlib(mock_import_module):
    mock_module = MagicMock()
    mock_import_module.return_value = mock_module

    result = require_module("test_module")

    mock_import_module.assert_called_once_with("test_module")
    assert result is mock_module
