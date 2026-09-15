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


def explain_utf8_conversion_error(value: object, context: str) -> None:
    """Explain a failed native string conversion without rescanning valid inputs."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"{context} is not valid UTF-8") from None


def format_name_for_error(name: str) -> str:
    """Keep a user-supplied name readable, escaped, and bounded in errors."""
    preview = repr(name[:32])
    return preview + "..." if len(name) > 32 else preview
