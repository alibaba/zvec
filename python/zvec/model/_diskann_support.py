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

from .param import DiskAnnIndexParam

__all__ = ["ensure_diskann_supported", "is_diskann_supported"]

# DiskAnn is currently only built and runnable on Linux x86 platforms. On other
# architectures (e.g. ARM / aarch64) the backend is not compiled in, so a
# DiskAnn index must be rejected at collection / index creation time instead of
# surfacing a confusing failure later during ``optimize()``. This mirrors how
# the RaBitQ index gates on ``platform.machine()``.
_DISKANN_SUPPORTED = platform.system() == "Linux" and platform.machine() in (
    "x86_64",
    "AMD64",
    "i686",
    "i386",
)


def is_diskann_supported() -> bool:
    """Return True if the DiskAnn index is supported on the current platform."""
    return _DISKANN_SUPPORTED


def ensure_diskann_supported(index_param: object) -> None:
    """Validate that a DiskAnn index can run on the current platform.

    The check is performed eagerly at collection / index creation time (e.g.
    ``create_and_open`` / ``create_index``) so that unsupported platforms fail
    fast with a clear message rather than erroring out during ``optimize()``.

    Args:
        index_param: The index configuration to validate. Non-DiskAnn params
            are ignored.

    Raises:
        RuntimeError: If a DiskAnn index is requested on an unsupported
            platform.
    """
    if isinstance(index_param, DiskAnnIndexParam) and not _DISKANN_SUPPORTED:
        raise RuntimeError(
            "DiskAnn index is not supported on this platform "
            f"({platform.system()} {platform.machine()}); it is only "
            "available on Linux x86_64. Please choose a different index "
            "type (e.g. HNSW)."
        )
