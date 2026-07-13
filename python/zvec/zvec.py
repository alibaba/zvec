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

from typing import Optional

from zvec._zvec import Initialize, _Collection

from .model import Collection
from .model.param import CollectionOption
from .model.schema import CollectionSchema

__all__ = ["create_and_open", "init", "open"]

from .typing.enum import LogLevel, LogType


def init(
    *,
    log_type: Optional[LogType] = None,
    log_level: Optional[LogLevel] = None,
    log_dir: Optional[str] = "./logs",
    log_basename: Optional[str] = "zvec.log",
    log_file_size: Optional[int] = 2048,
    log_overdue_days: Optional[int] = 7,
    query_threads: Optional[int] = None,
    optimize_threads: Optional[int] = None,
    invert_to_forward_scan_ratio: Optional[float] = None,
    brute_force_by_keys_ratio: Optional[float] = None,
    fts_brute_force_by_keys_ratio: Optional[float] = None,
    memory_limit_mb: Optional[int] = None,
    jieba_dict_dir: Optional[str] = None,
) -> None:
    """Initialize Zvec with configuration options.

    This function must be called before any other operation.
    It can only be called once — subsequent calls raise a ``RuntimeError``.

    Parameters set to ``None`` are **omitted** from the configuration and
    fall back to Zvec's internal defaults, which may be derived from the runtime
    environment (e.g., cgroup CPU/memory limits). Explicitly provided values
    always override defaults.

    Args:
        log_type (Optional[LogType], optional): Logger destination.
            - ``LogType.CONSOLE`` (default if omitted or set to this)
            - ``LogType.FILE``
            - If ``None``, uses internal default (currently ``CONSOLE``).
        log_level (Optional[LogLevel], optional): Minimum log severity.
            Default: ``LogLevel.WARN``.
            Accepted values: ``DEBUG``, ``INFO``, ``WARN``, ``ERROR``, ``FATAL``.
            If ``None``, uses internal default (``WARN``).
        log_dir (Optional[str], optional):
            Directory for log files (only used when ``log_type=FILE``).
            Parent directories are **not** created automatically.
            Default: ``"./logs"``.
            If ``None``, internal default is used.
        log_basename (Optional[str], optional):
            Base name for rotated log files (e.g., ``zvec.log.1``, ``zvec.log.2``).
            Default: ``"zvec.log"``.
        log_file_size (Optional[int], optional):
            Max size per log file in **MB** before rotation.
            Default: ``2048`` MB (2 GB).
        log_overdue_days (Optional[int], optional):
            Days to retain rotated log files before deletion.
            Default: ``7`` days.
        query_threads (Optional[int], optional):
            Number of threads for query execution.
            If ``None`` (default), inferred from available CPU cores (via cgroup).
            Must be ≥ 1 if provided.
        optimize_threads (Optional[int], optional):
            Threads for background tasks (e.g., compaction, indexing).
            If ``None``, defaults to same as ``query_threads`` or CPU count.
        invert_to_forward_scan_ratio (Optional[float], optional):
            Threshold to switch from inverted index to full forward scan.
            Range: [0.0, 1.0]. Higher → more aggressive index skipping.
            Default: ``0.9`` (if omitted).
        brute_force_by_keys_ratio (Optional[float], optional):
            Threshold to use brute-force key lookup over index.
            Lower → prefer index; higher → prefer brute-force.
            Range: [0.0, 1.0]. Default: ``0.1``.
        fts_brute_force_by_keys_ratio (Optional[float], optional):
            Threshold to switch FTS scan from posting-driven to
            candidate-driven (brute-force) when the invert filter is
            highly selective. Independent from ``brute_force_by_keys_ratio``
            because per-candidate FTS cost is higher.
            Range: [0.0, 1.0]. Default: ``0.05``.
        memory_limit_mb (Optional[int], optional):
            Soft memory cap in MB. Zvec may throttle or fail operations
            approaching this limit.
            If ``None``, inferred from cgroup memory limit * 0.8 (e.g., in Docker).
            Must be > 0 if provided.
        jieba_dict_dir (Optional[str], optional):
            Override the default directory containing ``jieba.dict.utf8`` and
            ``hmm_model.utf8`` for the jieba FTS tokenizer. When ``None``, the
            value previously registered by ``zvec.set_default_jieba_dict_dir``
            (called automatically on ``import zvec`` to point at the wheel's
            bundled dict) is preserved. JiebaTokenizer also honors the
            ``ZVEC_JIEBA_DICT_DIR`` environment variable and per-field
            ``FtsIndexParam.extra_params.jieba_dict_dir`` ahead of this value.

    Raises:
        RuntimeError: If Zvec is already initialized.
        ValueError: On invalid values (e.g., negative thread count, log level out of range).
        TypeError: If a value has incorrect type (e.g., string for ``query_threads``).

    Note:
        - All ``None`` arguments are **excluded** from the configuration payload,
          allowing the core library to apply environment-aware defaults.
        - This design ensures container-friendliness: in Kubernetes/Docker,
          omitting ``memory_limit_mb`` and thread counts lets Zvec auto-adapt.

    Examples:
        Initialize with defaults (log to console, auto-detect resources):
        >>> import zvec
        >>> zvec.init()

        Customize logging to file with rotation:
        >>> zvec.init(
        ...     log_type=LogType.FILE,
        ...     log_dir="/var/log/zvec",
        ...     log_file_size=1024,
        ...     log_overdue_days=30
        ... )

        Limit resources explicitly:
        >>> zvec.init(
        ...     memory_limit_mb=2048,
        ...     query_threads=4,
        ...     optimize_threads=2
        ... )

        Fine-tune query heuristics:
        >>> zvec.init(
        ...     invert_to_forward_scan_ratio=0.95,
        ...     brute_force_by_keys_ratio=0.05
        ... )
    """
    # Build config dict, skipping None values
    config_dict = {}
    if log_type is not None:
        if not isinstance(log_type, LogType):
            raise TypeError("log_type must be LogType")
        config_dict["log_type"] = log_type.name
    if log_level is not None:
        if not isinstance(log_level, LogLevel):
            raise TypeError("log_level must be LogLevel")
        config_dict["log_level"] = log_level.name
    if log_dir is not None:
        config_dict["log_dir"] = log_dir
    if log_basename is not None:
        config_dict["log_basename"] = log_basename
    if log_file_size is not None:
        config_dict["log_file_size"] = log_file_size
    if log_overdue_days is not None:
        config_dict["log_overdue_days"] = log_overdue_days
    if query_threads is not None:
        config_dict["query_threads"] = query_threads
    if optimize_threads is not None:
        config_dict["optimize_threads"] = optimize_threads
    if invert_to_forward_scan_ratio is not None:
        config_dict["invert_to_forward_scan_ratio"] = invert_to_forward_scan_ratio
    if brute_force_by_keys_ratio is not None:
        config_dict["brute_force_by_keys_ratio"] = brute_force_by_keys_ratio
    if fts_brute_force_by_keys_ratio is not None:
        config_dict["fts_brute_force_by_keys_ratio"] = fts_brute_force_by_keys_ratio
    if memory_limit_mb is not None:
        config_dict["memory_limit_mb"] = memory_limit_mb
    if jieba_dict_dir is not None:
        config_dict["jieba_dict_dir"] = jieba_dict_dir

    Initialize(config_dict)


def create_and_open(
    path: str,
    schema: CollectionSchema,
    option: Optional[CollectionOption] = None,
) -> Collection:
    """Create a new collection and open it for use.

    If a collection already exists at the given path, it may raise an error
    depending on the underlying implementation.

    Args:
        path (str): Path or name of the collection to create.
        schema (CollectionSchema): Schema defining the structure of the collection.
        option (Optional[CollectionOption]): Configuration options
            for opening the collection. Defaults to a default-constructed
            ``CollectionOption()`` if not provided.

    Returns:
        Collection: An opened collection instance ready for operations.

    Examples:
        >>> import zvec
        >>> schema = zvec.CollectionSchema(
        ...     name="my_collection",
        ...     fields=[zvec.FieldSchema("id", zvec.DataType.INT64, nullable=True)]
        ... )
        >>> coll = create_and_open("./my_collection", schema)
    """
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(schema, CollectionSchema):
        raise TypeError("schema must be a CollectionSchema")

    option = option or CollectionOption()
    if not isinstance(option, CollectionOption):
        raise TypeError("option must be a CollectionOption")

    _collection = _Collection.CreateAndOpen(path, schema._get_object(), option)
    return Collection._from_core(_collection)


def open(path: str, option: CollectionOption = CollectionOption()) -> Collection:
    """Open an existing collection from disk.

    The collection must have been previously created with ``create_and_open``.

    Args:
        path (str): Path or name of the existing collection.
        option (CollectionOption, optional): Configuration options
            for opening the collection. Defaults to a default-constructed
            ``CollectionOption()`` if not provided.

    Returns:
        Collection: An opened collection instance.

    Examples:
        >>> import zvec
        >>> coll = zvec.open("./my_collection")
    """
    _collection = _Collection.Open(path, option)
    return Collection._from_core(_collection)
