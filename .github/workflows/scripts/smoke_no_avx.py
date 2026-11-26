"""Exercise DLL loading or Python import and vector search under Intel SDE."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import tempfile
from pathlib import Path


def load_dll() -> None:
    # Locate the native image without importing zvec (and running PyInit).
    spec = importlib.util.find_spec("zvec")
    package = Path(next(iter(spec.submodule_search_locations)))
    native = next(package.glob("_zvec*.pyd"))
    ctypes.WinDLL(str(native))
    print(f"DLL load succeeded: {native.name}")  # noqa: T201


def search() -> None:
    import zvec  # noqa: PLC0415

    schema = zvec.CollectionSchema(
        name="no_avx",
        vectors=[
            zvec.VectorSchema(
                "dense",
                zvec.DataType.VECTOR_FP32,
                dimension=16,
                index_param=zvec.FlatIndexParam(),
            ),
            zvec.VectorSchema(
                "sparse",
                zvec.DataType.SPARSE_VECTOR_FP32,
                index_param=zvec.FlatIndexParam(),
            ),
        ],
    )
    with tempfile.TemporaryDirectory() as tmp:
        collection = zvec.create_and_open(str(Path(tmp) / "collection"), schema)
        collection.insert(
            [
                zvec.Doc(
                    id="a", vectors={"dense": [1.0] * 16, "sparse": {2: 1.0, 4: 2.0}}
                ),
                zvec.Doc(id="b", vectors={"dense": [0.0] * 16, "sparse": {3: 1.0}}),
            ]
        )
        collection.optimize()
        for field, vector in [("dense", [1.0] * 16), ("sparse", {2: 1.0, 4: 2.0})]:
            result = collection.query(zvec.Query(field, vector=vector), topk=1)
            assert result[0].id == "a", result
        collection.close()
    print(f"Import and dense/sparse search succeeded: zvec {zvec.__version__}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll-only", action="store_true")
    parser.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    if args.dll_only:
        load_dll()
    elif args.import_only:
        import zvec

        print(f"Python import succeeded: zvec {zvec.__version__}")  # noqa: T201
    else:
        search()
