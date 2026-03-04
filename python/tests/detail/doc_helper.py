from zvec import CollectionSchema, Doc

from support_helper import *

import numpy as np
from typing import Literal, Optional, Union, Tuple

import random
import string
import math


def generate_constant_vector(
    i: int, dimension: int, dtype: Literal["int8", "float16", "float32"] = "float32"
):
    if dtype == "int8":
        vec = [(i % 127)] * dimension
        vec[i % dimension] = (i + 1) % 127
    else:
        base_val = (i % 1000) / 256.0
        special_val = ((i + 1) % 1000) / 256.0
        vec = [base_val] * dimension
        vec[i % dimension] = special_val

    return vec


def generate_constant_vector_recall(
    i: int, dimension: int, dtype: Literal["int8", "float16", "float32"] = "float32"
):
    if dtype == "int8":
        vec = [(i % 127)] * dimension
        vec[i % dimension] = (i + 1) % 127
    else:
        base_val = math.sin((i) * 1000) / 256.0
        special_val = math.sin((i + 1) * 1000) / 256.0
        vec = [base_val] * dimension
        vec[i % dimension] = special_val

    return vec


def generate_sparse_vector(i: int):
    return {i: i + 0.1}


def generate_vectordict(i: int, schema: CollectionSchema) -> Doc:
    doc_fields = {}
    doc_vectors = {}
    doc_fields = {}
    doc_vectors = {}
    for field in schema.fields:
        if field.data_type == DataType.BOOL:
            doc_fields[field.name] = i % 2 == 0
        elif field.data_type == DataType.INT32:
            doc_fields[field.name] = i
        elif field.data_type == DataType.UINT32:
            doc_fields[field.name] = i
        elif field.data_type == DataType.INT64:
            doc_fields[field.name] = i
        elif field.data_type == DataType.UINT64:
            doc_fields[field.name] = i
        elif field.data_type == DataType.FLOAT:
            doc_fields[field.name] = float(i) + 0.1
        elif field.data_type == DataType.DOUBLE:
            doc_fields[field.name] = float(i) + 0.11
        elif field.data_type == DataType.STRING:
            doc_fields[field.name] = f"test_{i}"
        elif field.data_type == DataType.ARRAY_BOOL:
            doc_fields[field.name] = [i % 2 == 0, i % 3 == 0]
        elif field.data_type == DataType.ARRAY_INT32:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_UINT32:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_INT64:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_UINT64:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_FLOAT:
            doc_fields[field.name] = [float(i + 0.1), float(i + 1.1), float(i + 2.1)]
        elif field.data_type == DataType.ARRAY_DOUBLE:
            doc_fields[field.name] = [float(i + 0.11), float(i + 1.11), float(i + 2.11)]
        elif field.data_type == DataType.ARRAY_STRING:
            doc_fields[field.name] = [f"test_{i}", f"test_{i + 1}", f"test_{i + 2}"]
        else:
            raise ValueError(f"Unsupported field type: {field.data_type}")
    for vector in schema.vectors:
        if vector.data_type == DataType.VECTOR_FP16:
            doc_vectors[vector.name] = generate_constant_vector(
                i, vector.dimension, "float16"
            )
        elif vector.data_type == DataType.VECTOR_FP32:
            doc_vectors[vector.name] = generate_constant_vector(
                i, vector.dimension, "float32"
            )
        elif vector.data_type == DataType.VECTOR_INT8:
            doc_vectors[vector.name] = generate_constant_vector(
                i,
                vector.dimension,
                "int8",
            )
        elif vector.data_type == DataType.SPARSE_VECTOR_FP32:
            doc_vectors[vector.name] = generate_sparse_vector(i)
        elif vector.data_type == DataType.SPARSE_VECTOR_FP16:
            doc_vectors[vector.name] = generate_sparse_vector(i)
        else:
            raise ValueError(f"Unsupported vector type: {vector.data_type}")
    return doc_fields, doc_vectors


def generate_vectordict_recall(i: int, schema: CollectionSchema) -> Doc:
    doc_fields = {}
    doc_vectors = {}
    doc_fields = {}
    doc_vectors = {}
    for field in schema.fields:
        if field.data_type == DataType.BOOL:
            doc_fields[field.name] = i % 2 == 0
        elif field.data_type == DataType.INT32:
            doc_fields[field.name] = i
        elif field.data_type == DataType.UINT32:
            doc_fields[field.name] = i
        elif field.data_type == DataType.INT64:
            doc_fields[field.name] = i
        elif field.data_type == DataType.UINT64:
            doc_fields[field.name] = i
        elif field.data_type == DataType.FLOAT:
            doc_fields[field.name] = float(i) + 0.1
        elif field.data_type == DataType.DOUBLE:
            doc_fields[field.name] = float(i) + 0.11
        elif field.data_type == DataType.STRING:
            doc_fields[field.name] = f"test_{i}"
        elif field.data_type == DataType.ARRAY_BOOL:
            doc_fields[field.name] = [i % 2 == 0, i % 3 == 0]
        elif field.data_type == DataType.ARRAY_INT32:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_UINT32:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_INT64:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_UINT64:
            doc_fields[field.name] = [i, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_FLOAT:
            doc_fields[field.name] = [float(i + 0.1), float(i + 1.1), float(i + 2.1)]
        elif field.data_type == DataType.ARRAY_DOUBLE:
            doc_fields[field.name] = [float(i + 0.11), float(i + 1.11), float(i + 2.11)]
        elif field.data_type == DataType.ARRAY_STRING:
            doc_fields[field.name] = [f"test_{i}", f"test_{i + 1}", f"test_{i + 2}"]
        else:
            raise ValueError(f"Unsupported field type: {field.data_type}")
    for vector in schema.vectors:
        if vector.data_type == DataType.VECTOR_FP16:
            doc_vectors[vector.name] = generate_constant_vector_recall(
                i, vector.dimension, "float16"
            )
        elif vector.data_type == DataType.VECTOR_FP32:
            doc_vectors[vector.name] = generate_constant_vector_recall(
                i, vector.dimension, "float32"
            )
        elif vector.data_type == DataType.VECTOR_INT8:
            doc_vectors[vector.name] = generate_constant_vector_recall(
                i,
                vector.dimension,
                "int8",
            )
        elif vector.data_type == DataType.SPARSE_VECTOR_FP32:
            doc_vectors[vector.name] = generate_sparse_vector(i)
        elif vector.data_type == DataType.SPARSE_VECTOR_FP16:
            doc_vectors[vector.name] = generate_sparse_vector(i)
        else:
            raise ValueError(f"Unsupported vector type: {vector.data_type}")
    return doc_fields, doc_vectors


def generate_vectordict_update(i: int, schema: CollectionSchema) -> Doc:
    doc_fields = {}
    doc_vectors = {}
    doc_fields = {}
    doc_vectors = {}
    for field in schema.fields:
        if field.data_type == DataType.BOOL:
            doc_fields[field.name] = (i + 1) % 2 == 0
        elif field.data_type == DataType.INT32:
            doc_fields[field.name] = i + 1
        elif field.data_type == DataType.UINT32:
            doc_fields[field.name] = i + 1
        elif field.data_type == DataType.INT64:
            doc_fields[field.name] = i + 1
        elif field.data_type == DataType.UINT64:
            doc_fields[field.name] = i + 1
        elif field.data_type == DataType.FLOAT:
            doc_fields[field.name] = float(i + 1) + 0.1
        elif field.data_type == DataType.DOUBLE:
            doc_fields[field.name] = float(i + 1) + 0.11
        elif field.data_type == DataType.STRING:
            doc_fields[field.name] = f"test_{i + 1}"
        elif field.data_type == DataType.ARRAY_BOOL:
            doc_fields[field.name] = [(i + 1) % 2 == 0, (i + 1) % 3 == 0]
        elif field.data_type == DataType.ARRAY_INT32:
            doc_fields[field.name] = [i + 1, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_UINT32:
            doc_fields[field.name] = [i + 1, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_INT64:
            doc_fields[field.name] = [i + 1, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_UINT64:
            doc_fields[field.name] = [i + 1, i + 1, i + 2]
        elif field.data_type == DataType.ARRAY_FLOAT:
            doc_fields[field.name] = [float(i + 1.1), float(i + 2.1), float(i + 3.1)]
        elif field.data_type == DataType.ARRAY_DOUBLE:
            doc_fields[field.name] = [float(i + 1.11), float(i + 2.11), float(i + 3.11)]
        elif field.data_type == DataType.ARRAY_STRING:
            doc_fields[field.name] = [f"test_{i + 1}", f"test_{i + 2}", f"test_{i + 3}"]
        else:
            raise ValueError(f"Unsupported field type: {field.data_type}")
    for vector in schema.vectors:
        if vector.data_type == DataType.VECTOR_FP16:
            doc_vectors[vector.name] = generate_constant_vector(
                i + 1, vector.dimension, "float16"
            )
        elif vector.data_type == DataType.VECTOR_FP32:
            doc_vectors[vector.name] = generate_constant_vector(
                i + 1, vector.dimension, "float32"
            )
        elif vector.data_type == DataType.VECTOR_INT8:
            doc_vectors[vector.name] = generate_constant_vector(
                i + 1,
                vector.dimension,
                "int8",
            )
        elif vector.data_type == DataType.SPARSE_VECTOR_FP32:
            doc_vectors[vector.name] = generate_sparse_vector(i + 1)
        elif vector.data_type == DataType.SPARSE_VECTOR_FP16:
            doc_vectors[vector.name] = generate_sparse_vector(i + 1)
        else:
            raise ValueError(f"Unsupported vector type: {vector.data_type}")
    return doc_fields, doc_vectors


def generate_doc(i: int, schema: CollectionSchema) -> Doc:
    doc_fields = {}
    doc_vectors = {}
    doc_fields, doc_vectors = generate_vectordict(i, schema)
    doc = Doc(id=str(i), fields=doc_fields, vectors=doc_vectors)
    return doc


def generate_doc_recall(i: int, schema: CollectionSchema) -> Doc:
    doc_fields = {}
    doc_vectors = {}
    doc_fields, doc_vectors = generate_vectordict_recall(i, schema)
    doc = Doc(id=str(i), fields=doc_fields, vectors=doc_vectors)
    return doc


def generate_update_doc(i: int, schema: CollectionSchema) -> Doc:
    doc_fields = {}
    doc_vectors = {}
    doc_fields, doc_vectors = generate_vectordict_update(i, schema)
    doc = Doc(id=str(i), fields=doc_fields, vectors=doc_vectors)
    return doc


def generate_doc_random(i, schema: CollectionSchema) -> Doc:
    doc_fields = {}
    doc_vectors = {}

    random.seed(i)

    for field in schema.fields:
        if field.data_type == DataType.BOOL:
            doc_fields[field.name] = random.choice([True, False])
        elif field.data_type == DataType.INT32:
            doc_fields[field.name] = random.randint(-2147483648, 2147483647)
        elif field.data_type == DataType.UINT32:
            doc_fields[field.name] = random.randint(0, 4294967295)
        elif field.data_type == DataType.INT64:
            doc_fields[field.name] = random.randint(
                -9223372036854775808, 9223372036854775807
            )
        elif field.data_type == DataType.UINT64:
            doc_fields[field.name] = random.randint(0, 18446744073709551615)
        elif field.data_type == DataType.FLOAT:
            doc_fields[field.name] = random.uniform(-3.4028235e38, 3.4028235e38)
        elif field.data_type == DataType.DOUBLE:
            doc_fields[field.name] = random.uniform(
                -1.7976931348623157e308, 1.7976931348623157e308
            )
        elif field.data_type == DataType.STRING:
            length = random.randint(1, 999)
            doc_fields[field.name] = "".join(
                random.choices(string.ascii_letters + string.digits, k=length)
            )
        elif field.data_type == DataType.ARRAY_BOOL:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.choice([True, False]) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_INT32:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.randint(-2147483648, 2147483647) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_UINT32:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.randint(0, 4294967295) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_INT64:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.randint(-9223372036854775808, 9223372036854775807)
                for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_UINT64:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.randint(0, 18446744073709551615) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_FLOAT:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.uniform(-3.4028235e38, 3.4028235e38) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_DOUBLE:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.uniform(-1.7976931348623157e308, 1.7976931348623157e308)
                for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_STRING:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                "".join(
                    random.choices(
                        string.ascii_letters + string.digits, k=random.randint(1, 100)
                    )
                )
                for _ in range(array_length)
            ]
        else:
            raise ValueError(f"Unsupported field type: {field.data_type}")

    for vector in schema.vectors:
        if vector.data_type == DataType.VECTOR_FP16:
            doc_vectors[vector.name] = generate_constant_vector(
                random.randint(1, 100), DEFAULT_VECTOR_DIMENSION, "float16"
            )
        elif vector.data_type == DataType.VECTOR_FP32:
            doc_vectors[vector.name] = generate_constant_vector(
                random.randint(1, 100), DEFAULT_VECTOR_DIMENSION, "float32"
            )
        elif vector.data_type == DataType.VECTOR_INT8:
            doc_vectors[vector.name] = generate_constant_vector(
                random.randint(1, 100), DEFAULT_VECTOR_DIMENSION, "int8"
            )
        elif vector.data_type == DataType.SPARSE_VECTOR_FP32:
            doc_vectors[vector.name] = generate_sparse_vector(random.randint(1, 100))
        elif vector.data_type == DataType.SPARSE_VECTOR_FP16:
            doc_vectors[vector.name] = generate_sparse_vector(random.randint(1, 100))
        else:
            raise ValueError(f"Unsupported vector type: {vector.data_type}")

    doc = Doc(id=i, fields=doc_fields, vectors=doc_vectors)
    return doc


def generate_vectordict_random(schema: CollectionSchema):
    doc_fields = {}
    doc_vectors = {}
    for field in schema.fields:
        if field.data_type == DataType.BOOL:
            doc_fields[field.name] = random.choice([True, False])
        elif field.data_type == DataType.INT32:
            doc_fields[field.name] = random.randint(-2147483648, 2147483647)
        elif field.data_type == DataType.UINT32:
            doc_fields[field.name] = random.randint(0, 4294967295)
        elif field.data_type == DataType.INT64:
            doc_fields[field.name] = random.randint(
                -9223372036854775808, 9223372036854775807
            )
        elif field.data_type == DataType.UINT64:
            doc_fields[field.name] = random.randint(0, 18446744073709551615)
        elif field.data_type == DataType.FLOAT:
            doc_fields[field.name] = random.uniform(-3.4028235e38, 3.4028235e38)
        elif field.data_type == DataType.DOUBLE:
            doc_fields[field.name] = random.uniform(
                -1.7976931348623157e308, 1.7976931348623157e308
            )
        elif field.data_type == DataType.STRING:
            length = random.randint(1, 999)
            doc_fields[field.name] = "".join(
                random.choices(string.ascii_letters + string.digits, k=length)
            )
        elif field.data_type == DataType.ARRAY_BOOL:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.choice([True, False]) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_INT32:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.randint(-2147483648, 2147483647) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_UINT32:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.randint(0, 4294967295) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_INT64:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.randint(-9223372036854775808, 9223372036854775807)
                for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_UINT64:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.randint(0, 18446744073709551615) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_FLOAT:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.uniform(-3.4028235e38, 3.4028235e38) for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_DOUBLE:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                random.uniform(-1.7976931348623157e308, 1.7976931348623157e308)
                for _ in range(array_length)
            ]
        elif field.data_type == DataType.ARRAY_STRING:
            array_length = random.randint(0, 10)
            doc_fields[field.name] = [
                "".join(
                    random.choices(
                        string.ascii_letters + string.digits, k=random.randint(1, 100)
                    )
                )
                for _ in range(array_length)
            ]
        else:
            raise ValueError(f"Unsupported field type: {field.data_type}")

    for vector in schema.vectors:
        if vector.data_type == DataType.VECTOR_FP16:
            doc_vectors[vector.name] = generate_constant_vector(
                random.randint(1, 100), vector.dimension, "float16"
            )
        elif vector.data_type == DataType.VECTOR_FP32:
            doc_vectors[vector.name] = generate_constant_vector(
                random.randint(1, 100), vector.dimension, "float32"
            )
        elif vector.data_type == DataType.VECTOR_INT8:
            doc_vectors[vector.name] = generate_constant_vector(
                random.randint(1, 100), vector.dimension, "int8"
            )
        elif vector.data_type == DataType.SPARSE_VECTOR_FP32:
            doc_vectors[vector.name] = generate_sparse_vector(random.randint(1, 100))
        elif vector.data_type == DataType.SPARSE_VECTOR_FP16:
            doc_vectors[vector.name] = generate_sparse_vector(random.randint(1, 100))
        else:
            raise ValueError(f"Unsupported vector type: {vector.data_type}")

    return doc_fields, doc_vectors
