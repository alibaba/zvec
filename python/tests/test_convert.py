from __future__ import annotations

import math

import pytest
from zvec._zvec import _Doc
from zvec.model.convert import convert_to_py_doc, convert_to_cpp_doc
from zvec import Doc, CollectionSchema, DataType, FieldSchema, VectorSchema


# ----------------------------
# Convert Cpp Doc Test Case
# ----------------------------
class TestConvertCppDoc:
    def test_default(self):
        doc = Doc(id="1")
        schema = CollectionSchema(
            name="test_collection",
            fields=FieldSchema("name", DataType.STRING),
        )

        cpp_doc = convert_to_cpp_doc(doc, collection_schema=schema)

        assert cpp_doc is not None
        assert cpp_doc.pk() == doc.id

    def test_with_field_notin_schema(self):
        doc = Doc(id="1", fields={"name": "Tom"})
        schema = CollectionSchema(
            name="test_collection",
            fields=[
                FieldSchema("id", DataType.UINT64),
                FieldSchema("salary", DataType.UINT32),
                FieldSchema("age", DataType.INT32),
                FieldSchema("create_at", DataType.INT64),
                FieldSchema("author", DataType.STRING),
                FieldSchema("weight", DataType.FLOAT),
            ],
        )
        with pytest.raises(ValueError):
            convert_to_cpp_doc(doc, collection_schema=schema)

    def test_with_scalar_fields(self):
        schema = CollectionSchema(
            name="test_collection",
            fields=[
                FieldSchema("id", DataType.UINT64),
                FieldSchema("salary", DataType.UINT32),
                FieldSchema("age", DataType.INT32),
                FieldSchema("create_at", DataType.INT64),
                FieldSchema("author", DataType.STRING),
                FieldSchema("weight", DataType.FLOAT),
                FieldSchema("bmi", DataType.DOUBLE),
                FieldSchema("is_male", DataType.BOOL),
            ],
        )
        doc = Doc(
            id="1",
            fields={
                "id": 1,
                "salary": 1000,
                "age": 18,
                "create_at": 1640995200,
                "bmi": 80.0 / 200.0,
                "author": "Tom",
                "weight": 80.0,
                "is_male": True,
            },
        )
        cpp_doc = convert_to_cpp_doc(doc, collection_schema=schema)
        assert cpp_doc is not None
        assert cpp_doc.pk() == doc.id
        assert cpp_doc.get_any("id", DataType.UINT64) == 1
        assert cpp_doc.get_any("salary", DataType.UINT32) == 1000
        assert cpp_doc.get_any("age", DataType.INT32) == 18
        assert cpp_doc.get_any("create_at", DataType.INT64) == 1640995200
        assert cpp_doc.get_any("author", DataType.STRING) == "Tom"
        assert math.isclose(
            cpp_doc.get_any("weight", DataType.FLOAT), 80.0, rel_tol=1e-6
        )
        assert math.isclose(
            cpp_doc.get_any("bmi", DataType.DOUBLE), 80.0 / 200.0, rel_tol=1e-6
        )
        assert cpp_doc.get_any("is_male", DataType.BOOL) == True

    def test_with_array_fields(self):
        schema = CollectionSchema(
            name="test_collection",
            fields=[
                FieldSchema("tags", DataType.ARRAY_STRING),
                FieldSchema("ids", DataType.ARRAY_UINT64),
                FieldSchema("marks", DataType.ARRAY_UINT32),
                FieldSchema("x", DataType.ARRAY_INT32),
                FieldSchema("y", DataType.ARRAY_INT64),
                FieldSchema("scores", DataType.ARRAY_FLOAT),
                FieldSchema("ratios", DataType.ARRAY_DOUBLE),
                FieldSchema("results", DataType.ARRAY_BOOL),
            ],
        )

        doc = Doc(
            id="1",
            fields={
                "tags": ["tag1", "tag2", "tag3"],
                "ids": [111111111111, 222222222222, 333333333333],
                "marks": [100, 200, 300],
                "x": [1, 2, 3],
                "y": [100, 200, 300],
                "scores": [1.1, 2.2, 3.3],
                "ratios": [0.1, 0.2, 0.3],
                "results": [True, False, True],
            },
        )
        cpp_doc = convert_to_cpp_doc(doc, collection_schema=schema)

        assert cpp_doc is not None
        assert cpp_doc.pk() == doc.id
        assert cpp_doc.get_any("tags", DataType.ARRAY_STRING) == doc.field("tags")
        assert cpp_doc.get_any("ids", DataType.ARRAY_UINT64) == doc.field("ids")
        assert cpp_doc.get_any("marks", DataType.ARRAY_UINT32) == doc.field("marks")
        assert cpp_doc.get_any("x", DataType.ARRAY_INT32) == doc.field("x")
        assert cpp_doc.get_any("y", DataType.ARRAY_INT64) == doc.field("y")
        scores = cpp_doc.get_any("scores", DataType.ARRAY_FLOAT)
        for i in range(len(doc.field("scores"))):
            assert math.isclose(scores[i], doc.field("scores")[i], rel_tol=1e-1)
        ratios = cpp_doc.get_any("ratios", DataType.ARRAY_DOUBLE)
        for i in range(len(doc.field("ratios"))):
            assert math.isclose(ratios[i], doc.field("ratios")[i], rel_tol=1e-1)
        results = cpp_doc.get_any("results", DataType.ARRAY_BOOL)
        for i in range(len(doc.field("results"))):
            assert results[i] == doc.field("results")[i]

    def test_with_dense_vector_fields(self):
        schema = CollectionSchema(
            name="test_collection",
            vectors=[
                VectorSchema(
                    name="embedding",
                    data_type=DataType.VECTOR_FP16,
                    dimension=4,
                ),
                VectorSchema(
                    name="image",
                    data_type=DataType.VECTOR_FP32,
                    dimension=8,
                ),
                VectorSchema(
                    name="text",
                    data_type=DataType.VECTOR_INT8,
                    dimension=32,
                ),
            ],
        )

        doc = Doc(
            id="1",
            vectors={
                "embedding": [1.1] * 4,
                "image": [2.2] * 8,
                "text": [4] * 32,
            },
        )
        cpp_doc = convert_to_cpp_doc(doc, collection_schema=schema)
        assert cpp_doc is not None
        assert cpp_doc.pk() == doc.id

        embedding_vector = cpp_doc.get_any("embedding", DataType.VECTOR_FP16)
        assert len(embedding_vector) == 4
        for i in range(4):
            assert math.isclose(
                embedding_vector[i], doc.vector("embedding")[i], rel_tol=1e-1
            )

        image_vector = cpp_doc.get_any("image", DataType.VECTOR_FP32)
        assert len(image_vector) == 8
        for i in range(8):
            assert math.isclose(image_vector[i], doc.vector("image")[i], rel_tol=1e-1)

        text_vector = cpp_doc.get_any("text", DataType.VECTOR_INT8)
        assert len(text_vector) == 32
        for i in range(32):
            assert text_vector[i] == doc.vectors["text"][i]

    def test_with_sparse_vector_fields(self):
        schema = CollectionSchema(
            name="test_collection",
            vectors=[
                VectorSchema(
                    name="author",
                    data_type=DataType.SPARSE_VECTOR_FP32,
                ),
                VectorSchema(
                    name="content",
                    data_type=DataType.SPARSE_VECTOR_FP16,
                ),
            ],
        )
        doc = Doc(
            id="1",
            vectors={
                "author": {1: 1.1, 2: 2.2, 3: 3.3},
                "content": {4: 4.4, 5: 5.5, 6: 6.6},
            },
        )

        cpp_doc = convert_to_cpp_doc(doc, collection_schema=schema)
        assert cpp_doc is not None
        assert cpp_doc.pk() == doc.id

        author_vector = cpp_doc.get_any("author", DataType.SPARSE_VECTOR_FP32)
        assert isinstance(author_vector, dict)
        for key, value in doc.vector("author").items():
            assert math.isclose(author_vector[key], value, rel_tol=1e-1)

        content_vector = cpp_doc.get_any("content", DataType.SPARSE_VECTOR_FP16)
        assert isinstance(content_vector, dict)
        for key, value in doc.vector("content").items():
            assert math.isclose(content_vector[key], value, rel_tol=1e-1)

    def test_with_scalar_fields_error_datatype(self):
        schema = CollectionSchema(
            name="test_collection",
            fields=[
                FieldSchema("id", DataType.UINT64),
                FieldSchema("salary", DataType.UINT32),
                FieldSchema("age", DataType.INT32),
                FieldSchema("create_at", DataType.INT64),
                FieldSchema("author", DataType.STRING),
                FieldSchema("weight", DataType.FLOAT),
                FieldSchema("bmi", DataType.DOUBLE),
                FieldSchema("is_male", DataType.BOOL),
            ],
        )
        doc = Doc(
            id="1",
            fields={
                "id": "1",
            },
        )
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"salary": "1000"})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"age": "18"})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"create_at": "2021-01-01"})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"author": 1})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"weight": "80.5"})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"bmi": "25.0"})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"is_male": "true"})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

    def test_with_array_fields_error_datatype(self):
        schema = CollectionSchema(
            name="test_collection",
            fields=[
                FieldSchema("tags", DataType.ARRAY_STRING),
                FieldSchema("ids", DataType.ARRAY_UINT64),
                FieldSchema("marks", DataType.ARRAY_UINT32),
                FieldSchema("x", DataType.ARRAY_INT32),
                FieldSchema("y", DataType.ARRAY_INT64),
                FieldSchema("scores", DataType.ARRAY_FLOAT),
                FieldSchema("ratios", DataType.ARRAY_DOUBLE),
                FieldSchema("results", DataType.ARRAY_BOOL),
            ],
        )

        doc = Doc(id="1", fields={"tags": [1, 2, 3]})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"ids": ["1", "2", "3"]})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"marks": [1.1, 2.2, 3.3]})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"x": [1.1, 2.2, 3.3]})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"y": [1.1, 2.2, 3.3]})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"scores": ["1", "2", "3"]})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"ratios": ["1", "2", "3"]})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", fields={"results": ["1", "2", "3"]})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

    def test_with_vector_fields_error_datatype(self):
        schema = CollectionSchema(
            name="test_collection",
            vectors=[
                VectorSchema(
                    name="embedding",
                    data_type=DataType.VECTOR_FP16,
                    dimension=4,
                ),
                VectorSchema(
                    name="image",
                    data_type=DataType.VECTOR_FP32,
                    dimension=8,
                ),
                VectorSchema(
                    name="text",
                    data_type=DataType.VECTOR_INT8,
                    dimension=32,
                ),
            ],
        )

        doc = Doc(id="1", vectors={"image": ["1.1"] * 4})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", vectors={"text": ["1"] * 4})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(id="1", vectors={"embedding": ["1"] * 4})
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

    def test_with_sparse_vector_error_datatype(self):
        schema = CollectionSchema(
            name="test_collection",
            vectors=[
                VectorSchema(
                    name="author",
                    data_type=DataType.SPARSE_VECTOR_FP32,
                ),
                VectorSchema(
                    name="content",
                    data_type=DataType.SPARSE_VECTOR_FP16,
                ),
            ],
        )
        doc = Doc(
            id="1",
            vectors={
                "author": {"1": 1.1, "2": 2.2, "3": 3.3},
            },
        )
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(
            id="1",
            vectors={
                "content": {"1": 1.1, "2": 2.2, "3": 3.3},
            },
        )
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)

        doc = Doc(
            id="1",
            vectors={
                "author": {1: "1", 2: "2", 3: "3"},
            },
        )
        with pytest.raises(TypeError):
            convert_to_cpp_doc(doc, collection_schema=schema)


# ----------------------------
# Convert Py Doc Test Case
# ----------------------------
class TestConvertPyDoc:
    def test_default(self):
        doc = _Doc()
        doc.set_pk("1")
        doc.set_score(1.0)

        schema = CollectionSchema(
            name="test_collection",
            fields=FieldSchema("name", DataType.STRING),
        )

        py_doc = convert_to_py_doc(doc, schema)
        assert py_doc.id == "1"
        assert py_doc.score == 1.0

    def test_with_scalar_fields(self):
        schema = CollectionSchema(
            name="test_collection",
            fields=[
                FieldSchema("id", DataType.UINT64),
                FieldSchema("salary", DataType.UINT32),
                FieldSchema("age", DataType.INT32),
                FieldSchema("create_at", DataType.INT64),
                FieldSchema("author", DataType.STRING),
                FieldSchema("weight", DataType.FLOAT),
                FieldSchema("bmi", DataType.DOUBLE),
                FieldSchema("is_male", DataType.BOOL),
            ],
        )
        doc = _Doc()
        doc.set_pk("1")
        doc.set_any("id", schema.field("id")._get_object(), 1111111111111111)
        doc.set_any("salary", schema.field("salary")._get_object(), 1000)
        doc.set_any("age", schema.field("age")._get_object(), 18)
        doc.set_any("create_at", schema.field("create_at")._get_object(), 1640995200)
        doc.set_any("author", schema.field("author")._get_object(), "Tom")
        doc.set_any("weight", schema.field("weight")._get_object(), 80.0)
        doc.set_any("bmi", schema.field("bmi")._get_object(), 80.0 / 200.0)
        doc.set_any("is_male", schema.field("is_male")._get_object(), True)

        py_doc = convert_to_py_doc(doc, schema)
        assert py_doc.id == "1"
        assert py_doc.field("id") == 1111111111111111
        assert py_doc.field("salary") == 1000
        assert py_doc.field("age") == 18
        assert py_doc.field("create_at") == 1640995200
        assert py_doc.field("author") == "Tom"
        assert py_doc.field("weight") == 80.0
        assert py_doc.field("bmi") == 80.0 / 200.0
        assert py_doc.field("is_male") == True

    def test_with_array_fields(self):
        schema = CollectionSchema(
            name="test_collection",
            fields=[
                FieldSchema("tags", DataType.ARRAY_STRING),
                FieldSchema("ids", DataType.ARRAY_UINT64),
                FieldSchema("marks", DataType.ARRAY_UINT32),
                FieldSchema("x", DataType.ARRAY_INT32),
                FieldSchema("y", DataType.ARRAY_INT64),
                FieldSchema("scores", DataType.ARRAY_FLOAT),
                FieldSchema("ratios", DataType.ARRAY_DOUBLE),
                FieldSchema("results", DataType.ARRAY_BOOL),
            ],
        )

        doc = _Doc()
        doc.set_pk("1")
        doc.set_any(
            "tags", schema.field("tags")._get_object(), ["tag1", "tag2", "tag3"]
        )
        doc.set_any(
            "ids",
            schema.field("ids")._get_object(),
            [111111111111, 222222222222, 3333333333333],
        )
        doc.set_any("marks", schema.field("marks")._get_object(), [1000, 2000, 3000])
        doc.set_any("x", schema.field("x")._get_object(), [1, 2, 3])
        doc.set_any("y", schema.field("y")._get_object(), [100, 200, 300])
        doc.set_any("scores", schema.field("scores")._get_object(), [0.1, 0.2, 0.3])
        doc.set_any("ratios", schema.field("ratios")._get_object(), [0.1, 0.2, 0.3])
        doc.set_any(
            "results", schema.field("results")._get_object(), [True, False, True]
        )

        py_doc = convert_to_py_doc(doc, schema)
        assert py_doc.field("tags") == ["tag1", "tag2", "tag3"]
        assert py_doc.field("ids") == [111111111111, 222222222222, 3333333333333]
        assert py_doc.field("marks") == [1000, 2000, 3000]
        assert py_doc.field("x") == [1, 2, 3]
        assert py_doc.field("y") == [100, 200, 300]

        scores = doc.get_any("scores", DataType.ARRAY_FLOAT)
        for i in range(len(scores)):
            assert math.isclose(scores[i], py_doc.field("scores")[i], rel_tol=1e-1)
        ratios = doc.get_any("ratios", DataType.ARRAY_DOUBLE)
        for i in range(len(ratios)):
            assert math.isclose(ratios[i], py_doc.field("ratios")[i], rel_tol=1e-1)
        results = doc.get_any("results", DataType.ARRAY_BOOL)
        for i in range(len(results)):
            assert results[i] == py_doc.field("results")[i]

    def test_with_dense_vector_fields(self):
        schema = CollectionSchema(
            name="test_collection",
            vectors=[
                VectorSchema(
                    name="embedding",
                    data_type=DataType.VECTOR_FP16,
                    dimension=4,
                ),
                VectorSchema(
                    name="image",
                    data_type=DataType.VECTOR_FP32,
                    dimension=8,
                ),
                VectorSchema(
                    name="text",
                    data_type=DataType.VECTOR_INT8,
                    dimension=32,
                ),
            ],
        )

        doc = _Doc()
        doc.set_pk("1")
        doc.set_any("embedding", schema.vector("embedding")._get_object(), [1.1] * 4)
        doc.set_any("image", schema.vector("image")._get_object(), [2.2] * 8)
        doc.set_any("text", schema.vector("text")._get_object(), [4] * 32)

        py_doc = convert_to_py_doc(doc, schema)
        assert py_doc.id == "1"

        embedding_vector = py_doc.vector("embedding")
        assert len(embedding_vector) == 4
        for i in range(4):
            assert math.isclose(
                py_doc.vector("embedding")[i], embedding_vector[i], rel_tol=1e-1
            )

        image_vector = py_doc.vector("image")
        assert len(image_vector) == 8
        for i in range(8):
            assert math.isclose(
                py_doc.vector("image")[i], image_vector[i], rel_tol=1e-1
            )

        text_vector = py_doc.vector("text")
        assert len(text_vector) == 32
        for i in range(32):
            assert py_doc.vector("text")[i] == text_vector[i]

    def test_with_sparse_vector_fields(self):
        schema = CollectionSchema(
            name="test_collection",
            vectors=[
                VectorSchema(
                    name="author",
                    data_type=DataType.SPARSE_VECTOR_FP32,
                ),
                VectorSchema(
                    name="content",
                    data_type=DataType.SPARSE_VECTOR_FP16,
                ),
            ],
        )

        doc = _Doc()
        doc.set_pk("1")
        doc.set_any(
            "author", schema.vector("author")._get_object(), {1: 1.1, 2: 2.2, 3: 3.3}
        )
        doc.set_any(
            "content", schema.vector("content")._get_object(), {4: 4.4, 5: 5.5, 6: 6.6}
        )

        py_doc = convert_to_py_doc(doc, schema)
        assert py_doc.id == "1"

        author_vector = py_doc.vector("author")
        assert isinstance(author_vector, dict)
        for key, value in doc.get_any("author", DataType.SPARSE_VECTOR_FP32).items():
            assert math.isclose(author_vector[key], value, rel_tol=1e-1)

        content_vector = py_doc.vector("content")
        assert isinstance(content_vector, dict)
        for key, value in doc.get_any("content", DataType.SPARSE_VECTOR_FP16).items():
            assert math.isclose(content_vector[key], value, rel_tol=1e-1)
