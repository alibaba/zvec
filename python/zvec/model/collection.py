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

from typing import Optional, Union

from _zvec import _Collection

from ..executor import QueryContext, QueryExecutorFactory
from ..extension import ReRanker
from ..typing import Status
from .convert import convert_to_cpp_doc, convert_to_py_doc
from .doc import Doc
from .param import (
    AddColumnOption,
    AlterColumnOption,
    CollectionOption,
    FlatIndexParam,
    HnswIndexParam,
    IndexOption,
    InvertIndexParam,
    IVFIndexParam,
    OptimizeOption,
)
from .param.vector_query import VectorQuery
from .schema import CollectionSchema, CollectionStats, FieldSchema

__all__ = ["Collection"]

_VECTOR_INDEX_TYPES = (HnswIndexParam, IVFIndexParam, FlatIndexParam)


class Collection:
    """Represents an opened collection in Zvec.

    A `Collection` provides methods for data definition (DDL), data manipulation (DML),
    and querying (DQL). It is obtained via `create_and_open()` or `open()`.

    This class is not meant to be instantiated directly; use factory functions instead.
    """

    def __init__(self, obj: _Collection):
        self._obj = obj
        self._schema = None
        self._querier = None

    @classmethod
    def _from_core(cls, core_collection: _Collection) -> Collection:
        if not core_collection:
            raise ValueError("Collection is None")
        inst = cls.__new__(cls)
        inst._obj = core_collection
        schema = CollectionSchema._from_core(core_collection.Schema())
        inst._schema = schema
        inst._querier = QueryExecutorFactory.create(schema)
        return inst

    @property
    def path(self) -> str:
        """str: The filesystem path of the collection."""
        return self._obj.Path()

    @property
    def option(self) -> CollectionOption:
        """CollectionOption: The options used to open the collection."""
        return self._obj.Options()

    @property
    def schema(self) -> CollectionSchema:
        """CollectionSchema: The schema defining the structure of the collection."""
        return self._schema

    @property
    def stats(self) -> CollectionStats:
        """CollectionStats: Runtime statistics about the collection (e.g., doc count, size)."""
        return self._obj.Stats()

    # ========== Collection DDL Methods ==========
    def destroy(self) -> None:
        """Permanently delete the collection from disk.

        Warning:
            This operation is irreversible. All data will be lost.
        """
        self._obj.Destroy()

    def flush(self) -> None:
        """Force all pending writes to disk.

        Ensures durability of recent inserts/updates.
        """
        self._obj.Flush()

    # ========== Index DDL Methods ==========
    def create_index(
        self,
        field_name: str,
        index_param: Union[
            HnswIndexParam, IVFIndexParam, FlatIndexParam, InvertIndexParam
        ],
        option: IndexOption = IndexOption(),
    ) -> None:
        """Create an index on a field.

        Vector index types (HNSW, IVF, FLAT) can only be applied to vector fields.
        Inverted index (`InvertIndexParam`) is for scalar fields.

        Args:
            field_name (str): Name of the field to index.
            index_param (Union[HnswIndexParam, IVFIndexParam, FlatIndexParam, InvertIndexParam]):
                Index configuration.
            option (Optional[IndexOption], optional): Index creation options.
                Defaults to ``IndexOption()``.

        Raises:
            ValueError: If a vector index is applied to a non-vector field.
        """
        if index_param in _VECTOR_INDEX_TYPES and not self.schema.vector(field_name):
            supported_types = ", ".join(cls.__name__ for cls in _VECTOR_INDEX_TYPES)
            raise ValueError(
                f"Cannot apply vector index to non-vector field '{field_name}'. "
                f"The field must be of vector type to use index types like {supported_types}."
            )
        self._obj.CreateIndex(field_name, index_param, option)
        self._schema = CollectionSchema._from_core(self._obj.Schema())

    def drop_index(self, field_name: str) -> None:
        """Remove the index from a field.

        Args:
            field_name (str): Name of the indexed field.
        """
        self._obj.DropIndex(field_name)
        self._schema = CollectionSchema._from_core(self._obj.Schema())

    def optimize(self, option: OptimizeOption = OptimizeOption()) -> None:
        """Optimize the collection (e.g., merge segments, rebuild index).

        Args:
            option (Optional[OptimizeOption], optional): Optimization options.
                Defaults to ``OptimizeOption()``.
        """
        self._obj.Optimize(option)

    # ========== COLUMN DDL Methods ==========
    def add_column(
        self,
        field_schema: FieldSchema,
        expression: str = "",
        option: AddColumnOption = AddColumnOption(),
    ) -> None:
        """Add a new column to the collection.

        The column is populated using the provided expression (e.g., SQL-like formula).

        Args:
            field_schema (FieldSchema): Schema definition for the new column.
            expression (str): Expression to compute values for existing documents.
            option (Optional[AddColumnOption], optional): Options for the operation.
                Defaults to ``AddColumnOption()``.
        """
        self._obj.AddColumn(field_schema._get_object(), expression, option)
        self._schema = CollectionSchema._from_core(self._obj.Schema())

    def drop_column(self, field_name: str) -> None:
        """Remove a column from the collection.

        Args:
            field_name (str): Name of the column to drop.
        """
        self._obj.DropColumn(field_name)
        self._schema = CollectionSchema._from_core(self._obj.Schema())

    def alter_column(
        self,
        old_name: str,
        new_name: Optional[str] = None,
        field_schema: Optional[FieldSchema] = None,
        option: AlterColumnOption = AlterColumnOption(),
    ) -> None:
        """Rename a column, update its schema.

        This method supports three atomic operations:
          1. Rename only (when `field_schema` is None).
          2. Modify schema only (when `new_name` is None or empty string).

        Args:
            old_name (str): The current name of the column to be altered.
            new_name (Optional[str]): The new name for the column.
                - If provided and non-empty, the column will be renamed.
                - If `None` or empty string, no rename occurs.
            field_schema (Optional[FieldSchema]): The new schema definition.
                - If provided, the column's type, dimension, or other properties will be updated.
                - If `None`, only renaming (if requested) is performed.
            option (AlterColumnOption, optional): Options controlling the alteration behavior.
                Defaults to ``AlterColumnOption()``.

        **Limitation**: This operation **only supports scalar numeric columns**. such as:
        - `DOUBLE`, `FLOAT`,
        - `INT32`, `INT64`, `UINT32`, `UINT64`

        Note:
            - Schema modification may trigger data migration or index rebuild.

        Examples:
            >>> # Rename column only
            >>> results = collection.alter_column(old_name="id", new_name="doc_id")

            >>> # Modify schema only
            >>> new_schema = FieldSchema(name="doc_id", dtype=DataType.INT64)
            >>> collection.alter_column("id", field_schema=new_schema)
        """
        self._obj.AlterColumn(
            old_name,
            new_name or "",
            field_schema._get_object() if field_schema else None,
            option,
        )
        self._schema = CollectionSchema._from_core(self._obj.Schema())

    # ========== Collection DDL Methods ==========
    def insert(self, docs: Union[Doc, list[Doc]]) -> Union[Status, list[Status]]:
        """Insert new documents into the collection.

        Documents must have unique IDs and conform to the schema.

        Args:
            docs (Union[Doc, list[Doc]]): One or more documents to insert.

        Returns:
            Union[Status, list[Status]]: If a single Doc was given, returns its Status;
            if a list was given, returns a list of Status objects.
        """
        is_single = isinstance(docs, Doc)
        doc_list = [docs] if is_single else docs
        results = self._obj.Insert(
            [convert_to_cpp_doc(doc, self.schema) for doc in doc_list]
        )
        return results[0] if is_single else results

    def upsert(self, docs: Union[Doc, list[Doc]]) -> Union[Status, list[Status]]:
        """Insert new documents or update existing ones by ID.

        Args:
            docs (Union[Doc, list[Doc]]): Documents to upsert.

        Returns:
            Union[Status, list[Status]]: If a single Doc was given, returns its Status;
            if a list was given, returns a list of Status objects.
        """
        is_single = isinstance(docs, Doc)
        doc_list = [docs] if is_single else docs
        results = self._obj.Upsert(
            [convert_to_cpp_doc(doc, self.schema) for doc in doc_list]
        )
        return results[0] if is_single else results

    def update(self, docs: Union[Doc, list[Doc]]) -> Union[Status, list[Status]]:
        """Update existing documents by ID.

        Only specified fields are updated; others remain unchanged.

        Args:
            docs (Union[Doc, list[Doc]]): Documents containing updated fields.

        Returns:
            Union[Status, list[Status]]: If a single Doc was given, returns its Status;
            if a list was given, returns a list of Status objects.
        """
        is_single = isinstance(docs, Doc)
        doc_list = [docs] if is_single else docs
        results = self._obj.Update(
            [convert_to_cpp_doc(doc, self.schema) for doc in doc_list]
        )
        return results[0] if is_single else results

    def delete(self, ids: Union[str, list[str]]) -> Union[Status, list[Status]]:
        """Delete documents by ID.

        Args:
            ids (Union[str, list[str]]): One or more document IDs to delete.

        Returns:
            Union[Status, list[Status]]: If a single id was given, returns its Status;
            if a list was given, returns a list of Status objects.
        """
        is_single = isinstance(ids, str)
        id_list = [ids] if isinstance(ids, str) else ids
        results = self._obj.Delete(id_list)
        return results[0] if is_single else results

    def delete_by_filter(self, filter: str) -> None:
        """Delete documents matching a filter expression.

        Args:
            filter (str): Boolean expression (e.g., ``"age > 30"``).
        """
        self._obj.DeleteByFilter(filter)

    # ========== Collection DQL-fetch Methods ==========
    def fetch(self, ids: Union[str, list[str]]) -> dict[str, Doc]:
        """Retrieve documents by ID.

        Args:
            ids (Union[str, list[str]]): Document IDs to fetch.

        Returns:
            dict[str, Doc]: Mapping from ID to document. Missing IDs are omitted.
        """
        ids = [ids] if isinstance(ids, str) else ids
        docs = self._obj.Fetch(ids)
        return {
            doc_id: py_doc
            for doc_id, core_doc in docs.items()
            if (py_doc := convert_to_py_doc(core_doc, self.schema)) is not None
        }

    # ========== Collection DQL-Query Methods ==========

    def query(
        self,
        vectors: Optional[Union[VectorQuery, list[VectorQuery]]] = None,
        *,
        topk: int = 10,
        filter: Optional[str] = None,
        include_vector: bool = False,
        output_fields: Optional[list[str]] = None,
        reranker: Optional[ReRanker] = None,
    ) -> list[Doc]:
        """Perform vector similarity search with optional filtering and re-ranking.

        At least one `VectorQuery` must be provided.

        Args:
            vectors (Optional[Union[VectorQuery, list[VectorQuery]]], optional):
                One or more vector queries. Defaults to None.
            topk (int, optional): Number of nearest neighbors to return.
                Defaults to 10.
            filter (Optional[str], optional): Boolean expression to pre-filter candidates.
                Defaults to None.
            include_vector (bool, optional): Whether to include vector data in results.
                Defaults to False.
            output_fields (Optional[list[str]], optional): Scalar fields to include.
                If None, all fields are returned. Defaults to None.
            reranker (Optional[ReRanker], optional): Re-ranker to refine results.
                Defaults to None.

        Returns:
            list[Doc]: Top-k matching documents, sorted by relevance score.

        Examples:
            >>> from zvec import VectorQuery
            >>> results = collection.query(
            ...     vectors=VectorQuery("embedding", vector=[0.1, 0.2]),
            ...     topk=5,
            ...     filter="category == 'tech'",
            ...     output_fields=["title", "url"]
            ... )
        """
        ctx = QueryContext(
            topk=topk,
            filter=filter,
            queries=[vectors] if isinstance(vectors, VectorQuery) else vectors,
            include_vector=include_vector,
            output_fields=output_fields,
            reranker=reranker,
        )
        return self._querier.execute(ctx, self._obj)
