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

from zvec._zvec import _Doc

from ._validation import explain_utf8_conversion_error, format_name_for_error
from .doc import Doc
from .schema import CollectionSchema


def convert_to_cpp_doc(doc: Doc, collection_schema: CollectionSchema) -> _Doc:
    if not doc or not collection_schema:
        return None

    _doc = _Doc()

    # set pk
    try:
        _doc.set_pk(doc.id)
    except TypeError:
        explain_utf8_conversion_error(doc.id, "Invalid doc: id")
        raise

    # set scalar fields
    for k, v in doc.fields.items():
        field_schema = collection_schema.field(k)
        if not field_schema:
            raise ValueError(
                f"Invalid schema: {format_name_for_error(k)} not found in collection schema"
            )
        _doc.set_any(k, field_schema._get_object(), v)

    # set vector fields
    for k, v in doc.vectors.items():
        vector_schema = collection_schema.vector(k)
        if not vector_schema:
            raise ValueError(
                f"Invalid schema: {format_name_for_error(k)} not found in collection schema"
            )
        _doc.set_any(k, vector_schema._get_object(), v)
    return _doc


def convert_to_cpp_docs(
    docs: list[Doc], collection_schema: CollectionSchema
) -> list[_Doc]:
    converted = []
    for index, doc in enumerate(docs):
        try:
            converted.append(convert_to_cpp_doc(doc, collection_schema))
        except (TypeError, ValueError) as error:
            # Preserve the original exception and cause. Unicode error subclasses
            # carry structured arguments that must not be replaced with a string.
            if type(error) in (TypeError, ValueError):
                suffix = f" (document at index {index})"
                message = str(error)
                if not message.endswith(suffix):
                    error.args = (message + suffix,)
            raise
    return converted


def convert_to_py_doc(doc: _Doc, collection_schema: CollectionSchema) -> Doc:
    if not doc or not collection_schema:
        return None

    data_tuple = doc.get_all(collection_schema._get_object())
    return Doc._from_tuple(data_tuple)
