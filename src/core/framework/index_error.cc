// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <zvec/core/framework/index_error.h>

namespace zvec {
namespace core {

INDEX_ERROR_CODE_DEFINE(Success, 0, "Success");
INDEX_ERROR_CODE_DEFINE(Runtime, 1, "Runtime error");
INDEX_ERROR_CODE_DEFINE(Logic, 2, "Logic error");
INDEX_ERROR_CODE_DEFINE(Type, 3, "Type error");
INDEX_ERROR_CODE_DEFINE(System, 4, "System call error");
INDEX_ERROR_CODE_DEFINE(Cast, 5, "Cast error");
INDEX_ERROR_CODE_DEFINE(IO, 6, "IO error");
INDEX_ERROR_CODE_DEFINE(AuthExpired, 7, "Auth expired error");

INDEX_ERROR_CODE_DEFINE(NotImplemented, 11, "Not implemented");
INDEX_ERROR_CODE_DEFINE(Unsupported, 12, "Unsupported");
INDEX_ERROR_CODE_DEFINE(Denied, 13, "Permission denied");
INDEX_ERROR_CODE_DEFINE(Canceled, 14, "Operation canceled");
INDEX_ERROR_CODE_DEFINE(Overflow, 15, "Overflow");
INDEX_ERROR_CODE_DEFINE(Underflow, 16, "Underflow");
INDEX_ERROR_CODE_DEFINE(OutOfRange, 17, "Out of range");
INDEX_ERROR_CODE_DEFINE(NoBuffer, 18, "No buffer space available");
INDEX_ERROR_CODE_DEFINE(NoMemory, 19, "Not enough space");
INDEX_ERROR_CODE_DEFINE(NoParamFound, 20, "No parameter found");
INDEX_ERROR_CODE_DEFINE(NoReady, 21, "No ready");
INDEX_ERROR_CODE_DEFINE(NoExist, 22, "No exist");
INDEX_ERROR_CODE_DEFINE(Exist, 23, "Already exist");
INDEX_ERROR_CODE_DEFINE(Mismatch, 24, "Mismatch");
INDEX_ERROR_CODE_DEFINE(Duplicate, 25, "Duplicate");
INDEX_ERROR_CODE_DEFINE(Uninitialized, 26, "Uninitialized");

INDEX_ERROR_CODE_DEFINE(InvalidArgument, 31, "Invalid argument");
INDEX_ERROR_CODE_DEFINE(InvalidFormat, 32, "Invalid format");
INDEX_ERROR_CODE_DEFINE(InvalidLength, 33, "Invalid length");
INDEX_ERROR_CODE_DEFINE(InvalidChecksum, 34, "Invalid checksum");
INDEX_ERROR_CODE_DEFINE(InvalidValue, 35, "Invalid value");

INDEX_ERROR_CODE_DEFINE(CreateDirectory, 101, "Create directory error");
INDEX_ERROR_CODE_DEFINE(OpenDirectory, 102, "Open directory error");
INDEX_ERROR_CODE_DEFINE(Serialize, 105, "Serialize error");
INDEX_ERROR_CODE_DEFINE(Deserialize, 106, "Deserialize error");
INDEX_ERROR_CODE_DEFINE(CreateFile, 111, "Create file error");
INDEX_ERROR_CODE_DEFINE(OpenFile, 112, "Open file error");
INDEX_ERROR_CODE_DEFINE(SeekFile, 113, "Seek file error");
INDEX_ERROR_CODE_DEFINE(CloseFile, 114, "Close file error");
INDEX_ERROR_CODE_DEFINE(TruncateFile, 115, "TruncateFile file error");
INDEX_ERROR_CODE_DEFINE(MMapFile, 116, "MMap file error");
INDEX_ERROR_CODE_DEFINE(FlushFile, 117, "Flush file error");
INDEX_ERROR_CODE_DEFINE(WriteData, 121, "Write data error");
INDEX_ERROR_CODE_DEFINE(ReadData, 122, "Read data error");

INDEX_ERROR_CODE_DEFINE(PackIndex, 201, "Read data error");
INDEX_ERROR_CODE_DEFINE(UnpackIndex, 202, "Read data error");
INDEX_ERROR_CODE_DEFINE(IndexLoaded, 203, "Index loaded");
INDEX_ERROR_CODE_DEFINE(NoIndexLoaded, 204, "No index loaded");
INDEX_ERROR_CODE_DEFINE(NoTrained, 205, "No trained");
INDEX_ERROR_CODE_DEFINE(IndexFull, 206, "Index full");

}  // namespace core
}  // namespace zvec
