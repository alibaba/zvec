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
#pragma once

#include <map>
#include <zvec/ailego/pattern/expected.hpp>

namespace zvec {
namespace core {

/*! Error
 */

class ErrorCode;

template <typename T>
using Result = tl::expected<T, ErrorCode>;

template <typename T>
using result_unexpect = tl::unexpected<ErrorCode>;

/*! Index Error
 */
class IndexError {
 public:
  /*! Index Error Code
   */
  class Code {
   public:
    //! Constructor
    Code(int val, const char *str) : value_(-val), desc_(str) {
      IndexError::Instance()->emplace(this);
    }

    //! Retrieve the value of code
    operator int() const {
      return (this->value_);
    }

    //! Retrieve the value of code
    int value() const {
      return (this->value_);
    }

    //! Retrieve the description of code
    const char *desc() const {
      return (this->desc_);
    }

   private:
    int value_;
    const char *desc_;
  };

  //! Retrieve the description of code
  static const char *What(int val) {
    return IndexError::Instance()->what(val);
  }

 protected:
  //! Constructor
  IndexError(void) : map_() {}

  //! Inserts a new code into map
  void emplace(const IndexError::Code *code) {
    map_.emplace(code->value(), code);
  }

  //! Retrieve the description of code
  const char *what(int val) const {
    auto iter = map_.find(val);
    if (iter != map_.end()) {
      return iter->second->desc();
    }
    return "";
  }

  //! Retrieve the singleton
  static IndexError *Instance(void) {
    static IndexError error;
    return (&error);
  }

 private:
  //! Disable them
  IndexError(const IndexError &) = delete;
  IndexError(IndexError &&) = delete;
  IndexError &operator=(const IndexError &) = delete;

  //! Error code map
  std::map<int, const IndexError::Code *> map_;
};

//! Index Error Code Define
#define INDEX_ERROR_CODE_DEFINE(__NAME__, __VAL__, __DESC__)           \
  const IndexError::Code IndexError_##__NAME__((__VAL__), (__DESC__)); \
  const IndexError::Code &_IndexErrorCode_##__VAL__##_Register(        \
      IndexError_##__NAME__)

//! Index Error Code Declare
#define INDEX_ERROR_CODE_DECLARE(__NAME__) \
  extern const IndexError::Code IndexError_##__NAME__

//! Build-in error code
INDEX_ERROR_CODE_DECLARE(Success);      // Success
INDEX_ERROR_CODE_DECLARE(Runtime);      // Runtime error
INDEX_ERROR_CODE_DECLARE(Logic);        // Logic error
INDEX_ERROR_CODE_DECLARE(Type);         // Type error
INDEX_ERROR_CODE_DECLARE(System);       // System call error
INDEX_ERROR_CODE_DECLARE(Cast);         // Cast error
INDEX_ERROR_CODE_DECLARE(IO);           // IO error
INDEX_ERROR_CODE_DECLARE(AuthExpired);  // Auth expired error

INDEX_ERROR_CODE_DECLARE(NotImplemented);  // Not implemented
INDEX_ERROR_CODE_DECLARE(Unsupported);     // Unsupported
INDEX_ERROR_CODE_DECLARE(Denied);          // Permission denied
INDEX_ERROR_CODE_DECLARE(Canceled);        // Operation canceled
INDEX_ERROR_CODE_DECLARE(Overflow);        // Overflow
INDEX_ERROR_CODE_DECLARE(Underflow);       // Underflow
INDEX_ERROR_CODE_DECLARE(OutOfRange);      // Out of range
INDEX_ERROR_CODE_DECLARE(NoBuffer);        // No buffer space available
INDEX_ERROR_CODE_DECLARE(NoMemory);        // Not enough space
INDEX_ERROR_CODE_DECLARE(NoParamFound);    // No parameter found
INDEX_ERROR_CODE_DECLARE(NoReady);         // No ready
INDEX_ERROR_CODE_DECLARE(NoExist);         // No exist
INDEX_ERROR_CODE_DECLARE(Exist);           // Already exist
INDEX_ERROR_CODE_DECLARE(Mismatch);        // Mismatch
INDEX_ERROR_CODE_DECLARE(Duplicate);       // Duplicate
INDEX_ERROR_CODE_DECLARE(Uninitialized);   // Uninitialized

INDEX_ERROR_CODE_DECLARE(InvalidArgument);  // Invalid argument
INDEX_ERROR_CODE_DECLARE(InvalidFormat);    // Invalid format
INDEX_ERROR_CODE_DECLARE(InvalidLength);    // Invalid length
INDEX_ERROR_CODE_DECLARE(InvalidChecksum);  // Invalid checksum
INDEX_ERROR_CODE_DECLARE(InvalidValue);     // Invalid value

INDEX_ERROR_CODE_DECLARE(CreateDirectory);  // Create directory error
INDEX_ERROR_CODE_DECLARE(OpenDirectory);    // Open directory error
INDEX_ERROR_CODE_DECLARE(Serialize);        // Serialize error
INDEX_ERROR_CODE_DECLARE(Deserialize);      // Deserialize error
INDEX_ERROR_CODE_DECLARE(CreateFile);       // Create file error
INDEX_ERROR_CODE_DECLARE(OpenFile);         // Open file error
INDEX_ERROR_CODE_DECLARE(SeekFile);         // Seek file error
INDEX_ERROR_CODE_DECLARE(CloseFile);        // Close file error
INDEX_ERROR_CODE_DECLARE(TruncateFile);     // TruncateFile file error
INDEX_ERROR_CODE_DECLARE(MMapFile);         // MMap file error
INDEX_ERROR_CODE_DECLARE(FlushFile);        // Flush file error
INDEX_ERROR_CODE_DECLARE(WriteData);        // Write data error
INDEX_ERROR_CODE_DECLARE(ReadData);         // Read data error

INDEX_ERROR_CODE_DECLARE(PackIndex);      // Read data error
INDEX_ERROR_CODE_DECLARE(UnpackIndex);    // Read data error
INDEX_ERROR_CODE_DECLARE(IndexLoaded);    // Index loaded
INDEX_ERROR_CODE_DECLARE(NoIndexLoaded);  // No index loaded
INDEX_ERROR_CODE_DECLARE(NoTrained);      // No trained
INDEX_ERROR_CODE_DECLARE(IndexFull);      // Index full

}  // namespace core
}  // namespace zvec