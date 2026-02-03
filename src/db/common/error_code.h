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

/*! Error
 */

class ErrorCode;

class ErrorCode {
 public:
  /*! Error Code
   */
  class Code {
   public:
    //! Constructor
    Code(int val, const char *str) : value_(-val), desc_(str) {
      ErrorCode::Instance()->emplace(this);
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
  static const char *What(int val);

 protected:
  //! Constructor
  ErrorCode(void) : map_() {}

  //! Inserts a new code into map
  void emplace(const ErrorCode::Code *code) {
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
  static ErrorCode *Instance(void) {
    static ErrorCode error;
    return (&error);
  }

 private:
  //! Disable them
  ErrorCode(const ErrorCode &) = delete;
  ErrorCode(ErrorCode &&) = delete;
  ErrorCode &operator=(const ErrorCode &) = delete;

  //! Error code map
  std::map<int, const ErrorCode::Code *> map_;
};

//! Error Code Define
#define PROXIMA_ZVEC_ERROR_CODE_DEFINE(__NAME__, __VAL__, __DESC__)        \
  const zvec::ErrorCode::Code ErrorCode_##__NAME__((__VAL__), (__DESC__)); \
  const zvec::ErrorCode::Code &_ErrorCode_##__VAL__##_Register(            \
      ErrorCode_##__NAME__)

//! Proxima SE Error Code Declare
#define PROXIMA_ZVEC_ERROR_CODE_DECLARE(__NAME__) \
  extern const zvec::ErrorCode::Code ErrorCode_##__NAME__

//! Error code helper
#define PROXIMA_ZVEC_ERROR_CODE(__NAME__) zvec::ErrorCode_##__NAME__

// 0~999  [Builtin]
PROXIMA_ZVEC_ERROR_CODE_DECLARE(Success);

// 1000~1999 [Common Error]
PROXIMA_ZVEC_ERROR_CODE_DECLARE(RuntimeError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(LogicError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(StatusError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(LoadConfig);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(ConfigError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidArgument);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(NotInitialized);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(OpenFile);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(ReadData);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(WriteData);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(ExceedLimit);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(SerializeError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(DeserializeError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(StartServer);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(StoppedService);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(FileSystem);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(RpcError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InitChannelError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(AddSubChannelError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(NoNeedProcess);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EtcdError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(MessageQueueError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(KafkaSubTopicExistErr);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(KafkaUnSubTopicNotExistErr);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InitKafkaError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(KafkaPublishError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(ExceedKafkaMessageSizeLimit);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(NotImplemented);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(Timeout);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(MasterNoLeader);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(NeedRetry);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(Abort);

PROXIMA_ZVEC_ERROR_CODE_DECLARE(UnreadyQueue);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(ScheduleError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(TaskIsRunning);

PROXIMA_ZVEC_ERROR_CODE_DECLARE(DirectoryAlreadyExists);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(DirectoryNotExists);

// 2000~2999 [Client Check]
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyCollectionName);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyColumnName);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyPartitionName);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyColumns);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyPrimaryKey);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyDocList);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyDocFields);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyIndexField);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidRecord);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidQuery);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidWriteRequest);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidVectorFormat);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidDataType);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidIndexType);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidFeature);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidFilter);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidPrimaryKey);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidField);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(MismatchedIndexColumn);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(MismatchedDimension);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(MismatchedDataType);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InexistentCollection);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InexistentPartition);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InexistentColumn);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InexistentKey);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(DuplicateCollection);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(DuplicatePartition);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(DuplicateKey);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(DuplicateField);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(UnreadyPartition);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(UnreadyCollection);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(UnsupportedCondition);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(OrderbyNotInSelectItems);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(PbToSqlInfoError);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(ExceedRateLimit);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidSparseValues);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidBatchSize);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidDimension);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidExtraParam);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidRadius);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidLinear);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidTopk);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidCollectionName);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidPartitionName);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidFieldName);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidChannelCount);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidReplicaCount);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidJson);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidClusterConfig);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(DuplicateCluster);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InexistentCluster);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidClusterStatus);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(RpcTimedout);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidGroupBy);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(EmptyVectorField);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(VectorNotAllowed);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidReferenceCollection);
PROXIMA_ZVEC_ERROR_CODE_DECLARE(InvalidOrderBy);


// 40000~49999 [De Admin]
}  // namespace zvec
