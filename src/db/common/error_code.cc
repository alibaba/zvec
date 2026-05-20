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

#include "error_code.h"

namespace zvec {

// 0~999  [Builtin]
PROXIMA_ZVEC_ERROR_CODE_DEFINE(Success, 0, "Success");

// 1000~1999 [Common Error]
PROXIMA_ZVEC_ERROR_CODE_DEFINE(RuntimeError, 1000, "Runtime Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(LogicError, 1001, "Logic Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(StatusError, 1002, "Status Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(LoadConfig, 1003, "Load Config Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(ConfigError, 1004, "Config Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidArgument, 1005, "Invalid Argument");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(NotInitialized, 1006, "Not Initialized");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(OpenFile, 1007, "Open File Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(ReadData, 1008, "Read Data Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(WriteData, 1009, "Write Data Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(ExceedLimit, 1010, "Exceed Limit");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(SerializeError, 1011, "Serialize Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(DeserializeError, 1012, "Deserialize Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(StartServer, 1013, "Start Server Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(StoppedService, 1014, "Visit Stopped Service");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(FileSystem, 1015, "File System Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(RpcError, 1016, "RPC Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InitChannelError, 1017,
                               "Init brpc channel Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(AddSubChannelError, 1018,
                               "Add sub channel Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(NoNeedProcess, 1019, "No need process");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EtcdError, 1020, "Etcd Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(MessageQueueError, 1021, "Message Queue Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(KafkaSubTopicExistErr, 1022,
                               "Kafka topic subscribe already exist Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(KafkaUnSubTopicNotExistErr, 1023,
                               "Kafka topic unsubscribe not exist Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InitKafkaError, 1024, "Init kafka error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(KafkaPublishError, 1025, "Kafka publish error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(ExceedKafkaMessageSizeLimit, 1026,
                               "Exceed kafka message size limit");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(NotImplemented, 1027,
                               "The function is not implemented");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(Timeout, 1028, "Timeout");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(MasterNoLeader, 1029, "Master no leader");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(NeedRetry, 1030, "Need retry");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(Abort, 1031, "Abort");

PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyCollectionName, 2000,
                               "Empty Collection Name");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyColumnName, 2001, "Empty Column Name");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyPartitionName, 2002,
                               "Empty collection partition name");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyColumns, 2003, "Empty Columns");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyPrimaryKey, 2004, "Empty primary key");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyDocList, 2005, "Empty doc list");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyDocFields, 2006, "Empty doc fields");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyIndexField, 2007, "Empty index field");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidRecord, 2008, "Invalid Record");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidQuery, 2009, "Invalid Query");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidWriteRequest, 2010,
                               "Invalid Write Request");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidVectorFormat, 2011,
                               "Invalid Vector Format");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidDataType, 2012, "Invalid Data Type");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidIndexType, 2013, "Invalid Index Type");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidFeature, 2014, "Invalid Feature");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidFilter, 2015, "Invalid Filter");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidPrimaryKey, 2016, "Invalid primary key");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidField, 2017, "Invalid field");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(MismatchedIndexColumn, 2018,
                               "Mismatched Index Column");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(MismatchedDimension, 2019,
                               "Mismatched Dimension");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(MismatchedDataType, 2020,
                               "Mismatched Data Type");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InexistentCollection, 2021,
                               "Collection Not Exist");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InexistentPartition, 2022,
                               "Inexistent collection partition");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InexistentColumn, 2023, "Column Not Exist");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InexistentKey, 2024, "Key Not Exist");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(DuplicateCollection, 2025,
                               "Duplicate Collection");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(DuplicatePartition, 2026,
                               "Duplicate collection partition");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(DuplicateKey, 2027, "Duplicate Key");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(DuplicateField, 2028, "Duplicate field");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(UnreadyPartition, 2029,
                               "Status of collection partition is incorrect");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(UnreadyCollection, 2030,
                               "Status of collection is incorrect");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(UnsupportedCondition, 2031,
                               "Query condition has error or not supported");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(
    OrderbyNotInSelectItems, 2032,
    "Order by column must exists in select item list");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(PbToSqlInfoError, 2033, "Pb to sql info error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(ExceedRateLimit, 2034, "Exceed Rate Limit");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidSparseValues, 2035,
                               "Invalid Sparse Values");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidBatchSize, 2036, "Invalid batch size");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidDimension, 2037, "Invalid dimension");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidExtraParam, 2038, "Invalid extra param");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidRadius, 2039, "Invalid radius");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidLinear, 2040, "Invalid is linear");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidTopk, 2041, "Invalid topk");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidCollectionName, 2042,
                               "Invalid collection name");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidPartitionName, 2043,
                               "Invalid partition name");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidFieldName, 2044, "Invalid field name");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidChannelCount, 2045,
                               "Invalid channel count");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidReplicaCount, 2046,
                               "Invalid replica count");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidJson, 2047, "Invalid json");
// used by master
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidClusterConfig, 2048,
                               "Invalid cluster config");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(DuplicateCluster, 2049, "Duplicate Cluster");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InexistentCluster, 2050, "Inexistent Cluster");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidClusterStatus, 2051,
                               "Invalid Cluster Status");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(RpcTimedout, 2052, "Rpc Timedout");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidGroupBy, 2053, "Invalid GroupBy Request");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(EmptyVectorField, 2054, "Empty vector field");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(VectorNotAllowed, 2055, "Vector not allowed");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidReferenceCollection, 2056,
                               "Invalid reference collection");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(InvalidOrderBy, 2057, "Invalid OrderBy field");


PROXIMA_ZVEC_ERROR_CODE_DEFINE(UnreadyQueue, 5002,
                               "Compute Queue Is Unready Yet");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(ScheduleError, 5003, "Schedule Task Error");
PROXIMA_ZVEC_ERROR_CODE_DEFINE(TaskIsRunning, 5004,
                               "Task is running in other coroutine");

const char *ErrorCode::What(int val) {
  return ErrorCode::Instance()->what(val);
}

}  // namespace zvec
