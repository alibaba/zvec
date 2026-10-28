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

# Exercise the real tool binaries without Python, downloaded data, or plugins.
foreach(arg TXT2VECS LOCAL_BUILDER LOCAL_BUILDER_ORIGINAL RECALL BENCH WORK_DIR)
  if(NOT DEFINED ${arg} OR "${${arg}}" STREQUAL "")
    message(FATAL_ERROR "Missing -D${arg}")
  endif()
endforeach()
file(MAKE_DIRECTORY "${WORK_DIR}")

function(run_tool stage tool)
  set(log "${WORK_DIR}/${stage}.log")
  execute_process(
      COMMAND "${tool}" ${ARGN}
      WORKING_DIRECTORY "${WORK_DIR}"
      OUTPUT_FILE "${log}"
      ERROR_FILE "${log}"
      RESULT_VARIABLE result
      TIMEOUT 30
    )
  file(READ "${log}" output)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "${stage} failed (${result}):\n${output}")
  endif()
  set(tool_output "${output}" PARENT_SCOPE)
endfunction()

# Integer-valued FP32 inputs are exactly representable in FP16. Different
# records and non-contiguous keys expose reused iterator buffers or lost keys.
set(state 42)
set(vectors "")
set(queries "")
set(truth "")
foreach(row RANGE 0 255)
  math(EXPR key "${row} * 7 + 3")
  set(record "${key};")
  foreach(dim RANGE 0 15)
    math(EXPR state "(${state} * 1103515245 + 12345) & 0x7fffffff")
    math(EXPR value "((${state} >> 12) % 2001) - 1000")
    string(APPEND record "${value}.0 ")
  endforeach()
  string(APPEND vectors "${record}\n")
  math(EXPR query_offset "(${row} - 128) % 11")
  if(row GREATER_EQUAL 128 AND row LESS_EQUAL 205 AND query_offset EQUAL 0)
    string(APPEND queries "${record}\n")
    string(APPEND truth "${key};${key}\n")
  endif()
endforeach()
file(WRITE "${WORK_DIR}/vectors.txt" "${vectors}")
file(WRITE "${WORK_DIR}/queries.txt" "${queries}")
file(WRITE "${WORK_DIR}/truth.txt" "${truth}")
run_tool(convert "${TXT2VECS}"
    "--input=${WORK_DIR}/vectors.txt" "--output=${WORK_DIR}/vectors.vecs"
    --dimension=16 --type=float
  )

foreach(builder LOCAL_BUILDER LOCAL_BUILDER_ORIGINAL)
  foreach(mode fp32 fp16 selected_range)
    set(case "${builder}_${mode}")
    set(case_dir "${WORK_DIR}/${case}")
    file(MAKE_DIRECTORY "${case_dir}")
    set(extra "")
    set(count 256)
    if(mode STREQUAL "fp16")
      set(extra "    ConverterName: HalfFloatConverter\n")
    elseif(mode STREQUAL "selected_range")
      set(extra "    MaxDocs: 224\n    KeepDocs: 96\n")
      set(count 96)
    endif()
    file(WRITE "${case_dir}/build.yaml"
"BuilderCommon:
    BuilderClass: DiskAnnBuilder
    BuildFile: '${WORK_DIR}/vectors.vecs'
    NeedTrain: true
    TrainFile: '${WORK_DIR}/vectors.vecs'
    DumpPath: '${case_dir}/diskann.index'
    MetricName: SquaredEuclidean
    LogLevel: info
${extra}
BuilderParams:
    zvec.diskann.builder.max_degree: !!int 16
    zvec.diskann.builder.list_size: !!int 32
    zvec.diskann.builder.max_pq_chunk_num: !!int 4
    zvec.diskann.builder.thread_count: !!int 2
")
    run_tool("${case}_build" "${${builder}}" "${case_dir}/build.yaml")
    foreach(expected "Trained count\\[256\\]" "Built count\\[${count}\\]" "Dump count\\[${count}\\]"
                     "Copied source vectors into owned graph holder: count=${count}")
      if(NOT tool_output MATCHES "${expected}")
        message(FATAL_ERROR "${case}: missing ${expected}:\n${tool_output}")
      endif()
    endforeach()
    file(WRITE "${case_dir}/search.yaml"
"SearcherCommon:
    SearcherClass: DiskAnnSearcher
    ContainerType: MMapFileReadStorage
    IndexPath: '${case_dir}/diskann.index'
    TopK: '1'
    QueryFile: '${WORK_DIR}/queries.txt'
    QueryType: float
    QueryFirstSep: ';'
    QuerySecondSep: ' '
    GroundTruthFile: '${WORK_DIR}/truth.txt'
    GroundTruthFirstSep: ';'
    CompareById: true
    RecallThreadCount: 1
    RecallBatchCount: 1
    BenchThreadCount: 1
    BenchBatchCount: 1
    BenchIterCount: 16
    BenchSecs: 2
    LogLevel: info
SearcherParams:
    zvec.diskann.searcher.list_size: !!int 96
")
    run_tool("${case}_recall" "${RECALL}" "${case_dir}/search.yaml")
    # These tools can return zero even after a query failure. Check the work
    # and independently known self-query IDs, not just the process exit code.
    if(NOT tool_output MATCHES "Process query: 8[\r\n]"
        OR NOT tool_output MATCHES "Recall@1: 100(\\.0+)?[\r\n]")
      message(FATAL_ERROR "${case}: self-query validation failed:\n${tool_output}")
    endif()
    run_tool("${case}_bench" "${BENCH}" "${case_dir}/search.yaml")
    if(NOT tool_output MATCHES "Process query: 16,"
        OR tool_output MATCHES "Failed to knn search|Search results is small")
      message(FATAL_ERROR "${case}: benchmark queries failed:\n${tool_output}")
    endif()
    message(STATUS "${case}: built/dumped ${count}, recall@1=100%, 16 bench queries")
  endforeach()
endforeach()
