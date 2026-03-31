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

#include <stdio.h>
#include <string.h>
#include <limits>
#include <map>
#ifdef _MSC_VER
#include <chrono>
#else
#include <sys/time.h>
#endif
#include <ailego/parallel/lock.h>

namespace zvec {
namespace core {

class BenchResult {
 public:
  BenchResult() {
    total_query_count_ = 0;
    total_process_time_by_us_ = 0;
    min_time_by_us_ = std::numeric_limits<long>::max();
    max_time_by_us_ = 0;
  }
  ~BenchResult() {}

  void add_time(int query_count, long time_by_us) {
    lock_.lock();
    total_query_count_ += query_count;
    total_process_time_by_us_ += time_by_us;
    long time_val = time_by_us / 100;
    if (process_time_map_.find(time_val) != process_time_map_.end()) {
      ++process_time_map_[time_val];
    } else {
      process_time_map_[time_val] = 1;
    }
    if (time_by_us < min_time_by_us_) {
      min_time_by_us_ = time_by_us;
    } else if (time_by_us > max_time_by_us_) {
      max_time_by_us_ = time_by_us;
    }
    lock_.unlock();
  }
  void mark_start() {
#ifdef _MSC_VER
    start_ = std::chrono::steady_clock::now();
#else
    gettimeofday(&start_, NULL);
#endif
  }
  void mark_end() {
#ifdef _MSC_VER
    end_ = std::chrono::steady_clock::now();
#else
    gettimeofday(&end_, NULL);
#endif
  }
  long get_duration_by_ms() {
#ifdef _MSC_VER
    return static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_)
            .count());
#else
    long duration = (end_.tv_sec - start_.tv_sec) * 1000 +
                    (end_.tv_usec - start_.tv_usec) / 1000;
    return duration;
#endif
  }
  long get_total_query_count() {
    return total_query_count_;
  }
  std::map<long, long> &get_process_time_map() {
    return process_time_map_;
  }
  long get_total_process_time_by_ms() {
    return total_process_time_by_us_ / 1000;
  }
  void print() {
    fprintf(stdout,
            "Process query: %ld, total process time: %ldms, "
            "duration: %ldms, max: %ldms, min:%ldms\n",
            get_total_query_count(), get_total_process_time_by_ms(),
            get_duration_by_ms(), max_time_by_us_ / 1000,
            min_time_by_us_ / 1000);
    fprintf(stdout, "Avg latency: %0.1fms qps: %0.1f\n",
            ((float)get_total_process_time_by_ms()) / get_total_query_count(),
            get_total_query_count() / ((float)get_duration_by_ms() / 1000));

    int tot_num = 0;
    int percent[] = {25, 50, 75, 90, 95, 99};
    int index = 0;
    float max_time = 0.0;
    int last_num = 0;

    for (auto element : process_time_map_) {
      tot_num += element.second;
      if (tot_num >= total_query_count_ * percent[index] / 100) {
        if (last_num != tot_num) {
          max_time = (float)element.first / 10;
          last_num = tot_num;
        }
        fprintf(stdout, "%d Percentile:\t\t %.1f ms\n", percent[index],
                max_time);
        index++;
        if (index >= 6) {
          break;
        }
      }
    }
    for (; index < 6; index++) {
      fprintf(stdout, "%d Percentile:\t\t %.1f ms\n", percent[index], max_time);
    }
    fprintf(stdout, "\n");
  }

 private:
  long total_query_count_;
  long total_process_time_by_us_;
  long min_time_by_us_;
  long max_time_by_us_;
#ifdef _MSC_VER
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point end_;
#else
  struct timeval start_;
  struct timeval end_;
#endif
  ailego::SpinMutex lock_;
  std::map<long, long> process_time_map_;  // <processTimeBy100us, count>
};

}  // namespace core
}  // namespace zvec
