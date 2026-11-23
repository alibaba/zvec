// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <zvec/ailego/buffer/process_rss_budget.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR && !TARGET_OS_MACCATALYST
#include <mach/mach.h>
#include <os/proc.h>
#endif
#endif

namespace zvec::ailego {

inline ProcessRssBudget::NativeReader NativeApplicationMemoryReader() {
#if defined(__APPLE__) && TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR && \
    !TARGET_OS_MACCATALYST
  return +[](ProcessRssBudget::NativeSample *sample) -> bool {
    if (__builtin_available(iOS 13.0, tvOS 13.0, watchOS 6.0, *)) {
      task_vm_info_data_t info{};
      mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
      if (::task_info(::mach_task_self(), TASK_VM_INFO,
                      reinterpret_cast<task_info_t>(&info),
                      &count) != KERN_SUCCESS ||
          count < TASK_VM_INFO_REV1_COUNT)
        return false;
      sample->footprint = static_cast<size_t>(info.phys_footprint);
      sample->available = ::os_proc_available_memory();
      return true;
    }
    return false;
  };
#else
  // Simulator memory limits describe the host, not a device application.
  // It still exercises RSS and the explicit host-pressure notification path.
  return nullptr;
#endif
}

}  // namespace zvec::ailego
