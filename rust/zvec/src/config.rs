use std::ffi::CStr;

use crate::error::{check_error, to_cstring, Error, ErrorCode, Result};
use crate::types::LogLevel;

/// Global configuration for the zvec library.
///
/// Use this to configure memory limits, thread counts, and logging
/// before calling [`initialize`].
pub struct ConfigData {
    pub(crate) handle: *mut zvec_sys::zvec_config_data_t,
}

impl ConfigData {
    /// Creates a new configuration with default values.
    pub fn new() -> Result<Self> {
        let handle = unsafe { zvec_sys::zvec_config_data_create() };
        if handle.is_null() {
            return Err(Error {
                code: ErrorCode::InternalError,
                message: "failed to create config data".into(),
            });
        }
        Ok(ConfigData { handle })
    }

    /// Sets the memory limit in bytes.
    pub fn set_memory_limit(&mut self, bytes: u64) -> Result<()> {
        check_error(unsafe { zvec_sys::zvec_config_data_set_memory_limit(self.handle, bytes) })
    }

    /// Returns the memory limit in bytes.
    pub fn memory_limit(&self) -> u64 {
        unsafe { zvec_sys::zvec_config_data_get_memory_limit(self.handle) }
    }

    /// Sets the number of query threads.
    pub fn set_query_thread_count(&mut self, count: u32) -> Result<()> {
        check_error(unsafe {
            zvec_sys::zvec_config_data_set_query_thread_count(self.handle, count)
        })
    }

    /// Returns the number of query threads.
    pub fn query_thread_count(&self) -> u32 {
        unsafe { zvec_sys::zvec_config_data_get_query_thread_count(self.handle) }
    }

    /// Sets the number of optimize threads.
    pub fn set_optimize_thread_count(&mut self, count: u32) -> Result<()> {
        check_error(unsafe {
            zvec_sys::zvec_config_data_set_optimize_thread_count(self.handle, count)
        })
    }

    /// Returns the number of optimize threads.
    pub fn optimize_thread_count(&self) -> u32 {
        unsafe { zvec_sys::zvec_config_data_get_optimize_thread_count(self.handle) }
    }

    /// Configures console logging at the specified level.
    pub fn set_console_log(&mut self, level: LogLevel) -> Result<()> {
        let log_config =
            unsafe { zvec_sys::zvec_config_log_create_console(level as u32) };
        if log_config.is_null() {
            return Err(Error {
                code: ErrorCode::InternalError,
                message: "failed to create console log config".into(),
            });
        }
        // Ownership of log_config transfers to config_data
        check_error(unsafe {
            zvec_sys::zvec_config_data_set_log_config(self.handle, log_config)
        })
    }

    /// Configures file logging at the specified level.
    pub fn set_file_log(
        &mut self,
        level: LogLevel,
        dir: &str,
        basename: &str,
        file_size_mb: u32,
        overdue_days: u32,
    ) -> Result<()> {
        let c_dir = to_cstring(dir)?;
        let c_basename = to_cstring(basename)?;

        let log_config = unsafe {
            zvec_sys::zvec_config_log_create_file(
                level as u32,
                c_dir.as_ptr(),
                c_basename.as_ptr(),
                file_size_mb,
                overdue_days,
            )
        };
        if log_config.is_null() {
            return Err(Error {
                code: ErrorCode::InternalError,
                message: "failed to create file log config".into(),
            });
        }
        // Ownership of log_config transfers to config_data
        check_error(unsafe {
            zvec_sys::zvec_config_data_set_log_config(self.handle, log_config)
        })
    }
}

impl Drop for ConfigData {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Safety: handle was created by zvec_config_data_create
            unsafe { zvec_sys::zvec_config_data_destroy(self.handle) };
        }
    }
}

/// Initializes the zvec library with optional configuration.
///
/// Must be called before any other zvec operations. Pass `None` to use
/// default configuration.
pub fn initialize(config: Option<&ConfigData>) -> Result<()> {
    let c_config = config
        .map(|c| c.handle as *const _)
        .unwrap_or(std::ptr::null());
    check_error(unsafe { zvec_sys::zvec_initialize(c_config) })
}

/// Shuts down the zvec library and releases all resources.
pub fn shutdown() -> Result<()> {
    check_error(unsafe { zvec_sys::zvec_shutdown() })
}

/// Returns `true` if the library has been initialized.
pub fn is_initialized() -> bool {
    unsafe { zvec_sys::zvec_is_initialized() }
}

/// Returns the library version string.
pub fn version() -> String {
    unsafe {
        let ptr = zvec_sys::zvec_get_version();
        if ptr.is_null() {
            return String::new();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Checks if the current library version meets the minimum requirements.
pub fn check_version(major: i32, minor: i32, patch: i32) -> bool {
    unsafe { zvec_sys::zvec_check_version(major, minor, patch) }
}

/// Returns the major version number.
pub fn version_major() -> i32 {
    unsafe { zvec_sys::zvec_get_version_major() }
}

/// Returns the minor version number.
pub fn version_minor() -> i32 {
    unsafe { zvec_sys::zvec_get_version_minor() }
}

/// Returns the patch version number.
pub fn version_patch() -> i32 {
    unsafe { zvec_sys::zvec_get_version_patch() }
}
