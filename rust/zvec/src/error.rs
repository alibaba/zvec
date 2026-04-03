use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_void;

/// Error codes returned by zvec operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    NotFound,
    AlreadyExists,
    InvalidArgument,
    PermissionDenied,
    FailedPrecondition,
    ResourceExhausted,
    Unavailable,
    InternalError,
    NotSupported,
    Unknown,
}

impl From<u32> for ErrorCode {
    fn from(code: u32) -> Self {
        match code {
            1 => ErrorCode::NotFound,
            2 => ErrorCode::AlreadyExists,
            3 => ErrorCode::InvalidArgument,
            4 => ErrorCode::PermissionDenied,
            5 => ErrorCode::FailedPrecondition,
            6 => ErrorCode::ResourceExhausted,
            7 => ErrorCode::Unavailable,
            8 => ErrorCode::InternalError,
            9 => ErrorCode::NotSupported,
            _ => ErrorCode::Unknown,
        }
    }
}

impl From<ErrorCode> for u32 {
    fn from(code: ErrorCode) -> Self {
        match code {
            ErrorCode::NotFound => 1,
            ErrorCode::AlreadyExists => 2,
            ErrorCode::InvalidArgument => 3,
            ErrorCode::PermissionDenied => 4,
            ErrorCode::FailedPrecondition => 5,
            ErrorCode::ResourceExhausted => 6,
            ErrorCode::Unavailable => 7,
            ErrorCode::InternalError => 8,
            ErrorCode::NotSupported => 9,
            ErrorCode::Unknown => 10,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::NotFound => write!(f, "NotFound"),
            ErrorCode::AlreadyExists => write!(f, "AlreadyExists"),
            ErrorCode::InvalidArgument => write!(f, "InvalidArgument"),
            ErrorCode::PermissionDenied => write!(f, "PermissionDenied"),
            ErrorCode::FailedPrecondition => write!(f, "FailedPrecondition"),
            ErrorCode::ResourceExhausted => write!(f, "ResourceExhausted"),
            ErrorCode::Unavailable => write!(f, "Unavailable"),
            ErrorCode::InternalError => write!(f, "InternalError"),
            ErrorCode::NotSupported => write!(f, "NotSupported"),
            ErrorCode::Unknown => write!(f, "Unknown"),
        }
    }
}

/// An error returned by zvec operations.
#[derive(Debug, Clone)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

impl Error {
    /// Returns `true` if this is a "not found" error.
    pub fn is_not_found(&self) -> bool {
        self.code == ErrorCode::NotFound
    }

    /// Returns `true` if this is an "already exists" error.
    pub fn is_already_exists(&self) -> bool {
        self.code == ErrorCode::AlreadyExists
    }

    /// Returns `true` if this is an "invalid argument" error.
    pub fn is_invalid_argument(&self) -> bool {
        self.code == ErrorCode::InvalidArgument
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zvec error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// A specialized `Result` type for zvec operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Convert a C error code to a Rust `Result`.
///
/// Returns `Ok(())` if the code is `ZVEC_OK`, otherwise fetches the last error
/// message from the C library and returns an appropriate `Error`.
pub(crate) fn check_error(code: zvec_sys::zvec_error_code_t) -> Result<()> {
    if code == zvec_sys::ZVEC_OK {
        return Ok(());
    }

    let message = unsafe {
        let mut c_msg: *mut std::os::raw::c_char = std::ptr::null_mut();
        zvec_sys::zvec_get_last_error(&mut c_msg);

        let msg = if c_msg.is_null() {
            "unknown error".to_string()
        } else {
            let s = CStr::from_ptr(c_msg).to_string_lossy().into_owned();
            zvec_sys::zvec_free(c_msg as *mut c_void);
            s
        };
        msg
    };

    Err(Error {
        code: ErrorCode::from(code),
        message,
    })
}

/// Create a `CString` from a `&str`, returning an `InvalidArgument` error
/// if the string contains a null byte.
pub(crate) fn to_cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error {
        code: ErrorCode::InvalidArgument,
        message: "string contains null byte".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_from_u32_known_values() {
        assert_eq!(ErrorCode::from(1), ErrorCode::NotFound);
        assert_eq!(ErrorCode::from(2), ErrorCode::AlreadyExists);
        assert_eq!(ErrorCode::from(3), ErrorCode::InvalidArgument);
        assert_eq!(ErrorCode::from(4), ErrorCode::PermissionDenied);
        assert_eq!(ErrorCode::from(5), ErrorCode::FailedPrecondition);
        assert_eq!(ErrorCode::from(6), ErrorCode::ResourceExhausted);
        assert_eq!(ErrorCode::from(7), ErrorCode::Unavailable);
        assert_eq!(ErrorCode::from(8), ErrorCode::InternalError);
        assert_eq!(ErrorCode::from(9), ErrorCode::NotSupported);
    }

    #[test]
    fn error_code_from_u32_unknown_falls_back() {
        assert_eq!(ErrorCode::from(0), ErrorCode::Unknown);
        assert_eq!(ErrorCode::from(10), ErrorCode::Unknown);
        assert_eq!(ErrorCode::from(99), ErrorCode::Unknown);
        assert_eq!(ErrorCode::from(u32::MAX), ErrorCode::Unknown);
    }

    #[test]
    fn error_code_to_u32_roundtrip() {
        let codes = [
            ErrorCode::NotFound,
            ErrorCode::AlreadyExists,
            ErrorCode::InvalidArgument,
            ErrorCode::PermissionDenied,
            ErrorCode::FailedPrecondition,
            ErrorCode::ResourceExhausted,
            ErrorCode::Unavailable,
            ErrorCode::InternalError,
            ErrorCode::NotSupported,
            ErrorCode::Unknown,
        ];
        for code in codes {
            let numeric: u32 = code.into();
            assert!(numeric >= 1 && numeric <= 10, "code {:?} mapped to {}", code, numeric);
        }
    }

    #[test]
    fn error_code_display() {
        assert_eq!(ErrorCode::NotFound.to_string(), "NotFound");
        assert_eq!(ErrorCode::AlreadyExists.to_string(), "AlreadyExists");
        assert_eq!(ErrorCode::InvalidArgument.to_string(), "InvalidArgument");
        assert_eq!(ErrorCode::PermissionDenied.to_string(), "PermissionDenied");
        assert_eq!(ErrorCode::FailedPrecondition.to_string(), "FailedPrecondition");
        assert_eq!(ErrorCode::ResourceExhausted.to_string(), "ResourceExhausted");
        assert_eq!(ErrorCode::Unavailable.to_string(), "Unavailable");
        assert_eq!(ErrorCode::InternalError.to_string(), "InternalError");
        assert_eq!(ErrorCode::NotSupported.to_string(), "NotSupported");
        assert_eq!(ErrorCode::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn error_is_helpers() {
        let not_found = Error {
            code: ErrorCode::NotFound,
            message: "item missing".into(),
        };
        assert!(not_found.is_not_found());
        assert!(!not_found.is_already_exists());
        assert!(!not_found.is_invalid_argument());

        let already_exists = Error {
            code: ErrorCode::AlreadyExists,
            message: "duplicate".into(),
        };
        assert!(!already_exists.is_not_found());
        assert!(already_exists.is_already_exists());
        assert!(!already_exists.is_invalid_argument());

        let invalid_arg = Error {
            code: ErrorCode::InvalidArgument,
            message: "bad param".into(),
        };
        assert!(!invalid_arg.is_not_found());
        assert!(!invalid_arg.is_already_exists());
        assert!(invalid_arg.is_invalid_argument());
    }

    #[test]
    fn error_display_format() {
        let err = Error {
            code: ErrorCode::InternalError,
            message: "something broke".into(),
        };
        assert_eq!(err.to_string(), "zvec error InternalError: something broke");
    }

    #[test]
    fn error_implements_std_error() {
        let err = Error {
            code: ErrorCode::NotFound,
            message: "gone".into(),
        };
        let std_err: &dyn std::error::Error = &err;
        assert!(std_err.to_string().contains("NotFound"));
    }

    #[test]
    fn error_clone() {
        let err = Error {
            code: ErrorCode::Unavailable,
            message: "service down".into(),
        };
        let cloned = err.clone();
        assert_eq!(cloned.code, ErrorCode::Unavailable);
        assert_eq!(cloned.message, "service down");
    }

    #[test]
    fn to_cstring_valid_string() {
        let result = to_cstring("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().to_str().unwrap(), "hello");
    }

    #[test]
    fn to_cstring_empty_string() {
        let result = to_cstring("");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().to_str().unwrap(), "");
    }

    #[test]
    fn to_cstring_with_null_byte_returns_error() {
        let result = to_cstring("hello\0world");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_invalid_argument());
        assert!(err.message.contains("null byte"));
    }

    #[test]
    fn to_cstring_unicode() {
        let result = to_cstring("你好世界🌍");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().to_str().unwrap(), "你好世界🌍");
    }

    #[test]
    fn error_code_copy_semantics() {
        let code = ErrorCode::NotFound;
        let copied = code;
        assert_eq!(code, copied);
    }

    #[test]
    fn error_code_eq() {
        assert_eq!(ErrorCode::NotFound, ErrorCode::NotFound);
        assert_ne!(ErrorCode::NotFound, ErrorCode::AlreadyExists);
    }
}
