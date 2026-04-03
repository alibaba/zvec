package zvec

/*
#include "zvec/c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// ErrorCode represents a zvec error code.
type ErrorCode int

const (
	ErrOK                 ErrorCode = 0
	ErrNotFound           ErrorCode = 1
	ErrAlreadyExists      ErrorCode = 2
	ErrInvalidArgument    ErrorCode = 3
	ErrPermissionDenied   ErrorCode = 4
	ErrFailedPrecondition ErrorCode = 5
	ErrResourceExhausted  ErrorCode = 6
	ErrUnavailable        ErrorCode = 7
	ErrInternalError      ErrorCode = 8
	ErrNotSupported       ErrorCode = 9
	ErrUnknown            ErrorCode = 10
)

// Error represents a zvec error with code and message.
type Error struct {
	Code    ErrorCode
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("zvec error %d: %s", e.Code, e.Message)
}

// Sentinel errors for common error codes.
var (
	ErrNotFoundError           = &Error{Code: ErrNotFound, Message: "resource not found"}
	ErrAlreadyExistsError      = &Error{Code: ErrAlreadyExists, Message: "resource already exists"}
	ErrInvalidArgumentError    = &Error{Code: ErrInvalidArgument, Message: "invalid argument"}
	ErrPermissionDeniedError   = &Error{Code: ErrPermissionDenied, Message: "permission denied"}
	ErrFailedPreconditionError = &Error{Code: ErrFailedPrecondition, Message: "failed precondition"}
	ErrResourceExhaustedError  = &Error{Code: ErrResourceExhausted, Message: "resource exhausted"}
	ErrUnavailableError        = &Error{Code: ErrUnavailable, Message: "unavailable"}
	ErrInternalErrorError      = &Error{Code: ErrInternalError, Message: "internal error"}
	ErrNotSupportedError       = &Error{Code: ErrNotSupported, Message: "not supported"}
	ErrUnknownError            = &Error{Code: ErrUnknown, Message: "unknown error"}
)

// IsNotFound checks if the error is a not found error.
func IsNotFound(err error) bool {
	var zvecErr *Error
	return errors.As(err, &zvecErr) && zvecErr.Code == ErrNotFound
}

// IsAlreadyExists checks if the error is an already exists error.
func IsAlreadyExists(err error) bool {
	var zvecErr *Error
	return errors.As(err, &zvecErr) && zvecErr.Code == ErrAlreadyExists
}

// IsInvalidArgument checks if the error is an invalid argument error.
func IsInvalidArgument(err error) bool {
	var zvecErr *Error
	return errors.As(err, &zvecErr) && zvecErr.Code == ErrInvalidArgument
}

// toError converts a C error code to a Go error.
// Returns nil if the error code is ZVEC_OK.
func toError(code C.zvec_error_code_t) error {
	if code == C.ZVEC_OK {
		return nil
	}

	var cMsg *C.char
	C.zvec_get_last_error(&cMsg)
	defer func() {
		if cMsg != nil {
			C.zvec_free(unsafe.Pointer(cMsg))
		}
	}()

	message := "unknown error"
	if cMsg != nil {
		message = C.GoString(cMsg)
	}

	return &Error{
		Code:    ErrorCode(code),
		Message: message,
	}
}

// errorCodeToString converts an error code to its string representation.
func errorCodeToString(code ErrorCode) string {
	cStr := C.zvec_error_code_to_string(C.zvec_error_code_t(code))
	return C.GoString(cStr)
}
