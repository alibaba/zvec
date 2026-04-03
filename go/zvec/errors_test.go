package zvec

import (
	"errors"
	"fmt"
	"testing"
)

func TestErrorCodeConstants(t *testing.T) {
	tests := []struct {
		name     string
		got      ErrorCode
		expected int
	}{
		{"OK", ErrOK, 0},
		{"NotFound", ErrNotFound, 1},
		{"AlreadyExists", ErrAlreadyExists, 2},
		{"InvalidArgument", ErrInvalidArgument, 3},
		{"PermissionDenied", ErrPermissionDenied, 4},
		{"FailedPrecondition", ErrFailedPrecondition, 5},
		{"ResourceExhausted", ErrResourceExhausted, 6},
		{"Unavailable", ErrUnavailable, 7},
		{"InternalError", ErrInternalError, 8},
		{"NotSupported", ErrNotSupported, 9},
		{"Unknown", ErrUnknown, 10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int(tt.got) != tt.expected {
				t.Errorf("ErrorCode %s = %d, want %d", tt.name, tt.got, tt.expected)
			}
		})
	}
}

func TestErrorMessage(t *testing.T) {
	err := &Error{Code: ErrNotFound, Message: "item not found"}
	expected := "zvec error 1: item not found"
	if err.Error() != expected {
		t.Errorf("Error.Error() = %q, want %q", err.Error(), expected)
	}
}

func TestErrorMessageFormatting(t *testing.T) {
	tests := []struct {
		code    ErrorCode
		message string
		want    string
	}{
		{ErrOK, "success", "zvec error 0: success"},
		{ErrInternalError, "something broke", "zvec error 8: something broke"},
		{ErrUnknown, "", "zvec error 10: "},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("code_%d", tt.code), func(t *testing.T) {
			err := &Error{Code: tt.code, Message: tt.message}
			if err.Error() != tt.want {
				t.Errorf("Error.Error() = %q, want %q", err.Error(), tt.want)
			}
		})
	}
}

func TestErrorImplementsErrorInterface(t *testing.T) {
	var err error = &Error{Code: ErrNotFound, Message: "test"}
	if err == nil {
		t.Fatal("Error should not be nil")
	}
	if err.Error() == "" {
		t.Error("Error.Error() should not be empty")
	}
}

func TestIsNotFound(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil error", nil, false},
		{"not found error", &Error{Code: ErrNotFound, Message: "not found"}, true},
		{"other zvec error", &Error{Code: ErrInternalError, Message: "internal"}, false},
		{"standard error", errors.New("some error"), false},
		{"wrapped not found", fmt.Errorf("wrapped: %w", &Error{Code: ErrNotFound, Message: "not found"}), true},
		{"wrapped other", fmt.Errorf("wrapped: %w", &Error{Code: ErrAlreadyExists, Message: "exists"}), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsNotFound(tt.err); got != tt.want {
				t.Errorf("IsNotFound() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsAlreadyExists(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil error", nil, false},
		{"already exists error", &Error{Code: ErrAlreadyExists, Message: "exists"}, true},
		{"not found error", &Error{Code: ErrNotFound, Message: "not found"}, false},
		{"standard error", errors.New("some error"), false},
		{"wrapped already exists", fmt.Errorf("wrapped: %w", &Error{Code: ErrAlreadyExists, Message: "exists"}), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsAlreadyExists(tt.err); got != tt.want {
				t.Errorf("IsAlreadyExists() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsInvalidArgument(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil error", nil, false},
		{"invalid argument error", &Error{Code: ErrInvalidArgument, Message: "bad arg"}, true},
		{"not found error", &Error{Code: ErrNotFound, Message: "not found"}, false},
		{"standard error", errors.New("some error"), false},
		{"wrapped invalid argument", fmt.Errorf("wrapped: %w", &Error{Code: ErrInvalidArgument, Message: "bad"}), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsInvalidArgument(tt.err); got != tt.want {
				t.Errorf("IsInvalidArgument() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestErrorsAs(t *testing.T) {
	originalErr := &Error{Code: ErrNotFound, Message: "resource not found"}
	wrappedErr := fmt.Errorf("operation failed: %w", originalErr)

	var zvecErr *Error
	if !errors.As(wrappedErr, &zvecErr) {
		t.Fatal("errors.As should find *Error in wrapped error")
	}
	if zvecErr.Code != ErrNotFound {
		t.Errorf("extracted error code = %d, want %d", zvecErr.Code, ErrNotFound)
	}
	if zvecErr.Message != "resource not found" {
		t.Errorf("extracted error message = %q, want %q", zvecErr.Message, "resource not found")
	}
}

func TestSentinelErrors(t *testing.T) {
	sentinels := []*Error{
		ErrNotFoundError,
		ErrAlreadyExistsError,
		ErrInvalidArgumentError,
		ErrPermissionDeniedError,
		ErrFailedPreconditionError,
		ErrResourceExhaustedError,
		ErrUnavailableError,
		ErrInternalErrorError,
		ErrNotSupportedError,
		ErrUnknownError,
	}
	expectedCodes := []ErrorCode{
		ErrNotFound, ErrAlreadyExists, ErrInvalidArgument, ErrPermissionDenied,
		ErrFailedPrecondition, ErrResourceExhausted, ErrUnavailable,
		ErrInternalError, ErrNotSupported, ErrUnknown,
	}
	for i, sentinel := range sentinels {
		if sentinel.Code != expectedCodes[i] {
			t.Errorf("sentinel[%d].Code = %d, want %d", i, sentinel.Code, expectedCodes[i])
		}
		if sentinel.Message == "" {
			t.Errorf("sentinel[%d].Message should not be empty", i)
		}
	}
}

func BenchmarkErrorCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = &Error{Code: ErrNotFound, Message: "not found"}
	}
}

func BenchmarkIsNotFound(b *testing.B) {
	err := &Error{Code: ErrNotFound, Message: "not found"}
	for i := 0; i < b.N; i++ {
		IsNotFound(err)
	}
}

func BenchmarkErrorMessage(b *testing.B) {
	err := &Error{Code: ErrNotFound, Message: "not found"}
	for i := 0; i < b.N; i++ {
		_ = err.Error()
	}
}
