// Package zvec provides Go bindings for the zvec vector database library.
//
// Zvec is an open-source, in-process vector database — lightweight, lightning-fast,
// and designed to embed directly into applications. This Go SDK wraps the zvec C-API
// using cgo to provide idiomatic Go access to all zvec functionality.
//
// Basic usage:
//
//	// Initialize the library
//	if err := zvec.Initialize(nil); err != nil {
//	    log.Fatal(err)
//	}
//	defer zvec.Shutdown()
//
//	// Create a collection schema
//	schema := zvec.NewCollectionSchema("my_collection")
//	defer schema.Destroy()
//
//	// Add fields
//	idField := zvec.NewFieldSchema("id", zvec.DataTypeString, false, 0)
//	idField.SetIndexParams(zvec.NewInvertIndexParams(true, false))
//	schema.AddField(idField)
//
//	embeddingField := zvec.NewFieldSchema("embedding", zvec.DataTypeVectorFP32, false, 128)
//	hnswParams := zvec.NewHNSWIndexParams(zvec.MetricTypeCosine, 16, 200)
//	embeddingField.SetIndexParams(hnswParams)
//	schema.AddField(embeddingField)
//
//	// Create and open collection
//	collection, err := zvec.CreateAndOpen("./my_data", schema, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer collection.Close()
package zvec

/*
#cgo CFLAGS: -I${SRCDIR}/../../src/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/lib -lzvec_c_api -Wl,-rpath,${SRCDIR}/../../build/lib

#include "zvec/c_api.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// Initialize initializes the zvec library with optional configuration.
// Pass nil to use default configuration.
// Must be called before any other zvec operations.
func Initialize(config *ConfigData) error {
	var cConfig *C.zvec_config_data_t
	if config != nil {
		cConfig = config.handle
	}
	return toError(C.zvec_initialize(cConfig))
}

// Shutdown cleans up zvec library resources.
// Should be called when the library is no longer needed.
func Shutdown() error {
	return toError(C.zvec_shutdown())
}

// IsInitialized checks if the library has been initialized.
func IsInitialized() bool {
	return bool(C.zvec_is_initialized())
}

// GetVersion returns the library version string.
func GetVersion() string {
	return C.GoString(C.zvec_get_version())
}

// CheckVersion checks if the current library version meets the minimum requirements.
func CheckVersion(major, minor, patch int) bool {
	return bool(C.zvec_check_version(C.int(major), C.int(minor), C.int(patch)))
}

// GetVersionMajor returns the major version number.
func GetVersionMajor() int {
	return int(C.zvec_get_version_major())
}

// GetVersionMinor returns the minor version number.
func GetVersionMinor() int {
	return int(C.zvec_get_version_minor())
}

// GetVersionPatch returns the patch version number.
func GetVersionPatch() int {
	return int(C.zvec_get_version_patch())
}

// ClearError clears the last error status.
func ClearError() {
	C.zvec_clear_error()
}

// ConfigData represents the global configuration for the zvec library.
type ConfigData struct {
	handle *C.zvec_config_data_t
}

// NewConfigData creates a new configuration data instance.
func NewConfigData() *ConfigData {
	handle := C.zvec_config_data_create()
	if handle == nil {
		return nil
	}
	return &ConfigData{handle: handle}
}

// Destroy releases the configuration data resources.
func (c *ConfigData) Destroy() {
	if c.handle != nil {
		C.zvec_config_data_destroy(c.handle)
		c.handle = nil
	}
}

// SetMemoryLimit sets the memory limit in bytes.
func (c *ConfigData) SetMemoryLimit(bytes uint64) error {
	return toError(C.zvec_config_data_set_memory_limit(c.handle, C.uint64_t(bytes)))
}

// GetMemoryLimit returns the memory limit in bytes.
func (c *ConfigData) GetMemoryLimit() uint64 {
	return uint64(C.zvec_config_data_get_memory_limit(c.handle))
}

// SetQueryThreadCount sets the number of query threads.
func (c *ConfigData) SetQueryThreadCount(count uint32) error {
	return toError(C.zvec_config_data_set_query_thread_count(c.handle, C.uint32_t(count)))
}

// GetQueryThreadCount returns the number of query threads.
func (c *ConfigData) GetQueryThreadCount() uint32 {
	return uint32(C.zvec_config_data_get_query_thread_count(c.handle))
}

// SetOptimizeThreadCount sets the number of optimize threads.
func (c *ConfigData) SetOptimizeThreadCount(count uint32) error {
	return toError(C.zvec_config_data_set_optimize_thread_count(c.handle, C.uint32_t(count)))
}

// GetOptimizeThreadCount returns the number of optimize threads.
func (c *ConfigData) GetOptimizeThreadCount() uint32 {
	return uint32(C.zvec_config_data_get_optimize_thread_count(c.handle))
}

// SetConsoleLog configures console logging with the specified level.
func (c *ConfigData) SetConsoleLog(level LogLevel) error {
	logConfig := C.zvec_config_log_create_console(C.zvec_log_level_t(level))
	if logConfig == nil {
		return &Error{Code: ErrInternalError, Message: "failed to create console log config"}
	}
	return toError(C.zvec_config_data_set_log_config(c.handle, logConfig))
}

// SetFileLog configures file logging.
func (c *ConfigData) SetFileLog(level LogLevel, dir, basename string, fileSizeMB, overdueDays uint32) error {
	cDir := C.CString(dir)
	defer C.free(unsafe.Pointer(cDir))
	cBasename := C.CString(basename)
	defer C.free(unsafe.Pointer(cBasename))

	logConfig := C.zvec_config_log_create_file(
		C.zvec_log_level_t(level),
		cDir,
		cBasename,
		C.uint32_t(fileSizeMB),
		C.uint32_t(overdueDays),
	)
	if logConfig == nil {
		return &Error{Code: ErrInternalError, Message: "failed to create file log config"}
	}
	return toError(C.zvec_config_data_set_log_config(c.handle, logConfig))
}
