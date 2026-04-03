//go:build integration

package zvec

import (
	"testing"
)

func TestMain(m *testing.M) {
	if err := Initialize(nil); err != nil {
		panic("failed to initialize zvec: " + err.Error())
	}
	defer Shutdown()

	m.Run()
}

func TestGetVersion(t *testing.T) {
	version := GetVersion()
	if version == "" {
		t.Error("GetVersion() returned empty string")
	}
	t.Logf("Version: %s", version)
}

func TestGetVersionComponents(t *testing.T) {
	major := GetVersionMajor()
	minor := GetVersionMinor()
	patch := GetVersionPatch()

	if major < 0 {
		t.Errorf("GetVersionMajor() returned negative value: %d", major)
	}
	if minor < 0 {
		t.Errorf("GetVersionMinor() returned negative value: %d", minor)
	}
	if patch < 0 {
		t.Errorf("GetVersionPatch() returned negative value: %d", patch)
	}

	t.Logf("Version components: %d.%d.%d", major, minor, patch)
}

func TestCheckVersion(t *testing.T) {
	// CheckVersion(0,0,0) should always return true
	if !CheckVersion(0, 0, 0) {
		t.Error("CheckVersion(0,0,0) returned false, expected true")
	}

	// Check against current version
	major := GetVersionMajor()
	minor := GetVersionMinor()
	patch := GetVersionPatch()
	if !CheckVersion(major, minor, patch) {
		t.Errorf("CheckVersion(%d,%d,%d) returned false, expected true", major, minor, patch)
	}
}

func TestIsInitialized(t *testing.T) {
	if !IsInitialized() {
		t.Error("IsInitialized() returned false after Initialize()")
	}
}

func TestClearError(t *testing.T) {
	// Should not panic
	ClearError()
}

func TestConfigData(t *testing.T) {
	config := NewConfigData()
	if config == nil {
		t.Fatal("NewConfigData() returned nil")
	}
	defer config.Destroy()

	// Test MemoryLimit round-trip
	testMemoryLimit := uint64(1024 * 1024 * 1024) // 1GB
	if err := config.SetMemoryLimit(testMemoryLimit); err != nil {
		t.Errorf("SetMemoryLimit(%d) failed: %v", testMemoryLimit, err)
	}
	if got := config.GetMemoryLimit(); got != testMemoryLimit {
		t.Errorf("GetMemoryLimit() = %d, want %d", got, testMemoryLimit)
	}

	// Test QueryThreadCount round-trip
	testQueryThreadCount := uint32(4)
	if err := config.SetQueryThreadCount(testQueryThreadCount); err != nil {
		t.Errorf("SetQueryThreadCount(%d) failed: %v", testQueryThreadCount, err)
	}
	if got := config.GetQueryThreadCount(); got != testQueryThreadCount {
		t.Errorf("GetQueryThreadCount() = %d, want %d", got, testQueryThreadCount)
	}

	// Test OptimizeThreadCount round-trip
	testOptimizeThreadCount := uint32(2)
	if err := config.SetOptimizeThreadCount(testOptimizeThreadCount); err != nil {
		t.Errorf("SetOptimizeThreadCount(%d) failed: %v", testOptimizeThreadCount, err)
	}
	if got := config.GetOptimizeThreadCount(); got != testOptimizeThreadCount {
		t.Errorf("GetOptimizeThreadCount() = %d, want %d", got, testOptimizeThreadCount)
	}
}

func TestConfigDataConsoleLog(t *testing.T) {
	config := NewConfigData()
	if config == nil {
		t.Fatal("NewConfigData() returned nil")
	}
	defer config.Destroy()

	// Test setting console log with different levels
	levels := []LogLevel{LogLevelDebug, LogLevelInfo, LogLevelWarn, LogLevelError}
	for _, level := range levels {
		if err := config.SetConsoleLog(level); err != nil {
			t.Errorf("SetConsoleLog(%v) failed: %v", level, err)
		}
	}
}

func TestConfigDataDestroy(t *testing.T) {
	config := NewConfigData()
	if config == nil {
		t.Fatal("NewConfigData() returned nil")
	}

	// First Destroy should not panic
	config.Destroy()

	// Second Destroy should also not panic
	config.Destroy()
}

func TestConfigDataNil(t *testing.T) {
	config := NewConfigData()
	if config == nil {
		t.Error("NewConfigData() returned nil, expected non-nil")
	}
	if config != nil {
		config.Destroy()
	}
}
