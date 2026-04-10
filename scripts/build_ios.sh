#!/bin/bash
set -e
CURRENT_DIR=$(pwd)

# Platform options: OS (arm64 device), SIMULATOR64 (x86_64 sim), SIMULATORARM64 (arm64 sim)
PLATFORM=${1:-"OS"}
BUILD_TYPE=${2:-"Release"}
IOS_DEPLOYMENT_TARGET="13.0"

# Determine architecture based on platform
case "$PLATFORM" in
    "OS")
        ARCH="arm64"
        ;;
    "SIMULATOR64")
        ARCH="x86_64"
        ;;
    "SIMULATORARM64")
        ARCH="arm64"
        ;;
    *)
        echo "error: Unknown platform '$PLATFORM'"
        echo "Usage: $0 [OS|SIMULATOR64|SIMULATORARM64] [Release|Debug]"
        echo "  OS            - Build for iOS device (arm64)"
        echo "  SIMULATOR64   - Build for iOS Simulator (x86_64)"
        echo "  SIMULATORARM64- Build for iOS Simulator (arm64, Apple Silicon)"
        exit 1
        ;;
esac

echo "Building zvec for iOS"
echo "  Platform: $PLATFORM"
echo "  Architecture: $ARCH"
echo "  Build Type: $BUILD_TYPE"
echo "  iOS Deployment Target: $IOS_DEPLOYMENT_TARGET"

# step1: use host env to compile protoc
echo "step1: building protoc for host..."

git submodule foreach --recursive 'git stash --include-untracked'

HOST_BUILD_DIR="build_host"
mkdir -p $HOST_BUILD_DIR
cd $HOST_BUILD_DIR

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j protoc
PROTOC_EXECUTABLE=$CURRENT_DIR/$HOST_BUILD_DIR/bin/protoc
cd $CURRENT_DIR

echo "step1: Done!!!"

# step2: cross build zvec for iOS
echo "step2: building zvec for iOS..."

# reset thirdparty directory
git submodule foreach --recursive 'git stash --include-untracked'

BUILD_DIR="build_ios_${PLATFORM}"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Determine SDK and additional flags based on platform
if [ "$PLATFORM" = "OS" ]; then
    SDK_NAME="iphoneos"
else
    SDK_NAME="iphonesimulator"
fi

SDK_PATH=$(xcrun --sdk $SDK_NAME --show-sdk-path)

echo "configure CMake..."
cmake \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
    -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
    -DCMAKE_OSX_SYSROOT="$SDK_PATH" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_TOOLS=OFF \
    -DCMAKE_INSTALL_PREFIX="./install" \
    -DGLOBAL_CC_PROTOBUF_PROTOC=$PROTOC_EXECUTABLE \
    -DIOS=ON \
    ../

echo "building..."
CORE_COUNT=$(sysctl -n hw.ncpu)
make -j$CORE_COUNT

echo "step2: Done!!!"

# step3: build and run all unit tests on simulator
if [ "$PLATFORM" != "OS" ]; then
  echo "step3: building and running unit tests on simulator..."

  make -j$CORE_COUNT unittest

  # Boot simulator
  DEVICE_ID=$(xcrun simctl list devices available -j \
    | python3 -c "
import json, sys
data = json.load(sys.stdin)
for runtime, devices in data['devices'].items():
    if 'iOS' in runtime:
        for d in devices:
            if 'iPhone' in d['name'] and d['isAvailable']:
                print(d['udid'])
                sys.exit(0)
sys.exit(1)
")
  xcrun simctl boot "$DEVICE_ID"
  echo "Booted simulator: $DEVICE_ID"

  # Run all test .app bundles
  FAILED_TESTS=""
  PASSED=0
  TOTAL=0

  for APP in bin/*_test.app; do
    [ -d "$APP" ] || continue
    TEST_NAME=$(basename "$APP" .app)
    BUNDLE_ID="com.zvec.${TEST_NAME}"
    TOTAL=$((TOTAL + 1))

    echo "--- Running ${TEST_NAME} ---"
    xcrun simctl install "$DEVICE_ID" "$APP"
    set +e
    xcrun simctl launch --console "$DEVICE_ID" "$BUNDLE_ID" 2>&1 | tee /tmp/${TEST_NAME}.log
    LAUNCH_EXIT=${PIPESTATUS[0]}
    set -e

    if grep -q '\[  FAILED  \]' /tmp/${TEST_NAME}.log; then
      echo "FAIL: ${TEST_NAME}"
      FAILED_TESTS="${FAILED_TESTS} ${TEST_NAME}"
    elif grep -q '\[  PASSED  \]' /tmp/${TEST_NAME}.log; then
      PASSED=$((PASSED + 1))
    elif grep -qE 'Failed: 0$' /tmp/${TEST_NAME}.log; then
      # c_api_test uses a custom test framework (not GTest)
      PASSED=$((PASSED + 1))
    elif [ "$LAUNCH_EXIT" -eq 0 ]; then
      echo "WARN: ${TEST_NAME} exited 0 but produced no recognisable test summary"
      PASSED=$((PASSED + 1))
    else
      echo "FAIL: ${TEST_NAME} exited ${LAUNCH_EXIT} with no test summary"
      FAILED_TESTS="${FAILED_TESTS} ${TEST_NAME}"
    fi
  done

  # Shutdown simulator
  xcrun simctl shutdown "$DEVICE_ID" || true

  echo ""
  echo "Test summary: ${PASSED}/${TOTAL} passed"
  if [ -n "$FAILED_TESTS" ]; then
    echo "Failed tests:${FAILED_TESTS}"
    exit 1
  fi

  echo "step3: Done!!!"
else
  echo "Skipping tests (device build cannot run on simulator)"
fi

echo ""
echo "Build completed successfully!"
echo "Output directory: $CURRENT_DIR/$BUILD_DIR"

# Test On MacOS15
# 1: xcrun simctl boot "iPhone 16" 
# 2: cd $BUILD_DIR
# 3: xcrun simctl launch --console booted com.zvec.collection_test
