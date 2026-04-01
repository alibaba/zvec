#!/bin/bash
set -e
CURRENT_DIR=$(pwd)

ABI=${1:-"arm64-v8a"}
API_LEVEL=${2:-21}
BUILD_TYPE=${3:-"Release"}

# step1: use host env to compile protoc
echo "step1: building protoc for host..."
HOST_BUILD_DIR="build_host"
mkdir -p $HOST_BUILD_DIR
cd $HOST_BUILD_DIR

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j protoc
PROTOC_EXECUTABLE=$CURRENT_DIR/$HOST_BUILD_DIR/bin/protoc
cd $CURRENT_DIR

echo "step1: Done!!!"

# step2: cross build zvec based on android ndk
echo "step2: building zvec for android..."

# reset thirdparty directory
git submodule foreach --recursive 'git stash --include-untracked'

export ANDROID_SDK_ROOT=$HOME/Library/Android/sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export ANDROID_NDK_HOME=$ANDROID_SDK_ROOT/ndk/28.2.13676358
export CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake

export PATH=$PATH:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin
export PATH=$PATH:$ANDROID_SDK_ROOT/platform-tools
export PATH=$PATH:$ANDROID_NDK_HOME

if [ -z "$ANDROID_NDK_HOME" ]; then
    echo "error: ANDROID_NDK_HOME env not set"
    echo "please install NDK and set env variable ANDROID_NDK_HOME"
    exit 1
fi

BUILD_DIR="build_android_${ABI}"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "configure CMake..."
cmake \
    -DANDROID_NDK="$ANDROID_NDK_HOME" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_NATIVE_API_LEVEL="$API_LEVEL" \
    -DANDROID_STL="c++_static" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_TOOLS=OFF \
    -DCMAKE_INSTALL_PREFIX="./install" \
    -DGLOBAL_CC_PROTOBUF_PROTOC=$PROTOC_EXECUTABLE \
    ../

echo "building..."
CORE_COUNT=$(sysctl -n hw.ncpu)
make -j$CORE_COUNT

echo "step2: Done!!!"