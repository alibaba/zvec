#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  echo "usage: $0 <native-root-dir> [ffm|jni] [host|platform] [jni-source-dir]" >&2
  echo "platform: darwin-aarch64, darwin-x86_64, linux-aarch64, linux-x86_64, windows-x86_64" >&2
}

cpu_count() {
  if command -v getconf >/dev/null 2>&1; then
    local count
    count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
    if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
      echo "$count"
      return
    fi
  fi
  if command -v sysctl >/dev/null 2>&1; then
    local count
    count=$(sysctl -n hw.ncpu 2>/dev/null || true)
    if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
      echo "$count"
      return
    fi
  fi
  if [ -n "${NUMBER_OF_PROCESSORS:-}" ]; then
    echo "$NUMBER_OF_PROCESSORS"
    return
  fi
  echo 2
}

normalize_arch() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *)
      echo "error: unsupported architecture: $1" >&2
      exit 1
      ;;
  esac
}

detect_host_platform() {
  local uname_s uname_m os arch
  uname_s=$(uname -s 2>/dev/null || echo "${OS:-unknown}")
  uname_m=$(uname -m 2>/dev/null || echo "${PROCESSOR_ARCHITECTURE:-unknown}")
  arch=$(normalize_arch "$uname_m")

  case "$uname_s" in
    Darwin*) os="darwin" ;;
    Linux*) os="linux" ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT*) os="windows" ;;
    *)
      echo "error: unsupported operating system: $uname_s" >&2
      exit 1
      ;;
  esac

  echo "$(normalize_platform "$os-$arch")"
}

normalize_platform() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    host) detect_host_platform ;;
    darwin-aarch64|macos-aarch64|macos-arm64|osx-aarch64|osx-arm64) echo "darwin-aarch64" ;;
    darwin-x86_64|darwin-amd64|macos-x86_64|macos-amd64|osx-x86_64|osx-amd64) echo "darwin-x86_64" ;;
    linux-aarch64|linux-arm64) echo "linux-aarch64" ;;
    linux-x86_64|linux-amd64) echo "linux-x86_64" ;;
    windows-x86_64|windows-amd64|win32-x86_64|win32-amd64) echo "windows-x86_64" ;;
    *)
      echo "error: unsupported Java native platform: $1" >&2
      usage
      exit 1
      ;;
  esac
}

library_name() {
  local base=$1
  local platform=$2
  case "$platform" in
    darwin-*) echo "lib${base}.dylib" ;;
    linux-*) echo "lib${base}.so" ;;
    windows-*) echo "${base}.dll" ;;
    *)
      echo "error: unsupported Java native platform: $platform" >&2
      exit 1
      ;;
  esac
}

find_built_library() {
  local build_dir=$1
  local library_name=$2
  local path
  path=$(find "$build_dir" -name "$library_name" -print -quit)
  if [ -z "$path" ]; then
    echo "error: $library_name not found under $build_dir" >&2
    exit 1
  fi
  echo "$path"
}

fix_macos_install_names() {
  local core_source=$1
  local core_output=$2
  local jni_output=$3
  local core_name=$4
  local jni_name=$5

  if ! command -v install_name_tool >/dev/null 2>&1; then
    return
  fi

  install_name_tool -id "@rpath/$core_name" "$core_output" || true
  install_name_tool -id "@rpath/$jni_name" "$jni_output" || true
  install_name_tool -change "$core_source" "@loader_path/$core_name" "$jni_output" || true
  install_name_tool -change "$core_output" "@loader_path/$core_name" "$jni_output" || true
  install_name_tool -change "@rpath/$core_name" "@loader_path/$core_name" "$jni_output" || true
}

NATIVE_ROOT_DIR=${1:-}
MODE=${2:-ffm}
PLATFORM_ARG=${3:-host}
JNI_SRC_DIR=${4:-}

if [ -z "$NATIVE_ROOT_DIR" ]; then
  usage
  exit 1
fi

case "$MODE" in
  ffm|jni) ;;
  *)
    echo "error: unsupported mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

# Backward compatibility with the old jni signature:
# build_java_native.sh <native-dir> jni <jni-source-dir>
if [ "$MODE" = "jni" ] && [ -z "$JNI_SRC_DIR" ] && [ -d "$PLATFORM_ARG" ]; then
  JNI_SRC_DIR=$PLATFORM_ARG
  PLATFORM_ARG=host
fi

PLATFORM=$(normalize_platform "$PLATFORM_ARG")
HOST_PLATFORM=$(detect_host_platform)
if [ "$PLATFORM" != "$HOST_PLATFORM" ] && [ "${ZVEC_ALLOW_CROSS:-0}" != "1" ]; then
  echo "error: requested Java native platform $PLATFORM, but host platform is $HOST_PLATFORM" >&2
  echo "build on a matching host, or set ZVEC_ALLOW_CROSS=1 with an appropriate CMake toolchain" >&2
  exit 1
fi
CORE_COUNT=$(cpu_count)
OUTPUT_DIR="$NATIVE_ROOT_DIR/$PLATFORM"
BUILD_PLATFORM=${PLATFORM//-/_}
BUILD_DIR=${ZVEC_NATIVE_BUILD_DIR:-"$ROOT_DIR/build_java_native_$BUILD_PLATFORM"}
CMAKE_LOG="$BUILD_DIR/native-build.log"
CORE_LIBRARY_NAME=$(library_name zvec_c_api "$PLATFORM")
JNI_LIBRARY_NAME=$(library_name zvec_java_jni "$PLATFORM")
BUILD_JAVA_JNI_BINDING=OFF

if [ "$MODE" = "jni" ]; then
  BUILD_JAVA_JNI_BINDING=ON
  if [ -z "$JNI_SRC_DIR" ]; then
    echo "error: JNI source directory is required in jni mode" >&2
    exit 1
  fi
  if [ -z "${JAVA_HOME:-}" ]; then
    echo "error: JAVA_HOME must be set to build JNI library" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$BUILD_DIR"

if [ -e "$ROOT_DIR/.git" ]; then
  git -C "$ROOT_DIR" submodule update --init --recursive --jobs "$CORE_COUNT"
fi

if ! cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBUILD_C_BINDINGS=ON \
    -DBUILD_JAVA_JNI_BINDING="$BUILD_JAVA_JNI_BINDING" \
    -DZVEC_JAVA_HOME="${JAVA_HOME:-}" \
    -DZVEC_JAVA_JNI_SOURCE_DIR="$JNI_SRC_DIR" \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_TOOLS=OFF >"$CMAKE_LOG" 2>&1; then
  echo "error: failed to configure zvec native libraries for $PLATFORM; see $CMAKE_LOG" >&2
  exit 1
fi

if ! cmake --build "$BUILD_DIR" --target zvec_c_api --config Release -j"$CORE_COUNT" >>"$CMAKE_LOG" 2>&1; then
  echo "error: failed to build $CORE_LIBRARY_NAME; see $CMAKE_LOG" >&2
  exit 1
fi

CORE_PATH=$(find_built_library "$BUILD_DIR" "$CORE_LIBRARY_NAME")
CORE_OUTPUT="$OUTPUT_DIR/$CORE_LIBRARY_NAME"
cp "$CORE_PATH" "$CORE_OUTPUT"

if [ "$MODE" = "jni" ]; then
  if ! cmake --build "$BUILD_DIR" --target zvec_java_jni --config Release -j"$CORE_COUNT" >>"$CMAKE_LOG" 2>&1; then
    echo "error: failed to build $JNI_LIBRARY_NAME; see $CMAKE_LOG" >&2
    exit 1
  fi

  JNI_PATH=$(find_built_library "$BUILD_DIR" "$JNI_LIBRARY_NAME")
  JNI_OUTPUT="$OUTPUT_DIR/$JNI_LIBRARY_NAME"
  cp "$JNI_PATH" "$JNI_OUTPUT"

  if [[ "$PLATFORM" == darwin-* ]]; then
    fix_macos_install_names "$CORE_PATH" "$CORE_OUTPUT" "$JNI_OUTPUT" "$CORE_LIBRARY_NAME" "$JNI_LIBRARY_NAME"
  fi
fi
