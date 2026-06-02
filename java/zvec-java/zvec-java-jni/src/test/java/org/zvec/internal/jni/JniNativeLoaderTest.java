package org.zvec.internal.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class JniNativeLoaderTest {
  @Test
  void resolvesDarwinAarch64ResourceDir() {
    assertEquals(
        "/META-INF/native/darwin-aarch64",
        JniNativeLoader.platformResourceDir("Mac OS X", "aarch64"));
    assertEquals(
        "/META-INF/native/darwin-aarch64",
        JniNativeLoader.platformResourceDir("Darwin", "arm64"));
  }

  @Test
  void resolvesLinuxAndWindowsResourceDirs() {
    assertEquals(
        "/META-INF/native/linux-x86_64",
        JniNativeLoader.platformResourceDir("Linux", "amd64"));
    assertEquals(
        "/META-INF/native/linux-aarch64",
        JniNativeLoader.platformResourceDir("Linux", "aarch64"));
    assertEquals(
        "/META-INF/native/windows-x86_64",
        JniNativeLoader.platformResourceDir("Windows 10", "amd64"));
  }

  @Test
  void resolvesPlatformSpecificLibraryNames() {
    assertEquals("libzvec_c_api.dylib", JniNativeLoader.cApiLibraryName("darwin-aarch64"));
    assertEquals("libzvec_java_jni.dylib", JniNativeLoader.jniLibraryName("darwin-aarch64"));
    assertEquals("libzvec_c_api.so", JniNativeLoader.cApiLibraryName("linux-x86_64"));
    assertEquals("libzvec_java_jni.so", JniNativeLoader.jniLibraryName("linux-x86_64"));
    assertEquals("zvec_c_api.dll", JniNativeLoader.cApiLibraryName("windows-x86_64"));
    assertEquals("zvec_java_jni.dll", JniNativeLoader.jniLibraryName("windows-x86_64"));
  }

  @Test
  void rejectsUnsupportedPlatform() {
    IllegalStateException ex =
        assertThrows(
            IllegalStateException.class,
            () -> JniNativeLoader.platformResourceDir("Linux", "riscv64"));
    assertTrue(ex.getMessage().contains("Unsupported zvec-java-jni platform"));
  }

  @Test
  void extractionTargetUsesPlatformScopedTempDir() {
    Path target = JniNativeLoader.extractionTarget("/META-INF/native/darwin-aarch64");

    assertTrue(target.getFileName().toString().startsWith("zvec-java-darwin-aarch64-"));
  }
}
