package org.zvec.internal.ffm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class FfmNativeLoaderTest {
  @Test
  void mapsMacOsArm64ToBundledDylib() {
    assertEquals(
        "/META-INF/native/darwin-aarch64/libzvec_c_api.dylib",
        FfmNativeLoader.platformResourcePath("Mac OS X", "aarch64"));
  }

  @Test
  void mapsLinuxAndWindowsToBundledLibraries() {
    assertEquals(
        "/META-INF/native/linux-x86_64/libzvec_c_api.so",
        FfmNativeLoader.platformResourcePath("Linux", "amd64"));
    assertEquals(
        "/META-INF/native/linux-aarch64/libzvec_c_api.so",
        FfmNativeLoader.platformResourcePath("Linux", "aarch64"));
    assertEquals(
        "/META-INF/native/windows-x86_64/zvec_c_api.dll",
        FfmNativeLoader.platformResourcePath("Windows 10", "amd64"));
  }

  @Test
  void resolvesPlatformSpecificLibraryNames() {
    assertEquals("libzvec_c_api.dylib", FfmNativeLoader.cApiLibraryName("darwin-aarch64"));
    assertEquals("libzvec_c_api.so", FfmNativeLoader.cApiLibraryName("linux-x86_64"));
    assertEquals("zvec_c_api.dll", FfmNativeLoader.cApiLibraryName("windows-x86_64"));
  }

  @Test
  void rejectsUnsupportedPlatforms() {
    assertThrows(
        IllegalStateException.class,
        () -> FfmNativeLoader.platformResourcePath("Linux", "riscv64"));
  }

  @Test
  void createsUniqueExtractionTargets() {
    Path first = FfmNativeLoader.extractionTarget("/META-INF/native/darwin-aarch64/libzvec_c_api.dylib");
    Path second = FfmNativeLoader.extractionTarget("/META-INF/native/darwin-aarch64/libzvec_c_api.dylib");

    assertEquals("libzvec_c_api.dylib", first.getFileName().toString());
    assertEquals("libzvec_c_api.dylib", second.getFileName().toString());
    assertNotEquals(first, second);
    assertNotEquals(first.getParent(), second.getParent());
    assertTrue(first.getParent().getFileName().toString().startsWith("zvec-java-darwin-aarch64-"));
  }
}
