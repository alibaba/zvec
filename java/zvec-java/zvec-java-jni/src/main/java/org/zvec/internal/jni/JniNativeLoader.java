package org.zvec.internal.jni;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public final class JniNativeLoader {
  private static final AtomicBoolean LOADED = new AtomicBoolean(false);

  private JniNativeLoader() {}

  static String platformResourceDir(String osName, String arch) {
    return "/META-INF/native/" + platformId(osName, arch);
  }

  static String cApiLibraryName(String platformId) {
    if (platformId.startsWith("windows-")) {
      return "zvec_c_api.dll";
    }
    if (platformId.startsWith("darwin-")) {
      return "libzvec_c_api.dylib";
    }
    if (platformId.startsWith("linux-")) {
      return "libzvec_c_api.so";
    }
    throw new IllegalStateException("Unsupported zvec-java-jni platform: " + platformId);
  }

  static String jniLibraryName(String platformId) {
    if (platformId.startsWith("windows-")) {
      return "zvec_java_jni.dll";
    }
    if (platformId.startsWith("darwin-")) {
      return "libzvec_java_jni.dylib";
    }
    if (platformId.startsWith("linux-")) {
      return "libzvec_java_jni.so";
    }
    throw new IllegalStateException("Unsupported zvec-java-jni platform: " + platformId);
  }

  private static String platformId(String osName, String arch) {
    String normalizedOs = osName.toLowerCase(Locale.ROOT);
    String normalizedArch = normalizeArch(arch);
    if (normalizedOs.contains("mac") || normalizedOs.contains("darwin")) {
      return "darwin-" + normalizedArch;
    }
    if (normalizedOs.contains("linux")) {
      return "linux-" + normalizedArch;
    }
    if (normalizedOs.contains("windows")) {
      if (normalizedArch.equals("x86_64")) {
        return "windows-x86_64";
      }
      throw new IllegalStateException("Unsupported zvec-java-jni platform: " + osName + " / " + arch);
    }
    throw new IllegalStateException("Unsupported zvec-java-jni platform: " + osName + " / " + arch);
  }

  private static String normalizeArch(String arch) {
    String normalizedArch = arch.toLowerCase(Locale.ROOT);
    if (normalizedArch.equals("aarch64") || normalizedArch.equals("arm64")) {
      return "aarch64";
    }
    if (normalizedArch.equals("x86_64") || normalizedArch.equals("amd64")) {
      return "x86_64";
    }
    throw new IllegalStateException("Unsupported zvec-java-jni platform: " + arch);
  }

  public static void load() {
    if (LOADED.get()) {
      return;
    }
    synchronized (LOADED) {
      if (LOADED.get()) {
        return;
      }
      String resourceDir =
          platformResourceDir(System.getProperty("os.name"), System.getProperty("os.arch"));
      String platformId = resourceDir.substring(resourceDir.lastIndexOf('/') + 1);
      Path targetDir = extractionTarget(resourceDir);
      Path cApi = extract(resourceDir + "/" + cApiLibraryName(platformId), targetDir);
      Path jni = extract(resourceDir + "/" + jniLibraryName(platformId), targetDir);
      System.load(cApi.toAbsolutePath().toString());
      System.load(jni.toAbsolutePath().toString());
      LOADED.set(true);
    }
  }

  static Path extractionTarget(String resourceDir) {
    String platformId = resourceDir.substring(resourceDir.lastIndexOf('/') + 1);
    try {
      Path targetDir =
          Files.createTempDirectory(
              Path.of(System.getProperty("java.io.tmpdir")), "zvec-java-" + platformId + "-");
      targetDir.toFile().deleteOnExit();
      return targetDir;
    } catch (IOException e) {
      throw new IllegalStateException("Failed to create temp directory for native libraries", e);
    }
  }

  private static Path extract(String resourcePath, Path targetDir) {
    Path targetFile = targetDir.resolve(resourcePath.substring(resourcePath.lastIndexOf('/') + 1));

    try {
      try (InputStream in = JniNativeLoader.class.getResourceAsStream(resourcePath)) {
        if (in == null) {
          throw new IllegalStateException("Missing native resource: " + resourcePath);
        }
        Files.copy(in, targetFile);
      }
      targetFile.toFile().deleteOnExit();
      return targetFile;
    } catch (IOException e) {
      throw new IllegalStateException("Failed to extract native library: " + resourcePath, e);
    }
  }
}
