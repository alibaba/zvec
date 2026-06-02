package org.zvec.internal.ffm;

import org.zvec.internal.ZvecException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public final class FfmNativeLoader {
  private static final AtomicBoolean LOADED = new AtomicBoolean(false);

  private FfmNativeLoader() {}

  static String platformResourcePath(String osName, String arch) {
    String platformId = platformId(osName, arch);
    return "/META-INF/native/" + platformId + "/" + cApiLibraryName(platformId);
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
    throw new IllegalStateException("Unsupported zvec-java platform: " + platformId);
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
      throw new IllegalStateException("Unsupported zvec-java platform: " + osName + " / " + arch);
    }
    throw new IllegalStateException("Unsupported zvec-java platform: " + osName + " / " + arch);
  }

  private static String normalizeArch(String arch) {
    String normalizedArch = arch.toLowerCase(Locale.ROOT);
    if (normalizedArch.equals("aarch64") || normalizedArch.equals("arm64")) {
      return "aarch64";
    }
    if (normalizedArch.equals("x86_64") || normalizedArch.equals("amd64")) {
      return "x86_64";
    }
    throw new IllegalStateException("Unsupported zvec-java platform: " + arch);
  }

  public static void load() {
    if (LOADED.get()) {
      return;
    }
    synchronized (LOADED) {
      if (LOADED.get()) {
        return;
      }
      String resource =
          platformResourcePath(System.getProperty("os.name"), System.getProperty("os.arch"));
      Path extracted = extract(resource);
      System.load(extracted.toAbsolutePath().toString());
      LOADED.set(true);
    }
  }

  private static Path extract(String resourcePath) {
    Path targetFile = extractionTarget(resourcePath);

    try {
      try (InputStream in = FfmNativeLoader.class.getResourceAsStream(resourcePath)) {
        if (in == null) {
          throw new IllegalStateException("Missing native resource: " + resourcePath);
        }
        Files.copy(in, targetFile);
      }
      targetFile.toFile().deleteOnExit();
      targetFile.getParent().toFile().deleteOnExit();
      return targetFile;
    } catch (IOException e) {
      throw new IllegalStateException("Failed to extract native library: " + resourcePath, e);
    }
  }

  static Path extractionTarget(String resourcePath) {
    String fileName = resourcePath.substring(resourcePath.lastIndexOf('/') + 1);
    String[] segments = resourcePath.split("/");
    if (segments.length < 2) {
      throw new IllegalStateException("Unexpected native resource path: " + resourcePath);
    }

    String platformId = segments[segments.length - 2];
    try {
      Path targetDir =
          Files.createTempDirectory(
              Path.of(System.getProperty("java.io.tmpdir")), "zvec-java-" + platformId + "-");
      return targetDir.resolve(fileName);
    } catch (IOException e) {
      throw new IllegalStateException("Failed to create temp directory for native library", e);
    }
  }
}
