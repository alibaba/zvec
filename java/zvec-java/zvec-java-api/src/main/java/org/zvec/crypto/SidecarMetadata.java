package org.zvec.crypto;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Optional;

/** Reads/writes _zvec_enc.json inside a collection directory. Atomic on write. */
public final class SidecarMetadata {

  public static final String FILENAME = "_zvec_enc.json";

  private SidecarMetadata() {}

  public static Optional<EncryptionMetadata> read(Path collectionDir) {
    Path file = collectionDir.resolve(FILENAME);
    if (!Files.exists(file)) {
      return Optional.empty();
    }
    String text;
    try {
      text = Files.readString(file, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new EncryptionMetadataIOException("read sidecar failed: " + file, e);
    }
    return Optional.of(SidecarJson.read(text));
  }

  public static void write(Path collectionDir, EncryptionMetadata meta) {
    String text = SidecarJson.write(meta);
    Path file = collectionDir.resolve(FILENAME);
    Path tmp = collectionDir.resolve(FILENAME + ".tmp." + java.util.UUID.randomUUID());
    try {
      Files.createDirectories(collectionDir);
      Files.writeString(tmp, text, StandardCharsets.UTF_8);
      try {
        Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
      } catch (java.nio.file.AtomicMoveNotSupportedException e) {
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
      }
    } catch (IOException e) {
      try { Files.deleteIfExists(tmp); } catch (IOException ignored) {}
      throw new EncryptionMetadataIOException("write sidecar failed: " + file, e);
    }
  }
}
