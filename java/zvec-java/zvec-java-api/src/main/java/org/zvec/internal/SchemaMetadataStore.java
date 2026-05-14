package org.zvec.internal;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.zvec.CollectionSchema;
import org.zvec.TuningProfile;
import org.zvec.VectorSchema;

public final class SchemaMetadataStore {
  private static final String VERSION_KEY = "version";
  private static final String VERSION = "1";
  private static final String FILE_NAME = ".zvec-java-schema.properties";

  private SchemaMetadataStore() {}

  public static void write(String collectionPath, CollectionSchema schema) {
    Path metadataPath = metadataPath(collectionPath);
    Properties properties = new Properties();
    properties.setProperty(VERSION_KEY, VERSION);

    for (VectorSchema vector : schema.vectors()) {
      String prefix = vectorPrefix(vector.name());
      properties.setProperty(
          prefix + "rawIndexParamsExplicit", Boolean.toString(vector.hnswIndexParams() != null));
      if (vector.tuningProfile() != null) {
        properties.setProperty(prefix + "tuningProfile", vector.tuningProfile().name());
      }
      if (vector.expectedDocCount() != null) {
        properties.setProperty(prefix + "expectedDocCount", Long.toString(vector.expectedDocCount()));
      }
    }

    try {
      Files.createDirectories(metadataPath.getParent());
      try (OutputStream output = Files.newOutputStream(metadataPath)) {
        properties.store(output, "zvec-java schema metadata");
      }
    } catch (IOException e) {
      throw new IllegalStateException("Failed to write Java schema metadata", e);
    }
  }

  public static CollectionSchema merge(String collectionPath, CollectionSchema nativeSchema) {
    Path metadataPath = metadataPath(collectionPath);
    if (!Files.isRegularFile(metadataPath)) {
      return nativeSchema;
    }

    Properties properties = new Properties();
    try (InputStream input = Files.newInputStream(metadataPath)) {
      properties.load(input);
      if (!VERSION.equals(properties.getProperty(VERSION_KEY))) {
        return nativeSchema;
      }

      List<VectorSchema> vectors = new ArrayList<>(nativeSchema.vectors().size());
      for (VectorSchema vector : nativeSchema.vectors()) {
        vectors.add(mergeVector(properties, vector));
      }
      return new CollectionSchema(nativeSchema.name(), nativeSchema.fields(), vectors);
    } catch (IOException | RuntimeException ignored) {
      return nativeSchema;
    }
  }

  private static VectorSchema mergeVector(Properties properties, VectorSchema vector) {
    String prefix = vectorPrefix(vector.name());
    String rawExplicitValue = properties.getProperty(prefix + "rawIndexParamsExplicit");
    if (rawExplicitValue == null) {
      throw new IllegalStateException(
          "Missing raw state flag in Java schema metadata: " + prefix + "rawIndexParamsExplicit");
    }
    boolean rawIndexParamsExplicit =
        parseBoolean(rawExplicitValue, prefix + "rawIndexParamsExplicit");

    TuningProfile tuningProfile = vector.tuningProfile();
    String profileValue = properties.getProperty(prefix + "tuningProfile");
    if (profileValue != null) {
      try {
        tuningProfile = TuningProfile.valueOf(profileValue);
      } catch (IllegalArgumentException e) {
        throw new IllegalStateException("Invalid tuning profile in Java schema metadata: " + profileValue, e);
      }
    }

    Long expectedDocCount = vector.expectedDocCount();
    String expectedDocCountValue = properties.getProperty(prefix + "expectedDocCount");
    if (expectedDocCountValue != null) {
      expectedDocCount = parseLong(expectedDocCountValue, prefix + "expectedDocCount");
    }

    return new VectorSchema(
        vector.name(),
        vector.dataType(),
        vector.dimension(),
        rawIndexParamsExplicit ? vector.hnswIndexParams() : null,
        tuningProfile,
        expectedDocCount);
  }

  private static long parseLong(String value, String key) {
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException e) {
      throw new IllegalStateException("Invalid long in Java schema metadata: " + key, e);
    }
  }

  private static boolean parseBoolean(String value, String key) {
    if ("true".equalsIgnoreCase(value)) {
      return true;
    }
    if ("false".equalsIgnoreCase(value)) {
      return false;
    }
    throw new IllegalStateException("Invalid boolean in Java schema metadata: " + key);
  }

  private static String vectorPrefix(String vectorName) {
    return "vector." + vectorName + ".";
  }

  private static Path metadataPath(String collectionPath) {
    return Path.of(collectionPath).resolve(FILE_NAME);
  }
}
