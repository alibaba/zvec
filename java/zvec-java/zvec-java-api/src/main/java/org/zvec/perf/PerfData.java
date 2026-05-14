package org.zvec.perf;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.zvec.CollectionSchema;
import org.zvec.DataType;
import org.zvec.Doc;
import org.zvec.FieldSchema;
import org.zvec.HnswIndexParams;
import org.zvec.VectorSchema;

public final class PerfData {
  private PerfData() {}

  public static CollectionSchema schema(String name, int dimension) {
    return schema(name, dimension, null);
  }

  public static CollectionSchema schema(String name, int dimension, HnswIndexParams hnswIndexParams) {
    VectorSchema vector = new VectorSchema("embedding", DataType.VECTOR_FP32, dimension);
    if (hnswIndexParams != null) {
      vector = vector.withHnswIndex(hnswIndexParams);
    }
    return new CollectionSchema(
        name,
        List.of(
            new FieldSchema("title", DataType.STRING, false),
            new FieldSchema("bucket", DataType.INT64, false)),
        List.of(vector));
  }

  public static List<Doc> docs(int startDocIndex, int count, int dimension, long seed) {
    List<Doc> docs = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      int docIndex = startDocIndex + i;
      docs.add(doc(docIndex, dimension, seed));
    }
    return docs;
  }

  public static Doc doc(int docIndex, int dimension, long seed) {
    return Doc.of(docId(docIndex))
        .field("title", "doc-" + docIndex)
        .field("bucket", docIndex % 128L)
        .vector("embedding", vector(docIndex, dimension, seed));
  }

  public static VectorSample querySample(int docIndex, int dimension, long seed, int topK) {
    return new VectorSample(docId(docIndex), vector(docIndex, dimension, seed), topK);
  }

  public static String docId(int docIndex) {
    return "doc_" + docIndex;
  }

  public static float[] vector(int docIndex, int dimension, long seed) {
    float[] values = new float[dimension];
    fillVector(values, docIndex, seed);
    return values;
  }

  public static void fillVector(float[] values, int docIndex, long seed) {
    for (int i = 0; i < values.length; i++) {
      values[i] = hashedUnitFloat(docIndex, i, seed);
    }
  }

  private static float hashedUnitFloat(int docIndex, int dimIndex, long seed) {
    long mixed = seed;
    mixed ^= 0x9E3779B97F4A7C15L * (docIndex + 1L);
    mixed ^= 0xBF58476D1CE4E5B9L * (dimIndex + 1L);
    mixed = mix64(mixed);
    return ((mixed >>> 40) & 0xFFFFFFL) / (float) 0x1000000L;
  }

  private static long mix64(long z) {
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
    return z ^ (z >>> 31);
  }

  public static final class VectorSample {
    private final String expectedId;
    private final float[] vector;
    private final int topK;

    public VectorSample(String expectedId, float[] vector, int topK) {
      this.expectedId = Objects.requireNonNull(expectedId, "expectedId");
      this.vector = Objects.requireNonNull(vector, "vector");
      this.topK = topK;
    }

    public String expectedId() {
      return expectedId;
    }

    public float[] vector() {
      return vector;
    }

    public int topK() {
      return topK;
    }
  }
}
