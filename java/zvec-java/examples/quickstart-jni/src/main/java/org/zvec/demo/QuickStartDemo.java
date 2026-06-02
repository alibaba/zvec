package org.zvec.demo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.List;
import org.zvec.Collection;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.Zvec;
import org.zvec.ZvecSchemas;
import org.zvec.ZvecSearch;

public final class QuickStartDemo {
  private static final String VECTOR_FIELD = "embedding";

  private QuickStartDemo() {}

  public static void main(String[] args) throws Exception {
    Path collectionPath = collectionPath(args);
    boolean temporary = args.length == 0;
    if (!temporary && Files.exists(collectionPath)) {
      throw new IllegalArgumentException(
          "collection path already exists: " + collectionPath
              + System.lineSeparator()
              + "Choose an empty path or delete the existing directory first.");
    }

    CollectionSchema schema =
        ZvecSchemas.collection("demo_docs")
            .string("title")
            .string("category")
            .vector(VECTOR_FIELD, 4)
            .balanced()
            .build();

    try (Collection collection = Zvec.createAndOpen(collectionPath.toString(), schema)) {
      int inserted =
          collection.insert(
              List.of(
                  Doc.of("doc_1")
                      .field("title", "Vector search basics")
                      .field("category", "guide")
                      .vector(VECTOR_FIELD, new float[] {1.0f, 0.0f, 0.0f, 0.0f}),
                  Doc.of("doc_2")
                      .field("title", "Approximate nearest neighbors")
                      .field("category", "guide")
                      .vector(VECTOR_FIELD, new float[] {0.8f, 0.2f, 0.0f, 0.0f}),
                  Doc.of("doc_3")
                      .field("title", "Release checklist")
                      .field("category", "ops")
                      .vector(VECTOR_FIELD, new float[] {0.0f, 1.0f, 0.0f, 0.0f}),
                  Doc.of("doc_4")
                      .field("title", "Encrypted fields")
                      .field("category", "security")
                      .vector(VECTOR_FIELD, new float[] {0.0f, 0.0f, 1.0f, 0.0f})));
      collection.flush();
      System.out.println("Inserted " + inserted + " documents into " + collectionPath);
      printResults("Initial query", search(collection));
    }

    try (Collection reopened = Zvec.open(collectionPath.toString())) {
      printResults("Query after reopen", search(reopened));
    }

    if (temporary) {
      deleteRecursively(collectionPath.getParent());
    }
  }

  private static Path collectionPath(String[] args) throws Exception {
    if (args.length > 0) {
      return Paths.get(args[0]).toAbsolutePath().normalize();
    }
    Path runDir = Files.createTempDirectory("zvec-java-demo-");
    return runDir.resolve("docs");
  }

  private static List<Doc> search(Collection collection) {
    return collection.query(
        ZvecSearch.vector(VECTOR_FIELD, new float[] {1.0f, 0.0f, 0.0f, 0.0f})
            .topK(3)
            .project("title", "category")
            .build());
  }

  private static void printResults(String label, List<Doc> docs) {
    System.out.println();
    System.out.println(label + ":");
    for (Doc doc : docs) {
      System.out.printf(
          "  %s score=%.6f title=\"%s\" category=%s%n",
          doc.id(),
          doc.score(),
          doc.fields().get("title"),
          doc.fields().get("category"));
    }
  }

  private static void deleteRecursively(Path path) throws Exception {
    if (path == null || !Files.exists(path)) {
      return;
    }
    try (java.util.stream.Stream<Path> stream = Files.walk(path)) {
      stream.sorted(Comparator.reverseOrder()).forEach(QuickStartDemo::deleteOne);
    }
  }

  private static void deleteOne(Path path) {
    try {
      Files.deleteIfExists(path);
    } catch (Exception e) {
      throw new RuntimeException("failed to delete " + path, e);
    }
  }
}
