# zvec-java

Java bindings for zvec on the same desktop platforms supported by the native zvec build.

The recommended Java API is the fluent layer: `ZvecSchemas` for schema construction and `ZvecSearch` for query construction.

## Artifacts

- `org.zvec:zvec-java-jni`: JDK 11+ backend using JNI. This is the default choice for new Java users.
- `org.zvec:zvec-java-ffm`: JDK 25 backend using the Foreign Function & Memory API.
- `org.zvec:zvec-java-api`: public API and shared Java implementation. Backend artifacts bring this transitively.

The old `org.zvec:zvec-java` compatibility coordinate has been removed. Existing FFM users should depend on `org.zvec:zvec-java-ffm` directly.

Put exactly one backend artifact on the runtime classpath. If both JNI and FFM are present, startup fails unless `-Dorg.zvec.backend=jni` or `-Dorg.zvec.backend=ffm` is set.

## Requirements

- Java 25 for a full `java/zvec-java` reactor build
- Java 11+ for `zvec-java-jni`
- Java 25 for `zvec-java-ffm`
- Maven 3.8+
- CMake available on `PATH`

Native artifacts are packaged under `META-INF/native/<platform>`. Supported platform ids are
`darwin-aarch64`, `darwin-x86_64`, `linux-aarch64`, `linux-x86_64`, and `windows-x86_64`.
The Maven build uses `host` detection by default. Build each native package on a matching host
runner, or pass `-Dzvec.native.platform=<platform>` in that runner to make the package id explicit.

## Build

```bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
cd java/zvec-java

JAVA_HOME="$HOME/.sdkman/candidates/java/25.0.2-oracle" \
  mvn test

# JDK 11 JNI path
JAVA_HOME="$HOME/.sdkman/candidates/java/11.0.26-amzn" \
  mvn -pl zvec-java-jni -am test

# JDK 25 FFM path
JAVA_HOME="$HOME/.sdkman/candidates/java/25.0.2-oracle" \
  mvn -pl zvec-java-ffm -am test

# Explicit platform package on a Linux x86_64 CI runner
JAVA_HOME="$HOME/.sdkman/candidates/java/11.0.26-amzn" \
  mvn -pl zvec-java-jni -am test -Dzvec.native.platform=linux-x86_64
```

## Multi-Platform Release Artifacts

Do not commit built jars or native libraries to git. Keep `target/` local while preparing
release assets, then upload the assembled jars to GitHub Releases or publish them to a
Maven repository.

The release flow is:

1. Build each native platform on a matching runner or machine.
2. Copy the resulting `META-INF/native/<platform>` files into the backend module's
   `target/classes` directory.
3. Run `mvn package -DskipTests` without `clean` to preserve the copied platform files.
4. Upload `zvec-java-jni/target/zvec-java-jni-*.jar` and
   `zvec-java-ffm/target/zvec-java-ffm-*.jar` as release assets.

The JNI jar should contain `zvec_c_api` and `zvec_java_jni` for each packaged platform.
The FFM jar only needs `zvec_c_api` for each packaged platform.

## Example

An executable JNI quickstart is available in `examples/quickstart-jni`. It consumes
`org.zvec:zvec-java-jni` as a normal Maven dependency.

```bash
cd java/zvec-java
mvn -pl zvec-java-jni -am install -DskipTests

cd examples/quickstart-jni
mvn compile exec:java
```

## Quick Start

```java
import java.util.List;
import org.zvec.Doc;
import org.zvec.Collection;
import org.zvec.CollectionSchema;
import org.zvec.Zvec;
import org.zvec.ZvecSchemas;
import org.zvec.ZvecSearch;

CollectionSchema schema =
    ZvecSchemas.collection("docs").string("title").vector("embedding", 4).balanced().build();

try (Collection collection = Zvec.createAndOpen("./docs", schema)) {
  collection.insert(
      List.of(
          Doc.of("doc_1").field("title", "alpha").vector("embedding", new float[] {1f, 0f, 0f, 0f}),
          Doc.of("doc_2").field("title", "beta").vector("embedding", new float[] {0f, 1f, 0f, 0f})));

  List<Doc> results =
      collection.query(
          ZvecSearch.vector("embedding", new float[] {1f, 0f, 0f, 0f})
              .topK(2)
              .project("title")
              .build());
}
```

## Common Tuning

Use these fluent tuning methods immediately after `vector(name, dimension)`:

- `fast()` when you want the fastest index build and can trade off some recall
- `balanced()` for the default middle ground
- `accurate()` when search quality matters more than build speed
- `expectedDocCount(...)` when you know the collection size ahead of time

Example:

```java
CollectionSchema schema =
    ZvecSchemas.collection("docs")
        .string("title")
        .vector("embedding", 1536)
        .expectedDocCount(1_000_000L)
        .balanced()
        .build();
```

## Advanced Control

If you need direct HNSW configuration, use the compatibility layer:

```java
import org.zvec.HnswIndexParams;
import org.zvec.HnswQueryParams;
import org.zvec.VectorQuery;
import org.zvec.VectorSchema;

VectorSchema schema =
    new VectorSchema("embedding", org.zvec.DataType.VECTOR_FP32, 1536)
        .withHnswIndex(new HnswIndexParams(32, 300));

VectorQuery query =
    VectorQuery.of("embedding", new float[] {1f, 0f, 0f, 0f})
        .hnsw(new HnswQueryParams(128, 0.0f, false, true));
```

## Encrypted Fields

Mark a string field as encrypted in the schema; insert and query call sites stay identical to plaintext code.

```java
import org.zvec.crypto.KeyProvider;

KeyProvider keys = keyId -> myKms.fetchKey(keyId);   // 32 bytes for AES-256
CollectionSchema schema = ZvecSchemas.collection("docs")
    .string("title")
    .string("body").encrypted("body-key-v1")
    .vector("embed", 768).balanced()
    .build();

try (Collection col = Zvec.createAndOpen("./docs", schema, keys)) {
  col.insert(List.of(
      Doc.of("d1").field("title", "alpha")
                  .field("body", "plaintext stays plaintext at the call site")
                  .vector("embed", v)));

  List<Doc> results = col.query(
      ZvecSearch.vector("embed", q).topK(10).project("title", "body").build());

  // results.get(0).fields().get("body") is already plaintext
}
```

Reopen with the same provider:

```java
try (Collection col = Zvec.openWithKeys("./docs", keys)) { ... }
```

Key rotation (new writes use the new keyId; existing records keep their original):

```java
col.setActiveKeyId("body", "body-key-v2");
```

**Key things to know:**

- AES-256-GCM with a 12-byte random nonce per field per record. Nonce reuse under the same key would be catastrophic; use a `SecureRandom`-backed flow or never reuse keys across processes that don't coordinate nonces.
- `id`, field name, and collection name are bound into AAD automatically. Moving ciphertext between docs/fields/collections is detected.
- Queries cannot filter on encrypted fields. `ZvecSearch.filter("body = 'x'")` throws `IllegalArgumentException`.
- Decryption failures (tamper, missing key, AAD mismatch) abort the entire query — fail-loud by design.
- The library never logs key material, plaintext, or ciphertext. Caller adds logging in their own try/catch as needed.
- A static-key form `.encrypted(keyId, byte[])` is available for tests and demos. Key bytes are never persisted; reopening still requires a `KeyProvider`.
- Sidecar metadata lives at `<collection>/_zvec_enc.json`. Don't hand-edit unless you know what you're doing.

## Scope

Current support:

- create/open collection
- insert documents
- dense float vector query
- string / bool / int64 / double scalar fields

Deferred:

- update, upsert, delete, fetch
- sparse vectors
