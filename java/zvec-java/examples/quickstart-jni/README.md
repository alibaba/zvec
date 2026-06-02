# zvec-java JNI Quickstart

This example consumes `org.zvec:zvec-java-jni` as a normal Maven dependency.
It does not build native code itself; it expects the zvec-java jars to be installed
locally or available from a Maven repository.

## Run

From the repository root:

```bash
cd java/zvec-java
mvn -pl zvec-java-jni -am install -DskipTests

cd examples/quickstart-jni
mvn compile exec:java
```

The demo creates a collection, inserts a few documents, runs a vector search,
closes the collection, reopens it from disk, and runs the same search again.

To use a stable collection path instead of a temporary directory:

```bash
mvn compile exec:java -Dexec.args="/tmp/zvec-java-demo"
```

Use JDK 11 or newer for this JNI example. For the FFM backend, switch the
dependency to `org.zvec:zvec-java-ffm` and run on JDK 25.
