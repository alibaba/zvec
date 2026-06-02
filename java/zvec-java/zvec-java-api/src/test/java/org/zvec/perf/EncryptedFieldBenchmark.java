package org.zvec.perf;

import java.security.SecureRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.zvec.crypto.AesGcm256;
import org.zvec.crypto.AadEncoder;

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
public class EncryptedFieldBenchmark {

  @Param({"64", "1024", "1048576"})
  public int size;

  private byte[] key;
  private byte[] nonce;
  private byte[] plaintext;
  private byte[] aad;
  private byte[] ciphertext;
  private AesGcm256 aead;

  @Setup
  public void setup() {
    aead = new AesGcm256();
    key = new byte[32];
    new SecureRandom().nextBytes(key);
    nonce = new byte[12];
    new SecureRandom().nextBytes(nonce);
    plaintext = new byte[size];
    new SecureRandom().nextBytes(plaintext);
    aad = AadEncoder.encode("d1", "body", "docs");
    ciphertext = aead.seal(key, nonce, plaintext, aad);
  }

  @Benchmark
  public byte[] encrypt() {
    return aead.seal(key, nonce, plaintext, aad);
  }

  @Benchmark
  public byte[] decrypt() {
    return aead.open(key, nonce, ciphertext, aad);
  }
}
