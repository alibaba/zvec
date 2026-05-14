package org.zvec.crypto;

/**
 * Caller-implemented bridge between zvec's encryption layer and a key store.
 * The library never caches resolved keys across calls; callers manage their own
 * caching, KMS integration, rotation policy, and thread safety.
 */
public interface KeyProvider {

  /**
   * Resolve a keyId (as written into an envelope or scheduled in the sidecar)
   * to its raw bytes. Must return a 32-byte AES-256 key, or null if unknown.
   * Implementations may also throw — the library wraps any throwable in
   * {@link KeyResolutionException}.
   */
  byte[] resolve(String keyId);

  /**
   * Optional liveness check; the library does not call this. Provided for
   * caller-side rotation logic.
   */
  default boolean isActive(String keyId) { return true; }
}
