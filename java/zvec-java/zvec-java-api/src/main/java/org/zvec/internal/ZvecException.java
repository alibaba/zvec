package org.zvec.internal;

public final class ZvecException extends RuntimeException {
  private final int code;

  public ZvecException(int code, String message) {
    super(message);
    this.code = code;
  }

  public int code() {
    return code;
  }
}
