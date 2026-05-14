package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class AadEncoderTest {
  @Test
  void encodesLengthPrefixedConcatInBigEndian() {
    byte[] aad = AadEncoder.encode("d1", "body", "docs");

    byte[] id = "d1".getBytes(StandardCharsets.UTF_8);
    byte[] field = "body".getBytes(StandardCharsets.UTF_8);
    byte[] coll = "docs".getBytes(StandardCharsets.UTF_8);
    ByteBuffer expected = ByteBuffer.allocate(12 + id.length + field.length + coll.length);
    expected.putInt(id.length).put(id);
    expected.putInt(field.length).put(field);
    expected.putInt(coll.length).put(coll);

    assertArrayEquals(expected.array(), aad);
  }

  @Test
  void embeddedUnitSeparatorByteIsTreatedAsData() {
    // Field name with an embedded unit-separator (0x1F) byte; the length-prefix
    // encoding treats it as plain data, not as a delimiter.
    byte[] withSep = AadEncoder.encode("d1", "bodydocs", "docs");
    byte[] without = AadEncoder.encode("d1", "body", "docs");
    assertNotEquals(withSep.length, without.length);
  }

  @Test
  void rejectsNullArgs() {
    assertThrows(NullPointerException.class, () -> AadEncoder.encode(null, "f", "c"));
    assertThrows(NullPointerException.class, () -> AadEncoder.encode("d", null, "c"));
    assertThrows(NullPointerException.class, () -> AadEncoder.encode("d", "f", null));
  }

  @Test
  void supportsEmoji() {
    byte[] aad = AadEncoder.encode("👍", "body", "docs");
    byte[] thumb = "👍".getBytes(StandardCharsets.UTF_8);
    ByteBuffer first4 = ByteBuffer.wrap(aad, 0, 4);
    int len = first4.getInt();
    org.junit.jupiter.api.Assertions.assertEquals(thumb.length, len);
  }
}
