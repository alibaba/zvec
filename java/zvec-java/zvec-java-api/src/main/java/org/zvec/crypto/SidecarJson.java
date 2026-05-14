package org.zvec.crypto;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Minimal hand-rolled JSON codec for the _zvec_enc.json shape only.
 * Not a general-purpose JSON parser. Strings, integers, and nested objects
 * are the only types we emit or accept.
 */
final class SidecarJson {
  private SidecarJson() {}

  static String write(EncryptionMetadata meta) {
    StringBuilder sb = new StringBuilder(256);
    sb.append("{\n");
    sb.append("  \"version\": ").append(meta.version()).append(",\n");
    sb.append("  \"collection_name\": ").append(quote(meta.collectionName())).append(",\n");
    sb.append("  \"fields\": {");
    Map<String, EncryptionSpec> sorted = new TreeMap<>(meta.fields());
    boolean first = true;
    for (Map.Entry<String, EncryptionSpec> e : sorted.entrySet()) {
      if (!first) sb.append(",");
      first = false;
      sb.append("\n    ").append(quote(e.getKey())).append(": {");
      EncryptionSpec s = e.getValue();
      sb.append("\n      \"alg\": ").append(quote(s.alg())).append(",");
      sb.append("\n      \"active_key_id\": ").append(quote(s.activeKeyId())).append(",");
      sb.append("\n      \"created_at\": ").append(quote(s.createdAt().toString()));
      if (s.rotatedAt() != null) {
        sb.append(",\n      \"rotated_at\": ").append(quote(s.rotatedAt().toString()));
      }
      sb.append("\n    }");
    }
    if (!sorted.isEmpty()) sb.append("\n  ");
    sb.append("}\n}\n");
    return sb.toString();
  }

  static EncryptionMetadata read(String json) {
    Parser p = new Parser(json);
    try {
      Map<String, Object> root = p.parseObject();
      p.requireEof();
      int version = ((Number) require(root, "version")).intValue();
      String coll = (String) require(root, "collection_name");
      @SuppressWarnings("unchecked")
      Map<String, Object> fieldsObj = (Map<String, Object>) require(root, "fields");
      Map<String, EncryptionSpec> fields = new LinkedHashMap<>();
      for (Map.Entry<String, Object> e : fieldsObj.entrySet()) {
        @SuppressWarnings("unchecked")
        Map<String, Object> spec = (Map<String, Object>) e.getValue();
        String alg = (String) require(spec, "alg");
        String activeKeyId = (String) require(spec, "active_key_id");
        Instant created = Instant.parse((String) require(spec, "created_at"));
        Instant rotated = spec.containsKey("rotated_at") ? Instant.parse((String) spec.get("rotated_at")) : null;
        fields.put(e.getKey(), new EncryptionSpec(alg, activeKeyId, created, rotated));
      }
      return new EncryptionMetadata(version, coll, fields);
    } catch (IllegalArgumentException e) {
      throw e; // version / spec validation
    } catch (EncryptionMetadataIOException e) {
      throw e; // already correctly typed and messaged
    } catch (Exception e) {
      throw new EncryptionMetadataIOException("sidecar JSON parse failed: " + e.getMessage(), e);
    }
  }

  private static Object require(Map<String, Object> m, String k) {
    Object v = m.get(k);
    if (v == null) throw new EncryptionMetadataIOException("sidecar missing key: " + k, null);
    return v;
  }

  private static String quote(String s) {
    StringBuilder out = new StringBuilder(s.length() + 2);
    out.append('"');
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '"':
          out.append("\\\"");
          break;
        case '\\':
          out.append("\\\\");
          break;
        case '\n':
          out.append("\\n");
          break;
        case '\r':
          out.append("\\r");
          break;
        case '\t':
          out.append("\\t");
          break;
        default:
          if (c < 0x20) out.append(String.format("\\u%04x", (int) c));
          else out.append(c);
          break;
      }
    }
    out.append('"');
    return out.toString();
  }

  /** Single-pass recursive descent for the strict shape we emit. */
  private static final class Parser {
    private final String src;
    private int i;

    Parser(String src) { this.src = src; }

    Map<String, Object> parseObject() {
      skipWs();
      expect('{');
      Map<String, Object> m = new LinkedHashMap<>();
      skipWs();
      if (peek() == '}') { i++; return m; }
      while (true) {
        skipWs();
        String key = parseString();
        skipWs();
        expect(':');
        skipWs();
        m.put(key, parseValue());
        skipWs();
        if (peek() == ',') { i++; continue; }
        expect('}');
        return m;
      }
    }

    Object parseValue() {
      char c = peek();
      if (c == '"') return parseString();
      if (c == '{') return parseObject();
      if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
      throw new IllegalStateException("unexpected character at " + i + ": " + c);
    }

    String parseString() {
      expect('"');
      StringBuilder out = new StringBuilder();
      while (i < src.length()) {
        char c = src.charAt(i++);
        if (c == '"') return out.toString();
        if (c == '\\') {
          char esc = src.charAt(i++);
          switch (esc) {
            case '"':
              out.append('"');
              break;
            case '\\':
              out.append('\\');
              break;
            case 'n':
              out.append('\n');
              break;
            case 'r':
              out.append('\r');
              break;
            case 't':
              out.append('\t');
              break;
            case 'u':
              out.append((char) Integer.parseInt(src.substring(i, i + 4), 16));
              i += 4;
              break;
            default:
              throw new IllegalStateException("bad escape \\" + esc);
          }
        } else {
          out.append(c);
        }
      }
      throw new IllegalStateException("unterminated string");
    }

    Number parseNumber() {
      int start = i;
      if (peek() == '-') i++;
      while (i < src.length() && Character.isDigit(src.charAt(i))) i++;
      return Long.parseLong(src.substring(start, i));
    }

    void expect(char c) {
      if (i >= src.length() || src.charAt(i) != c) {
        throw new IllegalStateException("expected '" + c + "' at " + i);
      }
      i++;
    }

    void skipWs() {
      while (i < src.length() && Character.isWhitespace(src.charAt(i))) i++;
    }

    void requireEof() {
      skipWs();
      if (i < src.length()) {
        throw new IllegalStateException(
            "unexpected trailing content at offset " + i + ": '" + src.charAt(i) + "'");
      }
    }

    char peek() {
      if (i >= src.length()) throw new IllegalStateException("unexpected end of input");
      return src.charAt(i);
    }
  }
}
