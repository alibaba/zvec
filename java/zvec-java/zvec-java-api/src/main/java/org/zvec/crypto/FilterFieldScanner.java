package org.zvec.crypto;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Extracts identifier-like tokens from a filter string, skipping string
 * literals (single- or double-quoted, with backslash escapes). Returns
 * every identifier token it sees, including SQL keywords like AND/OR —
 * the caller intersects with the encrypted-field set, so keywords don't
 * cause false positives.
 */
public final class FilterFieldScanner {
  private FilterFieldScanner() {}

  public static Set<String> referencedFields(String filter) {
    Set<String> ids = new LinkedHashSet<>();
    if (filter == null || filter.isEmpty()) return ids;

    int i = 0;
    int n = filter.length();
    while (i < n) {
      char c = filter.charAt(i);
      if (c == '\'' || c == '"') {
        i = skipStringLiteral(filter, i, c);
      } else if (isIdentStart(c)) {
        int start = i;
        while (i < n && isIdentPart(filter.charAt(i))) i++;
        ids.add(filter.substring(start, i));
      } else {
        i++;
      }
    }
    return ids;
  }

  private static int skipStringLiteral(String s, int start, char quote) {
    int i = start + 1;
    int n = s.length();
    while (i < n) {
      char c = s.charAt(i);
      if (c == '\\' && i + 1 < n) { i += 2; continue; }
      if (c == quote) return i + 1;
      i++;
    }
    return n;
  }

  private static boolean isIdentStart(char c) {
    return Character.isLetter(c) || c == '_';
  }

  private static boolean isIdentPart(char c) {
    return Character.isLetterOrDigit(c) || c == '_';
  }
}
