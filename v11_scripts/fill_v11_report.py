#!/usr/bin/env python3
"""Apply v11 report updates to ~/bench/rabitq_refactor/comparison_report_ivf_rabitq_dot_reuse.md

Run from anywhere: python3 fill_v11_report.py
  1. In-place corrections (REPLACEMENTS from patch_v11_report_fixes.py, each
     count-asserted to match exactly once).
  2. Appends the v11 section from v11_report_append.md (idempotence guard:
     refuses to append twice).
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from patch_v11_report_fixes import REPLACEMENTS, REPORT  # noqa: E402

APPEND = (HERE / 'v11_report_append.md').read_text().rstrip('\n')


def main():
    text = REPORT.read_text()
    n = 0
    for old, new in REPLACEMENTS:
        cnt = text.count(old)
        assert cnt == 1, f'anchor matched {cnt} times (expect 1): {old[:60]!r}'
        text = text.replace(old, new)
        n += 1
    if '# v11 补充报告' in text:
        print('v11 section already present; in-place fixes were applied, '
              'append skipped (idempotence guard)')
        REPORT.write_text(text)
        print(f'applied {n}/{len(REPLACEMENTS)} in-place corrections')
        return 0
    if not text.endswith('\n'):
        text += '\n'
    text += '\n' + APPEND + '\n'
    REPORT.write_text(text)
    print(f'applied {n}/{len(REPLACEMENTS)} in-place corrections and '
          f'appended the v11 section ({len(APPEND.splitlines())} lines) '
          f'to {REPORT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
