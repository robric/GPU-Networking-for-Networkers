# -*- coding: utf-8 -*-
"""Pipe-justify a markdown table so the raw source is readable.

Reads rows from stdin. Accepts either real markdown ("| a | b |") or plain
TSV / "|"-separated rows; any existing separator row (---|---) is ignored and
regenerated. First row is treated as the header.

    python scripts/align_table.py < rows.txt

Example rows.txt (tab- or pipe-separated, header first):
    Model | What you issue | Notes
    NVLink | load/store     | memory-semantic
    RoCE   | send/recv      | message-passing

Keep cells narrow for readability; wrap long cells with <br> rather than
letting a column blow out the table width.
"""
import sys
import re

# Force UTF-8 on stdio: on Windows, stdin/stdout default to cp1252, which
# mis-decodes multi-byte glyphs (— → § █ ·) into several chars each, so the
# column-width math over-pads and the justified table comes out crooked.
for _stream in (sys.stdin, sys.stdout):
    try:
        _stream.reconfigure(encoding='utf-8')
    except AttributeError:  # Python < 3.7
        pass


def _split(line):
    line = line.strip()
    if '|' in line:
        # strip leading/trailing pipe, then split
        line = line.strip('|')
        return [c.strip() for c in line.split('|')]
    return [c.strip() for c in line.split('\t')]


def _is_sep(cells):
    return all(re.fullmatch(r':?-{2,}:?', c or '') for c in cells) and any(cells)


def main():
    rows = []
    for raw in sys.stdin.readlines():
        if not raw.strip():
            continue
        cells = _split(raw)
        if _is_sep(cells):
            continue
        rows.append(cells)
    if not rows:
        return
    n = max(len(r) for r in rows)
    rows = [r + [''] * (n - len(r)) for r in rows]
    w = [max(len(r[c]) for r in rows) for c in range(n)]

    def line(cells):
        return "| " + " | ".join(cells[c].ljust(w[c]) for c in range(n)) + " |"

    out = [line(rows[0]), "|" + "|".join("-" * (w[c] + 2) for c in range(n)) + "|"]
    for r in rows[1:]:
        out.append(line(r))
    print("\n".join(out))


if __name__ == '__main__':
    main()
