# -*- coding: utf-8 -*-
"""Aligned ASCII box generator for the GPU-networking diagrams.

Importable:
    from box import box
    print(box(["line one", "line two"], title="My box", border="="))

CLI (reads body lines from stdin, one per line):
    printf 'NVLink <-> peers\nPCIe <-> host\n' | python scripts/box.py --title "GPU" --border =

Keep label naming identical across related diagrams, and verify final
column widths with scripts/width_check.awk (display glyphs != bytes for
multi-byte chars like the box-drawing block, em/en dashes, the section sign).
"""
import sys
import argparse


def box(lines, title=None, border='-', lead='   '):
    """Return a single string: a padded, aligned ASCII box.

    lines  : list[str] interior rows (already display-width aware)
    title  : optional centered title in the top border
    border : top/bottom fill char ('-' or '=')
    lead   : left indent applied to every emitted line
    """
    w = max(len(l) for l in lines)
    if title:
        w = max(w, len(title) + 4)
    out = []
    if title:
        t = ' ' + title + ' '
        total = w + 2  # interior incl 1 space pad each side
        fill = total - len(t)
        left = fill // 2
        right = fill - left
        out.append(lead + '+' + border * left + t + border * right + '+')
    else:
        out.append(lead + '+' + border * (w + 2) + '+')
    for l in lines:
        out.append(lead + '| ' + l.ljust(w) + ' |')
    out.append(lead + '+' + ('=' if border == '=' else '-') * (w + 2) + '+')
    return '\n'.join(out)


def _main():
    ap = argparse.ArgumentParser(description="Generate an aligned ASCII box from stdin lines.")
    ap.add_argument('--title', default=None, help='centered title in the top border')
    ap.add_argument('--border', default='-', choices=['-', '='], help='top/bottom fill char')
    ap.add_argument('--lead', default='   ', help='left indent for every line')
    args = ap.parse_args()
    lines = [l.rstrip('\n') for l in sys.stdin.readlines()]
    if not lines:
        lines = ['']
    print(box(lines, title=args.title, border=args.border, lead=args.lead))


if __name__ == '__main__':
    _main()
