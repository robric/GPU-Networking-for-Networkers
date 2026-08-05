# Diagram helper scripts

Tooling for the aligned ASCII diagrams and tables in the main `README.md`.
**Reuse these — don't reinvent the box/table math each time.**

| Script | Does | Example |
| --- | --- | --- |
| `box.py` | Generates an aligned ASCII box (centered title, `-`/`=` border, indent) | `printf 'NVLink <-> peers\nPCIe <-> host\n' \| python scripts/box.py --title "GPU" --border =` |
| `align_table.py` | Pipe-justifies a markdown table so the raw source lines up | `python scripts/align_table.py < rows.txt` |
| `width_check.awk` | Reports each line's **display** width (normalizes multi-byte glyphs) so misaligned box edges stand out | `awk -f scripts/width_check.awk < diagram.txt` |

## Workflow
1. Build the box/table with `box.py` / `align_table.py` (importable too: `from box import box`).
2. Paste into `README.md`.
3. Verify edges with `width_check.awk` — every line of a given box should report the **same** width.

## Why the awk check exists
`awk length()` counts bytes. Glyphs we use a lot — block `█`, middle-dot `·`,
section `§`, arrow `→`, em/en dashes — are 2–3 bytes but render as **one**
column, so a naive byte count makes aligned boxes look broken (and vice-versa).
`width_check.awk` normalizes those to one byte before measuring. Add any new
multi-byte glyph to its `gsub` list.
