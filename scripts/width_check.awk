# Display-width checker for ASCII diagrams.
#
# awk counts BYTES, but several glyphs we use display as one column while
# occupying 2-3 bytes (block U+2588, middle-dot U+00B7, section U+00A7,
# arrow U+2192, em/en dashes). We normalize those to a single byte first,
# then report each line's display width so misaligned box edges stand out.
#
# Usage (run from the Bash tool / Git Bash):
#     awk -f scripts/width_check.awk < diagram.txt
#     # or check a width range only:
#     awk -f scripts/width_check.awk diagram.txt | sort -k2 -n | uniq -c -f1
{
    s = $0
    gsub(/\xe2\x96\x88/, "x", s)   # U+2588 full block
    gsub(/\xc2\xb7/,     "x", s)   # U+00B7 middle dot
    gsub(/\xc2\xa7/,     "x", s)   # U+00A7 section sign
    gsub(/\xe2\x86\x92/, "x", s)   # U+2192 rightwards arrow
    gsub(/\xe2\x80\x94/, "x", s)   # U+2014 em dash
    gsub(/\xe2\x80\x93/, "x", s)   # U+2013 en dash
    gsub(/\xc3\x97/,     "x", s)   # U+00D7 multiplication sign
    printf "%4d  %s\n", length(s), $0
}
