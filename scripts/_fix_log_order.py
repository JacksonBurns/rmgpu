"""Move the misplaced job-05/step-04 session-log entry to the BOTTOM of the
log (chronological, newest-last per STATUS.md convention)."""
import io

P = 'STATUS.md'
text = open(P, encoding='utf-8').read()
lines = text.split('\n')

# Find the start index of the step-04 entry.
start = None
for i, ln in enumerate(lines):
    if ln.startswith('### 2026-08-31 - job-05/step-04'):
        start = i
        break
assert start is not None, 'entry not found'

# Entry runs until the next '### ' heading.
end = None
for j in range(start + 1, len(lines)):
    if lines[j].startswith('### '):
        end = j
        break
assert end is not None, 'entry end not found'

block = lines[start:end]
# Remove the block (and any immediately-following blank line so we don't
# leave a double blank before the next header).
del lines[start:end]
# After deletion, if the line right before the next header is '' and the one
# before that is also '', collapse to a single blank.
# Find the position where the block used to be (the next header is now at start
# or start+1).
k = start
if k < len(lines) and lines[k].strip() == '':
    # check for double blank
    if k + 1 < len(lines) and lines[k + 1].strip() == '':
        del lines[k]

# Append the block at the very end (after a blank line, before EOF newline).
# Ensure file ends with exactly one newline handling.
out = '\n'.join(lines)
# Strip trailing blank lines, add one newline then the block.
while out.endswith('\n\n'):
    out = out[:-1]
out = out.rstrip('\n') + '\n\n' + '\n'.join(block).rstrip('\n') + '\n'

open(P, 'w', encoding='utf-8').write(out)
print('moved entry of %d lines to bottom' % len(block))
