"""Syntax-check every code cell of the training notebook (IPython `!` and `%` lines skipped)."""

import ast
import json
import sys

nb = json.load(open("notebooks/train_rvl_cdip.ipynb"))
checked = 0
for number, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    lines = "".join(cell["source"]).splitlines()
    kept, in_shell = [], False
    for line in lines:
        if in_shell:  # continuation of a `!command \` line
            in_shell = line.rstrip().endswith("\\")
            continue
        if line.lstrip().startswith(("!", "%")):
            in_shell = line.rstrip().endswith("\\")
            indent = line[: len(line) - len(line.lstrip())]
            kept.append(indent + "pass")  # a shell line is a statement; keep its block non-empty
            continue
        kept.append(line)
    try:
        ast.parse("\n".join(kept))
    except SyntaxError as exc:
        sys.exit(f"cell {number}: {exc}")
    checked += 1
print(f"{checked} code cells parse")
