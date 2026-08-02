"""Make the installed l5kit importable on modern numpy.

l5kit is archived upstream (last release Oct 2021) and its source still uses
the numpy scalar aliases (np.bool, np.int, np.float, np.object, np.str) that
numpy removed in 1.24. l5kit is installed with --no-deps because its pinned
dependencies no longer resolve on current Python/macOS arm64; this script then
rewrites the removed aliases to the plain builtins inside the installed
package. Word-boundary matching keeps valid names like np.bool_ untouched.
Idempotent: running it twice is a no-op.
"""

import importlib.util
import re
import sys
from pathlib import Path

ALIAS_RE = re.compile(r"\bnp\.(bool|int|float|object|str)\b")


def patch_tree(package_root: Path) -> int:
    changed = 0
    for path in sorted(package_root.rglob("*.py")):
        text = path.read_text()
        patched, count = ALIAS_RE.subn(r"\1", text)
        if count:
            tmp = path.with_suffix(".py.tmp")
            tmp.write_text(patched)
            tmp.replace(path)
            changed += count
            print(f"patched {count:2d} alias(es) in {path.relative_to(package_root)}")
    return changed


def main() -> int:
    spec = importlib.util.find_spec("l5kit")
    if spec is None or spec.origin is None:
        print(
            "l5kit is not installed; run `uv pip install --no-deps l5kit==1.5.0` first"
        )
        return 1
    root = Path(spec.origin).parent
    total = patch_tree(root)
    print(f"done: {total} replacement(s) in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
