"""Make the installed l5kit importable on modern numpy and protobuf.

l5kit is archived upstream (last release Oct 2021). It is installed with
--no-deps because its pinned dependencies no longer resolve on current
Python/macOS arm64, and this script then fixes two incompatibilities inside the
installed package:

- Its source still uses the numpy scalar aliases (np.bool, np.int, np.float,
  np.object, np.str) that numpy removed in 1.24. They are rewritten to the
  plain builtins; word-boundary matching keeps valid names like np.bool_
  untouched.
- Its semantic map protobuf module (data/proto/road_network_pb2.py) was
  generated in 2020 and does not import on the default upb backend of
  protobuf 4.21 or newer. It is replaced with the copy in third_party/l5kit/,
  regenerated from l5kit's own road_network.proto with a current protoc. The
  swap only happens when both modules embed the same serialized schema, byte
  for byte.

Idempotent: running it twice is a no-op.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path

ALIAS_RE = re.compile(r"\bnp\.(bool|int|float|object|str)\b")
PROTO_MODULE = Path("data/proto/road_network_pb2.py")
REGENERATED = (
    Path(__file__).resolve().parent.parent / "third_party/l5kit/road_network_pb2.py"
)


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


def embedded_schema(source: bytes) -> bytes:
    """The serialized FileDescriptorProto inside a generated _pb2 module, read
    without importing it: old-style modules pass it as serialized_pb=, current
    ones to AddSerializedFile()."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "AddSerializedFile":
            found += [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
        found += [
            kw.value.value
            for kw in node.keywords
            if kw.arg == "serialized_pb" and isinstance(kw.value, ast.Constant)
        ]
    if len(found) != 1 or not isinstance(found[0], bytes):
        raise ValueError(f"expected one embedded schema, found {len(found)}")
    return found[0]


def install_regenerated_proto(package_root: Path) -> bool:
    target = package_root / PROTO_MODULE
    installed = target.read_bytes()
    regenerated = REGENERATED.read_bytes()
    if installed == regenerated:
        print(f"{PROTO_MODULE} is already the regenerated module")
        return True
    if embedded_schema(installed) != embedded_schema(regenerated):
        print(
            f"{target} embeds a different schema than {REGENERATED}; "
            "regenerate that file for this l5kit version"
        )
        return False
    tmp = target.with_suffix(".py.tmp")
    tmp.write_bytes(regenerated)
    tmp.replace(target)
    print(f"replaced {PROTO_MODULE} with the regenerated module (same schema)")
    return True


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
    return 0 if install_regenerated_proto(root) else 1


if __name__ == "__main__":
    sys.exit(main())
