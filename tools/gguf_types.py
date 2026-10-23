"""tools/gguf_types.py - which storage type each kind of tensor of a GGUF has, from the header only (no model is loaded).

    python3 tools/gguf_types.py /data/model.gguf          # Python 3.8 or newer, standard library only

Prints, per tensor kind (blk.N. and the layer number left out), the types it appears in and how many layers have each: the list to send
when the engine stops at "blk.5.experts are Q4_K/Q4_K ..." or "blk.0.ssm_alpha.weight is Q6_K ...".  A file in several parts: the first.
"""
from __future__ import annotations

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402


def kind(name: str) -> str:
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit():
        return "blk.N." + ".".join(parts[2:])
    return name


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    g = GGUFFile(pathlib.Path(argv[1]))
    md = g.metadata
    for k in ("general.architecture", "general.name", "general.file_type", "qwen35moe.block_count",
              "qwen35moe.nextn_predict_layers"):
        print("%-34s %s" % (k, md.get(k, "(absent)")))
    kinds = collections.OrderedDict()
    for t in g.tensors:
        kinds.setdefault(kind(t.name), collections.Counter())[t.type_name] += 1
    print()
    for k, c in kinds.items():
        print("%-40s %s" % (k, ", ".join("%s x%d" % (ty, n) for ty, n in sorted(c.items(), key=lambda kv: -kv[1]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
