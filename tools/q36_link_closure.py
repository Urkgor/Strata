#!/usr/bin/env python3
"""tools/q36_link_closure.py - which sources of the static libraries strata-q36 really links (the lists of STRATA_Q36_ONLY in CMakeLists.txt).

A static link takes only the objects a program refers to, so the linker's map says exactly which sources of strata_kernels, strata_prefill,
strata_engine, strata_core and strata_mmq the Qwen3.6 engine needs.  To read it, link a FULL build (STRATA_Q36_ONLY off) with a map:

    cd build-rhel7                                           # a full build of strata-q36
    sed 's|-o strata-q36|-Wl,-Map=q36.map -o strata-q36-map|' CMakeFiles/strata-q36.dir/link.txt | sh
    python3 ../tools/q36_link_closure.py q36.map             # what to put in the lists, and what the lists say now

Run it again after a merge of `main`, with a new option, or for another CUDA architecture: a build whose lists are short of a file stops at
the link with the symbol's name, so a stale list is loud, never a wrong engine.
"""
import os
import re
import sys

LIBS = ["libstrata_kernels.a", "libstrata_prefill.a", "libstrata_engine.a", "libstrata_core.a", "libstrata_mmq.a", "libstrata_mmq_w4a4.a"]


def linked(map_path):
    text = open(map_path, errors="ignore").read()
    section = text.split("Archive member included to satisfy reference by file (symbol)")[1].split("Discarded input sections")[0]
    found = {}
    for lib, member in re.findall(r"^(\S+\.a)\((\S+\.o)\)", section, re.M):
        found.setdefault(os.path.basename(lib), set()).add(re.sub(r"\.o$", "", member))
    return found


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    found = linked(sys.argv[1])
    for lib in LIBS:
        members = sorted(found.get(lib, []))
        print(f"{lib}: {len(members)} linked")
        if members:
            print("   ", " ".join(members))
    kept = {"libstrata_kernels.a": 5, "libstrata_prefill.a": 1}
    for lib, count in kept.items():
        if len(found.get(lib, [])) != count:
            print(f"note: CMakeLists.txt keeps {count} source(s) of {lib}; this map links {len(found.get(lib, []))}: update the list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
