#!/usr/bin/env python3
"""tools/make_q36_only.py - the tree for ONE model: Qwen3.6-35B-A3B (qwen35moe) on a CUDA / Linux machine.

Strata is several engines (Flash-Next, DeepSeek-V4.1, Mac, Intel, AMD), their installers, apps and benchmarks.  For one model on one kind of machine
most of the repository is never built nor run.  This removes, with `git rm`, what the Qwen3.6 engine (src/qwen35 and what it links), its server (serve/)
and the build (CMakeLists.txt, tools/build_rhel7.sh) do not need:

    git checkout -B feature/q36-only feature/RHEL7_compatibility     # a branch derived from the one you work on
    python3 tools/make_q36_only.py                                   # a dry run: what goes, with line counts
    python3 tools/make_q36_only.py --apply                           # git rm + one commit

It is a script and not a one-off deletion so that the derived branch can be made again after `main` is merged into the source branch: delete
the derived branch and run the three commands again (a merge of the source into the derived branch would be one conflict per file `main`
changed in a removed folder).

What it keeps, so nothing that runs breaks:
  * src/, include/, cmake/, third_party/, CMakeLists.txt (the engines' source stays: CMake builds `strata-q36` through the common libraries; the
    sources of other backends and of the DeepSeek engine are behind options that are off - see the checks at the end of the run);
  * serve/ whole, plus the modules of strata_control/, strata_cli/ and tools/ that the server imports (found by reading the imports, not by a list);
    a serve test that needs a removed module is removed with it;
  * tools/qwen35/ (Qwen3.6 parity and gate scripts), tools/build_rhel7.sh, tools/check_glibc_symbols.sh, this script;
  * docs/ only for the files in KEEP_DOCS, tests/core/q36_* (the CPU tests of the engine's cache and attention).

Removes nothing the server imports; the checks at the end say what to look at (a CMake source path that is gone, a serve test that fails to import).
"""
import argparse
import ast
import fnmatch
import glob
import os
import re
import subprocess
import sys

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()

# Folders that go whole (tests/ and docs/ have exceptions, below).
REMOVE_DIRS = [
    "bench",                   # 480 k lines of measurements
    "sycl", "metal",           # the Intel Arc and Apple backends
    "apps",                    # the Strata app and its desktop shell
    "src/dsv41",               # the DeepSeek engine (its CMake targets are behind STRATA_BUILD_DSV41, off)
    "packaging", "docker", "integrations", ".github", "spark", "ref", "bin",
    "data/dsv41", "data/quant_quality",
    "tests",                   # except KEEP_TESTS
    "docs",                    # except KEEP_DOCS
]
REMOVE_FILES = [
    "setup.py", "setup.sh", "strata.py", "chat.py", "highlight.py", "START-HERE.bat", "UPDATE.bat", "update.sh", "Dockerfile", ".dockerignore",
    "flake.nix", "flake.lock", "CONTRIBUTING.md", "data/expert-profile-sycl.bin", "data/expert-profile-coder.bin",
    "README.de.md", "README.es.md", "README.fr.md", "README.ja.md", "README.pl.md", "README.pt-BR.md", "README.zh-CN.md",
]
# tests/core/dma_batch_parity.cu: CMake builds that test executable whether or not STRATA_BUILD_TESTS is on (found by configuring the CUDA build)
KEEP_TESTS = ["tests/core/q36_*", "tests/core/dma_batch_parity.cu"]
KEEP_DOCS = ["docs/QWEN36.md", "docs/RHEL7.md", "docs/DETAILS.md", "docs/BATCHING.md", "docs/OLDER_GPUS.md", "docs/QUANTS.md",
             "docs/INSTALL.md", "docs/TROUBLESHOOTING.md", "docs/HOW_IT_WORKS.md", "docs/LOGIT_BIAS.md", "docs/README.md"]
# tools/: only these (and what the server imports, found below) stay.
KEEP_TOOLS = ["tools/build_rhel7.sh", "tools/check_glibc_symbols.sh", "tools/make_q36_only.py", "tools/qwen35/*", "tools/strata_tokenizer.py",
              "tools/test_strata_tokenizer.py", "tools/unicode_classes.py", "tools/gen_unicode_classes.py", "tools/gguf_reader.py",
              "tools/gguf_types.py", "tools/strata_inspect.py", "tools/test_strata_inspect.py"]

SEARCH = [ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "serve")]


def resolve(module):
    parts = module.split(".")
    for base in SEARCH:
        path = os.path.join(base, *parts)
        if os.path.isfile(path + ".py"):
            return path + ".py"
        if os.path.isfile(os.path.join(path, "__init__.py")):
            return os.path.join(path, "__init__.py")
    return None


def imports_of(path):
    try:
        tree = ast.parse(open(path, encoding="utf8", errors="ignore").read())
    except (SyntaxError, OSError):
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = os.path.dirname(path)
                for _ in range(node.level - 1):
                    base = os.path.dirname(base)
                head = os.path.relpath(base, ROOT).replace(os.sep, ".")
                module = head + "." + node.module if node.module else head
            else:
                module = node.module or ""
            found.append(module)
            found += [module + "." + alias.name for alias in node.names]
    return found


def closure(seeds):
    """Every repository module reachable from `seeds` by imports (function-level imports included)."""
    seen, stack = set(), list(seeds)
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        for module in imports_of(path):
            target = resolve(module)
            if target and target.startswith(ROOT + os.sep) and target not in seen:
                stack.append(target)
    return seen


def lines(path):
    try:
        return sum(1 for _ in open(os.path.join(ROOT, path), "rb"))
    except OSError:
        return 0


def matches(path, patterns):
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def plan():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    tracked = [p for p in tracked if p]
    server = [p for p in glob.glob(os.path.join(ROOT, "serve", "*.py")) if not os.path.basename(p).startswith("test_")]
    needed = {os.path.relpath(p, ROOT) for p in closure(server)}
    remove = set()
    for path in tracked:
        top = path.split("/")[0]
        under = lambda d: path == d or path.startswith(d + "/")
        if path in REMOVE_FILES or any(under(d) for d in REMOVE_DIRS):
            remove.add(path)
        elif top == "tools" and path not in needed and not matches(path, KEEP_TOOLS):
            remove.add(path)
        elif top in ("strata_control", "strata_cli") and path.endswith(".py") and path not in needed:
            remove.add(path)
    for path in list(remove):
        if matches(path, KEEP_TESTS if path.startswith("tests/") else KEEP_DOCS if path.startswith("docs/") else []):
            remove.discard(path)
    # a serve test is kept only when every module it imports is kept
    for path in tracked:
        if path.startswith("serve/test_") and path.endswith(".py") and path not in remove:
            missing = [os.path.relpath(m, ROOT) for m in closure([os.path.join(ROOT, path)]) if os.path.relpath(m, ROOT) in remove]
            if missing:
                remove.add(path)
    # the same for the tools' tests
    for path in tracked:
        if path.startswith("tools/test_") and path.endswith(".py") and path not in remove:
            missing = [os.path.relpath(m, ROOT) for m in closure([os.path.join(ROOT, path)]) if os.path.relpath(m, ROOT) in remove]
            if missing:
                remove.add(path)
    return tracked, sorted(remove)


def checks():
    """What to look at after the removal: CMake source paths that no longer exist, outside the options that are off."""
    problems = []
    text = open(os.path.join(ROOT, "CMakeLists.txt"), encoding="utf8").read()
    for ref in sorted(set(re.findall(r"\b((?:src|include|tests|bench|tools|third_party)/[A-Za-z0-9_./-]+\.(?:cu|cuh|cpp|hpp|h|cmake|py|json|bin))", text))):
        if not os.path.exists(os.path.join(ROOT, ref)):
            problems.append(ref)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="git rm the files and commit (default: only say what would go)")
    args = parser.parse_args()
    tracked, remove = plan()
    total = sum(lines(p) for p in tracked)
    gone = sum(lines(p) for p in remove)
    by = {}
    for p in remove:
        key = p.split("/")[0] if "/" not in p or p.split("/")[0] not in ("src", "data", "tools", "tests", "docs") else "/".join(p.split("/")[:2])
        by.setdefault(key, [0, 0])
        by[key][0] += 1
        by[key][1] += lines(p)
    for key, (count, size) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {size:>8} lines  {count:>5} files  {key}")
    print(f"remove {len(remove)} of {len(tracked)} files, {gone} of {total} lines ({100 * gone // max(total, 1)} %)")
    if not args.apply:
        print("dry run: nothing was removed (--apply removes and commits)")
        return 0
    for start in range(0, len(remove), 400):
        subprocess.run(["git", "rm", "-q", "--ignore-unmatch", "--", *remove[start:start + 400]], cwd=ROOT, check=True)
    for path, _, files in os.walk(ROOT, topdown=False):      # git leaves no empty folders, but a __pycache__ may remain
        if path != ROOT and not files and not os.listdir(path):
            os.rmdir(path)
    # B (CMakeLists.txt, STRATA_Q36_ONLY): in this tree it is on by default - the engines it leaves uncompiled are not here to link anyway
    cmake = os.path.join(ROOT, "CMakeLists.txt")
    text = open(cmake, encoding="utf8").read()
    anchor = 'the target strata no longer links" OFF)'
    if anchor not in text:
        sys.exit("CMakeLists.txt has no STRATA_Q36_ONLY option to turn on: the option moved, update make_q36_only.py")
    open(cmake, "w", encoding="utf8").write(text.replace(anchor, 'the target strata no longer links" ON)'))
    build = os.path.join(ROOT, "tools", "build_rhel7.sh")        # ... and so is --q36-only of the build script (the target `strata` cannot link here)
    script = open(build, encoding="utf8").read()
    if 'Q36ONLY=""' not in script:
        sys.exit("tools/build_rhel7.sh has no Q36ONLY default to turn on: update make_q36_only.py")
    open(build, "w", encoding="utf8").write(script.replace('Q36ONLY=""', 'Q36ONLY="1"', 1))
    subprocess.run(["git", "add", "CMakeLists.txt", "tools/build_rhel7.sh"], cwd=ROOT, check=True)
    missing = checks()
    subprocess.run(["git", "commit", "-q", "-m", f"q36-only: remove what Qwen3.6-35B-A3B on CUDA does not use ({len(remove)} files, {gone} lines); "
                    "derived by tools/make_q36_only.py"], cwd=ROOT, check=True)
    print("committed.")
    if missing:
        print("CMake names these sources, which are gone (fine if they are behind an option that is off):")
        for ref in missing:
            print("  ", ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
