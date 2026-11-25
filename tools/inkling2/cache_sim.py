#!/usr/bin/env python3
"""Replays a decode trace of the expert cache (strata-inkling2 --trace-experts FILE) under several cache policies.

The file: a line "T" at the start of every decoded token, then a line per visit of a layer whose experts are in host memory: the layer, then its experts.
Prints, for several cache sizes, the share of expert uses served from the cache (hits) per policy, and the best any policy could do (Belady's, which knows the future).
usage: cache_sim.py TRACE [SLOTS ...]      (default sizes: 64 131 200 300 400)
"""
import sys
from collections import OrderedDict


def load(path):
    toks, cur = [], None
    for ln in open(path):
        f = ln.split()
        if not f:
            continue
        if f[0] == "T":
            cur = []
            toks.append(cur)
        elif cur is not None:
            ip = int(f[0])
            cur.append([(ip, int(e)) for e in f[1:]])
    return toks


def uses(toks):
    return [[k for v in t for k in v] for t in toks]


def sim_current(toks, slots, decay=0.97, admit=True):
    """The engine's policy: use counts that age by `decay` a token; a miss replaces the cache's least used expert, unless it was itself used no more (then it goes to the CPU or over PCIe)."""
    freq, cache, inc = {}, set(), 1.0
    hits = miss = 0
    for t in toks:
        inc /= decay
        for v in t:
            keys = v
            for k in keys:
                freq[k] = freq.get(k, 0.0) + inc
            taken = {k for k in keys if k in cache}
            hits += len(taken)
            for k in keys:
                if k in cache:
                    continue
                miss += 1
                cand = [c for c in cache if c not in taken]
                if len(cache) < slots:
                    cache.add(k); taken.add(k); continue
                if not cand:
                    continue
                low = min(cand, key=lambda c: freq[c])
                if admit and freq[k] <= freq[low]:
                    continue
                cache.discard(low); cache.add(k); taken.add(k)
    return hits, miss


def sim_lru(toks, slots):
    c = OrderedDict()
    hits = miss = 0
    for t in toks:
        for v in t:
            for k in v:
                if k in c:
                    hits += 1
                    c.move_to_end(k)
                else:
                    miss += 1
                    c[k] = 1
                    if len(c) > slots:
                        c.popitem(last=False)
    return hits, miss


def sim_belady(toks, slots):
    seq = [k for t in toks for v in t for k in v]
    nxt, last = [0] * len(seq), {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], 1 << 60)
        last[seq[i]] = i
    cache, hits, miss = {}, 0, 0
    for i, k in enumerate(seq):
        if k in cache:
            hits += 1
        else:
            miss += 1
            if len(cache) >= slots:
                far = max(cache, key=lambda c: cache[c])
                if cache[far] <= nxt[i]:
                    cache[k] = None
                    del cache[k]
                    continue
                del cache[far]
            cache[k] = None
        cache[k] = nxt[i]
    return hits, miss


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    toks = uses(load(sys.argv[1]))
    sizes = [int(a) for a in sys.argv[2:]] or [64, 131, 200, 300, 400]
    n = sum(len(t) for t in toks)
    # visits of one token: group back by 8 uses a visit
    vt = [[t[i:i + 8] for i in range(0, len(t), 8)] for t in toks]
    print(f"{len(toks)} tokens, {n} expert uses, {len({k for t in toks for k in t})} different experts used")
    print("slots  current(0.97)  current(0.90)  current(0.99)  no-admission  LRU  Belady(best possible)   [hit share, %]")
    for s in sizes:
        row = [sim_current(vt, s, 0.97), sim_current(vt, s, 0.90), sim_current(vt, s, 0.99), sim_current(vt, s, 0.97, False), sim_lru(vt, s), sim_belady(vt, s)]
        print(f"{s:5d}  " + "  ".join(f"{100.0 * h / max(1, h + m):12.1f}" for h, m in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
