// q36_ckpt.hpp - how the Qwen3.6 engine (q36_engine.cu) decides what of a prompt it reads again (--prompt-cache).
//
// WHY.  The engine holds a conversation as a recurrent state (the GDN layers cannot step back) plus a positional K / V cache.  A
// request used to reuse that state only when ALL the held tokens (the prompt and every token generated for it) were a proper
// start of the new prompt.  A chat client re-renders the previous reply (trimmed thinking, re-serialised tool calls), so one
// token of difference anywhere and the whole prompt was read again.  Two more reuse points fix that:
//   * the turn checkpoint: the state after the conversation so far, taken just before the last turn token (<|im_start|>) of
//     the prompt - what the next request of the same chat re-renders identically;
//   * the pin: the state after the first N tokens (pin=N, a start many requests share - a system prompt and a document),
//     kept apart from the rotating turn checkpoint, one for the engine.
// This header is the decision only (no CUDA): which held state a prompt continues, and where the read is cut for the new
// checkpoints.  tests/core/q36_ckpt_test.cpp holds it to the cases of STRATA_QWEN36_CHECKPOINTS.md.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace q36ck {

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

/// L, the number of held tokens, when `ids` has them as a PROPER start and the pictures in those cells are the held ones; else 0
/// (the held tokens end inside a picture, or a picture differs).  The rule of the live state, and of a checkpoint.
inline size_t match(const std::vector<int32_t>& held, const std::vector<ImgKey>& held_imgs, const std::vector<int32_t>& ids,
                    const std::vector<ImgKey>& imgs) {
    const size_t L = held.size();
    if (L == 0 || L >= ids.size() || !std::equal(held.begin(), held.end(), ids.begin())) return 0;
    std::vector<ImgKey> in_held;
    for (const ImgKey& k : imgs) {
        if (k.cell + k.n <= (int64_t) L) in_held.push_back(k);
        else if (k.cell < (int64_t) L) return 0;   // the held tokens end inside a picture
    }
    return in_held == held_imgs ? L : 0;
}

enum Source { kLive = 0, kTurn = 1, kPin = 2 };

/// A checkpoint as the decision sees it.
struct Held {
    bool valid = false;
    const std::vector<int32_t>* ids = nullptr;
    const std::vector<ImgKey>* imgs = nullptr;
};

struct Choice {
    size_t len = 0;        // tokens not read again
    Source src = kLive;    // whose state: the live state (nothing to restore), the session's turn checkpoint, the pin
};

/// The longest reusable start of `ids`: the live state first (it keeps the speculative drafts, and is the fast path when the
/// re-rendered reply happens to be identical), the turn checkpoint, the pin; the first of equal lengths wins.
inline Choice choose(const std::vector<int32_t>& live, const std::vector<ImgKey>& live_imgs, const Held& turn, const Held& pin,
                     const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) {
    Choice c{match(live, live_imgs, ids, imgs), kLive};
    if (turn.valid && turn.ids) {
        const size_t l = match(*turn.ids, *turn.imgs, ids, imgs);
        if (l > c.len) c = {l, kTurn};
    }
    if (pin.valid && pin.ids) {
        const size_t l = match(*pin.ids, *pin.imgs, ids, imgs);
        if (l > c.len) c = {l, kPin};
    }
    return c;
}

/// The cut `cut` falls inside a picture's cells (a state there cannot be continued).
inline bool inside_picture(int64_t cut, const std::vector<ImgKey>& imgs) {
    for (const ImgKey& k : imgs)
        if (k.cell < cut && cut < k.cell + k.n) return true;
    return false;
}

/// One place the read stops to keep a checkpoint: `at` tokens are read, then the state is saved as the turn checkpoint and / or the pin.
struct Cut {
    int64_t at = 0;
    bool turn = false, pin = false;
};

/// The cuts of a read from `reuse`: the last turn token of the prompt after `reuse` (the start of the new assistant turn, so the
/// header and the reply's thinking stay out of the checkpoint), and `pin` tokens when asked (pin > reuse: a pin at or before
/// `reuse` was not passed through, so it cannot be taken).  Ascending, merged when equal, never at 0, never at or past the end,
/// never inside a picture.  `turn_on` / `pin_on`: the kinds wanted.
inline std::vector<Cut> cuts(const std::vector<int32_t>& ids, int64_t reuse, int32_t turn_token, bool turn_on, int64_t pin,
                             bool pin_on, const std::vector<ImgKey>& imgs) {
    const int64_t n = (int64_t) ids.size();
    std::vector<Cut> out;
    auto usable = [&](int64_t at) { return at > reuse && at > 0 && at < n && !inside_picture(at, imgs); };
    if (turn_on && turn_token >= 0) {
        for (int64_t i = n - 1; i > reuse; --i)
            if (ids[(size_t) i] == turn_token) {
                if (usable(i)) out.push_back({i, true, false});
                break;
            }
    }
    if (pin_on && pin > 0 && usable(pin)) {
        bool merged = false;
        for (Cut& c : out)
            if (c.at == pin) { c.pin = merged = true; }
        if (!merged) out.push_back({pin, false, true});
    }
    std::sort(out.begin(), out.end(), [](const Cut& a, const Cut& b) { return a.at < b.at; });
    return out;
}

/// A turn checkpoint is attached to its session's K / V cache, whose cells at or past the first cell a read writes are overwritten:
/// after a read that starts at cell `reuse` the checkpoint at `pos` tokens still has its cells iff pos <= reuse.  (The pin keeps a
/// copy of its cells, so it survives anything.)
inline bool survives(int64_t pos, int64_t reuse) { return pos <= reuse; }

/// The pictures of `imgs` that lie entirely in the first `p` cells (what a checkpoint at p holds).
inline std::vector<ImgKey> imgs_before(const std::vector<ImgKey>& imgs, int64_t p) {
    std::vector<ImgKey> out;
    for (const ImgKey& k : imgs)
        if (k.cell + k.n <= p) out.push_back(k);
    return out;
}

/// One pitched copy between a K (or V) array on the device and the dense host buffer of the pin: `height` rows of `width` bytes, the
/// device rows `dev_pitch` apart starting at byte `dev_off` of the array, the host rows `width` apart starting at `host_off`.
struct Copy2D {
    size_t host_off = 0, dev_off = 0, dev_pitch = 0, width = 0, height = 0;
};

/// The copies that move the first `n` cells of every KV head of one K or V array of a layer (layout [HKV][ctx][HD] elements of `el`
/// bytes; q8_0: [HKV][ctx][HD] int8 quants then [HKV][ctx][HD / 32] half scales), appended to `out` with the host side starting at
/// `host_base`.  Returns the host bytes they fill: HKV * n * (one cell's bytes).
inline size_t kv_prefix_copies(size_t host_base, int64_t n, int64_t ctx, int HD, int HKV, size_t el, bool q8, std::vector<Copy2D>& out) {
    const size_t w = (size_t) n * (size_t) HD * el, row = (size_t) ctx * (size_t) HD * el;
    out.push_back({host_base, 0, row, w, (size_t) HKV});
    size_t used = w * (size_t) HKV;
    if (q8) {
        const size_t sw = (size_t) n * (size_t) (HD / 32) * 2, srow = (size_t) ctx * (size_t) (HD / 32) * 2;
        out.push_back({host_base + used, (size_t) HKV * (size_t) ctx * (size_t) HD, srow, sw, (size_t) HKV});
        used += sw * (size_t) HKV;
    }
    return used;
}

}  // namespace q36ck
