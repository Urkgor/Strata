// q36_ckpt.hpp on the cases of STRATA_QWEN36_CHECKPOINTS.md: what a request reads again (S1 a chat continues with its reply
// re-rendered, S2 a long shared start with a pin, the held state still the fast path, pictures), and where the read is cut.
// No GPU, no model: token ids are small numbers, 9 stands for <|im_start|> and 8 for <|im_end|>.
#include <cstdio>
#include <vector>

#include "q36_ckpt.hpp"

using namespace q36ck;
using Ids = std::vector<int32_t>;
using Imgs = std::vector<ImgKey>;

static int fails = 0;
#define EXPECT(c, ...)                                      \
    do {                                                    \
        if (!(c)) {                                         \
            ++fails;                                        \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
        }                                                   \
    } while (0)

static Ids cat(std::initializer_list<Ids> parts) {
    Ids out;
    for (const Ids& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}
static Ids rep(int32_t v, int n) { return Ids((size_t) n, v); }
static Held held(bool valid, const Ids& ids, const Imgs& imgs) { return {valid, &ids, &imgs}; }
// one turn checkpoint (the first versions' shape): the series of one
static Choice ch1(const Ids& live, const Imgs& live_imgs, const Held& turn, const Held& pin, const Ids& ids, const Imgs& imgs) {
    return choose(live, live_imgs, std::vector<Held>{turn}, pin, ids, imgs);
}

int main() {
    const int32_t T = 9;
    const Imgs none;
    // ---- S1: continuation with the reply re-rendered
    {
        const Ids sys = cat({{T, 1, 1, 1, 8}, {T, 2, 2, 8}});           // system, user 1
        const Ids hdr = {T, 3, 3};                                       // the assistant header (+ thinking opener)
        const Ids p1 = cat({sys, hdr});
        auto c1 = cuts(p1, 0, T, true, -1, false, none);
        EXPECT(c1.size() == 1 && c1[0].at == (int64_t) sys.size() && c1[0].turn && !c1[0].pin, "the turn cut is the last turn token (%zu cuts, at %lld)", c1.size(), c1.empty() ? -1LL : (long long) c1[0].at);
        const Ids turn_ids(p1.begin(), p1.begin() + (long) sys.size());  // what the checkpoint holds
        const Ids reply_gen = cat({{5, 5, 5, 5, 8}});                    // what the engine generated
        const Ids live = cat({p1, reply_gen});
        // request 2: the reply re-rendered with one token more, then user 2 and the new header
        const Ids rerendered = cat({hdr, {5, 5, 7, 5, 5, 8}, {T, 2, 4, 8}, {T, 3, 3}});
        const Ids p2 = cat({sys, rerendered});
        Choice ch = ch1(live, none, held(true, turn_ids, none), held(false, turn_ids, none), p2, none);
        EXPECT(ch.src == kTurn && ch.len == sys.size(), "S1: the turn checkpoint is taken (src %d, len %zu, wanted %zu)", (int) ch.src, ch.len, sys.size());
        // the same reply re-rendered identically: the live state is longer and wins (no restore)
        const Ids p2b = cat({live, {T, 2, 4, 8}, {T, 3, 3}});
        ch = ch1(live, none, held(true, turn_ids, none), held(false, turn_ids, none), p2b, none);
        EXPECT(ch.src == kLive && ch.len == live.size(), "S1 identical: the live state wins (src %d, len %zu)", (int) ch.src, ch.len);
        // the next cut of request 2, from the restored position: the last turn token of p2
        auto c2 = cuts(p2, (int64_t) ch.len == 0 ? 0 : (int64_t) sys.size(), T, true, -1, false, none);
        EXPECT(c2.size() == 1 && p2[(size_t) c2[0].at] == T && c2[0].at > (int64_t) sys.size(), "S1: request 2 cuts at its own last turn token");
        // a turn checkpoint taken after the state moved on is gone: it survives a read from its own position and not from before
        EXPECT(survives(10, 10) && survives(10, 12) && !survives(10, 9), "survives()");
    }
    // ---- S2: a long shared start, pinned
    {
        const Ids P = cat({{T, 1, 1}, rep(7, 40), {8}});                 // system + document, 44 tokens
        const Ids item1 = cat({{T, 2, 6, 8}, {T, 3, 3}}), item2 = cat({{T, 2, 4, 4, 8}, {T, 3, 3}}), item3 = cat({{T, 2, 9 - 3, 8}, {T, 3, 3}});
        const Ids p1 = cat({P, item1}), p2 = cat({P, item2}), p3 = cat({P, item3});
        const int64_t N = (int64_t) P.size();
        auto c1 = cuts(p1, 0, T, true, N, true, none);
        EXPECT(c1.size() == 2 && c1[0].at == N && c1[0].pin && !c1[0].turn && c1[1].turn && c1[1].at > N, "S2: a pin cut at |P| and a turn cut after it (%zu cuts)", c1.size());
        const Ids pin_ids(p1.begin(), p1.begin() + (long) N), turn_ids(p1.begin(), p1.begin() + (long) c1[1].at);
        const Ids live1 = cat({p1, {5, 5, 8}});
        Choice ch = ch1(live1, none, held(true, turn_ids, none), held(true, pin_ids, none), p2, none);
        EXPECT(ch.src == kPin && ch.len == (size_t) N, "S2: request 2 restores the pin (src %d, len %zu, wanted %lld)", (int) ch.src, ch.len, (long long) N);
        // an unrelated prompt: nothing matches; the pin is still there for item 3
        const Ids other = cat({{T, 7, 7, 8}, {T, 3, 3}});
        ch = ch1(other /*the session now holds the unrelated prompt*/, none, held(false, turn_ids, none), held(true, pin_ids, none), other, none);
        EXPECT(ch.len == 0, "an unrelated prompt matches nothing");
        ch = ch1(other, none, held(false, turn_ids, none), held(true, pin_ids, none), p3, none);
        EXPECT(ch.src == kPin && ch.len == (size_t) N, "S2: item 3 finds the pin again after the unrelated prompt");
        // a pin at or before the resumed position cannot be taken now; a pin past it can
        EXPECT(cuts(p2, N, T, false, N, true, none).empty(), "pin == reuse: no cut");
        EXPECT(cuts(p2, N + 2, T, false, N, true, none).empty(), "pin < reuse: no cut");
        auto c = cuts(p2, N - 5, T, false, N, true, none);
        EXPECT(c.size() == 1 && c[0].at == N && c[0].pin, "pin > reuse: a cut");
        // the pin equal to the turn boundary is one cut with both kinds
        const Ids q = cat({P, {T, 3, 3}});
        c = cuts(q, 0, T, true, N, true, none);
        EXPECT(c.size() == 1 && c[0].at == N && c[0].pin && c[0].turn, "pin == turn boundary: one merged cut (%zu)", c.size());
    }
    // ---- continuation still works, and a checkpoint is a proper start only
    {
        const Ids live = {T, 1, 2, 3, 4, 5};
        const Ids more = cat({live, {6, 7}});
        Choice ch = ch1(live, none, held(false, live, none), held(false, live, none), more, none);
        EXPECT(ch.src == kLive && ch.len == live.size(), "continuation: reuse == live.size() (%zu)", ch.len);
        ch = ch1(live, none, held(false, live, none), held(false, live, none), live, none);
        EXPECT(ch.len == 0, "the same prompt again: nothing is a PROPER start, the engine reads it (it needs the last logits)");
        const Ids pinned = {T, 1, 2};
        ch = ch1(Ids{}, none, held(false, pinned, none), held(true, pinned, none), more, none);
        EXPECT(ch.src == kPin && ch.len == 3, "a pin on an empty session");
    }
    // ---- pictures
    {
        const Ids ids = cat({{T, 1}, rep(50, 6), {2, 8}, {T, 3, 3}});    // a picture on cells 2 .. 7
        const Imgs pic = {{2, 6, 0xabc}}, pic2 = {{2, 6, 0xdef}};
        const Ids before(ids.begin(), ids.begin() + 10);                 // 10 tokens: the whole picture and two more
        EXPECT(ch1(Ids{}, none, held(true, before, pic), held(false, before, pic), ids, pic).len == 10, "the picture keys match: reuse");
        EXPECT(ch1(Ids{}, none, held(true, before, pic), held(false, before, pic), ids, pic2).len == 0, "a different picture: a full read");
        const Ids mid(ids.begin(), ids.begin() + 5);                     // the held tokens end inside the picture
        EXPECT(ch1(Ids{}, none, held(true, mid, none), held(false, mid, none), ids, pic).len == 0, "held tokens that end inside a picture: no reuse");
        EXPECT(inside_picture(5, pic) && !inside_picture(2, pic) && !inside_picture(8, pic), "inside_picture");
        EXPECT(cuts(ids, 0, T, false, 5, true, pic).empty(), "a pin inside a picture is not taken");
        EXPECT(imgs_before(pic, 5).empty() && imgs_before(pic, 8).size() == 1, "imgs_before");
    }
    // ---- the cuts: none when there is nothing to read before the boundary, the turn token being the last token
    {
        const Ids one_turn = {T, 1, 2, 3};
        auto c = cuts(one_turn, 0, T, true, -1, false, none);
        EXPECT(c.empty(), "a turn token at 0 is no cut (nothing before it)");
        const Ids ends_on_turn = {1, 2, 3, T};
        c = cuts(ends_on_turn, 0, T, true, -1, false, none);
        EXPECT(c.size() == 1 && c[0].at == 3, "the turn token as the last token: a cut before it");
        c = cuts(ends_on_turn, 3, T, true, -1, false, none);
        EXPECT(c.empty(), "T <= reuse: no cut");
        c = cuts(ends_on_turn, 0, T, false, -1, false, none);
        EXPECT(c.empty(), "turn off: no turn cut");
    }
    // ---- a series of turn checkpoints: the longest one that is still a start of the prompt (a long agent chat, then another call of the same skill)
    {
        // the chat: system+skill (13K) | documents (25K) | item 1 and its analysis (many turns) - checkpoints along it
        Ids chat = cat({{T, 1, 1}, rep(7, 100), {8}});                       // 0 .. 103: system + skill
        const size_t docs_at = chat.size();
        chat = cat({chat, {T, 2, 2}, rep(6, 200), {8}});                     // the documents
        const size_t item_at = chat.size();
        chat = cat({chat, {T, 2, 3, 3, 8}, {T, 3, 5, 5, 8}, {T, 2, 4, 8}, {T, 3, 3}});   // item 1 and its steps
        std::vector<Ids> ck_ids;
        std::vector<int64_t> at = {(int64_t) docs_at, (int64_t) item_at, (int64_t) item_at + 5, (int64_t) item_at + 10};
        for (int64_t a : at) ck_ids.push_back(Ids(chat.begin(), chat.begin() + (long) a));
        std::vector<Held> turns;
        for (const Ids& c : ck_ids) turns.push_back(held(true, c, none));
        // the second call: the same skill and documents, then ANOTHER item: it shares the start up to item_at
        const Ids second = cat({Ids(chat.begin(), chat.begin() + (long) item_at), {T, 2, 9, 9, 9, 8}, {T, 3, 3}});
        const Ids live = cat({chat, {5, 5, 8}});
        Choice ch = choose(live, none, turns, held(false, ck_ids[0], none), second, none);
        EXPECT(ch.src == kTurn && ch.idx == 1 && ch.len == item_at, "series: the checkpoint at the end of the documents is the longest that fits (idx %d len %zu, wanted %zu)", ch.idx, ch.len, item_at);
        // an invalid one is skipped
        turns[1].valid = false;
        ch = choose(live, none, turns, held(false, ck_ids[0], none), second, none);
        EXPECT(ch.src == kTurn && ch.idx == 0 && ch.len == docs_at, "series: without it, the one before (idx %d len %zu)", ch.idx, ch.len);
        // the one-checkpoint engine would have had only the newest, which is not a start of the second prompt
        EXPECT(ch1(live, none, held(true, ck_ids[3], none), held(false, ck_ids[0], none), second, none).len == 0, "a single latest checkpoint matches nothing here");
    }
    // ---- the cuts of a long read: the last turn token, and earlier turn tokens at least `every` apart
    {
        Ids ids = {T, 1};                                    // turn tokens at 0, 40, 100, 150, 400, 420 (the last)
        auto add_turn = [&](int64_t at) { while ((int64_t) ids.size() < at) ids.push_back(7); ids.push_back(T); ids.push_back(3); };
        for (int64_t a : {40, 100, 150, 400, 420}) add_turn(a);
        ids.push_back(3);
        auto c = cuts(ids, 0, T, true, -1, false, none, 100, 6);
        std::vector<int64_t> at;
        for (const Cut& x : c) at.push_back(x.at);
        const std::vector<int64_t> want = {100, 400, 420};   // 40 is too close to 0; 100 is 100 after reuse; 150 is 50 after 100; 400 is 300 after 100; 420 is the last
        EXPECT(at == want, "cuts every 100: %zu cuts (first %lld)", at.size(), at.empty() ? -1LL : (long long) at[0]);
        c = cuts(ids, 0, T, true, -1, false, none, 100, 1);
        EXPECT(c.size() == 2 && c[0].at == 100 && c[1].at == 420, "max_extra 1: one earlier cut and the last (%zu)", c.size());
        c = cuts(ids, 0, T, true, -1, false, none, 0, 6);
        EXPECT(c.size() == 1 && c[0].at == 420, "every 0: only the last turn token");
    }
    // ---- which checkpoint a full series drops
    {
        EXPECT(evict({10, 20, 30}, 6) == -1, "room: nothing dropped");
        EXPECT(evict({10, 20, 30, 40, 50, 60}, 6) == 1 || evict({10, 20, 30, 40, 50, 60}, 6) == 2 || evict({10, 20, 30, 40, 50, 60}, 6) == 3 || evict({10, 20, 30, 40, 50, 60}, 6) == 4, "an interior one is dropped (never the oldest or the newest)");
        EXPECT(evict({13384, 39257, 39644, 59385, 60701, 64597}, 6) == 4, "the log's series: 60701 goes (smallest gap 64597 - 59385), the oldest stays (%d)", evict({13384, 39257, 39644, 59385, 60701, 64597}, 6));
        EXPECT(evict({100, 200}, 2) == 1, "two: the newer goes, the oldest stays");
        EXPECT(evict({100}, 1) == 0, "one: replaced");
    }
    // ---- the pin's K / V copies: a pitched device array <-> a dense host buffer, f32 / f16 / q8_0, exact and nothing past the prefix touched
    {
        const int HD = 256, HKV = 2;
        const int64_t ctx = 300;
        for (int kind = 0; kind < 3; ++kind) {   // f32, f16, q8_0
            const bool q8 = kind == 2;
            const size_t el = kind == 0 ? 4 : kind == 1 ? 2 : 1;
            const size_t cell_bytes = q8 ? (size_t) HD + (size_t) (HD / 32) * 2 : (size_t) HD * el;
            const size_t dev_bytes = (size_t) HKV * (size_t) ctx * cell_bytes;
            for (int64_t n : {1, 17, 256, 299, 300}) {
                std::vector<uint8_t> dev(dev_bytes), back(dev_bytes, 0xEE);
                uint32_t s = 12345u + (uint32_t) n;
                for (uint8_t& b : dev) { s = s * 1664525u + 1013904223u; b = (uint8_t) (s >> 24); }
                std::vector<Copy2D> cp;
                const size_t used = kv_prefix_copies(0, n, ctx, HD, HKV, el, q8, cp);
                EXPECT(used == (size_t) HKV * (size_t) n * cell_bytes, "kind %d n %lld: %zu host bytes, wanted %zu", kind, (long long) n, used, (size_t) HKV * (size_t) n * cell_bytes);
                EXPECT(cp.size() == (q8 ? 2u : 1u), "copies");
                std::vector<uint8_t> host(used, 0);
                auto run = [&](bool to_host, std::vector<uint8_t>& devp) {   // cudaMemcpy2D on the CPU
                    for (const Copy2D& c : cp)
                        for (size_t r = 0; r < c.height; ++r) {
                            uint8_t* d = devp.data() + c.dev_off + r * c.dev_pitch;
                            uint8_t* h = host.data() + c.host_off + r * c.width;
                            if (to_host) std::copy(d, d + c.width, h);
                            else std::copy(h, h + c.width, d);
                        }
                };
                run(true, dev);
                std::vector<uint8_t> dev2 = back;   // every byte 0xEE
                run(false, dev2);
                // each head's first n cells equal the original; the quants and the scales of cells >= n stay 0xEE
                bool same = true, untouched = true;
                for (int h = 0; h < HKV; ++h)
                    for (int64_t c = 0; c < ctx; ++c) {
                        const size_t qoff = ((size_t) h * (size_t) ctx + (size_t) c) * (size_t) HD * el;
                        const size_t qn = (size_t) HD * el;
                        for (size_t b = 0; b < qn; ++b) {
                            (c < n ? same : untouched) &= c < n ? dev2[qoff + b] == dev[qoff + b] : dev2[qoff + b] == 0xEE;
                        }
                        if (q8) {
                            const size_t soff = (size_t) HKV * (size_t) ctx * (size_t) HD + ((size_t) h * (size_t) ctx + (size_t) c) * (size_t) (HD / 32) * 2;
                            for (size_t b = 0; b < (size_t) (HD / 32) * 2; ++b)
                                (c < n ? same : untouched) &= c < n ? dev2[soff + b] == dev[soff + b] : dev2[soff + b] == 0xEE;
                        }
                    }
                EXPECT(same, "kind %d n %lld: the prefix does not come back exactly", kind, (long long) n);
                EXPECT(untouched, "kind %d n %lld: a cell past the prefix was written", kind, (long long) n);
            }
        }
    }
    std::printf(fails ? "%d FAILED\n" : "q36_ckpt_test: all cases pass\n", fails);
    return fails ? 1 : 0;
}
