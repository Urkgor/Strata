// q36_copy.hpp - drafts copied from the conversation itself, for the Qwen3.6 engine's verify windows (--copy-drafts).
//
// When an answer repeats its input (an edit that writes back most of a file, a tool call that quotes a function, records returned
// with a few changes), the tokens after the last few written have usually been seen before, in the prompt or earlier in the answer:
// what followed them there is a good draft.  The idea and its two parts come from YASAKANI (https://github.com/yasakani-dev/yasakani,
// Apache-2.0: docs/TECHNOLOGY.md 5.2 and 5.4, src/copy_index.hpp); this is a separate implementation of them.
//
//  * Index: every 4-token window of the sequence is chained by its hash, newest first, so every earlier place with the same 4
//    tokens can be visited.  A lookup takes the place to copy from: the nearest one at or after where the last copy stopped (so a
//    copy that left its source for a moment resumes in order, and look-alike lines are not mixed up), unless its match backwards is
//    clearly shorter than the longest; else the longest match, the newest on ties.
//  * Cursor: while the answer follows the source, the next draft continues from the same place without a lookup.
//
// The drafts are only proposals: the verify window keeps the ones the model itself picks (spec_step), so the text is the text without
// them.  How many to propose adapts: start at `k0`, grow while every draft is kept, fall back to what was kept on a miss, and pause
// after misses at the start (a false match costs a whole window).  No CUDA here: tests/core/q36_copy_test.cpp.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace q36cp {

struct CopyIndex {
    static constexpr int N = 4;
    std::vector<int32_t> head, prev;
    uint32_t mask = 0;
    int upto = 0;   // the windows ending before `upto` are indexed

    static uint32_t hash4(const int32_t* s) {
        uint64_t h = 0x9E3779B97F4A7C15ull;
        for (int k = 0; k < N; ++k) {
            h ^= (uint64_t) (uint32_t) s[k];
            h *= 0xBF58476D1CE4E5B9ull;
            h ^= h >> 29;
        }
        return (uint32_t) (h >> 32);
    }
    void reset(int n) {
        size_t want = 1024;
        while (want < (size_t) n * 2) want <<= 1;
        head.assign(want, -1);
        mask = (uint32_t) (want - 1);
        prev.assign((size_t) std::max(n, 1), -1);
        upto = 0;
    }
    /// index the windows that end at positions [upto, end)
    void index_upto(const int32_t* S, int end) {
        if (end > (int) prev.size()) {   // longer than planned: twice the room, indexed again from the start
            reset(std::max(end, (int) prev.size() * 2));
        }
        for (; upto < end; ++upto) {
            if (upto < N - 1) { prev[(size_t) upto] = -1; continue; }
            const uint32_t h = hash4(S + upto - N + 1) & mask;
            prev[(size_t) upto] = head[h];
            head[h] = upto;
        }
    }
    /// An earlier place p < F whose tokens S[p-3 .. p] equal S[F-3 .. F] (see the file's comment for which one), or -1; its backward
    /// match length (up to max_back) in *len.
    int find(const int32_t* S, int F, int after, int max_cand, int max_back, int* len) const {
        *len = 0;
        if (F < N - 1 || head.empty() || F >= upto + 1) return -1;
        const uint32_t h = hash4(S + F - N + 1) & mask;
        int best = -1, best_len = 0, near = -1, near_len = 0, seen = 0, steps = 0;
        for (int p = head[h]; p >= 0 && seen < max_cand && steps < 4 * max_cand; p = prev[(size_t) p], ++steps) {
            if (p >= F) continue;
            int l = 0;
            while (l < max_back && p - l >= 0 && S[p - l] == S[F - l]) ++l;
            if (l < N) continue;   // another window in the same bucket
            ++seen;
            if (l > best_len) { best_len = l; best = p; }       // newest first: the newest wins a tie
            if (after >= 0 && p >= after) { near = p; near_len = l; }   // walking down: the last one seen is the nearest after
        }
        if (near >= 0 && near_len >= std::min(best_len, 8)) { *len = near_len; return near; }
        *len = best_len;
        return best;
    }
};

/// The drafter of one request: the sequence (prompt, then every token emitted), the index over it, the cursor and the draft length.
class CopyDrafter {
public:
    int k0 = 3, kmax = 7, min_back = 4;

    void reset(const std::vector<int32_t>& prompt, int room) {
        S_ = prompt;
        S_.reserve(prompt.size() + (size_t) std::max(room, 0) + 16);
        ix_.reset((int) S_.size() + std::max(room, 0) + 64);
        ix_.index_upto(S_.data(), (int) S_.size());
        src_ = -1;
        src_at_ = -1;
        stop_ = -1;
        k_ = k0;
        pause_ = 0;
        misses_ = 0;
    }
    /// a token emitted (the answer grows by one)
    void push(int32_t t) {
        S_.push_back(t);
        ix_.index_upto(S_.data(), (int) S_.size());
    }
    /// Up to min(k, limit) tokens that followed the last ones written where they were seen before (empty: none, or paused).  The
    /// last token of the sequence is the one the verify window starts with.
    std::vector<int32_t> propose(int limit) {
        std::vector<int32_t> out;
        const int F = (int) S_.size() - 1;
        last_p_ = -1;
        if (pause_ > 0) { --pause_; return out; }
        if (F < CopyIndex::N) return out;
        int p = -1;
        if (src_ >= 0) {   // the cursor: the source moved on by as many tokens as the answer did
            const int q = src_ + (F - src_at_);
            if (q >= 0 && q < F) {
                int l = 0;
                while (l < 64 && q - l >= 0 && S_[(size_t) (q - l)] == S_[(size_t) (F - l)]) ++l;
                if (l >= CopyIndex::N) { p = q; ++cursor_hits; }
            }
        }
        if (p < 0) {
            int len = 0;
            p = ix_.find(S_.data(), F, stop_, 32, 64, &len);
            if (p < 0 || len < min_back) {
                if (src_ >= 0) stop_ = src_ + (F - src_at_);   // the copy left its source here
                src_ = -1;
                return out;
            }
        }
        const int n = std::min({k_, limit, F - p});
        for (int i = 1; i <= n; ++i) out.push_back(S_[(size_t) (p + i)]);
        last_p_ = p;
        last_F_ = F;
        return out;
    }
    /// The verify window kept `kept` of the `offered` drafts of the last proposal.
    void result(int kept, int offered) {
        if (last_p_ < 0 || offered <= 0) return;
        if (kept >= offered) {
            k_ = std::min(kmax, k_ + 2);
            misses_ = 0;
        } else {
            k_ = std::max(2, kept + 1);
            if (kept == 0 && ++misses_ >= 2) {   // two false starts in a row: leave the copying for a few tokens
                pause_ = 4 * misses_;
                misses_ = std::min(misses_, 8);
            } else if (kept > 0) {
                misses_ = 0;
            }
        }
        src_ = last_p_;
        src_at_ = last_F_;
        if (kept < offered) stop_ = last_p_ + kept + 1;   // where the answer left the source
        last_p_ = -1;
    }
    int k() const { return k_; }
    long long cursor_hits = 0;

private:
    std::vector<int32_t> S_;
    CopyIndex ix_;
    int src_ = -1, src_at_ = -1, stop_ = -1;
    int last_p_ = -1, last_F_ = -1;
    int k_ = 3, pause_ = 0, misses_ = 0;
};

}  // namespace q36cp
