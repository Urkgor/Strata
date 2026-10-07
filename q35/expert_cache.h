// q35/expert_cache.h - which routed experts live in the card's cache.  Plain C++17, no llama.cpp, no GPU: the host
// side of the hybrid expert cache (third_party patch q35/patches/0001-hybrid-expert-cache.patch) and its unit test
// (q35/test_expert_cache.cpp) use the same code.
//
// The model has `n_layer` MoE layers of `n_expert` experts each; layer l has room for slots(l) of them on the card.
// After every decode step the engine tells the policy which experts each token asked for (observe); the policy keeps a
// decaying count of uses per expert, and end_step() returns the few swaps that put the most used uncached experts in
// place of the least used cached ones.  The count of a use halves after `half_life` tokens, so the cache follows the
// conversation; a newcomer has to beat the expert it replaces by `margin` uses, so it does not flap.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace q35 {

struct CacheParams {
    double half_life = 512.0;   // tokens after which a use counts half
    double margin = 1.0;        // uses a newcomer must have more than the expert it replaces
    int swaps_per_step = 2;     // experts moved after one step once the cache is full
    int fill_swaps = 24;        // ... while there are free slots
};

struct CacheSwap {
    int layer = 0;
    int slot = 0;
    int in = 0;     // the expert that goes into the slot
    int out = -1;   // the expert that was there (-1: the slot was free)
};

struct CacheStats {
    uint64_t hits = 0;      // expert uses found in the cache
    uint64_t lookups = 0;   // expert uses (tokens x experts per token x layers)
    uint64_t swaps = 0;     // experts copied into the cache
    double rate() const { return lookups ? (double) hits / (double) lookups : 0.0; }
};

class HotSet {
public:
    HotSet() = default;
    HotSet(int n_layer, int n_expert, const std::vector<int>& slots, const CacheParams& p = CacheParams())
        : n_layer_(n_layer), n_expert_(n_expert), params_(p) {
        slots_.assign((size_t) n_layer, 0);
        off_.assign((size_t) n_layer + 1, 0);
        for (int l = 0; l < n_layer; ++l) {
            const int s = l < (int) slots.size() ? std::max(0, std::min(slots[(size_t) l], n_expert)) : 0;
            slots_[(size_t) l] = s;
            off_[(size_t) l + 1] = off_[(size_t) l] + s;
        }
        slot_of_.assign((size_t) n_layer * (size_t) n_expert, -1);
        expert_at_.assign((size_t) off_[(size_t) n_layer], -1);
        locked_.assign((size_t) off_[(size_t) n_layer], 0);
        score_.assign((size_t) n_layer * (size_t) n_expert, 0.0);
        stamp_.assign((size_t) n_layer * (size_t) n_expert, 0);
        missed_.resize((size_t) n_layer);
        set_half_life(p.half_life);
    }

    int n_layer() const { return n_layer_; }
    int n_expert() const { return n_expert_; }
    int slots(int l) const { return slots_[(size_t) l]; }
    int total_slots() const { return off_.empty() ? 0 : off_.back(); }
    int used_slots() const { int n = 0; for (int e : expert_at_) n += e >= 0; return n; }
    bool full() const { return used_slots() == total_slots(); }
    const CacheParams& params() const { return params_; }
    const CacheStats& stats() const { return stats_; }
    void reset_stats() { stats_ = CacheStats(); }

    // -1: not cached
    int slot_of(int l, int e) const { return slot_of_[(size_t) l * (size_t) n_expert_ + (size_t) e]; }
    // -1: the slot is free
    int expert_at(int l, int s) const { return expert_at_[(size_t) off_[(size_t) l] + (size_t) s]; }
    // the uses counted for an expert, in "uses now" (older ones weigh less)
    double uses(int l, int e) const { return score_[(size_t) l * (size_t) n_expert_ + (size_t) e] / inc_; }

    // An expert placed by hand: it takes a free slot of layer l and is never replaced.  Returns true when a slot was filled (the caller copies
    // the expert's weights into it); false when it was already cached (it is locked now) or the layer has no free slot left.
    bool pin(int l, int e, CacheSwap& out) {
        const int have = slot_of(l, e);
        if (have >= 0) { locked_[(size_t) off_[(size_t) l] + (size_t) have] = 1; return false; }
        for (int s = 0; s < slots_[(size_t) l]; ++s) {
            const size_t at = (size_t) off_[(size_t) l] + (size_t) s;
            if (expert_at_[at] >= 0 || locked_[at]) continue;
            put(l, s, e);
            locked_[at] = 1;
            out.layer = l; out.slot = s; out.in = e; out.out = -1;
            return true;
        }
        return false;
    }
    bool locked(int l, int s) const { return locked_[(size_t) off_[(size_t) l] + (size_t) s] != 0; }
    // static: end_step() never moves anything (the cache is what was placed at the start)
    void set_static(bool on) { static_ = on; }
    void set_fill_swaps(int n) { params_.fill_swaps = std::max(0, n); }
    void set_half_life(double tokens) {
        params_.half_life = std::max(1.0, tokens);
        grow_per_token_ = std::pow(2.0, 1.0 / params_.half_life);
    }

    // Layer l of one step: ids[t * n_used + k] is the k-th expert of token t.  Counts the uses (for the ranking) and,
    // when `count`, the hits against the table as it is now.  Experts of tokens beyond the first `n_tok` are ignored.
    void observe(int l, const int32_t* ids, int n_tok, int n_used, bool count = true) {
        const size_t base = (size_t) l * (size_t) n_expert_;
        const bool cached_layer = slots_[(size_t) l] > 0;
        ++step_;
        for (int t = 0; t < n_tok; ++t) {
            // tokens are in order: a later token weighs a little more (the decay is per token)
            for (int k = 0; k < n_used; ++k) {
                const int e = ids[(size_t) t * (size_t) n_used + (size_t) k];
                if (e < 0 || e >= n_expert_) continue;
                score_[base + (size_t) e] += inc_;
                if (count) {
                    ++stats_.lookups;
                    if (slot_of_[base + (size_t) e] >= 0) ++stats_.hits;
                }
                if (cached_layer && slot_of_[base + (size_t) e] < 0 && stamp_[base + (size_t) e] != step_) {
                    stamp_[base + (size_t) e] = step_;
                    missed_[(size_t) l].push_back(e);
                }
            }
            inc_ *= grow_per_token_;
        }
        if (inc_ > 1e150) rescale();
    }

    // End of a step: the swaps to make, at most `budget` (< 0: the default for the cache's state), already applied to
    // this table.  The caller copies each swap's expert into its slot and rewrites the layer's tables.
    void end_step(std::vector<CacheSwap>& out, int budget = -1) {
        out.clear();
        if (static_) { for (auto& m : missed_) m.clear(); return; }
        if (budget < 0) budget = full() ? params_.swaps_per_step : params_.fill_swaps;
        struct Cand { double gain; bool fill; int layer, expert, slot; };
        std::vector<Cand> cands;
        for (int l = 0; l < n_layer_; ++l) {
            std::vector<int>& miss = missed_[(size_t) l];
            if (miss.empty()) continue;
            const size_t base = (size_t) l * (size_t) n_expert_;
            std::sort(miss.begin(), miss.end(), [&](int a, int b) {
                return score_[base + (size_t) a] != score_[base + (size_t) b] ? score_[base + (size_t) a] > score_[base + (size_t) b] : a < b;
            });
            // the free slots first, then the cached experts from the least used up
            std::vector<int> frees;
            std::vector<std::pair<double, int>> held;   // (score, slot)
            for (int s = 0; s < slots_[(size_t) l]; ++s) {
                const int e = expert_at_[(size_t) off_[(size_t) l] + (size_t) s];
                if (e < 0) { if (!locked_[(size_t) off_[(size_t) l] + (size_t) s]) frees.push_back(s); }
                else if (!locked_[(size_t) off_[(size_t) l] + (size_t) s]) held.emplace_back(score_[base + (size_t) e], s);
            }
            std::sort(held.begin(), held.end());
            size_t fi = 0, hi = 0;
            for (int e : miss) {
                const double sc = score_[base + (size_t) e];
                if (fi < frees.size()) {
                    cands.push_back({sc / inc_, true, l, e, frees[fi++]});
                } else if (hi < held.size() && sc - held[hi].first > params_.margin * inc_) {
                    cands.push_back({(sc - held[hi].first) / inc_, false, l, e, held[hi].second});
                    ++hi;
                } else {
                    break;   // the candidates only get weaker, the victims only stronger
                }
            }
            miss.clear();
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.fill != b.fill) return a.fill;
            if (a.gain != b.gain) return a.gain > b.gain;
            return a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
        });
        for (const Cand& c : cands) {
            if ((int) out.size() >= budget) break;
            CacheSwap sw;
            sw.layer = c.layer;
            sw.slot = c.slot;
            sw.in = c.expert;
            sw.out = expert_at_[(size_t) off_[(size_t) c.layer] + (size_t) c.slot];
            put(c.layer, c.slot, c.expert);
            out.push_back(sw);
        }
        stats_.swaps += out.size();
    }

    // Puts `e` into slot `s` of layer l (the host does this for the first fill, and end_step() for every swap).
    void put(int l, int s, int e) {
        int& at = expert_at_[(size_t) off_[(size_t) l] + (size_t) s];
        if (at >= 0) slot_of_[(size_t) l * (size_t) n_expert_ + (size_t) at] = -1;
        at = e;
        slot_of_[(size_t) l * (size_t) n_expert_ + (size_t) e] = s;
    }

    // ---- a profile: the uses of every expert, to start a later run with a warm cache
    std::vector<float> profile() const {
        std::vector<float> p(score_.size());
        for (size_t i = 0; i < p.size(); ++i) p[i] = (float) (score_[i] / inc_);
        return p;
    }
    bool load_profile(const std::vector<float>& p) {
        if (p.size() != score_.size()) return false;
        for (size_t i = 0; i < p.size(); ++i) score_[i] = std::isfinite(p[i]) && p[i] > 0 ? (double) p[i] : 0.0;
        inc_ = 1.0;
        return true;
    }
    // The swaps that fill every layer with its most used experts (after load_profile), applied to this table; for an
    // empty cache.
    void fill_from_profile(std::vector<CacheSwap>& out) {
        out.clear();
        for (int l = 0; l < n_layer_; ++l) {
            const size_t base = (size_t) l * (size_t) n_expert_;
            std::vector<int> order((size_t) n_expert_);
            for (int e = 0; e < n_expert_; ++e) order[(size_t) e] = e;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score_[base + (size_t) a] > score_[base + (size_t) b]; });
            size_t rank = 0;
            for (int s = 0; s < slots_[(size_t) l]; ++s) {
                if (expert_at(l, s) >= 0 || locked_[(size_t) off_[(size_t) l] + (size_t) s]) continue;
                while (rank < order.size() && score_[base + (size_t) order[rank]] > 0.0 && slot_of(l, order[rank]) >= 0) ++rank;
                if (rank >= order.size() || score_[base + (size_t) order[rank]] <= 0.0) break;
                CacheSwap sw;
                sw.layer = l; sw.slot = s; sw.in = order[rank++]; sw.out = -1;
                put(l, s, sw.in);
                out.push_back(sw);
            }
        }
        stats_.swaps += out.size();
    }

private:
    void rescale() {
        const double f = 1.0 / inc_;
        for (double& s : score_) s *= f;
        inc_ = 1.0;
    }

    int n_layer_ = 0, n_expert_ = 0;
    CacheParams params_;
    std::vector<int> slots_, off_;
    std::vector<int> slot_of_, expert_at_;
    std::vector<char> locked_;
    bool static_ = false;
    std::vector<double> score_;
    std::vector<uint64_t> stamp_;
    std::vector<std::vector<int>> missed_;
    uint64_t step_ = 0;
    double inc_ = 1.0, grow_per_token_ = 1.0;
    CacheStats stats_;
};

}  // namespace q35
