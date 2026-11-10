#pragma once
// The qwen35 engine's host sampler settings and the repeat / frequency / presence penalties, header-only so a unit test
// (src/qwen35/sampling_test.cpp) can run them without a GPU.
//
// Two ways to pick a token.  `apply_penalties` + `sample` on a whole logit row (the reference: 248,320 floats copied to
// the host, penalised and partially sorted there).  Or the device half (src/qwen35/q36_sample.cuh): the penalties as a
// list of (token, count) the engine uploads - `PenaltyWindow` keeps the counts of the window as the history grows, and
// `row_deltas` the few tokens a verify window's rows add to it - then the top k candidates of each row, and
// `sample_from_topk` finishes on those.  Same arithmetic, same draws: the token is the one `sample` picks, except where
// two logits are exactly equal (the device orders equal values by token id, std::partial_sort as it likes).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace strata::qwen35 {

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
    bool penalises() const { return last_n != 0 && (repeat != 1.f || freq != 0.f || present != 0.f); }
    /// the candidates a sampled pick needs: the arg-max at temperature 0, else top_k (64 at most, as `sample`)
    int candidates() const { return temperature <= 0.f ? 1 : top_k > 0 ? (top_k < 64 ? top_k : 64) : 64; }
};

/// The penalties on the logit row `lg` for the tokens of the last `s.last_n` (all when < 0) history tokens.  A row can be
/// shorter than the vocabulary (the MTP draft head picks among the first N token ids only, --draft-vocab N): a history
/// token that is not in the row (id < 0 or >= lg.size()) is not in it and takes no penalty - it used to index past the
/// row (#3911).
inline void apply_penalties(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            if (id < 0 || (size_t) id >= lg.size()) continue;
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
}

/// The draw over candidates sorted best first (`v` their logits after the penalties, `id` their tokens, `k` of them; an
/// id < 0 ends the list early): the tail of `sample` - temperature, min-p, top-p, one uniform draw.
inline int sample_from_topk(const float* v, const int32_t* id, int k, const Sampling& s, std::mt19937_64& rng) {
    int n = 0;
    while (n < k && id[n] >= 0) ++n;
    if (n <= 0) return 0;
    if (s.temperature <= 0.f) return id[0];
    std::vector<double> p((size_t) n);
    const double mx = v[0];
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += (p[(size_t) i] = std::exp((v[i] - mx) / s.temperature));
    size_t keep = (size_t) n;
    double cum = 0;
    for (int i = 0; i < n; ++i) {
        if (p[(size_t) i] / sum < s.min_p * (p[0] / sum)) { keep = (size_t) i; break; }
        cum += p[(size_t) i] / sum;
        if (cum >= s.top_p) { keep = (size_t) i + 1; break; }
    }
    keep = keep < 1 ? 1 : keep;
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= p[i]) <= 0) return id[i];
    return id[keep - 1];
}

/// The reference pick of one logit row: greedy at temperature 0 (the lowest id among equal maxima), else the top k by
/// std::partial_sort, then `sample_from_topk`.
inline int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    apply_penalties(lg, s, hist);
    if (s.temperature <= 0.f) {
        int best = 0;
        for (size_t i = 1; i < lg.size(); ++i)
            if (lg[i] > lg[(size_t) best]) best = (int) i;
        return best;
    }
    const int kk = s.candidates() < (int) lg.size() ? s.candidates() : (int) lg.size();
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    std::vector<float> v((size_t) kk);
    std::vector<int32_t> id((size_t) kk);
    for (int i = 0; i < kk; ++i) { v[(size_t) i] = lg[(size_t) idx[(size_t) i]]; id[(size_t) i] = idx[(size_t) i]; }
    return sample_from_topk(v.data(), id.data(), kk, s, rng);
}

/// The exact top k of a row by (value descending, id ascending): what the device kernels produce, for the tests.
inline void host_topk(const float* lg, int n, int k, std::vector<float>& v, std::vector<int32_t>& id) {
    std::vector<int> idx((size_t) n);
    for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
    const int kk = k < n ? k : n;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) {
        return lg[a] > lg[b] || (lg[a] == lg[b] && a < b);
    });
    v.assign((size_t) k, 0.f);
    id.assign((size_t) k, -1);
    for (int i = 0; i < kk; ++i) { v[(size_t) i] = lg[idx[(size_t) i]]; id[(size_t) i] = idx[(size_t) i]; }
}

/// (token, count): a penalty list entry
struct PenEntry {
    int32_t id = 0, cnt = 0;
};

/// The counts of the tokens in the penalty window (the last `last_n` of the history, all of it when last_n < 0) as the
/// history grows one token at a time, and the list of them with a count above zero (`base`).  Out-of-row ids (< 0 or
/// >= n_vocab) are not counted: apply_penalties skips them too.
class PenaltyWindow {
public:
    void reset(const Sampling& s, const std::vector<int32_t>& hist, int n_vocab) {
        s_ = s;
        nv_ = n_vocab;
        cnt_.assign((size_t) n_vocab, 0);
        where_.assign((size_t) n_vocab, -1);
        base_.clear();
        hist_ = hist;
        if (!s_.penalises()) return;
        const size_t from = s_.last_n < 0 || (size_t) s_.last_n >= hist_.size() ? 0 : hist_.size() - (size_t) s_.last_n;
        for (size_t i = from; i < hist_.size(); ++i) add(hist_[i], 1);
    }
    void push(int32_t tok) {
        hist_.push_back(tok);
        if (!s_.penalises()) return;
        add(tok, 1);
        if (s_.last_n > 0 && hist_.size() > (size_t) s_.last_n) add(hist_[hist_.size() - 1 - (size_t) s_.last_n], -1);
    }
    const std::vector<PenEntry>& base() const { return base_; }
    const std::vector<int32_t>& hist() const { return hist_; }
    /// The entries of row r of a verify window, whose history is the window's history then extra[0 .. r - 1]: the tokens
    /// whose count differs from `base` (final counts, 0 included).  The device skips these ids in the base pass and applies
    /// the final counts instead.
    void row_deltas(const int32_t* extra, int r, std::vector<PenEntry>& out) const {
        out.clear();
        if (!s_.penalises() || r <= 0) return;
        std::map<int32_t, int32_t> delta;
        for (int i = 0; i < r; ++i)
            if (extra[i] >= 0 && extra[i] < nv_) ++delta[extra[i]];
        if (s_.last_n > 0) {   // the window slides by r: the r oldest tokens of (history, extra) leave it
            std::vector<int32_t> all(hist_.begin(), hist_.end());
            all.insert(all.end(), extra, extra + r);
            const size_t n_old = hist_.size() > (size_t) s_.last_n ? hist_.size() - (size_t) s_.last_n : 0;   // already out
            const size_t n_new = all.size() > (size_t) s_.last_n ? all.size() - (size_t) s_.last_n : 0;
            for (size_t i = n_old; i < n_new; ++i)
                if (all[i] >= 0 && all[i] < nv_) --delta[all[i]];
        }
        for (const auto& [id, d] : delta) {
            if (d == 0) continue;
            out.push_back({id, cnt_[(size_t) id] + d});
        }
    }
    /// The full list of row r (base with the deltas applied): the reference for the tests and the host path.
    std::vector<PenEntry> row_list(const int32_t* extra, int r) const {
        std::vector<PenEntry> d;
        row_deltas(extra, r, d);
        std::map<int32_t, int32_t> m;
        for (const PenEntry& e : base_) m[e.id] = e.cnt;
        for (const PenEntry& e : d) m[e.id] = e.cnt;
        std::vector<PenEntry> out;
        for (const auto& [id, c] : m)
            if (c > 0) out.push_back({id, c});
        return out;
    }
    const Sampling& settings() const { return s_; }

private:
    void add(int32_t tok, int d) {
        if (tok < 0 || tok >= nv_) return;
        int32_t& c = cnt_[(size_t) tok];
        const int32_t before = c;
        c += d;
        if (before == 0 && c > 0) {
            where_[(size_t) tok] = (int32_t) base_.size();
            base_.push_back({tok, c});
        } else if (c == 0) {
            const int32_t w = where_[(size_t) tok];
            where_[(size_t) base_.back().id] = w;
            base_[(size_t) w] = base_.back();
            base_.pop_back();
            where_[(size_t) tok] = -1;
        } else {
            base_[(size_t) where_[(size_t) tok]].cnt = c;
        }
    }
    Sampling s_;
    int nv_ = 0;
    std::vector<int32_t> cnt_, where_, hist_;
    std::vector<PenEntry> base_;
};

/// The penalties of one list on a row (the device kernels' arithmetic, on the host: the reference for the tests).
inline void apply_list(float* lg, int n, const std::vector<PenEntry>& list, const Sampling& s) {
    for (const PenEntry& e : list) {
        if (e.id < 0 || e.id >= n || e.cnt <= 0) continue;
        float& v = lg[e.id];
        if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
        v -= (float) e.cnt * s.freq + s.present;
    }
}

}  // namespace strata::qwen35
