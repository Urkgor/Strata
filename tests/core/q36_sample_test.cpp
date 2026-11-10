// The sampler's device half of the Qwen3.6 engine (src/qwen35/q36_sample.cuh: the penalties from a (token, count) list, the
// top k of each logit row) run on the CPU emulation of CUDA (cuda_cpu_emu.hpp) against the host reference
// (include/strata/qwen35/sampling.hpp: apply_penalties on the full row, the top k by value then id): every penalised logit
// bit for bit, every candidate's value and id.  Rows of a verify window (the history plus the drafts before each row),
// windows of the whole history and of the last 50 tokens, k = 1 / 20 / 64.  No GPU.  Build: -ffp-contract=off.
#include "cuda_cpu_emu.hpp"
#define STRATA_Q36_EMU 1
#include "q36_sample.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using strata::qwen35::PenaltyWindow;
using strata::qwen35::PenEntry;
using strata::qwen35::Sampling;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } else { std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    const int V = 248320, R = 3;
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.f, 3.f);
    std::uniform_int_distribution<int> tok(0, V - 1), small(0, 99);
    // a history with repeats (tokens 0..99 come back often), and three drafts
    std::vector<int32_t> hist;
    for (int i = 0; i < 400; ++i) hist.push_back(small(rng) < 70 ? small(rng) : tok(rng));
    const int32_t drafts[2] = {hist[3], 123456};
    // device-side buffers (the emulation's cudaMalloc is calloc)
    q36s::Bufs B;
    const int pb = q36s::part_blocks(V);
    cudaMalloc((void**) &B.base, (size_t) V * sizeof(PenEntry));
    cudaMalloc((void**) &B.deltas, (size_t) R * q36s::kMaxDelta * sizeof(PenEntry));
    cudaMalloc((void**) &B.row_off, (size_t) (R + 1) * 4);
    cudaMalloc((void**) &B.cv, (size_t) R * pb * q36s::kMaxK * 4);
    cudaMalloc((void**) &B.ci, (size_t) R * pb * q36s::kMaxK * 4);
    cudaMalloc((void**) &B.ov, (size_t) R * q36s::kMaxK * 4);
    cudaMalloc((void**) &B.oi, (size_t) R * q36s::kMaxK * 4);
    float* lg = nullptr;
    cudaMalloc((void**) &lg, (size_t) R * V * 4);

    for (int last_n : {-1, 50, 0}) {
        for (int k : {64, 20, 1}) {
            Sampling s;
            s.temperature = k == 1 ? 0.f : 0.7f;
            s.top_k = k;
            s.repeat = last_n == 0 ? 1.f : 1.15f;
            s.freq = 0.2f;
            s.present = 0.3f;
            s.last_n = last_n;
            // the rows: random logits with a few exact ties (the same value at several ids, some among the best)
            std::vector<float> rows((size_t) R * V);
            for (auto& x : rows) x = nd(rng);
            for (int r = 0; r + 1 < R; ++r) {   // (the last row keeps distinct values: the draw is compared on it)
                float* row = rows.data() + (size_t) r * V;
                for (int t = 0; t < 30; ++t) row[tok(rng)] = 12.5f;          // ties at the top
                for (int t = 0; t < 200; ++t) row[tok(rng)] = row[hist[(size_t) t]];   // ties with penalised tokens
            }
            std::memcpy(lg, rows.data(), rows.size() * 4);
            PenaltyWindow pw;
            pw.reset(s, hist, V);
            // the lists as the engine uploads them
            std::vector<PenEntry> base = pw.base();
            if (!base.empty()) std::memcpy(B.base, base.data(), base.size() * sizeof(PenEntry));
            int off = 0;
            std::vector<PenEntry> d;
            for (int r = 0; r < R; ++r) {
                B.row_off[r] = off;
                pw.row_deltas(drafts, r, d);
                if (!d.empty()) std::memcpy(B.deltas + off, d.data(), d.size() * sizeof(PenEntry));
                off += (int) d.size();
            }
            B.row_off[R] = off;
            q36s::run(B, lg, V, V, R, k, (int) base.size(), s.penalises(), s.repeat, s.freq, s.present, (cudaStream_t) nullptr);
            // the reference: apply_penalties on each row with that row's history, then the top k by (value, id)
            int bad_pen = 0, bad_top = 0;
            for (int r = 0; r < R; ++r) {
                std::vector<float> ref(rows.begin() + (size_t) r * V, rows.begin() + (size_t) (r + 1) * V);
                std::vector<int32_t> h = hist;
                h.insert(h.end(), drafts, drafts + r);
                strata::qwen35::apply_penalties(ref, s, h);
                if (std::memcmp(ref.data(), lg + (size_t) r * V, (size_t) V * 4) != 0) ++bad_pen;
                std::vector<float> tv;
                std::vector<int32_t> ti;
                strata::qwen35::host_topk(ref.data(), V, k, tv, ti);
                for (int i = 0; i < k; ++i)
                    if (ti[(size_t) i] != B.oi[(size_t) r * k + i] || tv[(size_t) i] != B.ov[(size_t) r * k + i]) ++bad_top;
            }
            EXPECT(bad_pen == 0, "last_n %d, k %d: the penalised rows are the reference's bits (%d rows differ)", last_n, k, bad_pen);
            EXPECT(bad_top == 0, "last_n %d, k %d: the top %d of each row (value and id) are the reference's (%d differ)", last_n, k, k, bad_top);
            // the same draw: sample on the full row and sample_from_topk on the candidates, with one seed, where no two of the
            // candidates are equal (the order of equal values is std::partial_sort's there)
            if (k > 1) {
                int same = 0, tried = 0;
                for (int r = 0; r < R; ++r) {
                    std::vector<float> ref(rows.begin() + (size_t) r * V, rows.begin() + (size_t) (r + 1) * V);
                    std::vector<int32_t> h = hist;
                    h.insert(h.end(), drafts, drafts + r);
                    bool ties = false;
                    for (int i = 1; i < k; ++i) ties |= B.ov[(size_t) r * k + i] == B.ov[(size_t) r * k + i - 1];
                    if (ties) continue;
                    ++tried;
                    std::mt19937_64 a(99 + r), b(99 + r);
                    const int x = strata::qwen35::sample(ref, s, h, a);
                    const int y = strata::qwen35::sample_from_topk(B.ov + (size_t) r * k, B.oi + (size_t) r * k, k, s, b);
                    same += x == y;
                }
                EXPECT(same == tried && tried > 0, "last_n %d, k %d: the draw among the candidates is the full row's draw (%d of %d rows without ties)", last_n, k, same, tried);
            }
        }
    }
    {   // the window as the history grows: PenaltyWindow::push against a fresh reset
        Sampling s;
        s.repeat = 1.1f; s.freq = 0.1f; s.present = 0.f; s.last_n = 50;
        PenaltyWindow a;
        a.reset(s, hist, V);
        std::vector<int32_t> h2 = hist;
        for (int i = 0; i < 120; ++i) { const int32_t t = small(rng) < 50 ? small(rng) : tok(rng); a.push(t); h2.push_back(t); }
        PenaltyWindow b;
        b.reset(s, h2, V);
        std::vector<PenEntry> la = a.row_list(nullptr, 0), lb = b.row_list(nullptr, 0);
        EXPECT(la.size() == lb.size() && std::memcmp(la.data(), lb.data(), la.size() * sizeof(PenEntry)) == 0,
               "the window kept by push (%zu tokens) is the window of a reset", la.size());
    }
    std::printf(fails ? "%d FAILED\n" : "q36_sample_test: all checks pass\n", fails);
    return fails ? 1 : 0;
}
