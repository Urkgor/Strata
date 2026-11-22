// The Inkling 2 Mini CUDA engine (src/inkling2/i2_cuda.cu) run on the CPU emulation of CUDA (g4_emu.hpp) against the CPU reference (i2_ref.hpp) on the
// tiny random model (tools/inkling2/make_tiny.py).  No GPU.  Build: -DSTRATA_G4_EMU, see tools/inkling2/README.md; run under AddressSanitizer too.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "i2_cuda.hpp"
#include "i2_ref.hpp"

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: "); } else std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

static double rel_rms(const std::vector<float>& a, const float* b) {
    double d = 0, s = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (double) (a[i] - b[i]) * (a[i] - b[i]); m += a[i]; }
    m /= (double) a.size();
    for (size_t i = 0; i < a.size(); ++i) s += (double) (a[i] - m) * (a[i] - m);
    return std::sqrt(d / std::max(s, 1e-30));
}
static int argmax(const float* v, size_t n) { return (int) (std::max_element(v, v + n) - v); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: i2_emu_test tiny.gguf [quick]\n"); return 2; }
    const bool quick = argc > 2;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    i2::Model m;
    const std::string err = i2::load_model(argv[1], m);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const std::vector<int32_t> ids = {26, 202, 59, 235, 96, 129, 218, 196, 222, 51, 162, 224, 194, 48, 47, 216, 51, 213, 85, 174, 32, 80};   // 22 tokens (window 6); picked so that no
    // expert choice of the random model is a near-tie (a tie flipped by rounding noise changes the stream by 10%: chaos, not a bug)
    const int n = (int) ids.size(), nv = m.hp.n_vocab;
    i2::RefOpts ro;
    ro.cache_f16 = true;
    ro.threads = 4;
    i2::RefEngine ref(m, 64, ro);
    std::vector<float> all;
    ref.forward(ids.data(), n, nullptr, &all);
    auto row = [&](int t) { return all.data() + (size_t) t * (size_t) nv; };
    auto make = [&](int chunk, int slack, int host_experts = 0, int stage_slots = 32, int cache_slots = 0) {
        i2cuda::GpuOptions o;
        o.chunk = chunk; o.slack = slack; o.reserve_mib = 0; o.host_experts = host_experts; o.stage_slots = stage_slots; o.cache_slots = cache_slots; o.verbose = false;
        auto e = std::make_unique<i2cuda::GpuEngine>(m, 64, o);
        const std::string er = e->init();
        if (!er.empty()) { std::fprintf(stderr, "init: %s\n", er.c_str()); std::exit(1); }
        return e;
    };
    auto noop = [](int64_t) { return true; };
    std::vector<float> lg;
    const double tol = 2e-3;   // the K / V halves round a hair differently when the inputs differ by 1e-7: noise of 1e-3 after 48 visits
    {
        auto e = make(32, 1024);
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < tol && e->greedy() == argmax(row(n - 1), nv), "one chunk of %d tokens: within %.2e of the reference, same best token", n, r);
    }
    {   // local rings of window + chunk + slack = 6 + 4 + 2 = 12 cells wrap during the prompt
        auto e = make(4, 2);
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < tol && done == n, "chunks of 4 over 12-cell rings: within %.2e", r);
    }
    if (!quick) {
        auto e = make(8, 1024);
        int64_t done = 0;
        e->process(ids.data(), 5, noop, &done);
        double worst = 0;
        bool same = true;
        for (int t = 4; t < n; ++t) {
            if (t > 4) e->process(&ids[(size_t) t], 1, noop, &done);
            e->logits(lg);
            worst = std::max(worst, rel_rms(lg, row(t)));
            { const int g = e->greedy(); same &= g == argmax(lg.data(), lg.size()) && row(t)[g] >= row(t)[argmax(row(t), nv)] - 0.02f; }   // (a near-tie may go either way)
        }
        EXPECT(worst < tol && same, "decode, one token at a time for %d steps: worst %.2e", n - 4, worst);
    }
    {
        auto e = make(4, 6);
        int64_t done = 0;
        e->process(ids.data(), 16, noop, &done);
        EXPECT(e->can_keep(10) && e->can_keep(16) && !e->can_keep(17), "can_keep: 10 and 16 yes, 17 no");
        e->truncate(10);
        e->process(ids.data() + 10, n - 10, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < tol && e->pos() == n, "truncate to 10 and re-read: within %.2e", r);
    }
    {   // 22 tokens in 12 cells hold positions 10 .. 21; the token at q needs the keys from q - 5 on, so q >= 15
        auto e = make(4, 2);
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        EXPECT(!e->can_keep(2) && !e->can_keep(14) && e->can_keep(15) && e->can_keep(22), "can_keep after a wrap: 2 and 14 refused, 15 and 22 kept");
        e->truncate(15);
        e->process(ids.data() + 15, n - 15, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < tol, "truncate to 15 after a wrap and re-read: within %.2e", r);
    }
    {
        auto e = make(8, 1024, 4, 3);
        EXPECT(e->streamed_layers() == 4, "4 layers streamed (%d)", e->streamed_layers());
        int64_t done = 0;
        e->process(ids.data(), n - 3, noop, &done);
        double worst = 0;
        for (int t = n - 3; t < n; ++t) {
            e->process(&ids[(size_t) t], 1, noop, &done);
            e->logits(lg);
            worst = std::max(worst, rel_rms(lg, row(t)));
        }
        EXPECT(worst < tol, "4 layers' experts in host memory, 3 staging slots, prompt then 3 single tokens: worst %.2e", worst);
    }
    {
        auto e = make(32, 1024, 1000, 32);
        EXPECT(e->streamed_layers() == m.hp.n_layer, "all %d layers streamed (%d)", m.hp.n_layer, e->streamed_layers());
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < tol, "every layer streamed, one chunk: within %.2e", r);
    }
    {   // the expert cache: every layer streamed, 5 slots for 96 experts: a prompt through the staging slots, then single tokens through the cache (hits and evictions)
        auto e = make(8, 1024, 1000, 4, 5);
        int64_t done = 0;
        e->process(ids.data(), 6, noop, &done);
        double worst = 0;
        for (int t = 6; t < n; ++t) {
            e->process(&ids[(size_t) t], 1, noop, &done);
            e->logits(lg);
            worst = std::max(worst, rel_rms(lg, row(t)));
        }
        int slots = 0;
        long long hits = 0, misses = 0;
        e->cache_stats(&slots, &hits, &misses);
        EXPECT(worst < tol && slots == 5 && hits > 0 && misses > 0, "expert cache of %d slots over %d decoded tokens: worst %.2e, %lld hits, %lld misses", slots, n - 6, worst, hits, misses);
    }
    {   // and the same cache big enough for everything: after a first pass the steps of a repeated token are all hits
        auto e = make(8, 1024, 1000, 4, 96);
        int64_t done = 0;
        e->process(ids.data(), 6, noop, &done);
        for (int rep = 0; rep < 2; ++rep) {
            e->truncate(6);
            e->process(&ids[6], 1, noop, &done);
        }
        e->logits(lg);
        int slots = 0;
        long long hits = 0, misses = 0;
        e->cache_stats(&slots, &hits, &misses);
        EXPECT(rel_rms(lg, row(6)) < tol && hits >= 48 && hits + misses == 96, "a token decoded twice with a cache of everything: the second time is all hits (%lld hits, %lld misses of 96 lookups)", hits, misses);
    }
    std::printf(fails ? "%d FAILED\n" : "i2_emu_test: all scenarios pass\n", fails);
    return fails ? 1 : 0;
}
