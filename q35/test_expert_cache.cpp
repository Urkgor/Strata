// Unit test of q35/expert_cache.h (no llama.cpp, no GPU):
//   g++ -std=c++17 -O2 q35/test_expert_cache.cpp -o /tmp/test-expert-cache && /tmp/test-expert-cache
// or the CMake target `test-expert-cache`.
#include "expert_cache.h"

#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>

using namespace q35;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// a skewed router: expert popularity ~ 1 / (rank + 1)^s, a different ranking in every layer, 8 distinct experts a token
struct Router {
    int n_layer, n_expert, n_used;
    double s;
    std::vector<std::vector<int>> perm;      // perm[l][rank] = expert
    std::vector<std::vector<double>> cum;    // cumulative popularity by rank
    std::mt19937_64 rng;
    Router(int nl, int ne, int nu, double skew, uint64_t seed) : n_layer(nl), n_expert(ne), n_used(nu), s(skew), rng(seed) { reshuffle(); }
    void reshuffle() {
        perm.assign((size_t) n_layer, std::vector<int>((size_t) n_expert));
        cum.assign((size_t) n_layer, std::vector<double>((size_t) n_expert));
        for (int l = 0; l < n_layer; ++l) {
            std::iota(perm[(size_t) l].begin(), perm[(size_t) l].end(), 0);
            std::shuffle(perm[(size_t) l].begin(), perm[(size_t) l].end(), rng);
            double acc = 0;
            for (int r = 0; r < n_expert; ++r) { acc += 1.0 / std::pow(r + 1.0, s); cum[(size_t) l][(size_t) r] = acc; }
        }
    }
    void token(int l, int32_t* ids) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int k = 0; k < n_used; ++k) {
            for (;;) {
                const double x = u(rng) * cum[(size_t) l].back();
                const int r = (int) (std::lower_bound(cum[(size_t) l].begin(), cum[(size_t) l].end(), x) - cum[(size_t) l].begin());
                const int e = perm[(size_t) l][(size_t) std::min(r, n_expert - 1)];
                bool dup = false;
                for (int j = 0; j < k; ++j) dup = dup || ids[j] == e;
                if (!dup) { ids[k] = e; break; }
            }
        }
    }
    // the hit rate of a cache that holds each layer's `slots` most popular experts (what an all-knowing static cache gets)
    double oracle(int slots) const {
        double hit = 0, all = 0;
        for (int l = 0; l < n_layer; ++l) {
            double top = 0;
            for (int r = 0; r < slots; ++r) top += 1.0 / std::pow(r + 1.0, s);
            hit += top;
            all += cum[(size_t) l].back();
        }
        return hit / all;   // (an approximation: it ignores that an expert cannot be picked twice by one token)
    }
};

static void check_tables(const HotSet& h) {
    for (int l = 0; l < h.n_layer(); ++l) {
        int held = 0;
        for (int s = 0; s < h.slots(l); ++s) {
            const int e = h.expert_at(l, s);
            if (e < 0) continue;
            ++held;
            CHECK(h.slot_of(l, e) == s);
        }
        int cached = 0;
        for (int e = 0; e < h.n_expert(); ++e) {
            const int s = h.slot_of(l, e);
            if (s < 0) continue;
            ++cached;
            CHECK(s < h.slots(l) && h.expert_at(l, s) == e);
        }
        CHECK(cached == held);
        CHECK(held <= h.slots(l));
    }
}

// one decode step over all layers; returns the swaps made
static size_t step(HotSet& h, Router& r, int n_tok, std::vector<CacheSwap>& sw, int budget = -1) {
    std::vector<int32_t> ids((size_t) n_tok * (size_t) r.n_used);
    for (int l = 0; l < r.n_layer; ++l) {
        for (int t = 0; t < n_tok; ++t) r.token(l, &ids[(size_t) t * (size_t) r.n_used]);
        h.observe(l, ids.data(), n_tok, r.n_used);
    }
    h.end_step(sw, budget);
    return sw.size();
}

static double run(HotSet& h, Router& r, int steps, int measure_from, std::vector<CacheSwap>& sw) {
    for (int i = 0; i < steps; ++i) {
        if (i == measure_from) h.reset_stats();
        step(h, r, 1, sw);
        if (i % 97 == 0) check_tables(h);
    }
    return h.stats().rate();
}

int main() {
    const int NL = 12, NE = 256, NU = 8;

    // 1. the tables stay consistent, the budget is kept, a free slot is used before any expert is displaced
    {
        HotSet h(NL, NE, std::vector<int>((size_t) NL, 40));
        Router r(NL, NE, NU, 0.9, 1);
        std::vector<CacheSwap> sw;
        step(h, r, 1, sw);
        CHECK((int) sw.size() <= h.params().fill_swaps);
        for (const CacheSwap& s : sw) CHECK(s.out == -1);
        check_tables(h);
        for (int i = 0; i < 300; ++i) { step(h, r, 1 + i % 5, sw); check_tables(h); }
        CHECK(h.full());
        for (int i = 0; i < 50; ++i) { step(h, r, 1, sw); CHECK((int) sw.size() <= h.params().swaps_per_step); }
        step(h, r, 1, sw, 0);
        CHECK(sw.empty());
        std::printf("1. consistent tables, budgets kept: ok\n");
    }

    // 2. against an all-knowing static cache, and against the first-S experts of each layer, at several capacities
    {
        double prev = 0;
        for (int pct : {5, 10, 18, 30, 50}) {
            const int slots = NE * pct / 100;
            HotSet h(NL, NE, std::vector<int>((size_t) NL, slots));
            Router r(NL, NE, NU, 0.9, 7);
            std::vector<CacheSwap> sw;
            const double rate = run(h, r, 6000, 3000, sw);
            const double oracle = r.oracle(slots);
            // the static baseline: experts 0..S-1 of each layer
            double base = 0;
            {
                HotSet b(NL, NE, std::vector<int>((size_t) NL, slots));
                for (int l = 0; l < NL; ++l) for (int s = 0; s < slots; ++s) b.put(l, s, s);
                Router r2(NL, NE, NU, 0.9, 7);
                std::vector<int32_t> ids(NU);
                for (int i = 0; i < 3000; ++i) for (int l = 0; l < NL; ++l) { r2.token(l, ids.data()); b.observe(l, ids.data(), 1, NU); }
                base = b.stats().rate();
            }
            std::printf("2. %2d%% of the experts: hit rate %.3f   all-knowing static %.3f   arbitrary static %.3f\n", pct, rate, oracle, base);
            CHECK(rate > base);
            CHECK(rate > 0.80 * oracle);
            CHECK(rate >= prev);
            prev = rate;
        }
    }

    // 3. the conversation changes: the hit rate falls and comes back
    {
        const int slots = NE * 18 / 100;
        HotSet h(NL, NE, std::vector<int>((size_t) NL, slots));
        Router r(NL, NE, NU, 0.9, 11);
        std::vector<CacheSwap> sw;
        const double before = run(h, r, 4000, 2000, sw);
        r.reshuffle();
        h.reset_stats();
        for (int i = 0; i < 100; ++i) step(h, r, 1, sw);
        const double just_after = h.stats().rate();
        const double later = run(h, r, 4000, 3000, sw);
        std::printf("3. drift: before %.3f, right after %.3f, later %.3f\n", before, just_after, later);
        CHECK(just_after < before);
        CHECK(later > 0.9 * before);
    }

    // 4. a profile warms a new cache; an empty profile or a wrong size does nothing
    {
        const int slots = 30;
        HotSet a(NL, NE, std::vector<int>((size_t) NL, slots));
        Router r(NL, NE, NU, 0.9, 5);
        std::vector<CacheSwap> sw;
        run(a, r, 2000, 1000, sw);
        const std::vector<float> prof = a.profile();
        HotSet b(NL, NE, std::vector<int>((size_t) NL, slots));
        CHECK(!b.load_profile(std::vector<float>(3)));
        b.fill_from_profile(sw);
        CHECK(sw.empty());
        CHECK(b.load_profile(prof));
        b.fill_from_profile(sw);
        check_tables(b);
        CHECK(b.full());
        CHECK((int) sw.size() == NL * slots);
        // the new cache holds each layer's most used experts: it hits like the old one right away
        Router r2(NL, NE, NU, 0.9, 5);
        r2.perm = r.perm; r2.cum = r.cum;
        std::vector<int32_t> ids(NU);
        for (int i = 0; i < 1000; ++i) for (int l = 0; l < NL; ++l) { r2.token(l, ids.data()); b.observe(l, ids.data(), 1, NU); }
        std::printf("4. warm start from a profile: hit rate %.3f (the run that made it: %.3f)\n", b.stats().rate(), a.stats().rate());
        CHECK(b.stats().rate() > 0.9 * a.stats().rate());
    }

    // 5. layers without room, a layer asked for more experts than it has, long prompt steps
    {
        std::vector<int> slots((size_t) NL, 20);
        slots[0] = 0;
        slots[1] = 1000;
        HotSet h(NL, NE, slots);
        CHECK(h.slots(0) == 0 && h.slots(1) == NE);
        Router r(NL, NE, NU, 0.9, 3);
        std::vector<CacheSwap> sw;
        step(h, r, 512, sw, 4096);                 // a prompt chunk of 512 tokens teaches a lot at once
        check_tables(h);
        CHECK(h.slots(1) == NE);
        for (const CacheSwap& s : sw) CHECK(s.layer != 0);
        std::printf("5. a 512-token step made %zu swaps\n", sw.size());
    }

    // 6. the same trace gives the same swaps
    {
        HotSet a(NL, NE, std::vector<int>((size_t) NL, 25)), b(NL, NE, std::vector<int>((size_t) NL, 25));
        Router ra(NL, NE, NU, 0.8, 9), rb(NL, NE, NU, 0.8, 9);
        std::vector<CacheSwap> sa, sb;
        for (int i = 0; i < 500; ++i) {
            step(a, ra, 1, sa); step(b, rb, 1, sb);
            CHECK(sa.size() == sb.size());
            for (size_t j = 0; j < sa.size() && j < sb.size(); ++j) CHECK(sa[j].layer == sb[j].layer && sa[j].in == sb[j].in && sa[j].slot == sb[j].slot);
        }
        std::printf("6. deterministic: ok\n");
    }

    // 7. the decay counter does not overflow on a very long run
    {
        HotSet h(2, 64, {8, 8}, CacheParams{16.0, 1.0, 2, 8});
        std::mt19937 rng(1);
        std::vector<int32_t> ids(4);
        std::vector<CacheSwap> sw;
        for (int i = 0; i < 20000; ++i) {
            for (int l = 0; l < 2; ++l) { for (int& x : ids) x = (int) (rng() % 64); h.observe(l, ids.data(), 1, 4); }
            h.end_step(sw);
        }
        for (int l = 0; l < 2; ++l) for (int e = 0; e < 64; ++e) CHECK(std::isfinite(h.uses(l, e)));
        check_tables(h);
        std::printf("7. long run: ok\n");
    }

    // 8. an expert placed by hand is never replaced, however little the model asks for it; a static cache never moves
    {
        std::vector<int> slots((size_t) NL, 10);
        slots[3] = 0;
        HotSet h(NL, NE, slots);
        Router r(NL, NE, NU, 0.9, 21);
        const int rare = r.perm[0][(size_t) NE - 1];     // the least popular expert of layer 0
        CacheSwap sw;
        CHECK(h.pin(0, rare, sw));
        CHECK(sw.layer == 0 && sw.in == rare && sw.out == -1);
        CHECK(h.slot_of(0, rare) >= 0 && h.locked(0, h.slot_of(0, rare)));
        CHECK(!h.pin(0, rare, sw));                      // already there
        CHECK(!h.pin(3, 1, sw));                         // a layer without room
        int placed = 0;
        for (int e = 0; e < 20; ++e) if (e != rare && h.pin(1, e, sw)) ++placed;
        CHECK(placed == 10);                             // the layer's room, no more
        std::vector<CacheSwap> sws;
        for (int i = 0; i < 4000; ++i) step(h, r, 1, sws);
        CHECK(h.slot_of(0, rare) >= 0);                  // still there after 4000 steps of traffic that never asks for it
        int still = 0;
        for (int s = 0; s < h.slots(1); ++s) still += h.locked(1, s) && h.expert_at(1, s) >= 0;
        CHECK(still == 10);                              // layer 1 was filled by hand: nothing in it moved
        check_tables(h);

        HotSet st(NL, NE, std::vector<int>((size_t) NL, 10));
        st.set_static(true);
        Router r2(NL, NE, NU, 0.9, 22);
        size_t moved = 0;
        for (int i = 0; i < 300; ++i) moved += step(st, r2, 1, sws);
        CHECK(moved == 0 && st.used_slots() == 0);
        std::printf("8. placed experts stay, a static cache does not move: ok\n");
    }

    std::printf(g_fail ? "FAILED (%d)\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
