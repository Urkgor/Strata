// The copy drafter of the Qwen3.6 engine (src/qwen35/q36_copy.hpp) against scripted answers, no GPU: a "model" that writes a fixed
// answer, a verify window that keeps the drafts up to the first one the answer does not have (what spec_step does), and the counts.
//  1. an edit: the answer writes back a 2,000-token file with three small changes: nearly every token comes from a copy;
//  2. look-alike records (JSONL): the answer returns the records in order with one field changed each; the copy must follow the
//     records in order (the index's "nearest after the stop" rule), not jump to an earlier record that looks the same;
//  3. prose that repeats nothing: no draft is offered after the start (the pause), so no window is wasted;
//  4. the output is the scripted answer whatever the drafter proposes (it only proposes).
#include "q36_copy.hpp"

#include <cstdio>
#include <random>
#include <vector>

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: "); } else std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

struct Run { long long steps = 0, offered = 0, kept = 0; std::vector<int32_t> out; };

// The decode loop of the engine's GEN path: emit x, propose, verify (keep the drafts the answer has), emit the kept ones, the next x.
static Run decode(const std::vector<int32_t>& prompt, const std::vector<int32_t>& answer) {
    q36cp::CopyDrafter d;
    d.reset(prompt, (int) answer.size());
    Run r;
    size_t i = 0;
    while (i < answer.size()) {
        const int32_t x = answer[i];
        r.out.push_back(x);
        d.push(x);
        ++i;
        ++r.steps;
        if (i >= answer.size()) break;
        std::vector<int32_t> dr = d.propose(7);
        int a = 0;
        while (a < (int) dr.size() && i + (size_t) a < answer.size() && dr[(size_t) a] == answer[i + (size_t) a]) ++a;
        d.result(a, (int) dr.size());
        r.offered += (long long) dr.size();
        r.kept += a;
        for (int k = 0; k < a; ++k) { r.out.push_back(answer[i]); d.push(answer[i]); ++i; }
    }
    return r;
}

int main() {
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> tok(0, 50000);
    {   // 1. an edit
        std::vector<int32_t> file(2000);
        for (auto& t : file) t = tok(rng);
        std::vector<int32_t> prompt = {1, 2, 3};
        prompt.insert(prompt.end(), file.begin(), file.end());
        prompt.insert(prompt.end(), {7, 8, 9, 10});
        std::vector<int32_t> answer = file;
        for (int at : {300, 1100, 1700}) answer[(size_t) at] = tok(rng);   // three small changes
        const Run r = decode(prompt, answer);
        EXPECT(r.out == answer, "edit: the output is the answer");
        const double per_step = (double) answer.size() / (double) r.steps;
        EXPECT(per_step > 6.0, "edit of %zu tokens in %lld steps (%.2f tokens a step; 8 at most), drafts kept %lld of %lld", answer.size(), r.steps,
               per_step, r.kept, r.offered);
    }
    {   // 2. look-alike records: {"id": i, "name": N, "v": V} with V changed in the answer
        auto rec = [&](int i, int v) { return std::vector<int32_t>{100, 101, 102, 200 + i % 3, 103, 104, 300, 301, 302, 105, 106, 400 + v, 107}; };
        std::vector<int32_t> prompt, answer;
        for (int i = 0; i < 60; ++i) { auto a = rec(i, i % 2); prompt.insert(prompt.end(), a.begin(), a.end()); }
        prompt.insert(prompt.end(), {9, 9, 9, 9});
        for (int i = 0; i < 60; ++i) { auto a = rec(i, (i % 2) ^ (i % 5 == 0)); answer.insert(answer.end(), a.begin(), a.end()); }
        const Run r = decode(prompt, answer);
        EXPECT(r.out == answer, "records: the output is the answer");
        EXPECT((double) answer.size() / (double) r.steps > 3.5, "records: %zu tokens in %lld steps (%.2f a step), drafts kept %lld of %lld (%.0f%%)", answer.size(),
               r.steps, (double) answer.size() / (double) r.steps, r.kept, r.offered, r.offered ? 100.0 * r.kept / r.offered : 0.0);
    }
    {   // 3. prose that repeats nothing
        std::vector<int32_t> prompt(500), answer(800);
        for (auto& t : prompt) t = tok(rng);
        for (auto& t : answer) t = tok(rng);
        const Run r = decode(prompt, answer);
        EXPECT(r.out == answer && r.offered == 0, "nothing to copy: no draft offered (%lld), %lld steps for %zu tokens", r.offered, r.steps, answer.size());
    }
    {   // 4. a false match: a 4-token phrase seen once before, then the answer goes elsewhere; the drafter pauses after two misses
        std::vector<int32_t> prompt;
        for (int i = 0; i < 40; ++i) { prompt.insert(prompt.end(), {5, 6, 7, 8}); prompt.push_back(tok(rng)); }
        std::vector<int32_t> answer;
        for (int i = 0; i < 40; ++i) { answer.insert(answer.end(), {5, 6, 7, 8}); answer.push_back(60000 + i); }
        const Run r = decode(prompt, answer);
        EXPECT(r.out == answer, "false matches: the output is the answer; drafts offered %lld, kept %lld", r.offered, r.kept);
        EXPECT(r.offered < (long long) answer.size(), "false matches: the pauses keep the offers down (%lld offered for %zu tokens)", r.offered, answer.size());
    }
    std::printf(fails ? "%d FAILED\n" : "q36_copy_test: all checks pass\n", fails);
    return fails ? 1 : 0;
}
