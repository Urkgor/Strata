// g4_serve.hpp - the line protocol the Python server (serve/server.py) speaks to an engine, for the Gemma 4 engine.  One conversation held at a
// time: its tokens stay in the K / V cache, and the next request that begins with them reads only what follows.
//
//   engine -> server   INFO engine=<version>      READY <context> stop
//   server -> engine   GEN <max_new> [temperature=.. top_p=.. top_k=.. min_p=.. penalty_repeat=.. penalty_freq=.. penalty_present=..
//                                      penalty_last_n=.. seed=..] <id,id,...>          STOP (any time: ends the request in flight)
//   engine -> server   PP <read> <to read> <ms> <tok/s>   (while the prompt is read)    T <id>   (one per generated token)
//                      DONE <generated> <prompt tokens> <prompt ms> <decode ms> <stop|length|cancel> <accepted> <offered> <reused>
//                           0 0 0 0 0.0 <read> 0                                        ERR <message>
// The same lines as src/qwen35's engine, so the server's reader needs no new code.  `Engine` is any class with the members used below (the CPU
// reference and the CUDA engine both have them).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "strata/qwen35/sampling.hpp"

namespace g4 {

using Sampling = strata::qwen35::Sampling;

/// Pick the next token from a logit row: greedy at temperature 0, else top-k (at most 64) / min-p / top-p sampling, after the penalties.
inline int sample_token(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    strata::qwen35::apply_penalties(lg, s, hist);
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= p[i]) <= 0) return idx[i];
    return idx[keep - 1];
}

inline std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

template <class Engine>
class Serve {
public:
    using Clock = std::chrono::steady_clock;
    Serve(Engine& e, std::set<int32_t> eos, const char* version) : E(e), eos_(std::move(eos)), version_(version) {}

    int run() {
        std::printf("INFO engine=%s\n", version_);
        std::printf("READY %d stop\n", E.ctx());
        std::fflush(stdout);
        std::thread([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop_.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu_);
                lines_.push_back(l);
                cv_.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu_);
            eof_ = true;
            cv_.notify_one();
        }).detach();
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return eof_ || !lines_.empty(); });
                if (lines_.empty()) return 0;
                line = lines_.front();
                lines_.pop_front();
            }
            if (line.empty()) continue;
            if (line.rfind("GEN", 0) == 0 && line.rfind("GENI", 0) != 0 && line.rfind("GEN_", 0) != 0) request(line);
            else { std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n"); std::fflush(stdout); }
        }
    }

private:
    Engine& E;
    std::set<int32_t> eos_;
    const char* version_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> lines_;
    bool eof_ = false;
    std::atomic<bool> stop_{false};
    std::vector<int32_t> live_;           // the tokens the K / V cache holds, from cell 0

    static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
    void err(const std::string& m) { std::printf("ERR %s\n", m.c_str()); std::fflush(stdout); }

    void request(const std::string& line) {
        char* p = const_cast<char*>(line.c_str()) + 3;
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        bool prefill_only = false;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st, (size_t) (p - st));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "prefill") prefill_only = std::atoi(v) != 0;
            // (pin=, ckpt=, img_keys=: the Qwen engine's prompt-cache keys; this engine keeps one conversation and ignores them)
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 0 || ids.empty() || (max_new < 1 && !prefill_only)) { err("bad request: max_new or the ids"); return; }
        if ((long long) ids.size() + max_new > E.ctx()) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx());
            std::fflush(stdout);
            return;
        }
        for (int32_t t : ids)
            if (t < 0 || t >= E.n_vocab()) { err("a token id is outside the vocabulary"); return; }
        stop_.store(false);
        // what of the prompt the cache already holds: the common start, and at least the last token is read again (its logits are needed)
        size_t reuse = 0;
        while (reuse < live_.size() && reuse < ids.size() && live_[reuse] == ids[reuse]) ++reuse;
        if (reuse >= ids.size()) reuse = ids.size() - 1;
        if (!E.can_keep((int64_t) reuse)) reuse = 0;
        if (reuse == 0) { E.reset(); live_.clear(); }
        else { E.truncate((int64_t) reuse); live_.resize(reuse); }
        const auto t0 = Clock::now();
        const long long fresh = (long long) (ids.size() - reuse);
        bool cancelled = false;
        int64_t done_tokens = 0;
        auto on_piece = [&](int64_t done) -> bool {
            const double ms = ms_since(t0);
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, fresh, ms, (double) done / std::max(1e-9, ms / 1000.0));
            std::fflush(stdout);
            if (stop_.load() && done < fresh) { cancelled = true; return false; }
            return true;
        };
        const bool ok = E.process(ids.data() + reuse, (int64_t) fresh, on_piece, &done_tokens);
        live_.insert(live_.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) done_tokens);
        const double prompt_ms = ms_since(t0);
        if (!ok && !cancelled) { err("the engine failed while reading the prompt"); live_.clear(); E.reset(); return; }
        if (cancelled || max_new == 0) {
            std::printf("DONE 0 %zu %.1f 0.0 %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, cancelled ? "cancel" : "length", reuse, (long long) done_tokens);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed ? smp.seed : std::random_device{}());
        std::vector<int32_t> hist = ids;
        std::vector<float> lg;
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        for (;;) {
            int nx;
            if (smp.needs_logits()) { E.logits(lg); nx = sample_token(lg, smp, hist, rng); }
            else nx = E.greedy();
            std::printf("T %d\n", nx);
            std::fflush(stdout);
            ++produced;
            hist.push_back(nx);
            if (eos_.count(nx)) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (stop_.load()) { finish = "cancel"; break; }
            if (E.pos() + 1 > E.ctx()) break;
            int64_t did = 0;
            const int32_t one = nx;
            if (!E.process(&one, 1, [](int64_t) { return true; }, &did)) { err("the engine failed while decoding"); live_.clear(); E.reset(); return; }
            live_.push_back(nx);
        }
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms, ms_since(d0), finish, reuse, fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-gemma4: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), %lld generated in %.0f ms (%.1f tok/s) (%s)\n",
                     ids.size(), reuse, fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced, ms_since(d0),
                     ms_since(d0) > 0 ? 1000.0 * (double) produced / ms_since(d0) : 0.0, finish);
    }
};

}  // namespace g4
