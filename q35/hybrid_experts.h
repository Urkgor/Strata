// q35/hybrid_experts.h - the host side of the hybrid expert cache.
//
// What this does, in Strata's words: the experts stay in RAM (all of them), the card keeps a cache of the ones the
// model asks for most, and a decode step has the card compute the cached ones while the CPU computes only the rest.
// The graph is built by third_party/llama.cpp's build_moe_hybrid (q35/patches/0001-hybrid-expert-cache.patch, described
// in src/llama-hybrid.h); this class owns what the graph reads and writes:
//
//   * the cache tensors on the card and the four small tables of each layer (which experts are cached, where);
//   * the routing trace: the graph copies every token's experts into a tensor, after_decode() reads it back;
//   * the policy (expert_cache.h) that decides what to cache, and the copies from the GGUF file into the slots.
//
// Nothing here changes what the model computes: a cached expert is a byte-for-byte copy of the file's, and the graph
// adds the card's share and the CPU's share of each layer's mixture.
#pragma once

#include "expert_cache.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

namespace q35 {

struct HybridConfig {
    std::string mode = "off";   // off | sim (record the routing and simulate caches, compute as usual) | cache
    int64_t budget_mib = 0;     // cache > 0: this much card memory in all (0: what is free minus `margin_mib`)
    int slots_per_layer = 0;    // cache > 0: this many slots in every layer instead (tests, tuning)
    int margin_mib = 1024;      // card memory left free when the budget is taken from what is free
    bool in_ram = false;        // the cache lives in RAM (tests on a PC without a card): every layer takes part
    int max_tokens = 16;        // batches of up to this size use the hybrid graph
    int n_ubatch = 512;         // the largest batch llama_decode runs at once
    CacheParams policy;
    std::string profile;        // read at start, written at stop (and now and then): the uses of every expert
    std::string hot_file;       // experts the user places in the cache, locked there: lines `layer expert expert ...`
    std::string dump_file;      // written like the profile: the experts in the cache now, in the hot file's format
    bool static_cache = false;  // nothing is moved after the start: the cache is the hot file (and the profile's busiest)
    std::vector<int> sim_pct;   // capacities (% of the experts per layer) to simulate on the same traffic
    bool verbose = false;
};

class HybridExperts {
public:
    HybridExperts();
    ~HybridExperts();
    HybridExperts(const HybridExperts&) = delete;
    HybridExperts& operator=(const HybridExperts&) = delete;

    // Reads which layers take part, takes the card memory and allocates; does not attach to the model yet.
    bool init(llama_model* model, const std::string& gguf_path, const HybridConfig& cfg, std::string& err);
    void attach(llama_model* model);   // the graphs built from now on use it
    void detach(llama_model* model);

    bool ready() const;                // init() succeeded
    bool caching() const;              // mode cache: there are slots on the card
    void set_enabled(bool on);         // hybrid graphs (true) or the usual ones (false), for graphs built afterwards
    void set_fill_swaps(int n);        // experts moved per decode step while the cache has free slots
    int fill_swaps() const;

    // A llama_decode of `n` tokens (n <= n_ubatch) just ran: reads which experts its tokens used, counts, and moves the
    // experts the policy wants into the cache.
    void after_decode(int n);

    // the share of a request / of all requests (hits, lookups, swaps), the policy's own and the simulated ones
    CacheStats stats() const;
    struct Sim { int pct; int slots; CacheStats stats; };
    std::vector<Sim> sims() const;
    double update_ms() const;          // time spent in after_decode so far
    int layers_taking_part() const;
    int slots_total() const;
    int slots_used() const;
    size_t cache_bytes() const;
    std::string describe() const;      // one line for INFO / --verbose
    void reset_stats();
    // the uses of every expert, for a warm start next time; false: nothing to write
    bool save_profile() const;

    // test helpers: a layer's tables and the cached experts, to compare with what the graph does
    int experts_per_layer() const;
    int slot_of(int layer, int expert) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace q35
