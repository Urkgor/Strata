// q35/hybrid_experts.cpp - see hybrid_experts.h
#include "hybrid_experts.h"

#include "llama.h"
#include "llama-hybrid.h"   // third_party/llama.cpp/src, added by the patch
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace q35 {

namespace {

constexpr size_t MiB = 1024u * 1024u;

struct Shard {
    std::string path;
    int fd = -1;
    size_t size = 0;
    const uint8_t* map = nullptr;
};

struct Loc {   // where a tensor's bytes are in the GGUF file(s)
    int shard = -1;
    size_t off = 0;
    size_t bytes = 0;
};

struct Layer {
    int il = 0;
    llama_hybrid_experts t;
    int n_expert = 0;
    int slots = 0;
    int group = -1;
    size_t slot_bytes = 0;                     // up + gate + down (or gate_up + down) for one expert
    Loc loc_up, loc_gate, loc_gate_up, loc_down;
    ggml_tensor *c_up = nullptr, *c_gate = nullptr, *c_gate_up = nullptr, *c_down = nullptr;
    ggml_tensor *hit_f = nullptr, *slot_i = nullptr, *miss_f = nullptr, *cpu_i = nullptr;
    std::vector<float> hit, miss;              // host copies of the tables
    std::vector<int32_t> slot, cpuid;
};

struct Group {   // the layers whose compute is on one card (or all of them, for a CPU-only run)
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_backend_dev_t dev = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor* trace = nullptr;
    std::vector<int32_t> trace_host;
    std::vector<int> layers;                   // indexes into Impl::layers
    size_t budget = 0;                         // bytes for the cache
};

std::vector<std::string> shards_of(const std::string& first) {
    static const std::regex re("^(.*)-([0-9]{5})-of-([0-9]{5})\\.gguf$");
    std::smatch m;
    if (!std::regex_match(first, m, re)) return {first};
    const int n = std::atoi(m[3].str().c_str());
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char num[32];
        std::snprintf(num, sizeof num, "-%05d-of-%05d.gguf", i, n);
        out.push_back(m[1].str() + num);
    }
    return out;
}

std::string meta(const llama_model* m, const std::string& key) {
    char buf[256];
    const int n = llama_model_meta_val_str(m, key.c_str(), buf, sizeof buf);
    return n < 0 ? std::string() : std::string(buf);
}

}  // namespace

struct HybridExperts::Impl {
    HybridConfig cfg;
    bool ready = false;
    bool cache = false;
    bool enabled = true;
    int n_layer = 0, n_expert = 0, n_used = 0;
    std::vector<Layer> layers;
    std::vector<Group> groups;
    ggml_context* ctx_cpu = nullptr;           // the tables the CPU reads
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::vector<Shard> shards;
    std::unique_ptr<HotSet> hs;
    struct SimSet { int pct; std::unique_ptr<HotSet> set; };
    std::vector<SimSet> sims;
    llama_hybrid hy;
    double update_ms = 0;
    uint64_t decode_tokens = 0, saved_at = 0;
    std::vector<int32_t> ids;

    ~Impl() {
        for (Group& g : groups) {
            if (g.buf) ggml_backend_buffer_free(g.buf);
            if (g.ctx) ggml_free(g.ctx);
        }
        if (buf_cpu) ggml_backend_buffer_free(buf_cpu);
        if (ctx_cpu) ggml_free(ctx_cpu);
        for (Shard& s : shards) {
            if (s.map) munmap(const_cast<uint8_t*>(s.map), s.size);
            if (s.fd >= 0) close(s.fd);
        }
    }

    // ------------------------------------------------------------------------------------------ the GGUF file
    bool index_file(const std::string& first, std::string& err) {
        std::map<std::string, Loc> found;
        const std::vector<std::string> paths = shards_of(first);
        for (size_t si = 0; si < paths.size(); ++si) {
            gguf_init_params gp{true, nullptr};
            gguf_context* g = gguf_init_from_file(paths[si].c_str(), gp);
            if (!g) { err = "cannot read " + paths[si]; return false; }
            Shard sh;
            sh.path = paths[si];
            const size_t data = gguf_get_data_offset(g);
            for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
                Loc l;
                l.shard = (int) si;
                l.off = data + gguf_get_tensor_offset(g, i);
                l.bytes = gguf_get_tensor_size(g, i);
                found[gguf_get_tensor_name(g, i)] = l;
            }
            gguf_free(g);
            shards.push_back(sh);
        }
        auto name_of = [](const ggml_tensor* t) { return std::string(ggml_get_name(t)); };
        for (Layer& L : layers) {
            auto get = [&](const ggml_tensor* t, Loc& out) -> bool {
                if (!t) return true;
                auto it = found.find(name_of(t));
                if (it == found.end() || it->second.bytes != ggml_nbytes(t)) { err = "tensor " + name_of(t) + " is not in the GGUF file as the model has it"; return false; }
                out = it->second;
                return true;
            };
            if (!get(L.t.up, L.loc_up) || !get(L.t.gate, L.loc_gate) || !get(L.t.gate_up, L.loc_gate_up) || !get(L.t.down, L.loc_down)) return false;
        }
        std::vector<bool> need(shards.size(), false);
        for (const Layer& L : layers)
            for (const Loc* l : {&L.loc_up, &L.loc_gate, &L.loc_gate_up, &L.loc_down})
                if (l->shard >= 0) need[(size_t) l->shard] = true;
        for (size_t i = 0; i < shards.size(); ++i) {
            if (!need[i]) continue;
            Shard& s = shards[i];
            s.fd = open(s.path.c_str(), O_RDONLY);
            if (s.fd < 0) { err = "cannot open " + s.path; return false; }
            struct stat st;
            if (fstat(s.fd, &st) != 0) { err = "cannot stat " + s.path; return false; }
            s.size = (size_t) st.st_size;
            void* m = mmap(nullptr, s.size, PROT_READ, MAP_SHARED, s.fd, 0);
            if (m == MAP_FAILED) { err = "cannot map " + s.path; return false; }
            madvise(m, s.size, MADV_RANDOM);   // slices of 0.5 MB, not a sequential read
            s.map = (const uint8_t*) m;
        }
        return true;
    }

    // copies expert e of tensor `loc` into slot `slot` of `cache`
    bool copy_expert(ggml_tensor* cache, const Loc& loc, int e, int slot) {
        const size_t per = cache->nb[2];
        const Shard& s = shards[(size_t) loc.shard];
        const size_t off = loc.off + (size_t) e * per;
        if (off + per > s.size) return false;
        ggml_backend_tensor_set(cache, s.map + off, (size_t) slot * per, per);
        return true;
    }

    // ------------------------------------------------------------------------------------------ the tables
    void push_tables(Layer& L) {
        // the stand-in for the experts the CPU is not asked for: the most used one that is not cached anyway
        int stand_in = 0;
        double best = -1.0;
        for (int e = 0; e < L.n_expert; ++e) {
            if (hs->slot_of(L.il, e) >= 0) continue;
            const double u = hs->uses(L.il, e);
            if (u > best) { best = u; stand_in = e; }
        }
        for (int e = 0; e < L.n_expert; ++e) {
            const int s = hs->slot_of(L.il, e);
            L.hit[(size_t) e] = s >= 0 ? 1.0f : 0.0f;
            L.slot[(size_t) e] = s >= 0 ? s : 0;
            L.miss[(size_t) e] = s >= 0 ? 0.0f : 1.0f;
            L.cpuid[(size_t) e] = s >= 0 ? stand_in : e;
        }
        ggml_backend_tensor_set(L.hit_f, L.hit.data(), 0, L.hit.size() * sizeof(float));
        ggml_backend_tensor_set(L.slot_i, L.slot.data(), 0, L.slot.size() * sizeof(int32_t));
        ggml_backend_tensor_set(L.miss_f, L.miss.data(), 0, L.miss.size() * sizeof(float));
        ggml_backend_tensor_set(L.cpu_i, L.cpuid.data(), 0, L.cpuid.size() * sizeof(int32_t));
    }

    void apply(const std::vector<CacheSwap>& sw) {
        std::vector<bool> touched((size_t) n_layer, false);
        for (const CacheSwap& s : sw) {
            Layer* L = nullptr;
            for (Layer& x : layers) if (x.il == s.layer) { L = &x; break; }
            if (!L) continue;
            bool ok = copy_expert(L->c_down, L->loc_down, s.in, s.slot);
            if (L->c_gate_up) ok = ok && copy_expert(L->c_gate_up, L->loc_gate_up, s.in, s.slot);
            else ok = ok && copy_expert(L->c_up, L->loc_up, s.in, s.slot) && copy_expert(L->c_gate, L->loc_gate, s.in, s.slot);
            if (!ok) std::fprintf(stderr, "strata-q35: expert cache: could not read expert %d of layer %d from the file\n", s.in, s.layer);
            touched[(size_t) s.layer] = true;
        }
        for (Layer& L : layers) if (touched[(size_t) L.il]) push_tables(L);
    }

    // ------------------------------------------------------------------------------------------ setup
    bool init(llama_model* model, const std::string& gguf_path, const HybridConfig& c, std::string& err);
    void after_decode(int n);
    bool load_profile();
    bool read_profile(std::vector<float>& p) const;
    bool nonuniform = false;
    std::vector<std::vector<int>> hot;         // by layer: the experts of the hot file
    bool parse_hot(std::string& err);
    void save_dump() const;
    int moe_layers = 0;                        // layers with routed experts, whether or not they take part
    bool save_profile() const;
};

// -------------------------------------------------------------------------------------------------- profile

bool HybridExperts::Impl::save_profile() const {
    if (cfg.profile.empty() || !hs) return false;
    const std::vector<float> p = hs->profile();
    const std::string tmp = cfg.profile + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const int32_t head[4] = {0x50355133, n_layer, n_expert, n_used};   // "3Q5P"
    bool ok = std::fwrite(head, sizeof head, 1, f) == 1 && std::fwrite(p.data(), sizeof(float), p.size(), f) == p.size();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) { std::remove(tmp.c_str()); return false; }
    return std::rename(tmp.c_str(), cfg.profile.c_str()) == 0;
}

bool HybridExperts::Impl::read_profile(std::vector<float>& p) const {
    if (cfg.profile.empty()) return false;
    FILE* f = std::fopen(cfg.profile.c_str(), "rb");
    if (!f) return false;
    int32_t head[4] = {0, 0, 0, 0};
    p.assign((size_t) n_layer * (size_t) n_expert, 0.0f);
    const bool ok = std::fread(head, sizeof head, 1, f) == 1 && head[0] == 0x50355133 && head[1] == n_layer && head[2] == n_expert &&
                    head[3] == n_used && std::fread(p.data(), sizeof(float), p.size(), f) == p.size();
    std::fclose(f);
    return ok;
}

bool HybridExperts::Impl::load_profile() {
    if (!hs) return false;
    std::vector<float> p;
    if (!read_profile(p)) return false;
    if (!hs->load_profile(p)) return false;
    std::vector<CacheSwap> sw;
    hs->fill_from_profile(sw);
    apply(sw);
    for (auto& s : sims) {
        s.set->load_profile(p);
        std::vector<CacheSwap> ignore;
        s.set->fill_from_profile(ignore);
    }
    return true;
}

// The hot file: `layer expert expert ...` per line (spaces, commas or colons between; # starts a comment).
bool HybridExperts::Impl::parse_hot(std::string& err) {
    hot.assign((size_t) n_layer, std::vector<int>());
    if (cfg.hot_file.empty()) return true;
    std::ifstream f(cfg.hot_file);
    if (!f) { err = "cannot read the hot file " + cfg.hot_file; return false; }
    std::string line;
    int no = 0;
    while (std::getline(f, line)) {
        ++no;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::vector<long> v;
        const char* c = line.c_str();
        while (*c) {
            while (*c == ' ' || *c == '\t' || *c == ',' || *c == ':' || *c == '\r') ++c;
            if (!*c) break;
            char* e = nullptr;
            const long x = std::strtol(c, &e, 10);
            if (e == c) { err = cfg.hot_file + " line " + std::to_string(no) + ": not a number near \"" + std::string(c, std::min<size_t>(8, std::strlen(c))) + "\""; return false; }
            v.push_back(x);
            c = e;
        }
        if (v.empty()) continue;
        if (v[0] < 0 || v[0] >= n_layer) { err = cfg.hot_file + " line " + std::to_string(no) + ": layer " + std::to_string(v[0]) + " (the model has " + std::to_string(n_layer) + ")"; return false; }
        for (size_t i = 1; i < v.size(); ++i) {
            if (v[i] < 0 || v[i] >= n_expert) { err = cfg.hot_file + " line " + std::to_string(no) + ": expert " + std::to_string(v[i]) + " (a layer has " + std::to_string(n_expert) + ")"; return false; }
            std::vector<int>& h = hot[(size_t) v[0]];
            if (std::find(h.begin(), h.end(), (int) v[i]) == h.end()) h.push_back((int) v[i]);
        }
    }
    return true;
}

void HybridExperts::Impl::save_dump() const {
    if (cfg.dump_file.empty() || !hs) return;
    const std::string tmp = cfg.dump_file + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "# experts in the cache, layer by layer: a hot file for --cache-hot (edit it, or keep what the model asked for)\n");
    for (const Layer& L : layers) {
        std::vector<int> es;
        for (int e = 0; e < L.n_expert; ++e) if (hs->slot_of(L.il, e) >= 0) es.push_back(e);
        if (es.empty()) continue;
        std::fprintf(f, "%d", L.il);
        for (int e : es) std::fprintf(f, " %d", e);
        std::fprintf(f, "\n");
    }
    const bool ok = std::fclose(f) == 0;
    if (ok) std::rename(tmp.c_str(), cfg.dump_file.c_str()); else std::remove(tmp.c_str());
}

// -------------------------------------------------------------------------------------------------- init

bool HybridExperts::Impl::init(llama_model* model, const std::string& gguf_path, const HybridConfig& c, std::string& err) {
    cfg = c;
    const bool want_cache = cfg.mode == "cache";
    cache = want_cache;
    n_layer = llama_model_n_layer(model);
    const std::string arch = meta(model, "general.architecture");
    n_used = std::atoi(meta(model, arch + ".expert_used_count").c_str());
    if (n_used <= 0) { err = "the model does not say how many experts a token uses (" + arch + ".expert_used_count)"; return false; }

    // ---- the layers that take part
    int skipped_scales = 0, skipped_backend = 0, skipped_type = 0;
    for (int il = 0; il < n_layer; ++il) {
        Layer L;
        L.il = il;
        if (!llama_model_expert_tensors(model, il, &L.t)) continue;
        ++moe_layers;
        if (L.t.has_scales) { ++skipped_scales; continue; }
        if (!L.t.down->buffer || !L.t.gate_inp->buffer) continue;
        L.n_expert = (int) L.t.down->ne[2];
        if (n_expert == 0) n_expert = L.n_expert;
        if (L.n_expert != n_expert) continue;
        // in RAM: plain host memory, or llama.cpp's repacked CPU layout (its buffer type is not "host" because the data cannot be
        // read back as it was, but it is the CPU's; the cache copies from the GGUF file, not from the tensor)
        auto in_ram = [](const ggml_tensor* t) {
            if (!t || !t->buffer) return true;
            if (ggml_backend_buffer_is_host(t->buffer)) return true;
            ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
            return d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU;
        };
        const bool experts_in_ram = in_ram(L.t.down) && in_ram(L.t.up) && in_ram(L.t.gate) && in_ram(L.t.gate_up);
        ggml_backend_buffer_type_t lbuft = ggml_backend_buffer_get_type(L.t.gate_inp->buffer);
        const bool layer_on_card = !ggml_backend_buft_is_host(lbuft);
        if (want_cache && !cfg.in_ram && !(experts_in_ram && layer_on_card)) continue;   // nothing to cache for it
        if (want_cache && cfg.in_ram && !experts_in_ram) continue;
        if (want_cache && !cfg.in_ram) {
            // what the card's kernels were checked for: CUDA's (ROCm's are the same code), and quantized experts: for other
            // types CUDA picks mul_mat_id kernels that cannot take a token naming an expert twice, as the cache does
            ggml_backend_buffer_type_t b = ggml_backend_buffer_get_type(L.t.gate_inp->buffer);
            ggml_backend_dev_t d = ggml_backend_buft_get_device(b);
            const char* dn = d ? ggml_backend_dev_name(d) : "";
            if (std::strncmp(dn, "CUDA", 4) != 0 && std::strncmp(dn, "ROCm", 4) != 0) { ++skipped_backend; continue; }
            bool quantized = true;
            for (const ggml_tensor* t : {L.t.up, L.t.gate, L.t.gate_up, L.t.down}) if (t && !ggml_is_quantized(t->type)) quantized = false;
            if (!quantized) { ++skipped_type; continue; }
        }
        L.slot_bytes = (L.t.gate_up ? L.t.gate_up->nb[2] : L.t.up->nb[2] + L.t.gate->nb[2]) + L.t.down->nb[2];
        L.slot_bytes = std::max<size_t>(L.slot_bytes, 1);
        // the group of its card
        ggml_backend_buffer_type_t gbuft = (want_cache && cfg.in_ram) ? ggml_backend_cpu_buffer_type() : lbuft;
        int gi = -1;
        for (size_t i = 0; i < groups.size(); ++i) if (groups[i].buft == gbuft) gi = (int) i;
        if (gi < 0) {
            Group g;
            g.buft = gbuft;
            g.dev = ggml_backend_buft_get_device(gbuft);
            groups.push_back(g);
            gi = (int) groups.size() - 1;
        }
        L.group = gi;
        groups[(size_t) gi].layers.push_back((int) layers.size());
        layers.push_back(L);
    }
    if (layers.empty()) {
        err = want_cache ? (skipped_backend ? "the card's backend is not CUDA (the cache was written for CUDA's kernels)"
                           : skipped_type ? "the experts are not quantized (CUDA would pick a mul_mat_id kernel the cache cannot use)"
                           : skipped_scales ? "the experts have per-expert scales (not supported by the cache)"
                                           : "no layer has its routed experts in RAM and the rest on a card (use --cpu-moe)")
                         : "the model has no routed experts";
        return false;
    }

    // ---- how many experts each layer can hold
    if (want_cache) {
        if (!parse_hot(err)) return false;
        for (Group& g : groups) {
            size_t avail = 0;
            if (cfg.slots_per_layer > 0) {
                avail = SIZE_MAX;
            } else if (cfg.budget_mib > 0) {
                avail = (size_t) ((double) cfg.budget_mib * (double) MiB * (double) g.layers.size() / (double) layers.size());
            } else if (g.dev && !cfg.in_ram) {
                size_t free_b = 0, total_b = 0;
                ggml_backend_dev_memory(g.dev, &free_b, &total_b);
                const size_t margin = (size_t) cfg.margin_mib * MiB;
                avail = free_b > margin ? free_b - margin : 0;
            } else {
                err = "the cache needs a size here: --expert-cache MIB (or --cache-slots N)";
                return false;
            }
            g.budget = avail;
            // the experts the user placed come first; the rest of the memory is for the policy
            size_t locked_bytes = 0;
            for (int li : g.layers) locked_bytes += hot[(size_t) layers[(size_t) li].il].size() * layers[(size_t) li].slot_bytes;
            if (avail != SIZE_MAX && locked_bytes > avail) {
                err = "the hot file asks for " + std::to_string(locked_bytes >> 20) + " MiB of cache and there are " + std::to_string(avail >> 20) + " MiB";
                return false;
            }
            const size_t avail_rest = avail == SIZE_MAX ? SIZE_MAX : avail - locked_bytes;
            for (int li : g.layers) {
                Layer& L = layers[(size_t) li];
                const size_t nh = hot[(size_t) L.il].size();
                size_t slots = cfg.slots_per_layer > 0 ? (size_t) cfg.slots_per_layer : nh + avail_rest / g.layers.size() / L.slot_bytes;
                L.slots = (int) std::min<size_t>(std::max<size_t>(slots, nh), (size_t) L.n_expert);
            }
            // With a profile of an earlier run, the layers do not get the same share: the experts most used over all the layers
            // of the card fill it (a layer whose routing is concentrated gets more of the cache than one that spreads over all
            // its experts), so the same memory serves more of the uses.
            std::vector<float> prof;
            if (cfg.slots_per_layer == 0 && avail_rest != SIZE_MAX && read_profile(prof)) {
                struct Cand { double v; size_t li; };
                std::vector<Cand> cands;
                for (int li : g.layers) {
                    const Layer& L = layers[(size_t) li];
                    for (int e = 0; e < L.n_expert; ++e) {
                        const double sc = prof[(size_t) L.il * (size_t) n_expert + (size_t) e];
                        const std::vector<int>& h = hot[(size_t) L.il];
                        if (sc > 0 && std::find(h.begin(), h.end(), e) == h.end()) cands.push_back({sc / (double) L.slot_bytes, (size_t) li});
                    }
                }
                std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.v != b.v ? a.v > b.v : a.li < b.li; });
                std::vector<int> count(layers.size(), 0);
                size_t used = 0;
                for (int li : g.layers) count[(size_t) li] = (int) hot[(size_t) layers[(size_t) li].il].size();   // the placed ones are counted
                for (const Cand& c : cands) {
                    const Layer& L = layers[c.li];
                    if (count[c.li] >= L.n_expert || used + L.slot_bytes > avail_rest) continue;
                    ++count[c.li];
                    used += L.slot_bytes;
                }
                // what the profile does not ask for (experts never seen): spread over the layers that can still take some
                for (bool progress = true; progress;) {
                    progress = false;
                    for (int li : g.layers) {
                        const Layer& L = layers[(size_t) li];
                        if (count[(size_t) li] < L.n_expert && used + L.slot_bytes <= avail_rest) { ++count[(size_t) li]; used += L.slot_bytes; progress = true; }
                    }
                }
                int lo = 1 << 30, hi = 0;
                for (int li : g.layers) {
                    layers[(size_t) li].slots = count[(size_t) li];
                    lo = std::min(lo, count[(size_t) li]);
                    hi = std::max(hi, count[(size_t) li]);
                }
                if (lo != hi) nonuniform = true;
            }
        }
        int total = 0;
        for (const Layer& L : layers) total += L.slots;
        if (total == 0) { err = "no room for even one expert per layer on the card (--expert-cache MIB, or free some VRAM)"; return false; }
    }

    // ---- allocate: per card its cache, its tables and its trace; in RAM the tables the CPU reads
    const size_t per_layer_tensors = 8;
    auto make_ctx = [&](size_t n_tensors) {
        ggml_init_params ip{ggml_tensor_overhead() * (n_tensors + 4) + 4096, nullptr, true};
        return ggml_init(ip);
    };
    for (Group& g : groups) {
        g.ctx = make_ctx(g.layers.size() * per_layer_tensors + 4);
        if (!g.ctx) { err = "ggml_init failed"; return false; }
        g.trace = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, n_used, n_layer, cfg.n_ubatch);
        ggml_set_name(g.trace, "hybrid_trace");
    }
    ctx_cpu = make_ctx(layers.size() * per_layer_tensors + 4);
    if (!ctx_cpu) { err = "ggml_init failed"; return false; }
    for (Layer& L : layers) {
        Group& g = groups[(size_t) L.group];
        const std::string sfx = "." + std::to_string(L.il);
        if (want_cache && L.slots > 0) {
            auto mk = [&](const ggml_tensor* src, const char* nm) {
                ggml_tensor* t = ggml_new_tensor_3d(g.ctx, src->type, src->ne[0], src->ne[1], L.slots);
                ggml_set_name(t, (std::string(nm) + sfx).c_str());
                return t;
            };
            L.c_down = mk(L.t.down, "hot_down");
            if (L.t.gate_up) L.c_gate_up = mk(L.t.gate_up, "hot_gate_up");
            else { L.c_up = mk(L.t.up, "hot_up"); L.c_gate = mk(L.t.gate, "hot_gate"); }
        }
        L.hit_f = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 1, L.n_expert);
        L.slot_i = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, 1, L.n_expert);
        L.miss_f = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, 1, L.n_expert);
        L.cpu_i = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_I32, 1, L.n_expert);
        ggml_set_name(L.hit_f, ("hot_hit" + sfx).c_str());
        ggml_set_name(L.slot_i, ("hot_slot" + sfx).c_str());
        ggml_set_name(L.miss_f, ("cold_miss" + sfx).c_str());
        ggml_set_name(L.cpu_i, ("cold_id" + sfx).c_str());
        L.hit.assign((size_t) L.n_expert, 0.0f);
        L.miss.assign((size_t) L.n_expert, 1.0f);
        L.slot.assign((size_t) L.n_expert, 0);
        L.cpuid.resize((size_t) L.n_expert);
        for (int e = 0; e < L.n_expert; ++e) L.cpuid[(size_t) e] = e;
    }
    for (Group& g : groups) {
        g.buf = ggml_backend_alloc_ctx_tensors_from_buft(g.ctx, g.buft);
        if (!g.buf) { err = "could not allocate the expert cache on the card (out of memory: a smaller --expert-cache?)"; return false; }
        ggml_backend_buffer_set_usage(g.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // the scheduler runs their operations beside them
        ggml_backend_buffer_clear(g.buf, 0);                                       // zero blocks dequantize to 0: no NaN in a slot never filled
        g.trace_host.assign((size_t) n_used * (size_t) n_layer * (size_t) cfg.n_ubatch, 0);
    }
    buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type());
    if (!buf_cpu) { err = "could not allocate the expert tables in RAM"; return false; }
    ggml_backend_buffer_set_usage(buf_cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // ---- the policy, the file, the first tables
    std::vector<int> slots((size_t) n_layer, 0);
    for (const Layer& L : layers) slots[(size_t) L.il] = L.slots;
    if (want_cache) {
        hs.reset(new HotSet(n_layer, n_expert, slots, cfg.policy));
        if (!index_file(gguf_path, err)) return false;
        for (Layer& L : layers) push_tables(L);
        {   // the experts the user placed: copied in first and never replaced
            std::vector<CacheSwap> pins;
            int ignored = 0;
            for (int il = 0; il < n_layer; ++il) {
                const bool takes_part = std::any_of(layers.begin(), layers.end(), [&](const Layer& L) { return L.il == il; });
                for (int e : hot[(size_t) il]) {
                    CacheSwap sw;
                    if (!takes_part) { ++ignored; continue; }
                    if (hs->pin(il, e, sw)) pins.push_back(sw);
                }
            }
            apply(pins);
            if (ignored) std::fprintf(stderr, "strata-q35: the hot file names %d experts in layers that take no part in the cache (their experts are not in RAM): ignored\n", ignored);
        }
    } else {
        // the tables are never read: the graph is the usual one
        for (Layer& L : layers) {
            ggml_backend_tensor_set(L.hit_f, L.hit.data(), 0, L.hit.size() * sizeof(float));
            ggml_backend_tensor_set(L.slot_i, L.slot.data(), 0, L.slot.size() * sizeof(int32_t));
            ggml_backend_tensor_set(L.miss_f, L.miss.data(), 0, L.miss.size() * sizeof(float));
            ggml_backend_tensor_set(L.cpu_i, L.cpuid.data(), 0, L.cpuid.size() * sizeof(int32_t));
        }
    }
    std::vector<int> sim_in = cfg.sim_pct;
    for (int pct : sim_in) {
        if (pct <= 0 || pct > 100) continue;
        std::vector<int> ss((size_t) n_layer, 0);
        for (const Layer& L : layers) ss[(size_t) L.il] = std::max(1, n_expert * pct / 100);
        sims.push_back({pct, std::unique_ptr<HotSet>(new HotSet(n_layer, n_expert, ss, cfg.policy))});
    }
    if (want_cache) {
        load_profile();
        if (cfg.static_cache) {
            hs->set_static(true);
            if (hs->used_slots() == 0) std::fprintf(stderr, "strata-q35: --cache-static with nothing placed (no --cache-hot, no profile): the cache stays empty\n");
        }
    }

    // ---- what the graph sees
    hy.enabled = false;
    hy.max_tokens = cfg.max_tokens;
    hy.trace_tokens = cfg.n_ubatch;
    hy.layers.assign((size_t) n_layer, llama_hybrid_layer());
    for (Layer& L : layers) {
        llama_hybrid_layer& h = hy.layers[(size_t) L.il];
        h.up = L.c_up; h.gate = L.c_gate; h.gate_up = L.c_gate_up; h.down = L.c_down;
        h.hit_f = L.hit_f; h.slot_i = L.slot_i; h.miss_f = L.miss_f; h.cpu_i = L.cpu_i;
        h.trace = groups[(size_t) L.group].trace;
    }
    ready = true;
    return true;
}

// -------------------------------------------------------------------------------------------------- after a decode

void HybridExperts::Impl::after_decode(int n) {
    if (!ready || n <= 0 || n > cfg.n_ubatch) return;
    const auto t0 = std::chrono::steady_clock::now();
    for (Group& g : groups) {
        const size_t bytes = (size_t) n * (size_t) n_layer * (size_t) n_used * sizeof(int32_t);
        ggml_backend_tensor_get(g.trace, g.trace_host.data(), 0, bytes);
    }
    const bool decode_like = n <= cfg.max_tokens;
    const bool count_hits = decode_like && enabled;
    ids.resize((size_t) n * (size_t) n_used);
    for (const Layer& L : layers) {
        const int32_t* src = groups[(size_t) L.group].trace_host.data();
        for (int t = 0; t < n; ++t)
            std::memcpy(&ids[(size_t) t * (size_t) n_used], src + ((size_t) t * (size_t) n_layer + (size_t) L.il) * (size_t) n_used,
                        (size_t) n_used * sizeof(int32_t));
        if (hs) hs->observe(L.il, ids.data(), n, n_used, count_hits);
        for (auto& s : sims) s.set->observe(L.il, ids.data(), n, n_used, decode_like);
    }
    // a long prompt teaches a lot at once: it may fill the free slots and replace some more; a decode step moves a few
    int budget = -1;
    if (!decode_like) budget = 64;
    std::vector<CacheSwap> sw;
    if (hs) {
        if (!decode_like) budget += hs->total_slots() - hs->used_slots();
        hs->end_step(sw, std::min(budget, budget < 0 ? budget : 4096));
        apply(sw);
    }
    for (auto& s : sims) {
        std::vector<CacheSwap> ignore;
        int b = budget < 0 ? -1 : 64 + s.set->total_slots() - s.set->used_slots();
        s.set->end_step(ignore, b);
    }
    if (decode_like) decode_tokens += (uint64_t) n;
    if ((cfg.profile.size() || cfg.dump_file.size()) && decode_tokens - saved_at >= 4096) { saved_at = decode_tokens; save_dump(); save_profile(); }
    update_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// -------------------------------------------------------------------------------------------------- the public face

HybridExperts::HybridExperts() : p_(new Impl) {}
HybridExperts::~HybridExperts() { if (p_ && p_->ready) { p_->save_dump(); p_->save_profile(); } }

bool HybridExperts::init(llama_model* model, const std::string& gguf_path, const HybridConfig& cfg, std::string& err) {
    return p_->init(model, gguf_path, cfg, err);
}
void HybridExperts::attach(llama_model* model) {
    p_->hy.enabled = p_->cache && p_->enabled;
    llama_model_set_hybrid(model, &p_->hy);
}
void HybridExperts::detach(llama_model* model) { llama_model_set_hybrid(model, nullptr); }
bool HybridExperts::ready() const { return p_->ready; }
bool HybridExperts::caching() const { return p_->ready && p_->cache; }
void HybridExperts::set_enabled(bool on) {
    p_->enabled = on;
    p_->hy.enabled = p_->cache && on;
}
void HybridExperts::set_fill_swaps(int n) {
    p_->cfg.policy.fill_swaps = n;
    if (p_->hs) p_->hs->set_fill_swaps(n);
}
int HybridExperts::fill_swaps() const { return p_->cfg.policy.fill_swaps; }
void HybridExperts::after_decode(int n) { p_->after_decode(n); }
CacheStats HybridExperts::stats() const { return p_->hs ? p_->hs->stats() : CacheStats(); }
std::vector<HybridExperts::Sim> HybridExperts::sims() const {
    std::vector<Sim> out;
    for (const auto& s : p_->sims) out.push_back({s.pct, s.set->total_slots() / std::max(1, (int) p_->layers.size()), s.set->stats()});
    return out;
}
double HybridExperts::update_ms() const { return p_->update_ms; }
int HybridExperts::layers_taking_part() const { return (int) p_->layers.size(); }
int HybridExperts::slots_total() const { return p_->hs ? p_->hs->total_slots() : 0; }
int HybridExperts::slots_used() const { return p_->hs ? p_->hs->used_slots() : 0; }
size_t HybridExperts::cache_bytes() const {
    size_t b = 0;
    for (const auto& L : p_->layers) b += (size_t) L.slots * L.slot_bytes;
    return b;
}
int HybridExperts::experts_per_layer() const { return p_->n_expert; }
int HybridExperts::slot_of(int layer, int expert) const { return p_->hs ? p_->hs->slot_of(layer, expert) : -1; }
void HybridExperts::reset_stats() {
    if (p_->hs) p_->hs->reset_stats();
    for (auto& s : p_->sims) s.set->reset_stats();
    p_->update_ms = 0;
}
bool HybridExperts::save_profile() const {
    p_->save_dump();
    return p_->save_profile();
}

std::string HybridExperts::describe() const {
    char buf[512];
    if (!p_->ready) return "off";
    if (!p_->cache) {
        std::snprintf(buf, sizeof buf, "sim (routing recorded in %d layers; no cache)", (int) p_->layers.size());
        return buf;
    }
    const int slots = p_->hs->total_slots();
    const double pct = 100.0 * slots / std::max(1.0, (double) p_->layers.size() * p_->n_expert);
    int lo = 1 << 30, hi = 0;
    for (const auto& L : p_->layers) { lo = std::min(lo, L.slots); hi = std::max(hi, L.slots); }
    std::snprintf(buf, sizeof buf, "%d slots in %d of %d layers (%.0f%% of their experts%s), %.2f GiB %s",
                  slots, (int) p_->layers.size(), p_->moe_layers, pct,
                  p_->nonuniform ? (", shared out by the profile: " + std::to_string(lo) + " to " + std::to_string(hi) + " per layer").c_str() : "",
                  (double) cache_bytes() / (1024.0 * MiB), p_->cfg.in_ram ? "in RAM" : "on the card");
    return buf;
}

}  // namespace q35
