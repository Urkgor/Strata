// q35/strata_q35.cpp - Strata's engine for the Qwen3.5 / Qwen3.6 family (Qwen3.6-35B-A3B and its siblings).
//
// WHY A SECOND ENGINE.  `strata` (src/) is built for ONE architecture, qwen4exp (Qwen3.8-Flash-Next): 48 layers,
// hyper-connections, a sparse-attention indexer, a PLE table, 512 experts with top-10.  Qwen3.6-35B-A3B has none of
// that: 40 layers (30 Gated DeltaNet + 10 attention), a plain residual, dense attention, 256 experts with top-8.
// Its architecture is "qwen35moe" in llama.cpp, whose graph and CUDA/CPU kernels this repository already builds
// from (third_party/ggml/VERSION.txt pins the commit).  This engine runs that graph through libllama and gives it
// Strata's side of the contract:
//
//   * the same `--serve` protocol as `strata --serve` (serve/server.py talks to it unchanged): READY, INFO, GEN,
//     T, PP, RESUME, DONE, STOP, QUIT, SAVE/RESTORE;
//   * the same idea for the hardware: the dense part of the model on the graphics card, the routed experts in RAM
//     computed by the CPU (--cpu-moe, --n-cpu-moe, or fitted automatically to the free VRAM);
//   * a conversation cache that works for a recurrent model: the Gated DeltaNet state cannot be rolled back one
//     token at a time, so the engine keeps the live sequence and a few state checkpoints, and a request that
//     shares a prefix with them reads only what is new.
//
// Without --serve it generates from a prompt and prints the text (`strata-q35 --native model.gguf -p "..."`).
//
// A debugging command for validation against a reference implementation: `LOGITS <id,id,...>` (serve mode) clears
// the context, reads the ids and prints `LOGITS <n> <id>:<logit> ...` for the 8 best tokens of the last position.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef STRATA_Q35_FIT
#include "fit.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#ifndef STRATA_Q35_VERSION
#define STRATA_Q35_VERSION "0.1.0"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ----------------------------------------------------------------------------------------------------- options

struct Options {
    std::string model;
    bool serve = false;
    bool info = false;
    bool verbose = false;
    bool any_arch = false;
    int64_t max_context = 4096;
    int threads = 0;             // 0: the physical cores
    int threads_batch = 0;       // 0: same as threads
    int gpu_layers = -1;         // -1: all (libllama's default); the fit may lower it
    bool gpu_layers_set = false;
    bool cpu_moe = false;
    int n_cpu_moe = 0;
    int fit = 1;                 // 1: fit to the free VRAM when nothing was placed by hand
    int fit_margin_mib = 1024;
    std::string kv = "f16";
    int flash = -1;              // -1 auto, 0 off, 1 on
    int n_batch = 2048;
    int n_ubatch = 512;
    std::string load = "auto";   // auto | mmap | mlock | none | direct
    std::string numa;            // distribute | isolate | numactl
    std::string split_mode;      // none | layer | row
    int main_gpu = 0;
    std::vector<float> tensor_split;
    bool no_op_offload = false;
    bool no_kv_offload = false;
    bool eos_set = false;
    std::vector<int64_t> eos_ids;
    int ckpt_slots = 4;          // state checkpoints kept for the conversation cache
    int ckpt_every = 8192;       // tokens between checkpoints inside a long prompt (0: only at its end)
    int n_rs_seq = 0;            // recurrent snapshots for rollback (llama.cpp experimental)
    // one-shot generation (no --serve)
    std::string prompt;
    std::string prompt_file;
    std::string system;
    bool chat = false;
    int think = 1;               // --chat: 1 = the model thinks first (its default), 0 = answer at once
    int n_predict = 256;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 20;
    float min_p = 0.0f;
    uint64_t seed = 0;
};

void usage() {
    std::fputs(
        "strata-q35 " STRATA_Q35_VERSION " - Strata's engine for Qwen3.6-35B-A3B (llama.cpp's qwen35moe graph)\n"
        "\n"
        "  strata-q35 --serve --native MODEL.gguf [options]     the engine the Strata server drives (stdin/stdout)\n"
        "  strata-q35 --native MODEL.gguf -p \"text\" [options]    generate and print\n"
        "  strata-q35 --native MODEL.gguf --info                 what the file is\n"
        "\n"
        "model and context\n"
        "  --native, -m FILE      the GGUF (the first file of a split model)\n"
        "  --max-context N        tokens of context (default 4096; the model was trained for 262144)\n"
        "  --kv TYPE              K/V cache type: f16 (default), q8_0, q4_0, bf16 ...\n"
        "  --flash on|off|auto    flash attention (default auto)\n"
        "  --batch N, --ubatch N  prompt batch sizes (default 2048 / 512)\n"
        "\n"
        "where the model runs\n"
        "  --gpu-layers N|all     layers on the graphics card (default: all, then fitted to the free VRAM)\n"
        "  --cpu-moe              all routed experts in RAM (computed by the CPU); the rest on the card\n"
        "  --n-cpu-moe N          the routed experts of the first N layers in RAM\n"
        "  --fit on|off           fit the placement to the free VRAM (default on when nothing is placed by hand)\n"
        "  --fit-margin-mib N     VRAM to leave free per card when fitting (default 1024)\n"
        "  --threads N            CPU threads for generation (default: the physical cores)\n"
        "  --threads-batch N      CPU threads for the prompt (default: same)\n"
        "  --numa MODE            distribute | isolate | numactl (multi-socket PCs)\n"
        "  --split-mode MODE      none | layer | row (several cards); --main-gpu N; --tensor-split a,b,..\n"
        "  --load MODE            auto | mmap | mlock | none | direct (how the file is read)\n"
        "  --no-op-offload        do not let the card compute the CPU's weights during long prompts\n"
        "  --no-kv-offload        keep the K/V cache in RAM\n"
        "\n"
        "serving\n"
        "  --eos-ids a,b          token ids that end an answer (default: the model's end-of-turn tokens)\n"
        "  --ckpt-slots N         state checkpoints kept for the conversation cache (default 4; 0 = none)\n"
        "  --ckpt-every N         tokens between checkpoints inside a long prompt (default 8192)\n"
        "\n"
        "generating without --serve\n"
        "  -p, --prompt TEXT      the prompt;  -f, --file FILE  the prompt from a file\n"
        "  --chat                 wrap the prompt as one user message (ChatML);  --system TEXT\n"
        "  --think on|off         with --chat: think before answering (default on)\n"
        "  -n, --n-predict N      tokens to generate (default 256)\n"
        "  --temp F, --top-k N, --top-p F, --min-p F, --seed N     sampling (default: greedy)\n"
        "\n"
        "  --any-arch             run an architecture other than qwen35moe / qwen35 / qwen3next\n"
        "  --verbose              llama.cpp's own log\n"
        "  --version, --help\n",
        stderr);
}

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    while (*s == ' ') ++s;
    if (*s == '\0') { err = "no token ids"; return false; }
    while (*s) {
        char* e = nullptr;
        errno = 0;
        const long long v = std::strtoll(s, &e, 10);
        if (e == s || errno == ERANGE) { err = "bad token id list"; return false; }
        out.push_back(v);
        s = e;
        if (*s == ',') { ++s; if (*s == '\0') { err = "bad token id list"; return false; } }
        else if (*s != '\0' && *s != ' ' && *s != '\r' && *s != '\n') { err = "bad token id list"; return false; }
        else break;
    }
    return !out.empty();
}

int physical_cores() {
#if defined(__linux__)
    // one entry per core: the hyper-threads of a core share a thread_siblings_list
    std::set<std::string> cores;
    for (int i = 0; i < 4096; ++i) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
        if (!f) {
            if (i >= (int) std::thread::hardware_concurrency()) break;
            continue;
        }
        std::string s;
        std::getline(f, s);
        if (!s.empty()) cores.insert(s);
    }
    if (!cores.empty()) return (int) cores.size();
#endif
    const int hw = (int) std::thread::hardware_concurrency();
    return std::max(1, hw > 8 ? hw / 2 : hw);
}

bool parse_args(int argc, char** argv, Options& o, std::string& err) {
    auto need = [&](int& i, const char* flag) -> const char* {
        if (i + 1 >= argc) { err = std::string(flag) + " needs a value"; return nullptr; }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = nullptr;
        auto num = [&](const char* flag, long long lo, long long hi, long long& out) -> bool {
            v = need(i, flag);
            if (!v) return false;
            char* e = nullptr;
            const long long x = std::strtoll(v, &e, 10);
            if (e == v || *e != '\0' || x < lo || x > hi) { err = std::string("bad value for ") + flag + ": " + v; return false; }
            out = x;
            return true;
        };
        auto fnum = [&](const char* flag, float& out) -> bool {
            v = need(i, flag);
            if (!v) return false;
            char* e = nullptr;
            const float x = std::strtof(v, &e);
            if (e == v || *e != '\0' || !std::isfinite(x)) { err = std::string("bad value for ") + flag + ": " + v; return false; }
            out = x;
            return true;
        };
        long long n = 0;
        if (a == "--serve") o.serve = true;
        else if (a == "--info") o.info = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--any-arch") o.any_arch = true;
        else if (a == "--native" || a == "-m" || a == "--model") { if (!(v = need(i, a.c_str()))) return false; o.model = v; }
        else if (a == "--max-context" || a == "-c") { if (!num(a.c_str(), 64, 16LL << 20, n)) return false; o.max_context = n; }
        else if (a == "--threads" || a == "-t") { if (!num(a.c_str(), 1, 4096, n)) return false; o.threads = (int) n; }
        else if (a == "--threads-batch" || a == "-tb") { if (!num(a.c_str(), 1, 4096, n)) return false; o.threads_batch = (int) n; }
        else if (a == "--gpu-layers" || a == "-ngl" || a == "--n-gpu-layers") {
            if (!(v = need(i, a.c_str()))) return false;
            if (std::string(v) == "all" || std::string(v) == "auto") { o.gpu_layers = -1; }
            else {
                char* e = nullptr;
                const long x = std::strtol(v, &e, 10);
                if (e == v || *e != '\0' || x < 0) { err = std::string("bad value for ") + a + ": " + v; return false; }
                o.gpu_layers = (int) x;
            }
            o.gpu_layers_set = true;
        }
        else if (a == "--cpu-moe" || a == "-cmoe") o.cpu_moe = true;
        else if (a == "--n-cpu-moe" || a == "-ncmoe") { if (!num(a.c_str(), 0, 4096, n)) return false; o.n_cpu_moe = (int) n; }
        else if (a == "--fit") {
            if (!(v = need(i, a.c_str()))) return false;
            const std::string s = v;
            if (s == "on") o.fit = 1; else if (s == "off") o.fit = 0;
            else { err = "--fit takes on or off"; return false; }
        }
        else if (a == "--fit-margin-mib") { if (!num(a.c_str(), 0, 1 << 20, n)) return false; o.fit_margin_mib = (int) n; }
        else if (a == "--kv" || a == "--kv-type") { if (!(v = need(i, a.c_str()))) return false; o.kv = v; }
        else if (a == "--flash" || a == "--flash-attn") {
            if (!(v = need(i, a.c_str()))) return false;
            const std::string s = v;
            if (s == "on") o.flash = 1; else if (s == "off") o.flash = 0; else if (s == "auto") o.flash = -1;
            else { err = "--flash takes on, off or auto"; return false; }
        }
        else if (a == "--batch" || a == "-b") { if (!num(a.c_str(), 1, 1 << 20, n)) return false; o.n_batch = (int) n; }
        else if (a == "--ubatch" || a == "-ub") { if (!num(a.c_str(), 1, 1 << 20, n)) return false; o.n_ubatch = (int) n; }
        else if (a == "--load") { if (!(v = need(i, a.c_str()))) return false; o.load = v; }
        else if (a == "--no-mmap") o.load = "none";
        else if (a == "--mlock") o.load = "mlock";
        else if (a == "--numa") { if (!(v = need(i, a.c_str()))) return false; o.numa = v; }
        else if (a == "--split-mode" || a == "-sm") { if (!(v = need(i, a.c_str()))) return false; o.split_mode = v; }
        else if (a == "--main-gpu" || a == "-mg") { if (!num(a.c_str(), 0, 64, n)) return false; o.main_gpu = (int) n; }
        else if (a == "--tensor-split" || a == "-ts") {
            if (!(v = need(i, a.c_str()))) return false;
            std::stringstream ss(v);
            std::string tok;
            o.tensor_split.clear();
            while (std::getline(ss, tok, ',')) o.tensor_split.push_back(std::strtof(tok.c_str(), nullptr));
        }
        else if (a == "--no-op-offload") o.no_op_offload = true;
        else if (a == "--no-kv-offload") o.no_kv_offload = true;
        else if (a == "--eos-ids") {
            if (!(v = need(i, a.c_str()))) return false;
            std::string e;
            if (!parse_i64_list(v, o.eos_ids, e)) { err = "--eos-ids: " + e; return false; }
            o.eos_set = true;
        }
        else if (a == "--ckpt-slots") { if (!num(a.c_str(), 0, 64, n)) return false; o.ckpt_slots = (int) n; }
        else if (a == "--ckpt-every") { if (!num(a.c_str(), 0, 1LL << 30, n)) return false; o.ckpt_every = (int) n; }
        else if (a == "--rs-snapshots") { if (!num(a.c_str(), 0, 64, n)) return false; o.n_rs_seq = (int) n; }
        else if (a == "--prompt" || a == "-p") { if (!(v = need(i, a.c_str()))) return false; o.prompt = v; }
        else if (a == "--file" || a == "-f") { if (!(v = need(i, a.c_str()))) return false; o.prompt_file = v; }
        else if (a == "--system") { if (!(v = need(i, a.c_str()))) return false; o.system = v; }
        else if (a == "--chat") o.chat = true;
        else if (a == "--think") {
            if (!(v = need(i, a.c_str()))) return false;
            const std::string s = v;
            if (s == "on") o.think = 1; else if (s == "off") o.think = 0; else { err = "--think takes on or off"; return false; }
        }
        else if (a == "--n-predict" || a == "-n") { if (!num(a.c_str(), 1, 1LL << 30, n)) return false; o.n_predict = (int) n; }
        else if (a == "--temp" || a == "--temperature") { if (!fnum(a.c_str(), o.temperature)) return false; }
        else if (a == "--top-p") { if (!fnum(a.c_str(), o.top_p)) return false; }
        else if (a == "--min-p") { if (!fnum(a.c_str(), o.min_p)) return false; }
        else if (a == "--top-k") { if (!num(a.c_str(), 0, 1 << 20, n)) return false; o.top_k = (int) n; }
        else if (a == "--seed") { if (!num(a.c_str(), 0, LLONG_MAX, n)) return false; o.seed = (uint64_t) n; }
        else if (a == "--version") { std::printf("strata-q35 %s (llama.cpp %s)\n", STRATA_Q35_VERSION, llama_version()); std::exit(0); }
        else if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else { err = "unknown option " + a; return false; }
    }
    if (o.model.empty()) { err = "--native MODEL.gguf is required"; return false; }
    return true;
}

// ----------------------------------------------------------------------------------------------------- llama glue

bool g_verbose = false;
std::atomic<bool> g_stop{false};

void log_cb(ggml_log_level level, const char* text, void*) {
    if (!g_verbose && level < GGML_LOG_LEVEL_WARN) return;   // ERROR=4 WARN=3 INFO=2 DEBUG=1 (CONT=5 continues the last)
    std::fputs(text, stderr);
}

bool abort_cb(void*) { return g_stop.load(); }

bool parse_kv_type(const std::string& s, ggml_type& t) {
    static const struct { const char* n; ggml_type t; } k[] = {
        {"f32", GGML_TYPE_F32}, {"f16", GGML_TYPE_F16}, {"bf16", GGML_TYPE_BF16}, {"q8_0", GGML_TYPE_Q8_0},
        {"q4_0", GGML_TYPE_Q4_0}, {"q4_1", GGML_TYPE_Q4_1}, {"iq4_nl", GGML_TYPE_IQ4_NL},
        {"q5_0", GGML_TYPE_Q5_0}, {"q5_1", GGML_TYPE_Q5_1}};
    for (const auto& e : k)
        if (s == e.n) { t = e.t; return true; }
    return false;
}

std::string meta_str(const llama_model* m, const char* key) {
    char buf[512];
    const int n = llama_model_meta_val_str(m, key, buf, sizeof buf);
    return n < 0 ? std::string() : std::string(buf);
}

struct Ckpt {
    int32_t n = 0;                 // tokens in memory when it was taken
    std::vector<uint8_t> data;     // the recurrent state (llama_state_seq_*_ext, partial only)
};

struct Request {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 20;
    unsigned long long seed = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    int penalty_last_n = 0;
    bool ckpt = true;
};

class Engine {
public:
    Options o;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_memory_t mem = nullptr;
    int32_t n_vocab = 0;
    std::vector<llama_token> live;     // the tokens whose state is in memory: positions 0 .. live.size()-1
    std::vector<Ckpt> ckpts;           // each is the state after a prefix of `live`
    std::set<llama_token> eos;
    llama_batch batch{};
    std::list<std::string> override_strings;                           // keep the regexes alive
    std::vector<llama_model_tensor_buft_override> overrides;
    std::vector<float> tensor_split;
    std::vector<size_t> margins;
    std::string placement = "all on the card";
    std::string offload_tag = "gpu";   // one word for the INFO line

    ~Engine() {
        if (batch.token) llama_batch_free(batch);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    bool load(std::string& err) {
        if (!llama_supports_gpu_offload()) { placement = "CPU only (this build has no GPU backend)"; offload_tag = "cpu"; }
        const size_t max_ovr = llama_max_tensor_buft_overrides();
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        auto add_override = [&](const std::string& re) {
            override_strings.push_back(re);
            overrides.push_back({override_strings.back().c_str(), cpu_buft});
        };
        if (o.cpu_moe) {
            add_override("\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
            placement = "routed experts in RAM, the rest on the card";
            offload_tag = "cpu-moe";
        } else if (o.n_cpu_moe > 0) {
            for (int i = 0; i < o.n_cpu_moe; ++i)
                add_override("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
            placement = "routed experts of the first " + std::to_string(o.n_cpu_moe) + " layers in RAM";
            offload_tag = "n-cpu-moe=" + std::to_string(o.n_cpu_moe);
        }
        if (overrides.size() + 1 > max_ovr) { err = "too many tensor overrides (--n-cpu-moe)"; return false; }
        while (overrides.size() < max_ovr) overrides.push_back({nullptr, nullptr});

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = o.gpu_layers_set ? o.gpu_layers : -1;
        mp.main_gpu = o.main_gpu;
        if (o.split_mode == "none") mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        else if (o.split_mode == "layer") mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        else if (o.split_mode == "row") mp.split_mode = LLAMA_SPLIT_MODE_ROW;
        else if (!o.split_mode.empty()) { err = "--split-mode takes none, layer or row"; return false; }
        if (o.load == "auto") mp.load_mode = LLAMA_LOAD_MODE_AUTO;
        else if (o.load == "mmap") mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        else if (o.load == "mlock") mp.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
        else if (o.load == "none") mp.load_mode = LLAMA_LOAD_MODE_NONE;
        else if (o.load == "direct") mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
        else { err = "--load takes auto, mmap, mlock, none or direct"; return false; }
        tensor_split.assign(llama_max_devices(), 0.0f);
        for (size_t i = 0; i < o.tensor_split.size() && i < tensor_split.size(); ++i) tensor_split[i] = o.tensor_split[i];
        mp.tensor_split = tensor_split.data();
        mp.tensor_buft_overrides = overrides.data();   // an all-null list is the same as none
        if (o.info) mp.n_gpu_layers = 0;                // --info: the file's facts; nothing goes to the card

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) o.max_context;
        cp.n_batch = (uint32_t) o.n_batch;
        cp.n_ubatch = (uint32_t) std::min(o.n_ubatch, o.n_batch);
        cp.n_seq_max = 1;
        cp.n_rs_seq = (uint32_t) o.n_rs_seq;
        cp.n_threads = o.threads > 0 ? o.threads : physical_cores();
        cp.n_threads_batch = o.threads_batch > 0 ? o.threads_batch : cp.n_threads;
        cp.flash_attn_type = o.flash < 0 ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                         : (o.flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED);
        cp.offload_kqv = !o.no_kv_offload;
        cp.op_offload = !o.no_op_offload;
        cp.no_perf = true;
        ggml_type kt = GGML_TYPE_F16;
        if (!parse_kv_type(o.kv, kt)) { err = "--kv: unknown cache type " + o.kv; return false; }
        cp.type_k = kt;
        cp.type_v = kt;

#ifdef STRATA_Q35_FIT
        const bool placed_by_hand = o.cpu_moe || o.n_cpu_moe > 0 || o.gpu_layers_set;
        if (o.fit && !placed_by_hand && llama_supports_gpu_offload()) {
            margins.assign(llama_max_devices(), (size_t) o.fit_margin_mib << 20);
            std::vector<llama_model_tensor_buft_override> fit_ovr(max_ovr, llama_model_tensor_buft_override{nullptr, nullptr});
            const common_params_fit_status st =
                common_fit_params(o.model.c_str(), &mp, &cp, tensor_split.data(), fit_ovr.data(), margins.data(), 4096, nullptr,
                                  o.verbose ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_ERROR);
            if (st == COMMON_PARAMS_FIT_STATUS_SUCCESS) {
                // the fit may add overrides (experts of some layers to the CPU): they must outlive the model load
                overrides = fit_ovr;
                mp.tensor_buft_overrides = overrides.data();
                size_t n = 0;
                while (n < overrides.size() && overrides[n].pattern) ++n;
                placement = "fitted to the free VRAM: " + std::to_string(mp.n_gpu_layers) + " layers on the card, " +
                            std::to_string(n) + " expert overrides to RAM";
                offload_tag = "fit";
                if (cp.n_ctx != (uint32_t) o.max_context) {
                    // the fit may only lower a context size of 0; ours was set, so it must be unchanged
                    cp.n_ctx = (uint32_t) o.max_context;
                }
            } else {
                std::fprintf(stderr, "strata-q35: the VRAM fit found no placement (%s): loading as asked\n",
                             st == COMMON_PARAMS_FIT_STATUS_FAILURE ? "it does not fit" : "error");
            }
        }
#endif
        model = llama_model_load_from_file(o.model.c_str(), mp);
        if (!model) { err = "could not load " + o.model; return false; }
        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);

        const std::string arch = meta_str(model, "general.architecture");
        if (!o.any_arch && arch != "qwen35moe" && arch != "qwen35" && arch != "qwen3next") {
            err = "architecture is '" + arch + "', this engine runs qwen35moe (Qwen3.6-35B-A3B), qwen35 and qwen3next" +
                  (arch == "qwen4exp" ? "; Qwen3.8-Flash-Next runs on the `strata` engine" : "") +
                  " (--any-arch to try another)";
            return false;
        }
        if (o.info) return true;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) { err = "could not create the context (not enough memory for --max-context " + std::to_string(o.max_context) + "?)"; return false; }
        mem = llama_get_memory(ctx);
        llama_set_abort_callback(ctx, abort_cb, nullptr);
        batch = llama_batch_init((int32_t) cp.n_batch, 0, 1);

        if (o.eos_set) for (int64_t t : o.eos_ids) eos.insert((llama_token) t);
        else {
            for (llama_token t = 0; t < n_vocab; ++t)
                if (llama_vocab_is_eog(vocab, t)) eos.insert(t);
            // the Qwen3.5 / 3.6 vocabulary: <|endoftext|> and <|im_end|> end an answer, and serve/server.py stops on both
            if (arch != "qwen4exp" && n_vocab > 248046 && (arch == "qwen35moe" || arch == "qwen35" || arch == "qwen3next")) {
                eos.insert(248044);
                eos.insert(248046);
            }
        }
        return true;
    }

    // ------------------------------------------------------------------------------ the conversation cache

    void clear_all() {
        llama_memory_clear(mem, true);
        live.clear();
        ckpts.clear();
    }

    void drop_ckpts_above(int32_t n) {
        ckpts.erase(std::remove_if(ckpts.begin(), ckpts.end(), [&](const Ckpt& c) { return c.n > n; }), ckpts.end());
    }

    void take_ckpt() {
        if (o.ckpt_slots <= 0 || live.empty()) return;
        const int32_t n = (int32_t) live.size();
        for (const Ckpt& c : ckpts) if (c.n == n) return;
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (sz == 0) return;
        Ckpt c;
        c.n = n;
        c.data.resize(sz);
        const size_t got = llama_state_seq_get_data_ext(ctx, c.data.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (got == 0) return;
        c.data.resize(got);
        while ((int) ckpts.size() >= o.ckpt_slots) ckpts.erase(ckpts.begin());
        ckpts.push_back(std::move(c));
    }

    // Makes memory hold exactly live[0, keep') for the largest keep' <= keep it can reach, and returns keep'.
    // An attention-only cache is cut at any position; the Gated DeltaNet state is not, so a recurrent model goes back
    // to the newest checkpoint at or before `keep`, or to nothing.
    int32_t rewind_to(int32_t keep) {
        if (keep >= (int32_t) live.size()) return (int32_t) live.size();
        if (keep <= 0) { clear_all(); return 0; }
        if (llama_memory_seq_rm(mem, 0, keep, -1)) {   // a hybrid cache that refuses has not changed
            live.resize((size_t) keep);
            drop_ckpts_above(keep);
            return keep;
        }
        for (auto it = ckpts.rbegin(); it != ckpts.rend(); ++it) {
            if (it->n > keep) continue;
            const int32_t n = it->n;
            if (llama_state_seq_set_data_ext(ctx, it->data.data(), it->data.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0)
                continue;
            if (!llama_memory_seq_rm(mem, 0, n, -1)) break;   // cannot cut the attention part to match: start over
            live.resize((size_t) n);
            drop_ckpts_above(n);
            return n;
        }
        clear_all();
        return 0;
    }

    // ------------------------------------------------------------------------------ decoding

    // Feeds toks[0, n) at positions live.size() ...; `want_logits` asks for the last one's logits.
    // Returns 0, or the llama_decode code; on an abort (2) `live` has grown by what the memory really holds.
    int decode(const llama_token* toks, int n, bool want_logits) {
        const int32_t p0 = (int32_t) live.size();
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = toks[i];
            batch.pos[i] = p0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (want_logits && i == n - 1) ? 1 : 0;
        }
        const int rc = llama_decode(ctx, batch);
        if (rc == 0) {
            live.insert(live.end(), toks, toks + n);
        } else if (rc == 2) {   // aborted: the ubatches that finished stay in memory
            const int32_t pmax = llama_memory_seq_pos_max(mem, 0);
            const int32_t done = std::max(0, std::min(n, pmax + 1 - p0));
            live.insert(live.end(), toks, toks + done);
        } else {
            // the memory may hold part of this batch: forget everything rather than trust it
            clear_all();
        }
        return rc;
    }

    llama_sampler* make_sampler(const Request& r, const std::vector<llama_token>& prompt) {
        llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
        const bool pen = r.penalty_repeat != 1.0f || r.penalty_freq != 0.0f || r.penalty_present != 0.0f;
        int last_n = r.penalty_last_n;
        if (pen && last_n <= 0) last_n = 64;   // a penalty without a window counts over nothing
        if (pen) llama_sampler_chain_add(s, llama_sampler_init_penalties(n_vocab, last_n, r.penalty_repeat, r.penalty_freq, r.penalty_present));
        if (r.temperature <= 0.0f) {
            llama_sampler_chain_add(s, llama_sampler_init_greedy());
        } else {
            if (r.top_k > 0) llama_sampler_chain_add(s, llama_sampler_init_top_k(r.top_k));
            if (r.top_p < 1.0f) llama_sampler_chain_add(s, llama_sampler_init_top_p(r.top_p, 1));
            if (r.min_p > 0.0f) llama_sampler_chain_add(s, llama_sampler_init_min_p(r.min_p, 1));
            llama_sampler_chain_add(s, llama_sampler_init_temp(r.temperature));
            uint32_t seed = (uint32_t) r.seed;
            if (r.seed == 0) seed = std::random_device{}();
            llama_sampler_chain_add(s, llama_sampler_init_dist(seed));
        }
        if (pen) {   // the window starts inside the prompt
            const size_t from = prompt.size() > (size_t) last_n ? prompt.size() - (size_t) last_n : 0;
            for (size_t i = from; i < prompt.size(); ++i) llama_sampler_accept(s, prompt[i]);
        }
        return s;
    }

    // One request: reads the prompt (reusing what memory holds), samples, and reports through `out`.
    // `emit_token` is called with each sampled id.  Returns false after printing an ERR.
    struct Result {
        long long produced = 0;
        double prompt_ms = 0, decode_ms = 0;
        const char* finish = "length";
        long long resume = 0, read_n = 0;
    };

    struct Hooks {
        std::function<void(llama_token)> token;                       // each sampled id
        std::function<void(int32_t resume)> resume;                   // before the prompt is read
        std::function<void(int32_t done, int32_t total, double ms)> progress;   // after each prompt chunk
    };

    bool run(const std::vector<llama_token>& ids, const Request& r, Result& res, const Hooks& hk, std::string& err) {
        const int32_t n = (int32_t) ids.size();
        const auto t0 = Clock::now();
        g_stop.store(false);

        int32_t common = 0;
        const int32_t lim = (int32_t) std::min<size_t>(live.size(), ids.size());
        while (common < lim && live[(size_t) common] == ids[(size_t) common]) ++common;
        int32_t keep = std::min(common, n - 1);                 // the last prompt token is always read: it gives the logits
        keep = rewind_to(keep);
        // memory may hold fewer than `keep` now (a recurrent model without a checkpoint): read the rest
        res.resume = keep;
        if (hk.resume) hk.resume(keep);

        // ---- the prompt
        int32_t pos = keep;
        int32_t since_ck = 0;
        while (pos < n - 1) {
            const int32_t chunk = std::min<int32_t>(o.n_batch, n - 1 - pos);
            const int rc = decode(ids.data() + pos, chunk, false);
            if (rc != 0) {
                if (rc == 2) {
                    pos = (int32_t) live.size();
                    res.finish = "cancel";
                    res.read_n = std::max<long long>(0, pos - keep);
                    res.prompt_ms = ms_since(t0);
                    return true;
                }
                err = "decode failed while reading the prompt (code " + std::to_string(rc) + ")";
                return false;
            }
            pos += chunk;
            since_ck += chunk;
            if (hk.progress) hk.progress(pos, n, ms_since(t0));
            if (g_stop.load()) {
                res.finish = "cancel";
                res.read_n = pos - keep;
                res.prompt_ms = ms_since(t0);
                return true;
            }
            if (r.ckpt && o.ckpt_every > 0 && since_ck >= o.ckpt_every && pos < n - 1) { take_ckpt(); since_ck = 0; }
        }
        // the state just before the last prompt token: what a repeat of this prompt, or an edit of its end, resumes from
        if (r.ckpt && pos == n - 1 && pos > 0) take_ckpt();
        {
            const int rc = decode(ids.data() + pos, 1, true);
            if (rc != 0) {
                if (rc == 2) {
                    res.finish = "cancel";
                    res.read_n = std::max<long long>(0, (long long) live.size() - keep);
                    res.prompt_ms = ms_since(t0);
                    return true;
                }
                err = "decode failed while reading the prompt (code " + std::to_string(rc) + ")";
                return false;
            }
        }
        res.read_n = n - keep;
        if (hk.progress) hk.progress(n, n, ms_since(t0));
        res.prompt_ms = ms_since(t0);

        // ---- the answer
        llama_sampler* smpl = make_sampler(r, ids);
        const auto t1 = Clock::now();
        bool ok = true;
        for (;;) {
            const llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (hk.token) hk.token(tok);
            ++res.produced;
            if (eos.count(tok)) { res.finish = "stop"; break; }
            if (g_stop.load()) { res.finish = "cancel"; break; }
            if (res.produced >= r.max_new) { res.finish = "length"; break; }
            const int rc = decode(&tok, 1, true);
            if (rc == 2) { res.finish = "cancel"; break; }
            if (rc != 0) { err = "decode failed (code " + std::to_string(rc) + ")"; ok = false; break; }
        }
        res.decode_ms = ms_since(t1);
        llama_sampler_free(smpl);
        return ok;
    }

    // ------------------------------------------------------------------------------ serve

    std::vector<llama_token> to_tokens(const std::vector<int64_t>& v) {
        std::vector<llama_token> t(v.size());
        for (size_t i = 0; i < v.size(); ++i) t[i] = (llama_token) v[i];
        return t;
    }

    static std::string fail_kind_line(const char* kind, bool published, const std::string& why) {
        std::string w = why;
        for (char& c : w) if (c == '\n' || c == '\r') c = ' ';
        return std::string("SERR ") + kind + " " + (published ? "1" : "0") + " " + w;
    }

    void serve() {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> lines;
        bool eof = false;
        std::thread([&] {
            // read(2) directly: a thread blocked in std::getline(std::cin) holds stdin's lock, and exit() then
            // waits for that lock for ever
            std::string buf;
            char chunk[65536];
            bool running = true;
            while (running) {
                const ssize_t got = ::read(0, chunk, sizeof chunk);
                if (got < 0 && errno == EINTR) continue;
                if (got <= 0) running = false;
                else buf.append(chunk, (size_t) got);
                size_t nl;
                while ((nl = buf.find('\n')) != std::string::npos || (!running && !buf.empty())) {
                    std::string l = nl == std::string::npos ? buf : buf.substr(0, nl);
                    buf.erase(0, nl == std::string::npos ? buf.size() : nl + 1);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP") { g_stop.store(true); continue; }
                    std::lock_guard<std::mutex> lk(mu);
                    lines.push_back(l);
                    cv.notify_one();
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        }).detach();
        auto next = [&](std::string& out) -> bool {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return !lines.empty() || eof; });
            if (lines.empty()) return false;
            out = std::move(lines.front());
            lines.pop_front();
            return true;
        };

        size_t vfree = 0, vtotal = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                size_t f = 0, t = 0;
                ggml_backend_dev_memory(d, &f, &t);
                vfree += f;
                vtotal += t;
            }
        }
        const std::string arch = meta_str(model, "general.architecture");
        std::printf("INFO context=%lld kv=%s kv_resident=0 expert_slots=0 expert_cache_mib=0 spec=0 vram_free_mib=%lld "
                    "arch=%s n_expert=%s n_expert_used=%s offload=%s engine=q35-" STRATA_Q35_VERSION "\n",
                    (long long) o.max_context, o.kv.c_str(), (long long) (vfree >> 20), arch.c_str(),
                    meta_str(model, (arch + ".expert_count").c_str()).c_str(),
                    meta_str(model, (arch + ".expert_used_count").c_str()).c_str(), offload_tag.c_str());
        (void) vtotal;
        std::printf("READY %lld stop\n", (long long) o.max_context);
        std::fflush(stdout);

        std::string line;
        while (next(line)) {
            if (line == "QUIT") break;
            if (line.empty()) continue;
            if (line.rfind("VRAM", 0) == 0) {
                std::printf("ERR VRAM needs the `strata` engine started with --vram-elastic (not available here)\n");
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
                session_file(line);
                continue;
            }
            if (line.rfind("LOGITS ", 0) == 0) {
                debug_logits(line.c_str() + 7);
                continue;
            }
            if (line.rfind("GENI ", 0) == 0) {
                std::printf("ERR this engine was started without --vision\n");
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("GEN ", 0) != 0) {
                std::printf("ERR expected: GEN <max_new> <id,id,...>\n");
                std::fflush(stdout);
                continue;
            }
            handle_gen(line);
        }
    }

    void handle_gen(const std::string& line) {
        const char* p = line.c_str() + 4;
        char* endp = nullptr;
        Request r;
        r.max_new = std::strtoll(p, &endp, 10);
        if (endp == p || r.max_new < 1) {
            std::printf("ERR bad request: max_new\n");
            std::fflush(stdout);
            return;
        }
        // optional keys between max_new and the ids; the ids start at the first token without '='
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq);
            const char* val = tok.c_str() + eq + 1;
            const float fv = std::strtof(val, nullptr);
            if (key == "temperature") r.temperature = fv;
            else if (key == "top_p") r.top_p = fv;
            else if (key == "top_k") r.top_k = std::atoi(val);
            else if (key == "min_p") r.min_p = fv;
            else if (key == "penalty_last_n") r.penalty_last_n = std::atoi(val);
            else if (key == "penalty_repeat") r.penalty_repeat = fv;
            else if (key == "penalty_freq") r.penalty_freq = fv;
            else if (key == "penalty_present") r.penalty_present = fv;
            else if (key == "seed") r.seed = std::strtoull(val, nullptr, 10);
            else if (key == "ckpt") r.ckpt = std::atoi(val) != 0;
            // other keys (cvec, pcie_frac, spec_min_p, ...) belong to the `strata` engine: ignored
        }
        std::vector<int64_t> raw;
        std::string pe;
        if (!parse_i64_list(endp, raw, pe)) {
            std::printf("ERR bad request: %s\n", pe.c_str());
            std::fflush(stdout);
            return;
        }
        const int64_t n = (int64_t) raw.size();
        if (n + r.max_new + 8 > o.max_context) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                        (long long) r.max_new, (long long) o.max_context);
            std::fflush(stdout);
            return;
        }
        for (int64_t t : raw)
            if (t < 0 || t >= n_vocab) {
                std::printf("ERR a token id is outside the vocabulary\n");
                std::fflush(stdout);
                return;
            }
        const std::vector<llama_token> ids = to_tokens(raw);
        Result res;
        std::string err;
        Hooks hk;
        hk.token = [&](llama_token t) { std::printf("T %d\n", (int) t); std::fflush(stdout); };
        hk.resume = [&](int32_t resume) { std::printf("RESUME %d\n", (int) resume); std::fflush(stdout); };
        hk.progress = [&](int32_t done, int32_t total, double ms) {
            const double rate = ms > 0 ? (double) (done - res.resume) * 1000.0 / ms : 0.0;
            std::printf("PP %d %d %.0f %.1f\n", done, total, ms, rate);
            std::fflush(stdout);
        };
        const bool ok = run(ids, r, res, hk, err);
        if (!ok) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            return;
        }
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
        //      [hits] [lookups] [RAM blobs] [file blobs] [file MB] [prompt tokens read] [offloaded]
        std::printf("DONE %lld %lld %.1f %.1f %s 0 0 %lld 0 0 0 0 0.0 %lld 0\n", res.produced, (long long) n, res.prompt_ms,
                    res.decode_ms, res.finish, res.resume, res.read_n);
        std::fflush(stdout);
    }

    void debug_logits(const char* idlist) {
        std::vector<int64_t> raw;
        std::string pe;
        if (!parse_i64_list(idlist, raw, pe)) { std::printf("ERR %s\n", pe.c_str()); std::fflush(stdout); return; }
        for (int64_t t : raw)
            if (t < 0 || t >= n_vocab) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); return; }
        clear_all();
        g_stop.store(false);
        const std::vector<llama_token> ids = to_tokens(raw);
        size_t pos = 0;
        while (pos < ids.size()) {
            const int chunk = (int) std::min<size_t>((size_t) o.n_batch, ids.size() - pos);
            const bool last = pos + (size_t) chunk == ids.size();
            if (decode(ids.data() + pos, chunk, last) != 0) { std::printf("ERR decode failed\n"); std::fflush(stdout); return; }
            pos += (size_t) chunk;
        }
        const float* lg = llama_get_logits_ith(ctx, -1);
        std::vector<int> idx((size_t) n_vocab);
        for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
        const int k = std::min(8, (int) n_vocab);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        std::printf("LOGITS %d", k);
        for (int i = 0; i < k; ++i) std::printf(" %d:%.6f", idx[(size_t) i], lg[idx[(size_t) i]]);
        std::printf("\n");
        std::fflush(stdout);
    }

    void session_file(const std::string& line) {
        const bool save = line.rfind("SAVE ", 0) == 0;
        const std::string path = line.substr(save ? 5 : 8);
        const auto t0 = Clock::now();
        if (path.empty()) { std::printf("%s\n", fail_kind_line("invalid", false, "no file name").c_str()); std::fflush(stdout); return; }
        if (save) {
            if (live.empty()) { std::printf("%s\n", fail_kind_line("invalid", false, "there is no conversation to save").c_str()); std::fflush(stdout); return; }
            const size_t bytes = llama_state_seq_save_file(ctx, path.c_str(), 0, live.data(), live.size());
            if (bytes == 0) {
                std::printf("%s\n", fail_kind_line("io", false, "could not write " + path).c_str());
            } else {
                std::printf("SAVED %zu %zu %.1f\n", live.size(), bytes, ms_since(t0));
            }
            std::fflush(stdout);
            return;
        }
        struct stat st;
        if (stat(path.c_str(), &st) != 0) { std::printf("%s\n", fail_kind_line("invalid", false, "no such file: " + path).c_str()); std::fflush(stdout); return; }
        clear_all();
        std::vector<llama_token> toks((size_t) o.max_context);
        size_t ntok = 0;
        const size_t bytes = llama_state_seq_load_file(ctx, path.c_str(), 0, toks.data(), toks.size(), &ntok);
        if (bytes == 0 || ntok == 0 || ntok > toks.size()) {
            clear_all();
            std::printf("%s\n", fail_kind_line("invalid", false, "not a conversation this engine can load: " + path).c_str());
            std::fflush(stdout);
            return;
        }
        toks.resize(ntok);
        live = std::move(toks);
        std::printf("RESTORED %zu %zu %.1f\n", ntok, bytes, ms_since(t0));
        std::fflush(stdout);
    }

    // ------------------------------------------------------------------------------ one-shot generation

    std::string piece(llama_token t) {
        char buf[256];
        int n = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, false);
        if (n < 0) {
            std::string big((size_t) -n, '\0');
            n = llama_token_to_piece(vocab, t, &big[0], (int32_t) big.size(), 0, false);
            if (n < 0) return std::string();
            big.resize((size_t) n);
            return big;
        }
        return std::string(buf, (size_t) n);
    }

    int generate_text() {
        std::string text = o.prompt;
        if (!o.prompt_file.empty()) {
            std::ifstream f(o.prompt_file, std::ios::binary);
            if (!f) { std::fprintf(stderr, "strata-q35: cannot read %s\n", o.prompt_file.c_str()); return 1; }
            std::stringstream ss;
            ss << f.rdbuf();
            text = ss.str();
        }
        if (text.empty()) { std::fprintf(stderr, "strata-q35: give a prompt with -p or -f (or --serve)\n"); return 1; }
        if (o.chat) {
            std::string t;
            if (!o.system.empty()) t += "<|im_start|>system\n" + o.system + "<|im_end|>\n";
            t += "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
            t += o.think ? "<think>\n" : "<think>\n\n</think>\n\n";
            text = t;
        }
        std::vector<llama_token> ids((size_t) text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), true, true);
        if (n < 0) {
            ids.resize((size_t) -n);
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), true, true);
        }
        if (n <= 0) { std::fprintf(stderr, "strata-q35: the prompt has no tokens\n"); return 1; }
        ids.resize((size_t) n);
        if (n + o.n_predict + 8 > o.max_context) {
            std::fprintf(stderr, "strata-q35: prompt (%d tokens) + --n-predict (%d) exceeds --max-context (%lld)\n", n,
                         o.n_predict, (long long) o.max_context);
            return 1;
        }
        Request r;
        r.max_new = o.n_predict;
        r.temperature = o.temperature;
        r.top_p = o.top_p;
        r.top_k = o.top_k;
        r.min_p = o.min_p;
        r.seed = o.seed;
        r.ckpt = false;
        Result res;
        std::string err;
        Hooks hk;
        hk.token = [&](llama_token t) {
            if (eos.count(t)) return;
            const std::string s = piece(t);
            std::fwrite(s.data(), 1, s.size(), stdout);
            std::fflush(stdout);
        };
        const bool ok = run(ids, r, res, hk, err);
        std::printf("\n");
        if (!ok) { std::fprintf(stderr, "strata-q35: %s\n", err.c_str()); return 1; }
        const double ptok = res.prompt_ms > 0 ? (double) res.read_n * 1000.0 / res.prompt_ms : 0.0;
        const double dtok = res.decode_ms > 0 ? (double) res.produced * 1000.0 / res.decode_ms : 0.0;
        std::fprintf(stderr, "strata-q35: prompt %lld tokens at %.1f tokens/s, answer %lld tokens at %.1f tokens/s (%s)\n",
                     (long long) res.read_n, ptok, (long long) res.produced, dtok, res.finish);
        return 0;
    }

    void print_info() {
        char desc[256];
        llama_model_desc(model, desc, sizeof desc);
        const std::string arch = meta_str(model, "general.architecture");
        std::printf("file          %s\n", o.model.c_str());
        std::printf("architecture  %s\n", arch.c_str());
        std::printf("model         %s\n", desc);
        std::printf("size          %.2f GiB, %.2f B parameters\n", (double) llama_model_size(model) / (1024.0 * 1024 * 1024),
                    (double) llama_model_n_params(model) / 1e9);
        std::printf("layers        %d (hybrid: %s)\n", (int) llama_model_n_layer(model), llama_model_is_hybrid(model) ? "yes" : "no");
        std::printf("embedding     %d\n", (int) llama_model_n_embd(model));
        std::printf("experts       %s, %s per token\n", meta_str(model, (arch + ".expert_count").c_str()).c_str(),
                    meta_str(model, (arch + ".expert_used_count").c_str()).c_str());
        std::printf("context       %d trained\n", (int) llama_model_n_ctx_train(model));
        std::printf("vocabulary    %d tokens\n", (int) n_vocab);
        std::printf("placement     %s\n", placement.c_str());
    }
};

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::string err;
    if (!parse_args(argc, argv, o, err)) {
        std::fprintf(stderr, "strata-q35: %s\n\n", err.c_str());
        usage();
        return 2;
    }
    g_verbose = o.verbose;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    if (!o.numa.empty()) {
        ggml_numa_strategy s = GGML_NUMA_STRATEGY_DISABLED;
        if (o.numa == "distribute") s = GGML_NUMA_STRATEGY_DISTRIBUTE;
        else if (o.numa == "isolate") s = GGML_NUMA_STRATEGY_ISOLATE;
        else if (o.numa == "numactl") s = GGML_NUMA_STRATEGY_NUMACTL;
        else { std::fprintf(stderr, "strata-q35: --numa takes distribute, isolate or numactl\n"); return 2; }
        llama_numa_init(s);
    }
    auto engp = std::make_unique<Engine>();
    Engine& eng = *engp;
    eng.o = o;
    if (!eng.load(err)) {
        if (o.serve) std::printf("ERR %s\n", err.c_str());
        std::fprintf(stderr, "strata-q35: %s\n", err.c_str());
        std::fflush(stdout);
        return 1;
    }
    int rc = 0;
    if (o.info) eng.print_info();
    else if (o.serve) eng.serve();
    else rc = eng.generate_text();
    engp.reset();   // the model goes before the backends do
    llama_backend_free();
    return rc;
}
