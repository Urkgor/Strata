// tools/gemma4/llama_ref.cpp - the reference side of the Gemma 4 checks: llama.cpp's own tokenizer and logits.
//
//   llama_ref tok    MODEL.gguf [--no-special]   < lines of hex   (each line: a UTF-8 string, hex encoded)  -> a line of ids each
//   llama_ref logits MODEL.gguf OUT.bin [--all]  < lines of ids   (comma separated)                          -> float32 logits
//
// `tok` loads the vocabulary only (a GGUF whose tensors were never downloaded is enough).  `logits` runs the prompt of ids on the
// CPU in one batch with an f32 K / V cache and writes n_vocab floats for the last position (all positions with --all).
// Built against a llama.cpp checkout (see tools/gemma4/README.md); it is not part of Strata's own CMake.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "llama.h"

static std::string unhex(const std::string& h) {
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back((char) std::stoi(h.substr(i, 2), nullptr, 16));
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: llama_ref tok|logits MODEL [OUT.bin] [--all|--no-special]\n"); return 2; }
    const std::string mode = argv[1];
    bool all = false, special = true;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--all")) all = true;
        if (!std::strcmp(argv[i], "--no-special")) special = false;
    }
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = mode == "tok";
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[2], mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (mode == "tok") {
        std::string line;
        while (std::getline(std::cin, line)) {
            const std::string text = unhex(line);
            std::vector<llama_token> ids(text.size() + 16);
            int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), /*add_special*/ false, special);
            if (n < 0) { ids.resize((size_t) -n); n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), false, special); }
            for (int i = 0; i < n; ++i) std::printf(i ? ",%d" : "%d", ids[(size_t) i]);
            std::printf("\n");
        }
        return 0;
    }
    if (mode == "logits") {
        if (argc < 4) return 2;
        std::string line;
        std::getline(std::cin, line);
        std::vector<llama_token> ids;
        for (size_t p = 0; p < line.size();) {
            char* e = nullptr;
            ids.push_back((llama_token) std::strtol(line.c_str() + p, &e, 10));
            p = (size_t) (e - line.c_str());
            while (p < line.size() && (line[p] == ',' || line[p] == ' ')) ++p;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) std::max<size_t>(512, ids.size() + 8);
        cp.n_batch = cp.n_ubatch = (uint32_t) ids.size();
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.n_threads = cp.n_threads_batch = 4;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) { std::fprintf(stderr, "no context\n"); return 1; }
        llama_batch b = llama_batch_init((int32_t) ids.size(), 0, 1);
        for (size_t i = 0; i < ids.size(); ++i) {
            b.token[i] = ids[i];
            b.pos[i] = (llama_pos) i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = all || i + 1 == ids.size();
        }
        b.n_tokens = (int32_t) ids.size();
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
        const int nv = llama_vocab_n_tokens(vocab);
        std::FILE* f = std::fopen(argv[3], "wb");
        for (size_t i = 0; i < ids.size(); ++i)
            if (all || i + 1 == ids.size()) std::fwrite(llama_get_logits_ith(ctx, (int32_t) i), sizeof(float), (size_t) nv, f);
        std::fclose(f);
        std::fprintf(stderr, "ok: %zu tokens, %d vocab\n", ids.size(), nv);
        return 0;
    }
    return 2;
}
