// logits of a prompt on the reference engine: g4_ref_logits MODEL.gguf OUT.bin [--all] [--table] < ids (one line, comma separated)
#include <cstdio>
#include <cstring>
#include <iostream>
#include "g4_ref.hpp"
int main(int argc, char** argv) {
    g4::Model m;
    const std::string e = g4::load_model(argv[1], m);
    if (!e.empty()) { std::fprintf(stderr, "ERROR: %s\n", e.c_str()); return 1; }
    bool all = false;
    g4::RefOpts o;
    for (int i = 3; i < argc; ++i) { if (!std::strcmp(argv[i], "--all")) all = true; if (!std::strcmp(argv[i], "--table")) o.gelu_table = true; if (!std::strcmp(argv[i], "--f16")) o.cache_f16 = true; }
    std::string line;
    std::getline(std::cin, line);
    std::vector<int32_t> ids;
    for (size_t p = 0; p < line.size();) { char* en = nullptr; ids.push_back((int32_t) std::strtol(line.c_str() + p, &en, 10)); p = (size_t) (en - line.c_str()); while (p < line.size() && (line[p] == ',' || line[p] == ' ')) ++p; }
    g4::RefEngine eng(m, (int) ids.size() + 8, o);
    std::vector<float> last, every;
    eng.forward(ids.data(), (int) ids.size(), &last, all ? &every : nullptr);
    std::FILE* f = std::fopen(argv[2], "wb");
    if (all) std::fwrite(every.data(), 4, every.size(), f); else std::fwrite(last.data(), 4, last.size(), f);
    std::fclose(f);
    std::fprintf(stderr, "ok: %zu tokens\n", ids.size());
    return 0;
}
