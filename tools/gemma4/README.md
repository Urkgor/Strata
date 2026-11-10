# tools/gemma4 - checking the Gemma 4 engine

Nothing here needs a GPU. What you need: a llama.cpp checkout built for the CPU (it is the reference) and the header of a Gemma 4 GGUF.

```sh
# the header (vocabulary, template, tensor list) of the real file, 64 MB: no need for the 17 GB
URL=https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
SIZE=$(curl -sIL $URL | awk 'tolower($1)=="content-length:" {s=$2} END {print s+0}')
curl -L -r 0-67108863 -o header.part $URL && truncate -s $SIZE header.gguf && dd if=header.part of=header.gguf conv=notrunc

# llama.cpp v0.9.0 (or the commit in third_party/ggml/VERSION.txt for the engines), CPU only
cmake -S llama.cpp -B llb -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF && cmake --build llb --target llama llama-quantize -j4
g++ -std=c++17 -O2 -Illama.cpp/include -Illama.cpp/ggml/include tools/gemma4/llama_ref.cpp -o llama_ref \
    llb/src/libllama.a llb/ggml/src/libggml.a llb/ggml/src/libggml-cpu.a llb/ggml/src/libggml-base.a -lpthread -ldl -lm
```

1. **The tokenizer** against llama.cpp, id for id: `python3 tools/gemma4/tokenizer_parity.py --gguf header.gguf --llama-ref ./llama_ref --fuzz 3000 --files README-RHEL7.md serve/server.py`
   (`llama_ref tok` loads the vocabulary only, so the header is enough). The unit tests (no GGUF): `python3 -m unittest tools.test_gemma4_tokenizer serve.test_gemma4`.
2. **A small model** with the real layout and vocabulary: `python3 tools/gemma4/make_tiny.py --tokenizer-from header.gguf --out tiny-f16.gguf --quantize llb/bin/llama-quantize --out-q tiny.gguf`
   (`--iq` for the code-book types, `--vocab 512` for the model the emulation test reads; `--dequantize tiny.gguf tiny-f32.gguf` for an F32 copy with the same values).
3. **The quantization** against ggml: `tests/gemma4/g4_quant_test.cpp` (link ggml: `g++ -std=c++20 -Isrc/gemma4 -Illama.cpp/ggml/include tests/gemma4/g4_quant_test.cpp llb/ggml/src/libggml-base.a ... -o quant_test`).
4. **The reference** against llama.cpp: `g++ -std=c++20 -O2 -Isrc/gemma4 -Iinclude tools/gemma4/g4_cli_test.cpp -o g4_ref_logits -lpthread`, then
   `echo 2,9259,236764,1902,236888,105,2364,107 > ids.txt; ./llama_ref logits tiny-f32.gguf L.bin --all < ids.txt; ./g4_ref_logits tiny.gguf G.bin --all --table < ids.txt; python3 tools/gemma4/compare_logits.py L.bin G.bin 262144`
   (llama.cpp on the F32 copy has no activation quantization: the logits agree to ~1e-5 with `--table`, ~1e-4 without. Against the quantized file expect 1 to 2 %).
5. **The CUDA engine without a GPU**: `g++ -std=c++20 -O1 -DSTRATA_G4_EMU -Isrc/gemma4 -Iinclude tests/gemma4/g4_emu_test.cpp -x c++ src/gemma4/g4_cuda.cu -o emu_test -lpthread`
   (add `-fsanitize=address,undefined` for the memory checks), then `./emu_test tiny-v512.gguf` (`quick` skips the token-by-token scenario). The emulation runs the
   engine's kernels as fibers on one thread: a minute or two for the whole test.
6. **On a machine with a card**: `strata-gemma4 --model F.gguf --tokens ... --check`.
