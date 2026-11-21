# Inkling 2 Mini

`strata-inkling2` runs Thinking Machines' Inkling 2 Mini (GGUF architecture `inkling2_loop`) on CUDA, one card, with the same line protocol as the other
engines. Text only. It is its own engine (`src/inkling2/`), built on the Gemma 4 engine's GGUF reader, quantized kernels and server protocol
(`src/gemma4/`); it needs no llama.cpp.

**Status: not run on a real card.** The CPU reference equals llama.cpp on a tiny random model (F32, F16, Q8_0); the CUDA engine equals the reference on the CPU
emulation of CUDA. Speed, VRAM figures and the real 17.6 GB file are untested. Tokenizer, chat template and server glue are not written yet (see below).

## What the model is (llama.cpp `src/models/inkling2.cpp`)

- 24 physical layers; layers 6-17 run 3 times (48 visits), each visit with its own K/V and convolution state. Hidden 2304, 24 query heads over 8 KV heads of 128.
- 128 experts, 8 used (width 1152), no shared expert; sigmoid router with a selection bias, weights renormalised and scaled by 8; SITU activation.
- Sliding window 1024 except layers 5, 11, 17, 23 (global). No RoPE: a learned relative-position bias table per layer, an attention sink, an output gate,
  a QK clip, log-scaled attention on the global layers.
- Short convolutions (width 4) on K, V, the attention output and the FFN output.
- Block attention residuals (blocks of 6 layers): each layer mixes the blocks so far with learned queries. Later passes of the loop receive cross-pass messages and
  rank-64 controllers.
- Logits capped at 30; vocabulary 200 058 (padded 201 024).

## Files

| file | what |
|---|---|
| `src/inkling2/i2_model.hpp` | hyper-parameters and tensors from the GGUF |
| `src/inkling2/i2_ref.hpp` | CPU reference, token by token |
| `src/inkling2/i2_kernels.cuh`, `i2_cuda.hpp`, `i2_cuda.cu` | the CUDA engine |
| `src/inkling2/i2_main.cpp` | `strata-inkling2` (serve, `--tokens`, `--check`, `--bench-pp/-tg`) |
| `tools/inkling2/make_tiny.py`, `llama_ref.cpp` | a tiny random model with the real layout; the llama.cpp comparison tool |
| `tests/inkling2/i2_emu_test.cpp` | engine on the CUDA emulation vs the reference |

## Build and run

```sh
tools/build_rhel7.sh --inkling2 --arch "61" --cuda-home /usr/local/cuda-12.4        # build-rhel7/strata-inkling2
strata-inkling2 --model Inkling-2-Mini-IQ4_NL.gguf --tokens 1,2,3 --check            # GPU vs CPU reference
strata-inkling2 --model F.gguf --bench-pp 512 --bench-tg 64
```

Options: `--max-context N`, `--gpu N` (one card only), `--prefill N` (512), `--swa-slack N` (1024), `--vram-reserve-mib N` (700), `--host-experts N`, `--cpu`,
`--eos ID,...`.

## Memory

The IQ4_NL file is 17.6 GB: it does not fit an 11 GB card. The experts of the layers outside the loop stream from pinned host memory first, then the loop's
(each visited three times, so costlier). Per visit a decode step copies 8 experts; the real speed is unmeasured and expected to be low (tens of tokens a second at
best on PCIe 3). No expert cache and no CUDA graphs yet.

## Not done

- Tokenizer (o200k-style pre-tokenizer), chat template (`<|message_user|>`, `<|end_message|>`, reasoning effort), tool-call format and `serve/` glue.
- Expert cache (LFU) and prefetch for the streamed layers; CUDA graphs.
- No MTP/speculative drafting: the GGUF has none.

## Checks

```sh
python3 tools/inkling2/make_tiny.py --out tiny-f32.gguf
g++ -std=c++20 -O1 -fsanitize=address -DSTRATA_G4_EMU -Isrc/gemma4 -Isrc/inkling2 -Iinclude tests/inkling2/i2_emu_test.cpp -x c++ src/inkling2/i2_cuda.cu -o i2_emu_test -lpthread
./i2_emu_test tiny-f32.gguf            # "quick" skips the token-by-token decode scenario
```
