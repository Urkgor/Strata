# Inkling 2 Mini

`strata-inkling2` runs Thinking Machines' Inkling 2 Mini (GGUF architecture `inkling2_loop`) on CUDA, one card, with the same line protocol as the other
engines. Text only. It is its own engine (`src/inkling2/`), built on the Gemma 4 engine's GGUF reader, quantized kernels and server protocol
(`src/gemma4/`); it needs no llama.cpp.

**Status: not run on a real card.** The CPU reference equals llama.cpp on a tiny random model (F32, F16, Q8_0); the CUDA engine equals the reference on the CPU
emulation of CUDA. The tokenizer and the server glue match llama.cpp / the model's template on small cases. Speed, VRAM figures and the real 17.6 GB file are untested.

## Which GGUF

Two spellings of the same model are read: **ggml-org's** (`Inkling-2-Mini-IQ4_NL.gguf`: arch `inkling2_loop`, `block_count` 24, the loop's messages as `blk.N.loop_msg_query`) and **Unsloth's**
(`Inkling-2-Mini-UD-*.gguf`: arch `inkling2`, `block_count` 48 = the layers as visited, `num_loops` and `loop.core_start / core_end`, tensors for the 24 physical layers, the messages as the two global tensors
`loop_res_query` / `loop_kv_res_query` and the controllers as `blk.N.loop_attn|k|v|mlp_down|gain|forget`). The Unsloth files keep their experts in NVFP4 (ggml type 40) with a per-expert
`ffn_gate_up_exps.scale` / `ffn_down_exps.scale`, which the engine reads (NVFP4 rows: 64 values, four E4M3 scales, E2M1 nibbles; the scalars multiply the gate+up and down products). The other types of those files
(Q4_K, Q5_K, Q6_K, IQ4_XS, BF16, ...) were already read. Checked on a tiny model written in both spellings (identical logits) and in NVFP4 with scales (the CPU reference against llama.cpp
reading the same values as F32: 5e-6); not on the real files.

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
| `serve/inkling2.py`, `serve/fixtures/inkling2/chat_template.jinja` | the server side: thinking, answer and tool-call messages read into the form the server's parser takes; the repo's own chat template |
| `tools/strata_tokenizer.py` (`GPT4O_PATTERN`) | the tokenizer: byte-level BPE, o200k-style pre-tokenizer (`tokenizer.ggml.pre` = `inkling2`), no BOS |
| `serve/test_inkling2.py` | translator, parser, template and chat API against a fake engine |

## Build and run

```sh
tools/build_rhel7.sh --inkling2 --arch "61" --cuda-home /usr/local/cuda-12.4        # build-rhel7/strata-inkling2
strata-inkling2 --model Inkling-2-Mini-IQ4_NL.gguf --tokens 1,2,3 --check            # GPU vs CPU reference
strata-inkling2 --model F.gguf --bench-pp 512 --bench-tg 64
```

Options: `--max-context N`, `--gpu N` (one card only), `--prefill N` (512), `--swa-slack N` (1024), `--vram-reserve-mib N` (700), `--host-experts N`, `--cpu`,
`--eos ID,...`, `--expert-cache-mib N` (VRAM for the expert cache, default: what is left; 0 = none), `--cache-decay X` (0.97), `--cpu-experts auto|N` (the CPU expert pool: auto = on if faster here, 0 = off, N threads).

## Memory of the machine

The streamed layers' experts are copied once into pinned host memory (about 9 GiB of the 17.6 GiB file with an 11 GB card), next to the file's page cache (which the system gives back when it needs it): 32 GB of RAM is enough.
The engine prints the size of the host copy at start. If the pinning fails (a locked-memory limit) it falls back to ordinary memory and says so (slower copies).

## Serve it

```sh
python3 tools/strata_tokenizer.py --gguf Inkling-2-Mini-IQ4_NL.gguf --out $HOME/strata-data/packs/inkling2     # tokenizer + the GGUF's chat template
```

`strata-inkling2.json` for `python3 serve/server.py --engine strata --config strata-inkling2.json`:

```json
{
 "exe": "/path/to/Strata/build-rhel7/strata-inkling2",
 "args": ["--model", "/data/Inkling-2-Mini-IQ4_NL.gguf", "--max-context", "16384"],
 "tokenizer": "/home/you/strata-data/packs/inkling2/tokenizer",
 "arch": "inkling2",
 "lib_dirs": ["/usr/local/cuda-12.4/lib64"],
 "port": 8080
}
```

How the text goes: the template has no generation prompt. The prompt ends with the last message's `<|end_message|>` and the model writes its own
`<|message_model|><|content_thinking|>...<|end_message|>`, `<|message_model|><|content_text|>...<|end_message|>` and
`<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>`, then `<|content_model_end_sampling|>` (id 200006, the only stop).
`serve/inkling2.py` turns that into thinking, answer and tool calls. Reasoning effort: `none`, `minimal`, `low`, `medium`, `high` (default, 0.9), `xhigh`, `max`, or a
number 0 .. 0.99; `none` also turns the reasoning off. Sampling default: temperature 1.0 (the repo's `generation_config.json`). Images and audio are refused.

## Memory

The IQ4_NL file is 17.6 GB: it does not fit an 11 GB card (20 layers' experts are IQ4_NL, the last 4 layers' Q8_0). The experts of the layers outside the loop stream
from pinned host memory first, then the loop's (each visited three times, so costlier).

**Expert cache.** What VRAM is left after the weights, the K/V and the staging slots becomes a cache of the streamed layers' experts. A decode step looks up the 8 experts
the router chose; the ones the cache has are used where they are, the others are copied in over the experts with the lowest decayed use count (every use adds 1, every
decoded token multiplies all counts by 0.97). The design is that of the official Strata engine's "decayed-frequency promotion ring"
(github.com/Niko1221/Strata, `engine/src/inkling2/cuda/ink_hybrid.cuh`), done on the host side because this engine already reads the routing back at every streamed layer.
**CPU expert pool.** A decode step's expert that is not in the cache and not used more than the cache's least used one is not copied: it is computed on the CPU from the host copy
(`src/inkling2/i2_cpu_pool.cpp`: AVX2 + FMA rows for IQ4_NL and Q8_0 chosen at run time, any other type through the portable dequantization, same float arithmetic as the GPU kernels)
while the GPU runs the others; only the 2304 input activations go down and the expert's 2304 outputs come back. At start the engine times an expert on the pool and the same bytes over PCIe,
and uses the pool only if it is faster on this machine (it says so). The cold experts of a step are then shared between the CPU and the PCIe (into a few transient slots outside the cache's books), each one going to whichever would finish it first given the measured times and what is already queued (the official engine's "miss-sharing"). The idea is the official engine's CPU expert pool (`ink_hybrid.cuh`, `hybrid_doorbell.cu`); that one runs inside a CUDA graph through
a mapped-memory doorbell, this one is plain host code between the layers' kernels. Prompt reading does not use the cache or the pool (it streams through the staging slots). `strata-inkling2 --bench-tg 64` prints the hit rate. The real speed is unmeasured.
No CUDA graphs yet.

## Not done

- Checked only on small cases: the tokenizer against llama.cpp on a tiny vocabulary (the real 200 058-token vocabulary was not available here), the server glue against a fake engine.
- Taken from the official Strata engine, not yet here: DSpark speculative decoding (needs `dspark-Inkling-2-Mini-Q8_0.gguf`; it would run on a GTX 1080 Ti - 0.4 GB, plain FP32 kernels - but verifying 4 tokens touches about 3.6 times the experts of one, which is what a streaming card is short of, so it is not expected to speed this setup up), the vision tower (`mmproj-*.gguf`), K8V8/K8V4 KV compression. CUDA graphs.
- No MTP/speculative drafting: the GGUF has none.

## Checks

```sh
python3 tools/inkling2/make_tiny.py --out tiny-f32.gguf
g++ -std=c++20 -O1 -fsanitize=address -DSTRATA_G4_EMU -Isrc/gemma4 -Isrc/inkling2 -Iinclude tests/inkling2/i2_emu_test.cpp -x c++ src/inkling2/i2_cuda.cu -o i2_emu_test -lpthread
./i2_emu_test tiny-f32.gguf            # "quick" skips the token-by-token decode scenario
```
