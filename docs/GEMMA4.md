# Gemma 4 26B-A4B

`strata-gemma4` runs Google's Gemma 4 26B-A4B (a mixture of 128 experts, 8 used per token, 3.8 B parameters active) from a GGUF file, on CUDA, with the
same line protocol as the other engines, so `serve/server.py` (OpenAI and Anthropic API, web page) serves it. Text only: no pictures or audio, no
speculative drafting (the MTP file of the model is not used).

It is its own engine (`src/gemma4/`), not a mode of the Qwen one: Gemma 4's layers are not Qwen's. It shares no library with the other engines, needs no
llama.cpp to build, and reads the GGUF's quantized blocks directly.

## What the model is (read from the GGUF of `unsloth/gemma-4-26B-A4B-it-GGUF`, and from llama.cpp's `src/models/gemma4.cpp`)

- 30 layers, 2816 wide, 16 query heads. Five layers in six are sliding-window layers (window 1024, 8 KV heads of 256 values, rope base 10 000); every sixth
  is a global layer (2 KV heads of 512 values, rope base 1 000 000 with a "proportional" rope: only the first quarter of the head turns, the others carry a
  divisor of 1e30 in `rope_freqs`). A global layer has no `attn_v`: its V is its K projection.
- Each layer adds a dense GELU MLP (2112 wide) and the 8 experts (704 wide, gate and up fused in one tensor, a scale per expert on the down projection), each
  with its norms before and after, the router norm and its scale, and a layer scalar. No attention scale (the head norms carry it). The embeddings are scaled by
  sqrt(2816), the output is the tied token embedding, and the logits are capped: 30 tanh(x / 30).
- Context 262 144. Vocabulary 262 144 (`tokenizer.ggml.model` = `gemma4`: a BPE over raw UTF-8 with `▁` for the spaces).

## Files

| file | what |
|---|---|
| `src/gemma4/g4_quant.hpp`, `g4_iq_tables.hpp` | ggml's block formats and their dequantization, one implementation for the host and the device |
| `src/gemma4/g4_model.hpp` | the GGUF's hyper-parameters and tensors, bound by name with their shapes checked |
| `src/gemma4/g4_ref.hpp` | the CPU reference: one token at a time, written for clarity (about a second a token on four cores) |
| `src/gemma4/g4_kernels.cuh`, `g4_cuda.cu` | the CUDA engine |
| `src/gemma4/g4_serve.hpp` | the server's line protocol (`GEN`, `PP`, `T`, `DONE`, `STOP`), one conversation kept at a time |
| `src/gemma4/g4_emu.hpp` | CUDA on the CPU, to test the kernels without a GPU |
| `serve/gemma4.py` | the thinking channel and the native tool calls, translated to the form the server's parser reads |
| `tools/strata_tokenizer.py` (`Gemma4Tokenizer`) | the tokenizer, id for id llama.cpp's |

## Build and run

```sh
tools/build_rhel7.sh --gemma4 --arch "61;75" --cuda-home /usr/local/cuda-12.4      # build-rhel7/strata-gemma4; no llama.cpp needed
python3 tools/strata_tokenizer.py --gguf gemma-4-26B-A4B-it-UD-IQ4_XS.gguf --out $HOME/strata-data/packs/gemma4     # the tokenizer and the chat template
```

`strata-gemma4.json` for `python3 serve/server.py --engine strata --config strata-gemma4.json`:

```json
{
 "exe": "/path/to/Strata/build-rhel7/strata-gemma4",
 "args": ["--model", "/data/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", "--max-context", "32768"],
 "tokenizer": "/home/you/strata-data/packs/gemma4/tokenizer",
 "arch": "gemma4",
 "lib_dirs": ["/usr/local/cuda-12.4/lib64"],
 "port": 8080
}
```

Engine options (`strata-gemma4 --help`): `--max-context N` (default 8192), `--gpu 0,1` (devices in layer order; default: every device, the one with the
most free memory first), `--prefill N` (tokens read per chunk, 512), `--swa-slack N` (extra cells the sliding-window layers keep so a conversation can step
back, 1024), `--vram-reserve-mib N` (left free on each device, 700), `--cpu` (the CPU reference), `--eos ID,...` (default 1, 50, 106: `<eos>`,
`<|tool_response>`, `<turn|>`).

Checks you can run on your machine: `strata-gemma4 --model F.gguf --tokens 2,9259,236764 --check` runs the prompt on the GPU and on the CPU reference and says
how far apart the logits are (the engine's own gate: a difference above 2e-3 or another best token is reported as FAILED); `--bench-pp 512` and `--bench-tg 64`
time the prompt and the generation.

## Which GGUF

The engine reads F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and MXFP4: every
file Unsloth publishes for this model (headers read, not the files). Not IQ1_S, IQ1_M or the ternary types. The layers are spread over the cards in order (the
output head, 0.7 GiB, on the last one); nothing is offloaded to the CPU, so the file, the K/V cache and about 1 GiB of buffers must fit the VRAM of all cards.

Measured on the real file headers (weights of the layers + the output head; the K/V cache is 0.64 GiB at a context of 8192, 1.1 GiB at 32 768, 3.0 GiB at
131 072, with the defaults):

| file | size | needs |
|---|---|---|
| UD-Q2_K_XL | 10.5 GB | about 11 GiB with a long context |
| UD-IQ3_S | 11.3 GB | about 12 GiB |
| UD-Q3_K_XL | 12.9 GB | about 13.5 GiB |
| **UD-IQ4_XS** | 13.6 GB | about 14.5 GiB: fits a 11 + 8 GB pair with room for 32K of context |
| UD-Q4_K_S / M / XL | 16.5 / 16.9 / 17.0 GB | 17.5 to 18.5 GiB: a 11 + 8 GB pair only just, with `--vram-reserve-mib 300` and a context of 8K |
| UD-Q5_K_M, Q6_K, Q8_0 | 21 to 27 GB | more than two such cards |

## What was checked, and what was not

Checked, without a GPU (`tools/gemma4/README.md` says how to repeat each):
- the tokenizer against llama.cpp on the real 262 144-token vocabulary: 0 differences over 3 000 random strings and 970 pieces of real text and code, with and
  without special tokens;
- every quantization (21 types) against ggml's own dequantization on random blocks, bit for bit;
- the CPU reference against llama.cpp: on a small random model with this layout and the real vocabulary, the logits agree to 1e-4 of their spread (4e-5
  once ggml's half-precision GELU table is imitated; 7e-5 with the code-book types), the same best token and top five at every position; against the same
  model quantized the difference is 1 to 2 %, what llama.cpp's 8-bit quantization of the activations costs;
- the CUDA engine's kernels and host code (`g4_cuda.cu`, unchanged) on the CPU emulation of CUDA, against the reference: logits within 2e-5 for one chunk,
  chunks with a wrapping ring, one token at a time (6e-5), two devices, a step back; under AddressSanitizer and UBSan;
- the server: 32 tests (`serve/test_gemma4.py`: tool calls and thinking over both APIs, forced tools, the chat template against Jinja2), and a real
  server + engine process on the small model.
- it builds with nvcc 12.4 for sm_61 and sm_75 (`tools/build_rhel7.sh --gemma4`, no spills, 32 KB of shared memory at most per block).

Not checked: any run on a GPU (there was none), so no speed figure and no check of the arithmetic on a real card. `--check` is there for that. Nothing was run
with the real 26B weights (the headers of the real files were read; the arithmetic was checked on a model of the same layout).

## The conversation cache

A request that begins with the tokens the cache holds reads only what follows. It may also step back: the next prompt of a chat re-renders the previous
answer (the template drops its empty thinking channel), so the cache must return to the first token that differs. A global layer keeps every cell, a
sliding-window layer a ring of `window + chunk + slack` cells (1024 + 512 + 1024 by default), so the way back is at most `chunk + slack` = 1536 tokens: beyond
that the whole prompt is read again (about 300 to 500 tokens a second on a card of that class, a guess: not measured). An agent whose answers or tool results are
longer than that should raise `--swa-slack` with the VRAM it has left: 25 sliding-window layers at 8 KB a cell make 0.2 GiB per 1024 cells (`--swa-slack 4096`:
+0.6 GiB).

## Limits

One conversation at a time (no batch, no slots), K/V as f16 (no q8 / q4 cache), no CUDA graphs, no vision or audio, no draft model, no CPU experts. The
E2B and E4B sizes (per-layer embeddings, K/V shared between layers) are refused with a message; the dense 31B is read by the same code but was not tried.
