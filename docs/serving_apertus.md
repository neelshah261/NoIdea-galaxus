# Serving Apertus at 512 px, cheaply (plan #40, option 4)

Swarup's conditions for every method: **images at 512 x 512 pixels of area, always**, and **low cost** (our Hugging
Face credits). This page is how the serve lane runs Apertus for the smoke test (#13), the benchmark (#16) and the
final run, whichever predictor the model lane ships.

## Where Apertus can run

| Route | Images | What it costs us | Use for |
| --- | --- | --- | --- |
| Hack Apertus gateway (sprintd Perks key) | Q-0001, not answered yet | event-provided; limits unknown | zero-shot, smoke test |
| HF Inference Providers (PR #30 key ring) | depends on the provider | credits per request | only if the gateway takes no images |
| **Own GPU + vLLM** (`bench/vllm_apertus.sh`) | yes (Swiss AI image) | per GPU hour while it runs | generation methods, speed numbers, Saturday |
| Own GPU + transformers (`model.apertus.local`) | yes | per GPU hour | hidden states for the heads (#41) |

## Keeping 512 px everywhere

- The data lane's cache holds every image at 512 px of area (`$GALAXUS_DATA/cache/images/512/`, #27); train,
  heads, LoRA and the final run read the same pixels.
- `bench/vllm_apertus.sh` pins the server's image processor to 512 x 512 pixels (`max_pixels` = `min_pixels` =
  262144). Apertus' vision tokenizer uses 16 px per token, so one image is about 1,024 image tokens; the smoke test
  checks this (Q-0002).

## Keeping the cost low

- **One pass per image.** The heads (#41) read Apertus once and write no text, so the GPU time per image is the
  forward pass only. Generation methods pay for every output token on top.
- **Many images at once.** vLLM batches up to `MAX_NUM_SEQS` requests on the GPU; on our side
  `--workers N` (PR #37) keeps N batches in flight. Pick N with `bench --workers 1/4/8/16` in the dry run.
- **The shared prompt is computed once.** `--enable-prefix-caching`: the system prompt is the same for every image.
- **FP8 as an option** (#43): same 512 px, smaller weights, measured against bf16 before it is used.
- **Stop the GPU** after every session. Cost per 100,000 images (PLAN.md formula, from `galaxus.serve.bench`) only
  counts busy time; idle time is paid too.

## Order on the GPU, first session

1. `bash bench/vllm_apertus.sh` (bf16).
2. Smoke test with `--backend vllm`: images taken, schema output, tokens per image (Q-0001, Q-0002).
3. `galaxus.serve.bench` with the zero-shot predictor at `--workers 1, 4, 8, 16`: img/s and cost per 100k.
4. Stop the GPU; write the numbers into the issues (#13, #16, #43).

Nothing here was run yet: there was no GPU access when it was written (27 Nov). Fix what the first session shows.
