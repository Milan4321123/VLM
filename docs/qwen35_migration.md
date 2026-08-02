# Replacing VLM-FO1's Decoder with Qwen3.5-0.8B

This document explains the architectural change made on this branch: swapping
VLM-FO1's Qwen2.5-VL-3B decoder for `Qwen/Qwen3.5-0.8B`, so the model fits a
free Colab T4 GPU. It covers the motivation, exactly what changed vs. what
didn't, the untrained components, and what was verified to actually run.

## 1. Motivation

VLM-FO1's contribution is not "a fine-tuned Qwen2.5-VL" — it's the **Hybrid
Fine-grained Region Encoder (HFRE)**: a dual vision-encoder pipeline that turns
bounding-box proposals into region tokens a language model can reason over. The
paper's plug-and-play framing means the decoder is, in principle, swappable
without touching that region-encoding machinery.

On a free 16 GB T4, the full pipeline — two vision towers plus a 3B decoder —
leaves little headroom for activations and long generations, and iteration is
slow. Replacing just the decoder with a 0.8B model cuts the memory and latency
budget substantially while keeping every other component from the original
checkpoint. Training the new components this introduces (section 6) is future
work; this branch delivers the wiring.

## 2. What stays identical

| Component | File |
|---|---|
| Primary vision tower (`Qwen2_5_VlVisionTower`) | `vlm_fo1/model/multimodal_encoder/qwen2_5_vl_encoder.py` |
| Auxiliary vision tower (DaViT) | `vlm_fo1/model/multimodal_encoder/davit_aux_encoder.py` |
| Region encoder (HFRE / `object_vp_extractor`) | `vlm_fo1/model/multimodal_visual_prompt_encoder/hybrid_finegrained_region_encoder.py` |
| `mm_projector` / `mm_projector_aux` | trained VLM-FO1 checkpoint weights, untouched |
| `encode_images` / `encode_regions` | `vlm_fo1/model/language_model/omchat_qwen2_5_vl.py` |
| Marker-splitting tokenization (`tokenizer_image_token` / `tokenizer_image_region_token`) | `vlm_fo1/mm_utils.py` |

The load-bearing fact: `encode_images`/`encode_regions` only call the vision
tower, aux tower, `object_vp_extractor`, and the projectors — never
`self.model.layers` (the decoder). This is why removing the decoder in
section 3 doesn't break feature extraction. (Other parts of `mm_utils.py` did
change — the prompt-assembly and stopping-criteria code, see section 4 — but
the marker-splitting helpers the region pipeline depends on are untouched.)

## 3. What's removed

`_strip_original_qwen25_decoder` (`vlm_fo1/model/builder.py`) empties out the
Qwen2.5-VL decoder stack once loaded: `model.layers`, `embed_tokens`, final
`norm`, and `lm_head`. These are replaced with no-op modules (`nn.ModuleList()`,
`nn.Embedding(1,1)`, `nn.Identity()`) — the vision-tower-holding object still
needs a valid module tree, but its own text-generation path is never called
again after this point.

## 4. What's new

- **The swap is opt-in.** `load_pretrained_model` takes an `llm_model_path`
  argument that defaults to `None` (original single-model behaviour, used by
  the UPN/SAM3 scripts and the Gradio demos unchanged). `inference.py` passes
  it explicitly via `--llm-model-path`, which defaults to `Qwen/Qwen3.5-0.8B`;
  passing `''` there runs the original decoder instead.
- **Generator**: `Qwen/Qwen3.5-0.8B`, loaded via `AutoModelForImageTextToText`
  (it's a natively vision-language `Qwen3_5ForConditionalGeneration` model —
  its own vision tower is loaded but never used, since VLM-FO1 supplies its own
  region-aware embeddings directly).
- **`feature_bridge`** (`vlm_fo1/model/language_model/qwen35_fo1_wrapper.py`):
  a single `nn.Linear(2048, 1024, bias=False)`. Qwen2.5-VL's hidden size is
  2048; Qwen3.5-0.8B's is 1024 (from its `text_config`). Every image and
  region feature is projected through this before being spliced into Qwen3.5's
  input embeddings. It is the only new trainable *module* the swap introduces,
  and it starts randomly initialized — see section 6.
- **100 new `<regionN>` tokens** added to Qwen3.5's tokenizer
  (`_ensure_qwen35_fo1_tokens`). Their embedding rows are the other untrained
  addition. The embedding table is only ever grown, never shrunk — Qwen
  checkpoints pad it past the tokenizer length, and a blind resize-to-match
  would discard pretrained rows.
- **`Qwen35FO1InferenceWrapper`**: replaces the old single-model
  `forward`/`generate` flow. It runs the (decoder-stripped) vision backbone to
  get image/region features, projects them through `feature_bridge`, manually
  interleaves them with Qwen3.5's text embeddings, and calls
  `Qwen3.5.generate(inputs_embeds=...)` directly — bypassing `input_ids`-based
  generation entirely, since the spliced sequence contains real embeddings, not
  token ids.
- **Tokenizer switched to Qwen3.5's own tokenizer** (not the VLM-FO1
  checkpoint's), so all special-token ids (`<|im_start|>`, `<|im_end|>`, eos/pad)
  are resolved dynamically per-tokenizer in `mm_utils.py` rather than hardcoded
  to Qwen2.5-VL's fixed ids.
- **Qwen3.5's `<think>...</think>` chat-template preamble** is handled
  explicitly, since Qwen2.5-VL's template didn't have one.

## 5. Why not use Qwen3.5's own vision pathway?

Qwen3.5 is *itself* natively vision-language (its config carries its own
`vision_config`, a 12-layer ViT). This raises the natural question: why not
just use Qwen3.5's own vision pathway? Answer: VLM-FO1's HFRE/region-encoder is
what was purpose-trained to fuse the two vision encoders' features into
region-grounded tokens (`<ground>label</ground><objects><regionN>...</objects>`
output format). Discarding it in favor of Qwen3.5's native vision tower would
mean re-deriving that grounding behavior from scratch — the whole point of this
exercise is to keep it and swap only the decoder.

## 6. Known limitation

`feature_bridge` and the new `<regionN>` token embeddings are randomly
initialized and not trained. A `RuntimeWarning` fires at load time to make
this visible. The pipeline is dimensionally and mechanically correct, but its
outputs are not meaningfully grounded in the image until those components are
trained (via `bridge_state_path` / `VLM_FO1_QWEN35_BRIDGE_PATH` once a trained
state exists). Everything else in the pipeline is pretrained.

## 7. Compatibility with Transformers 5

The `qwen3_5` architecture first shipped in Transformers 5.2.0; under any 4.x
release, loading `Qwen/Qwen3.5-0.8B` fails with `ValueError: ... model type
qwen3_5 ... does not recognize this architecture`. Moving the repository from
its `transformers==4.50.1` pin to `>=5.2.0` required two follow-up fixes:

- Transformers 5 removed `SlidingWindowCache` from `transformers.cache_utils`,
  which the vendored `modeling_qwen2_5_vl.py` imports at module level. The
  class only appears in the text decoder's causal-mask preparation — code the
  swapped pipeline never runs — so the import now falls back to an inert
  placeholder class.
- Transformers 5 removed the `load_in_4bit`/`load_in_8bit` keyword arguments
  from `from_pretrained` (they now reach the model constructor and raise a
  `TypeError`). Both loading sites in `builder.py` pass an explicit
  `BitsAndBytesConfig` as `quantization_config` instead.

Separately, `builder.py` had a latent bug: `aux_image_processor` was assigned
only inside a conditional branch but read unconditionally afterwards. The read
was safe only because two string-matching guards happened to be textually
identical; the variables are now initialized to `None` up front so the code is
locally correct.

## 8. Testing

`tests/smoke_qwen35_wrapper.py` (plain script, `python
tests/smoke_qwen35_wrapper.py`) runs on CPU with the real Qwen3.5-0.8B weights
and a stub vision backbone that returns correctly shaped 2048-dim features. It
covers: the package import chain (including the vendored Qwen2.5-VL file under
Transformers 5), adding the 100 region tokens and the grow-only embedding
check, the `feature_bridge` projection, the embedding splice at the
image/region sentinel positions, and a real
`generate(inputs_embeds=..., use_cache=True)` call against Qwen3.5's hybrid
attention/linear-attention architecture. The decoded output is fluent but
ungrounded, as expected from section 6.

The quantized path (`--load-4bit`/`--load-8bit`) needs `bitsandbytes` and a
CUDA GPU, and the full checkpoint run needs the 6GB+ VLM-FO1 weights; both are
exercised on Colab via `inference.py --load-4bit` per the README, which
produces the bounding-box visualization in `demo/vlm_fo1_result.jpg`.
