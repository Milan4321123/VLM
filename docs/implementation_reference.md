# Implementation reference: VLM-FO1 with a Qwen3.5 decoder

**Reference prepared: 17 September 2026.** This document describes the current source tree and the recorded verification evidence. It is a technical source for writing a report, rather than a new experiment, a claim of model accuracy, or a statement of individual authorship. The pretrained execution measurements retain their original **15 September 2026** validation date; the cleanup checks are dated **16 September 2026**.

## 1. Scope and interpretation

The implemented task is an **inference-only replacement of VLM-FO1's language decoder with `Qwen/Qwen3.5-0.8B`**, together with usable command-line and browser interfaces and Docker packaging. The original VLM-FO1 visual pipeline remains responsible for image and region features: its Qwen2.5-VL vision encoder, auxiliary DaViT encoder, Hybrid Fine-grained Region Encoder (HFRE), and projectors.

There is no fine-tuning, alignment training, LoRA adapter training, training dataset preparation, optimizer, backward pass, or training loop in the supported execution path. The supplied assignment's broader training description is separate from this implementation's clarified no-training scope. Existing pretrained parameters are loaded; new feature alignment is not learned.

The central limitation is semantic alignment. The original projectors emit vectors in the space used by the old decoder. A new linear layer makes their width compatible with Qwen3.5, but it does not teach Qwen3.5 what those vectors mean. The newly introduced region-token meanings also have no trained alignment. Consequently:

- Successful loading, finite tensors and successful generation establish software execution.
- A generated region tag selects one of the input proposals; it does not establish that the selection is correct.
- An empty parsed detection list does not establish that no objects are present.
- Original VLM-FO1 benchmark scores cannot be attributed to this decoder replacement.
- No trained Qwen3.5-HFRE checkpoint, quantitative accuracy result, or baseline comparison is included.

See [migration notes](qwen35_migration.md), [validation record](validation.md), and the unchanged [execution evidence](evidence/README.md).

## 2. Provenance: what was inherited and what changed

The repository contains upstream architecture code and earlier local migration commits. Source inspection alone cannot establish which person authored every line. It does establish the following technical boundaries.

| Component | Provenance and current role |
|---|---|
| Qwen2.5-VL visual architecture, DaViT, HFRE, SimpleFP and original image/region projectors | Inherited VLM-FO1 architecture. Retained, with specific compatibility changes described below. HFRE is not a newly designed module in this project. |
| Region-index task templates and example fruit image/boxes | Based on the existing inference/demo workflow. The final interfaces expose and validate them. |
| UPN and its custom deformable-attention extension | Inherited optional proposal model under `detect_tools/upn/`. The final runtime makes loading/downloads conditional on actual use. |
| Qwen3.5 wrapper and initial decoder migration | Already present in the supplied repository history. Current changes harden loading, feature insertion, reproducibility, cache/stopping behavior and error reporting. |
| CLI and initial Docker support | Already present in repository history; refined into the shared runtime, portable CPU path and optional NVIDIA configuration. |
| Shared runtime, checkpoint/offline tests, acceptance evidence, submission scripts and current documentation | Added/refined during completion of the submission. These are separate from pretrained model research or model training. |

Relevant commits in the current branch's history include `fa7868f` (wrapper), `0fb9a64` (Qwen3.5 decoder loading), `3dde42a` (tokenizer-based ChatML marker IDs), `bc07b31` (assistant turn formatting), `63b4e62` (runtime attention selection), `0a47323` (prompt-driven CLI), and `7d7d14f` (Docker packaging). At preparation time, `7d7d14f` is the checked-out commit and the final completion/refinement work is present as working-tree changes. The commit alone is therefore **not** an identifier for the final source snapshot. Unrelated alternative branches are not evidence that their architecture is active here.

On 16 September, 26 unused files were moved to a checked local backup, `archive/legacy-files-2026-09-16.zip`: SAM3 demos/media, evaluation utilities/datasets, superseded scripts, old result images, Colab requirements and original patch files. The active submission retains the supported inference path, optional UPN, four example inputs, tests and evidence. This cleanup did not perform an accuracy evaluation. The historical [upstream README](upstream_README.md) preserves context; the root [README](../README.md) is the current runbook.

## 3. Source map

| File | Principal classes/functions and responsibilities |
|---|---|
| [cli.py](../cli.py) | `build_parser`, `make_prompt`, `main`: validate arguments, choose proposals, run inference, save image/JSON, emit clean JSON to stdout. |
| [inference.py](../inference.py) | `parse_args`, `main`: translate the bundled fruit example into the common CLI; accept a JSON boxes file for other images. |
| [demo/gradio_demo.py](../demo/gradio_demo.py) | `create_demo`/`launch_demo`, `process`, `upload_defaults`, `extract_bbox_and_original_image`: lazy Gradio interface and manual/drawn/UPN proposal selection. |
| [vlm_fo1/runtime.py](../vlm_fo1/runtime.py) | `RuntimeConfig`, `validate_boxes`, `parse_boxes`, `resolve_device`, `preflight`, `get_proposals`, `parse_answer`, `InferenceRuntime`, `get_runtime`: shared inference policy and metadata. |
| [vlm_fo1/model/builder.py](../vlm_fo1/model/builder.py) | `load_pretrained_model`, `_load_qwen35_language_model`, `_ensure_qwen35_fo1_tokens`: checkpoint construction, tokenizer selection, text-only Qwen loading and integrity checks. |
| [qwen35_fo1_wrapper.py](../vlm_fo1/model/language_model/qwen35_fo1_wrapper.py) | `Qwen35FO1InferenceWrapper`, `_PromptAwareStoppingCriteria`: frozen bridge, visual/text sequence construction, generation and output prefix contract. |
| [omchat_qwen2_5_vl.py](../vlm_fo1/model/language_model/omchat_qwen2_5_vl.py) | `OmChatQwen25VLConfig`, `OmChatQwen25VLForCausalLM`, `encode_images`, `encode_regions`: retained FO1 modules, optional original decoder, vision-only construction. |
| [omchat_arch.py](../vlm_fo1/model/omchat_arch.py) | `OmChatMetaModel`: constructs the primary/auxiliary towers, `object_vp_extractor` and original projectors. |
| [qwen2_5_vl_encoder.py](../vlm_fo1/model/multimodal_encoder/qwen2_5_vl_encoder.py) | `Qwen2_5_VlVisionTower`, `VisionFeaturesGather`, `custom_forward`: primary image encoding and intermediate feature collection. |
| [davit_aux_encoder.py](../vlm_fo1/model/multimodal_encoder/davit_aux_encoder.py) | `DavitVisionTower`: auxiliary image preprocessing and four-stage DaViT features. |
| [hybrid_finegrained_region_encoder.py](../vlm_fo1/model/multimodal_visual_prompt_encoder/hybrid_finegrained_region_encoder.py) | `HFREModule`, `extract_vt_region_feature`, `gen_sineembed_for_position`: ROIAlign, feature pooling/concatenation and box-position embeddings. |
| [simple_fpn.py](../vlm_fo1/model/multimodal_visual_prompt_encoder/simple_fpn.py) | `SimpleFP`: multiscale features from the primary vision map. |
| [multimodal_projector/builder.py](../vlm_fo1/model/multimodal_projector/builder.py) | `build_vision_projector`, `build_vision_projector_aux`: checkpoint-configured image and region projection modules. |
| [mm_utils.py](../vlm_fo1/mm_utils.py) | `prepare_inputs`, `make_message_context`, tokenization helpers, coordinate resizing, `KeywordsStoppingCriteria`, `draw_bboxes_and_save`. |
| [constants.py](../vlm_fo1/constants.py), [task_templates.py](../vlm_fo1/task_templates.py), [attn_utils.py](../vlm_fo1/attn_utils.py) | Sentinel constants, task strings and attention backend selection. |

## 4. Checkpoints and architectural dimensions

### Checkpoint identities

| Setting | Value / interpretation |
|---|---|
| Default FO1 checkpoint identifier | `omlab/VLM-FO1_Qwen2.5-VL-3B-v01`; the runbook identifies it as a Hub alias for `omlab/VLM-FO1-3B-v01`. |
| Decoder checkpoint | `Qwen/Qwen3.5-0.8B`. |
| FO1 revision used in recorded tests | `15788f0d303509abacba5f97b52c9ff625080319`. |
| Qwen3.5 revision used in recorded tests | `2fc06364715b967f1860aea9cf38778875588b17`. |
| FO1 path in full-pipeline evidence | `resources/VLM-FO1-3B-v01`, a complete locally staged checkpoint. |
| Normal revision selection | The loader calls `from_pretrained` without a fixed `revision=` argument. The recorded hashes identify tested weights; future Hub downloads are not automatically locked to those hashes. |

`default_model_path()` gives priority to a nonempty `VLM_FO1_MODEL`, then looks for `config.json` in `./resources/VLM-FO1-3B-v01` and `./resources/VLM-FO1_Qwen2.5-VL-3B-v01`, then uses the Hub default. Base Compose explicitly sets the Hub default unless overridden. Decoder IDs and vision-checkpoint IDs are separate settings; the Qwen2.5 text in the FO1 checkpoint name does not identify the active generation decoder.

The [metadata fixture](../tests/fixtures/fo1/README.md) contains the published configuration, processor configuration and safetensors index, not the weights. Its configuration records:

- Original FO1 decoder width: **2048**; 36 layers, 16 attention heads and 2 key/value heads. Those decoder layers are omitted in the replacement path.
- Primary vision encoder: 32 blocks, hidden width **1280**, 16 heads, output width **2048**, spatial patch size **14**, temporal patch size **2**, spatial merge size **2**.
- Primary full-attention/intermediate collection blocks: zero-based indices **7, 15, 23, 31**.
- Region limit: **100**; `mm_use_region_index_token=true`.
- HFRE: `mm_use_vision_tower_region_feature=true`, `mm_use_simpleFPN_for_vt=true`, `mm_region_feature_combination="concat"`, ROI output size **14**, box-based positional embeddings enabled, region layer normalization disabled.
- Primary projector: `identity`.
- Region projector: `mlp2x_gelu`, input **5888**, output **2048**.

### Actual feature widths

```text
Image
  ├─ primary Qwen2.5-VL vision encoder ─ image tokens: L × 2048 ─ identity projector ─┐
  │                                                                              │
  ├─ primary intermediate map ─ SimpleFP / ROIAlign ─ N × 2048 ─┐                  │
  └─ DaViT four stages ─ resize / ROIAlign ─ N × 3840 ───────────┤                  │
                                                             └─ concatenate      │
                                                                 N × 5888        │
                                                                 + box positions │
                                                                 region MLP      │
                                                                 N × 2048 ───────┤
                                                                                 ↓
                                                   shared new linear bridge: 2048 → 1024
                                                                                 ↓
                                             visual vectors + Qwen3.5 text embeddings
                                                                                 ↓
                                              Qwen3_5ForCausalLM.generate(inputs_embeds=...)
```

`L` varies with primary image preprocessing; `N` is the number of supplied/proposed boxes. DaViT stage channel counts are **256 + 512 + 1024 + 2048 = 3840**. Four primary FPN levels each contribute 512 channels, totaling **2048**. Concatenation produces **5888** region channels. The original region MLP is `Linear(5888,2048) → GELU → Linear(2048,2048)`. Its pretrained parameters remain part of FO1.

The new bridge is one **shared** `nn.Linear(2048,1024,bias=False)` for both global and region features: **2,097,152 parameters**, or **8 MiB** of float32 parameter storage. It is not a separate learned projection per region. An equal-width alternative decoder would receive an identity bridge; identity likewise would not establish feature alignment.

## 5. Complete request path

### 5.1 Validation and image loading

The CLI validates the task/prompt, numerical settings and manual-box JSON before expensive model loading. `validate_boxes()` requires a list of 1–100 boxes. Each box must contain four numeric, finite coordinates; strings, booleans, NaN, infinity, negative starts, zero area and reversed coordinates are rejected. Required ordering is `0 <= x1 < x2` and `0 <= y1 < y2`. After reading the image, right/bottom bounds must fit its width/height.

`load_input_image()` accepts a local image, PIL image, or HTTP(S) URL, applies EXIF orientation, and returns RGB. URL opening has a 30-second timeout. Standard inference is a **single-image request**. Lower-level prompt construction can retain conversation text/history with that image, but explicitly rejects more than one image. The submission does not expose a video or general batch-processing interface.

### 5.2 Proposal source

- **Manual JSON:** `--boxes` or `--boxes-file` bypasses UPN entirely. No UPN import, allocation or checkpoint download occurs.
- **Drawn regions in Gradio:** connected nontransparent brush components, after dilation with `disk(5)`, become rectangular bounding boxes. The original background image is sent to the model; brush strokes define proposals. These are user proposals, not segmentation predictions.
- **Automatic UPN:** the CLI requests UPN when no boxes argument is supplied; Gradio requires explicit selection. `UPNWrapper` is imported lazily and cached. CUDA availability is checked first. Scores are filtered at default **0.3**, at most **100** boxes are retained, coordinates are clipped to the image, and degenerate boxes are removed.

UPN's default checkpoint is `./resources/upn_large.pth`. Missing weights download only when `UPN_AUTO_DOWNLOAD=1` and an actual UPN request occurs. The source URL is the ChatRex GitHub `upn-large/upn_large.pth` release. Downloading uses a temporary file followed by atomic replacement; this is protection against an incomplete destination file, not a cryptographic authenticity check. UPN is a CUDA-only optional path in this submission and was not executed in the recorded validation.

### 5.3 Loading and initialization

`InferenceRuntime` resolves the device, checks quantization requirements, seeds the run and calls `load_pretrained_model()` with both checkpoint IDs and the bridge seed. `get_runtime()` caches one initialized configuration/model for subsequent demo requests.

The builder uses the vendored, flat `OmChatQwen25VLConfig`, verifies that `mm_vision_tower` exists, selects the tokenizer belonging to the **generating decoder**, and adds region tokens. In the Qwen3.5 path it constructs FO1 with `vision_only=True` before materializing checkpoint parameters. The old decoder's layers, token embedding, final text norm and language head are not loaded and then discarded; they are excluded from the retained model construction. Visual parameter names and dimensions remain those of the published checkpoint.

The retained FO1 stack contains **826 weight entries and 1,064,196,928 parameters**; the complete original index has **1,261 entries**. Loading rejects missing or shape-mismatched retained FO1 weights.

For Qwen3.5, `_load_qwen35_language_model()` uses the official Transformers **`Qwen3_5ForCausalLM`** class. Its checkpoint loading extracts the text configuration and remaps `model.language_model.*` into the text model's `model.*` namespace. The native Qwen3.5 vision tower is omitted. The loader rejects missing/mismatched decoder weights and reported loading errors. It increases the embedding table only when `len(tokenizer)` exceeds its existing row count, never shrinks it.

The supported builder uses `low_cpu_mem_usage=True` and explicit device maps. Omitting Qwen's unused native visual weights avoids approximately **384 MiB of float32 parameter allocation**. This is a parameter-storage saving, not a measured peak-process-memory reduction or a reduced checkpoint download size.

### 5.4 Tokenization and prompt assembly

`_ensure_qwen35_fo1_tokens()` adds `<region0>` through `<region99>` as special tokens. If needed, the pad token is set to EOS, or a pad token is created. The region token's numerical ID is not assumed to match the old tokenizer. A padded pretrained embedding table may already have rows at newly assigned IDs; those rows still have no trained meaning as FO1 region labels.

The two negative IDs in [constants.py](../vlm_fo1/constants.py) are internal placeholders:

| Marker | Internal value | Replacement |
|---|---|---|
| `<image>` | `IMAGE_TOKEN_INDEX = -200` | All projected image-token vectors for the image. |
| `<regionfeat>` | `DEFAULT_REGION_INDEX = -300` | One projected region vector. |
| `<regionN>` | Actual Qwen3.5 tokenizer vocabulary ID after addition | Ordinary text embedding for the region's index/name. |

Negative placeholders are removed during embedding construction; they are not looked up in the decoder's token embedding table.

`make_message_context()` wraps roles with ChatML, inserts `<|vision_start|><image><|vision_end|>` for the Qwen-backed pipeline and adds the sequence `<region0><regionfeat><region1><regionfeat>...` alongside the image. Task text follows. The region-feature list is created whenever boxes are present, independently of the CLI's optional `--region-tokens` flag. That flag additionally appends index labels to the task text; it does not enable feature extraction.

`prepare_inputs()` prepends a default `You are a helpful assistant.` system message when none exists. It resolves ChatML marker IDs through the active tokenizer. The default Qwen3.5 assistant prefix includes an empty, closed thinking block, `<think>\n\n</think>\n\n`, because `enable_thinking=False`. Task templates may contain their own requested answer format; the legacy reasoning template does not amount to a new reasoning-algorithm implementation.

### 5.5 Image preprocessing and coordinate transforms

`resize_shortest_edge_images_and_bboxes()` resizes images and boxes together. Without an explicit `image_size`, it preserves dimensions unless the longest edge exceeds **2048**; dimensions have a lower bound of **28** pixels. With `image_size`, the helper targets the shortest edge while enforcing the 2048 maximum. Resampling is bicubic, and x/y coordinates scale by their respective actual dimension ratios. The lower-level image loader preserves original dimensions so this paired resize remains responsible for small-image upscaling and coordinate conversion. That behavior was corrected on 17 September: previously the loader could upscale an image before its box coordinates were scaled. The recorded example is **500 × 399**, well above that boundary, and its evidence files were not changed.

The primary `Qwen2VLImageProcessor` produces flattened image patches and `image_grid_thw`. Its metadata specifies minimum pixels **3136**, maximum pixels **4,194,304**, patch size **14**, temporal patch size **2**, merge size **2**, mean `[0.48145466,0.4578275,0.40821073]` and standard deviation `[0.26862954,0.26130258,0.27577711]`.

The auxiliary branch uses the vendored CLIP-style image processor and ImageNet mean `[0.485,0.456,0.406]` / standard deviation `[0.229,0.224,0.225]`, with rescaling by `1/255` and no center crop. Boxes are converted to the auxiliary tensor's actual size via `adjust_bbox()`.

**Configured versus effective auxiliary sizing:** the FO1 JSON declares `aux_image_size=1024` and `aux_image_aspect_ratio="dynamic"`. However, the current inherited constructor passes the configuration as `args` to `DavitVisionTower`, whose initialization does not read these two fields and eagerly loads with defaults **768 × 768, `squash`**. The top-level builder applies the JSON settings only if `is_loaded` is false; this constructor already sets it true. The actual default path therefore uses the 768-square processor. A report must not claim that this path dynamically preprocesses at 1024 merely by reading the checkpoint JSON. This discrepancy was identified by source review; no preprocessing change or new accuracy claim was introduced for this reference.

### 5.6 Primary visual features and HFRE

`encode_images()` calls the primary tower and its original image projector. The primary custom forward collects intermediate states at blocks 7/15/23/31, restores spatial/window ordering and exposes feature maps. For the active `mm_use_simpleFPN_for_vt=true` path, `encode_regions()` passes the last gathered primary feature map to HFRE.

The auxiliary DaViT encoder returns its four spatial feature stages. HFRE interpolates them to the largest auxiliary stage size using bilinear interpolation, concatenates their channels and applies `torchvision.ops.roi_align` in float32 with spatial scale **1/4** and output **14 × 14**. Mean pooling across the ROI's spatial dimensions yields each 3840-channel auxiliary region vector.

The primary `SimpleFP` generates four 512-channel levels. ROIAlign uses feature strides **3.5, 7, 14, 28** and the same 14-square output, followed by mean pooling. Their concatenation yields a 2048-channel primary region vector. Coordinates for the primary branch are obtained from auxiliary-box coordinates using the primary grid size times the patch size relative to the auxiliary image tensor dimensions.

HFRE concatenates auxiliary and primary vectors, adds sine/cosine positional information based on normalized bounding-box center and size, and returns a 5888-channel vector per proposal. The pretrained region MLP converts it to 2048. These operations and their checkpoint parameters are inherited from FO1; this project does not retrain HFRE or claim a new region encoder.

### 5.7 Bridge, generation and output contract

`Qwen35FO1InferenceWrapper` creates its bridge under `torch.random.fork_rng(devices=[])` with `bridge_seed` (default **0**), so initializing the bridge is reproducible without consuming the caller's CPU RNG sequence. All wrapper parameters, including the retained encoders, new bridge and decoder, are frozen with `requires_grad_(False)` and placed in evaluation mode. Execution uses `torch.inference_mode()`.

`_bridge_features()` moves the bridge and features to the decoder embedding's device/dtype and checks both input and projected features for finite values. NaN/Inf causes a clear error before language generation. `_prepare_qwen_inputs()` embeds ordinary text using Qwen3.5's own embedding layer, replaces image/region sentinels with projected vectors and checks that image/region counts match the prompt. It supports left padding internally and creates the attention mask for the expanded embedding sequence. The public runtime sends one request/image at a time.

`generate()` calls the language model through `inputs_embeds`, with cache enabled. The standard interfaces use greedy generation (`do_sample=false`, temperature 0), default **256** maximum new tokens, and EOS/PAD IDs from the active tokenizer. Sampling arguments are passed only if sampling is explicitly requested at wrapper level. The wrapper preserves the underlying decoder's cache implementation, including Qwen3.5's hybrid attention state.

Because Transformers returns newly generated tokens when supplied only `inputs_embeds`, the wrapper prepends the **original logical prompt IDs**, not the longer expanded visual sequence. Runtime slicing by the original prompt length therefore selects the generated suffix correctly. `_PromptAwareStoppingCriteria` prepends that logical prompt before evaluating caller criteria. Returned generation objects with `.sequences` are handled as well as plain tensors.

Stage diagnostics report image encoding, region encoding and generation elapsed times to stderr. They do not provide accuracy/confidence estimates.

`seed_inference()` seeds Python, NumPy, PyTorch and CUDA RNGs, and disables cuDNN benchmarking. This supports repeatable initialization and controlled experiments; it is not a promise of bit-identical results across hardware, kernels or library versions, and deterministic algorithms are not globally forced.

An optional `bridge_state_path` or `VLM_FO1_QWEN35_BRIDGE_PATH` loads a bridge state with `torch.load(...,weights_only=True)`. Metadata identifies it as `external_state_unverified`; the code does not set `bridge_is_trained=true` merely because a file was provided. No such state was used in recorded results.

### 5.8 Parsing and persisted results

`parse_answer()` first reads `<ground>label</ground><objects><regionN>...</objects>` groups. If no such complete groups exist, it accepts bare `<regionN>` selections only when the entire trimmed output consists of region tags and whitespace, commas or square brackets. Prose mentions such as `Do not select <region0>.` do not become detections. Indexes are deduplicated per label, sorted, and retained only within the proposal list. Output boxes are taken from the **original input-coordinate proposal list**, not from a generated coordinate string or an internal resized tensor. There is no second detector or bounding-box regression stage after the language answer. This bare-tag restriction was refined on 17 September; the earlier empty-detection evidence files are unchanged.

CLI stdout contains one UTF-8 JSON document; diagnostic printing is redirected to stderr. `--json` writes the same result, and the default image location is `outputs/<image-stem>_result.jpg`. `--no-image` disables image saving. Image-write failures propagate and do not produce a success JSON. Empty detections produce an image without detection rectangles.

The result fields are:

| Field | Meaning |
|---|---|
| `image`, `prompt` | Input identifier and actual task text. |
| `proposal_source`, `num_proposals`, `proposals` | `user`/`upn`, proposal count and original-coordinate boxes. |
| `raw_output` | Decoded generated suffix before detection parsing. |
| `detections` | List of `{label, region_index, bbox}` for valid parsed indexes. |
| `model.vision_checkpoint`, `model.decoder_checkpoint` | Selected checkpoint paths/IDs. |
| `device`, `seed`, `do_sample`, `max_new_tokens` | Execution settings. |
| `bridge_is_trained`, `inference_status`, `limitations` | Explicit no-training/unverified-alignment metadata. |
| `inference_seconds` | Runtime prediction time, including preparation and generation, excluding runtime construction. |
| `elapsed_seconds` | CLI request time starting before image loading; includes proposals, model construction/loading and requested image saving. It is captured before JSON serialization/writing. |
| `output_image` | Present when an image was requested/saved. |

Gradio additionally returns `proposal_method` and interactive annotated/proposal images. `inference_status` for the replacement path records `backend="fo1_hfre_qwen35"`, `fine_tuning_performed=false`, bridge seed/state/alignment, untrained region-token alignment and `grounding_quality_validated=false`.

## 6. Compatibility changes for Transformers 5.2

| Problem encountered in integration | Current change and evidence |
|---|---|
| Public Transformers Qwen2.5 configuration became nested under `text_config`; FO1 expects flat attributes. | `omchat_qwen2_5_vl.py` imports its vendored `Qwen2_5_VLConfig`. Full architecture keys are compared with the published index. |
| Legacy token-ID/default configuration behavior differs. | Vendored configuration explicitly supplies missing pad/bos/eos attributes and uses the available RoPE validation API. |
| Default RoPE initializer moved out of the old registry. | `_default_rope()` implements the original full-head inverse-frequency formula, rather than silently changing rotary semantics. |
| Non-persistent rotary buffers can be allocated without valid contents after meta-device loading. | `_init_weights()` rebuilds vision/language rotary buffers with the Transformers initialization helpers; original inverse frequencies are also registered as non-persistent buffers. The regression poisons materialized buffers with NaNs and verifies recovery, weight preservation and finite vision output. |
| Transformers 5 tied-weight metadata uses a dictionary. | `_tied_weights_keys={"lm_head.weight":"model.embed_tokens.weight"}`; vision-only construction clears irrelevant text tying. |
| DaViT construction called `.item()` on a meta tensor while building a Python drop-path schedule. | `torch.linspace(...,device="cpu").tolist()` creates these scalar hyperparameters on CPU. |
| Importing a simple CLI loaded the entire model stack. | Model package exports are lazy; runtime imports heavyweight dependencies only when needed. Preflight deliberately exercises the main import chain without loading weights. |
| Original decoder treated DynamicCache as an indexable tuple. | The optional original path uses `get_seq_length()` when available, expands masks according to actual cached visual length and preserves text cache positions. Tests compare cached/full logits and execute tiny visual generation. |
| Stopping criteria and suffix slicing saw inconsistent lengths after visual token expansion. | The Qwen3.5 wrapper preserves the logical prompt prefix for both generated tensors and stopping criteria. |
| Input errors could surface late or image-saving failures could be swallowed. | Early shape/range validation, explicit single-image handling, finite-feature checks and propagated image-save errors. |

The optional original decoder is selected using CLI `--original-decoder`, or an empty `--llm-model-path` for the example script. The low-level builder's `llm_model_path=None` preserves the original decoder; the supported interfaces deliberately default to Qwen3.5. Only tiny actual-architecture tests validate that original-decoder path here; no complete pretrained baseline comparison was performed.

## 7. Device and precision policy

- `auto` resolves to CUDA when available, otherwise CPU. It does **not** choose Apple MPS automatically.
- CPU uses **float32**. CUDA selects bfloat16 when reported supported, otherwise float16. The loader accepts explicit `mps`, but MPS execution is not validated by the evidence.
- 4-bit and 8-bit options are mutually exclusive and require CUDA in the supported path. They use explicit `BitsAndBytesConfig` objects; optional GPU requirements pin `bitsandbytes==0.46.1`.
- Quantization settings are supplied to checkpoint loading for both retained FO1 and replacement decoder. This does not imply every operation or tensor becomes 4/8 bit; for example HFRE's ROI operations explicitly use float32 and the bridge follows the decoder embedding dtype.
- [attn_utils.py](../vlm_fo1/attn_utils.py) defaults to `sdpa`; `VLM_FO1_ATTN_IMPL` accepts `sdpa`, `eager`, or `flash_attention_2`. Explicit FlashAttention with a non-CUDA device is rejected. Docker does not install FlashAttention.
- Retained FO1 weights alone occupy approximately **3.964 GiB in float32**, derived from 1,064,196,928 parameters × 4 bytes. This excludes Qwen text weights, activations, buffers, images, libraries and the runtime. Severe memory pressure on a 4 GiB VM is therefore unsurprising; the recorded slow runs are not deployment benchmarks.

## 8. Interface contracts and configuration

### CLI and example script

`cli.py` accepts `--image` plus either `--prompt` or `--task`. Supported tasks are `od`, `count`, `ocr`, `brief-caption`, `detailed-caption`, `reason`, `grounding`, and `all`. All but `grounding` and `all` require `--target`. The task templates are inherited strings, not separately trained task heads.

Other controls include `--boxes` / `--boxes-file`, `--region-tokens`, `--threshold` (0–1), `--max-proposals` (1–100), `--model` / `--model-path`, `--llm-model-path` / `--original-decoder`, `--upn-ckpt`, `--device`, `--load-4bit` / `--load-8bit`, `--max-tokens` / `--max-new-tokens` (1–8192), `--seed` (0–4294967295), `--output`, `--json`, and `--no-image`.

`inference.py` defaults to `demo/demo_image.jpg`, label `orange`, and the seven boxes below. It forwards settings to the common CLI. Supplying another image requires `--bboxes-json`, avoiding accidental reuse of fruit coordinates:

```json
[[161,11,292,127],[268,61,428,226],[12,100,140,227],
 [205,188,332,320],[326,202,478,357],[136,106,269,233],[25,206,200,383]]
```

### Gradio

`create_demo()` constructs the page without loading checkpoints. It shows the configured decoder/device/token limit/seed and the untrained-alignment limitation. The fruit example is initialized with the seven manual boxes. `upload_defaults()` preserves those boxes when Gradio initializes that exact image; other uploads clear them and select drawn regions. Alternative examples do not receive fabricated proposal boxes.

The interface offers an image editor, proposal-source selector, JSON box input, task-template selector, prompt, UPN threshold, proposal visualization, selected-region image, raw text and result JSON. Submissions use `concurrency_limit=1`, and model loading is deferred to the first valid submission. `main()` uses `share=False`, default port **8000**, and default bind address `0.0.0.0` inside the container.

### Environment variables

| Variable | Role |
|---|---|
| `VLM_FO1_MODEL`, `VLM_FO1_LLM_MODEL` | Vision and decoder checkpoint selection. |
| `VLM_FO1_DEVICE` | `auto` for local runtime defaults; set to CPU/CUDA by the corresponding container configuration. |
| `VLM_FO1_LOAD_4BIT`, `VLM_FO1_LOAD_8BIT` | Boolean flags activated by the string `1`. |
| `VLM_FO1_MAX_TOKENS`, `VLM_FO1_SEED` | Default 256 and 0. |
| `VLM_FO1_ATTN_IMPL` | Attention backend, default SDPA. |
| `VLM_FO1_QWEN35_BRIDGE_PATH` | Optional externally supplied, unverified bridge state. |
| `UPN_CKPT`, `UPN_AUTO_DOWNLOAD` | UPN location and explicit download permission. |
| `PORT`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` | Browser server port/address; `PORT` has priority over the Gradio port variable. |
| `VLM_FO1_PORT` | Compose host-side port mapping, default 8000. |
| `HF_TOKEN`, `HF_HOME`, `HF_HUB_OFFLINE` | Hugging Face authentication/cache/offline operation. |

The base Compose file explicitly forwards its listed model/attention/UPN/token settings; other runtime variables can be supplied with `docker compose run -e NAME=value ...`. A host environment variable is not automatically inherited by a container unless configured or passed.

## 9. Dependencies and container packaging

[requirements.txt](../requirements.txt) pins Python dependencies: torch 2.6.0, torchvision 0.21.0, Transformers 5.2.0, accelerate 1.4.0, timm 1.0.9, Gradio 5.49.1, einops 0.8.1, NumPy 2.2.6, Pillow 11.3.0, requests 2.32.5, scikit-image 0.25.2, scikit-learn 1.7.2, matplotlib 3.10.6, mmengine 0.8.2 and ninja 1.13.0. Development requirements add pytest 8.4.2. GPU requirements add bitsandbytes 0.46.1. Report-building packages are separate and are not needed for inference.

[Dockerfile](../Dockerfile) has two targets:

| Target | Construction |
|---|---|
| `cpu` (last/default stage) | `python:3.10-slim-bookworm`; matching torch/torchvision CPU wheels; application requirements; no UPN custom CUDA extension. |
| `gpu` | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`; Python 3.10; CUDA 12.4 wheels; GPU/UPN requirements; compilation of `detect_tools/upn/ops` with `FORCE_CUDA=1`. Default compiler jobs 2; architecture list `7.0;7.5;8.0;8.6;8.9;9.0+PTX`. |

Both stages run `pip check`, use `/app`, and start `python demo/gradio_demo.py` through [docker/entrypoint.sh](../docker/entrypoint.sh). The entrypoint simply executes its arguments; it does not download weights. Build tooling pins pip 25.0.1, setuptools 75.8.0 and wheel 0.45.1.

[docker-compose.yml](../docker-compose.yml) exposes only `127.0.0.1:<host-port>:8000`, binds `./resources` and `./outputs`, persists Hugging Face cache in the `hf-cache` named volume, sets `shm_size: 2gb`, and uses `init: true`. The optional [GPU override](../docker-compose.gpu.yml) selects the GPU build/image, sets CUDA and reserves NVIDIA GPU devices. Its syntax was checked; actual NVIDIA execution was not.

[.dockerignore](../.dockerignore) excludes local virtual environments, Git/agent state, `.env` files, caches, weights, generated outputs, archives, temporary files and prebuilt native extensions. Image construction does not embed downloaded checkpoints. Normal `docker compose down` preserves the named model cache.

[scripts/package_submission.py](../scripts/package_submission.py) builds an allowlisted source/report ZIP rooted at `vlm-fo1-qwen35/`, requires the container entrypoint and build inputs, preserves source notices, excludes local secrets/caches/weights/symlinks, checks ZIP integrity, and writes a `.zip.sha256` sidecar. It requires the report at `output/pdf/project_report.pdf`. The optional legacy backup is not part of the submission ZIP.

## 10. Reproduction commands

These commands describe supported usage. They do not imply that every listed hardware path was tested.

```bash
# Local offline regression checks after dependencies are installed.
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python cli.py --check

# CPU Docker startup; page: http://localhost:8000
docker compose build
docker compose run --rm vlm-fo1 python cli.py --check
docker compose up

# Bundled image + seven manual boxes.
docker compose run --rm vlm-fo1 python inference.py \
  --max-new-tokens 32 --output outputs/example.jpg --json outputs/example.json

# Custom task with explicit manual regions.
docker compose run --rm vlm-fo1 python cli.py \
  --image demo/demo_image.jpg --task od --target orange \
  --boxes '[[161,11,292,127],[268,61,428,226]]' \
  --max-tokens 32 --output outputs/orange.jpg --json outputs/orange.json

# Full receiving-machine execution check, with a 16-token limit.
bash scripts/verify_container.sh cpu

# Optional NVIDIA path; requires configured NVIDIA host support.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
bash scripts/verify_container.sh gpu
```

`verify_container.sh` validates Compose, builds, saves a no-weight preflight, runs the real bundled inference with a **16-token cap**, then verifies the intended decoder, seven user proposals, nonempty raw text, untrained/limitation metadata, valid parsed region references and readable output image. It does **not** require any detection to exist or compare with ground-truth labels. GPU mode supplies `--load-4bit`.

For offline receiving-machine use, transfer a populated Hugging Face cache or complete local checkpoints and explicitly select both paths, for example:

```bash
docker compose run --rm -e HF_HUB_OFFLINE=1 vlm-fo1 python inference.py \
  --model-path /app/resources/VLM-FO1-3B-v01 \
  --llm-model-path /app/resources/Qwen3.5-0.8B \
  --max-new-tokens 16 --json outputs/offline-example.json
```

Initial downloads are approximately 8.9 GB for FO1 and 1.6 GB for Qwen3.5; optional UPN is approximately 875 MB. The README's 32 GB system RAM and 30 GB free-disk recommendation is a conservative planning estimate, **not a measured minimum**. Bash commands can be used on Windows through WSL2 with Docker's Linux-container integration.

## 11. Test inventory and what each test establishes

The recorded 15–16 September suite has **47 cases**, including the 16 malformed-box parametrizations. The table below identifies that recorded inventory; subsequent 17 September refinements add tests as described after the table. Tests do not download pretrained weights and do not compute accuracy metrics.

| Test file | Cases | Evidence established |
|---|---:|---|
| [test_cli.py](../tests/test_cli.py) | 29 | Malformed/finite/ordered/count/bounds validation; decoder defaults; validation before loading; task requirements; safe index parsing; manual proposals bypass UPN; stdout JSON; no-weight preflight; lightweight help; lazy UI; CPU UPN error; fruit-upload initialization; image-write failure handling. |
| [test_fo1_checkpoint.py](../tests/test_fo1_checkpoint.py) | 2 | Full published key-set comparison on meta device; retained parameter count; small save/load round-trip; poisoned rotary-buffer recovery; finite real primary-vision output and matching weights/features. |
| [test_qwen35_mechanics.py](../tests/test_qwen35_mechanics.py) | 9 | Frozen/reproducible bridge and preserved RNG; actual tiny Qwen3.5 hybrid-cache generation; finite-feature rejection; logical prompt-aware stopping; missing-region errors; left padding; single-image history; 100-region limit; image/box resizing. The visual features in wrapper tests are synthetic. |
| [test_qwen35_text_loader.py](../tests/test_qwen35_text_loader.py) | 3 | Exact decoder-weight remapping from a tiny full multimodal checkpoint, tied embeddings and no native vision; identical cached generation scores under left padding; rejection of deliberately removed pretrained decoder weight. |
| [test_original_decoder.py](../tests/test_original_decoder.py) | 2 | Actual tiny original vision/decoder generation using DynamicCache; finite scores and correct cache length; cached text logits matching full-sequence logits. |
| [test_package_submission.py](../tests/test_package_submission.py) | 2 | Required entrypoint/notice/build-file preservation, local-state exclusion and SHA256 output; refusal to package without the entrypoint. |

On 17 September, seven additional parser cases were added: three complete bare-tag selection formats and four negated/uncertain/quoted prose mentions. The existing structured-grounded and out-of-range-index test remains in place. These additions increase `test_cli.py` from 29 to 36 cases; they are not part of the unchanged 47-test historical evidence logs.

Two additional mechanics cases verify that PIL/file/base64/URL image loading preserves a small image's original size, and that preprocessing scales a small image's boxes together with the image (also checking an ordinary-size case and preserving caller boxes). These increase `test_qwen35_mechanics.py` from 9 to 11 cases. The combined source inventory is therefore **56 cases**; the historical 47-test logs remain unchanged. The 36-case CLI subset passed in **5.04 seconds** during this reference's parser refinement; the full final suite passed **56 tests in 9.30 seconds** on 17 September, with one upstream timm warning, and the no-download CPU preflight passed. See [final review tests](evidence/review-tests.txt) and [final review preflight](evidence/review-preflight.json).

The separate [smoke_qwen35_wrapper.py](../tests/smoke_qwen35_wrapper.py) is not automatically collected by pytest. It loads the real Qwen3.5 tokenizer/decoder through the shared loader and synthetic 2048-dimensional visual/region features, then runs generation. This tests real decoder execution with the wrapper without loading FO1's large visual stack. Its recorded success is distinct from both the tiny-model suite and complete pretrained inference. It is not a visual grounding test.

## 12. Recorded execution evidence, dates and exact outputs

The [validation record](validation.md) is the source of the dated test environment and command history. The JSON/text files linked below are the primary saved artifacts. The JSON does not itself contain timestamps or pinned revisions; those are recorded in the validation document. Do not silently re-date these runs to this reference's preparation date.

### 15 September 2026: container and native execution

The host was an Apple Silicon Mac with **8 GB physical RAM**. The successful Docker verification used Linux arm64 in an isolated VM with **2 CPUs, 4 GiB RAM and 6 GiB swap**, one inference thread, Python **3.10.21**, torch **2.6.0+cpu**, torchvision **0.21.0**, Transformers **5.2.0** and Gradio **5.49.1**. Its image identifier was:

```text
sha256:efbc8e816027bcc6469b4fe6589b8e93f50c05afe496c395d94bb42b8058971e
```

Native execution used Darwin/arm64, Python **3.12.3**, torch **2.6.0**, the same torchvision/Transformers/Gradio versions, **two CPU threads**, and the Docker VM **stopped**. See [native environment](evidence/native-environment.json).

| Recorded measure | Full Docker pipeline | Full native macOS pipeline |
|---|---:|---:|
| Maximum new tokens | **1** | **16** |
| Image/proposals | Fruit, 500 × 399; seven manual boxes | Same |
| Seed / generation | 0 / greedy | 0 / greedy |
| Primary image stage | 122.2 s | 44.8 s |
| Region stage | 135.6 s | 45.6 s |
| Generation stage | 96.0 s | 294.8 s |
| `inference_seconds` | **357.167** | **386.428** |
| `elapsed_seconds` including loading | **537.366** | **418.895** |
| Parsed detections | `[]` | `[]` |
| Saved image | Readable, 500 × 399, no detection rectangles | Readable, 500 × 399, no detection rectangles |

Docker's complete raw decoded output was:

```text
Based
```

Native execution's complete raw decoded output, preserving its line breaks and incomplete final word, was:

```text
Based on the visual analysis of the image provided:

1.  **Ident
```

Both are truncated execution samples, not completed answers to “Please detect orange in this image. Answer the question with object indexes.” Neither selected a region. The recorded pretrained runs validate that the real encoders, HFRE, projectors, bridge and decoder execute together and save outputs. They do not validate that the decoder understands the image or detects oranges.

Primary evidence:

- [Docker JSON](evidence/cpu-acceptance.json), [Docker image](evidence/cpu-acceptance.jpg), [Docker stage timings](evidence/docker-stages.txt).
- [Native JSON](evidence/native-acceptance.json), [native image](evidence/native-acceptance.jpg), [native stage timings](evidence/native-stages.txt).
- [CPU no-weight preflight](evidence/cpu-preflight.json): `status=ok`, `weights_loaded=false`, CPU and intended decoder.
- [Docker tests](evidence/docker-tests.txt): **47 passed in 34.95 seconds**, one upstream timm deprecation warning.

The test log does not turn the synthetic tests into 47 pretrained inference runs. The one-token Docker file retains the name `cpu-acceptance.json`; the cap in its content is authoritative. It is not the same as completing the receiving-machine script's 16-token default.

Earlier constrained Docker attempts with 32-token and 16-token limits were stopped during generation. They are not counted as successful acceptance. Read-only profiling of one earlier run observed that the visual stages had finished and generation progressed to 10 tokens. No numerical model changes were justified by the small arithmetic checks. Timings from different caps and thread/VM conditions are not a controlled performance comparison.

Browser verification established page startup on port 8000, configured Qwen3.5 controls, the fruit image and manual boxes. It did not establish a completed browser-driven inference run; an optional browser click test was blocked after the browser tool's usage limit was reached.

### 16 September 2026: cleanup verification

After removing unused files/dead code, the native offline suite again passed **47 tests in 9.64 seconds**, with the same timm warning, and the no-download preflight passed. See [cleanup test log](evidence/cleanup-tests.txt) and [cleanup preflight](evidence/cleanup-preflight.json).

The Docker image build and full pretrained execution were not repeated for that cleanup. The final source was verified through local regression/preflight checks; the Docker image hash identifies the earlier tested build. This distinction should remain visible in any report.

### 17 September 2026: final review and bounded fixes

The final review fixed premature tiny-image resizing and false detection parsing
from region tags mentioned in prose. The full offline suite passed **56 tests in
9.30 seconds**, with one upstream timm deprecation warning. The CPU preflight
passed with `weights_loaded=false`; dependency consistency and shell checks
also passed. These local checks do not re-date or replace the 15 September
Docker/full-pretrained evidence. No new Docker build or full-weight inference
was performed for the two edge-case fixes.

### 17 September 2026: subsequent 32-token laptop run

The current source subsequently completed native CPU output generation using the real checkpoints, default fruit image and seven boxes, seed 0 and two CPU threads. Request time was **231.785 seconds** and inference time **194.022 seconds**. Image/region/generation stages were 35.5/34.2/123.2 seconds. JSON and the 500 x 399 JPEG were verified. The answer incorrectly described a **television** in the fruit image, ended at the 32-token cap, and selected no regions. This is evidence of **failed qualitative detection**, alongside completed pipeline execution. It does not establish the exact cause experimentally; the known bridge and region embeddings remain untrained.

See [raw result](evidence/laptop-32-token.json), [image](evidence/laptop-32-token.jpg), [stage timings](evidence/laptop-32-token-stages.txt) and the [validation record](validation.md) for the command and resource observations. The outer macOS timing tool returned an unrelated sandbox clock-query error after output generation; a trivial successful Python command reproduced that instrumentation failure. Use the application's timings and distinguish saved application outputs from the outer wrapper status. No additional Docker or NVIDIA execution was performed.

## 13. Supported conclusions and unresolved coverage

The evidence supports these statements:

1. The default generation decoder is Qwen3.5-0.8B, while retained FO1 modules provide visual features.
2. A deterministic, frozen bridge makes the tensor widths compatible; no alignment training was performed.
3. CPU Docker builds and starts, and the full pretrained pipeline completed a one-token Docker request.
4. The same implementation completed a separate 16-token native CPU request.
5. Interfaces validate inputs, keep model/proposal metadata, preserve raw output, and report failures without fabricating detections.
6. The receiving computer has documented build/run/verification commands and persistent model/output storage.

Coverage that remains absent or limited:

- NVIDIA image compilation, CUDA inference, UPN inference and 4-/8-bit execution were **not run**. Compose configuration parsing is their only recorded deployment check.
- Actual container execution was Linux **arm64**; Linux x86_64, Windows/WSL and MPS were not separately executed in this evidence.
- No full pretrained original-decoder baseline, accuracy benchmark, ablation, robustness evaluation, adversarial evaluation or security certification was performed.
- Neither successful pretrained output contains a parsed object selection, and both stop at their configured token limit.
- The bridge and region-token semantics remain untrained; the code cannot turn this no-training migration into a validated replacement for pretrained FO1 grounding.
- Recorded runs use one bundled image and manually supplied proposals; they do not validate UPN quality or general input-image coverage.
- Normal downloads are not revision-pinned by the application; the recorded checkpoint hashes identify the tested revisions only.
- The effective DaViT processor settings differ from the top-level checkpoint's auxiliary sizing fields, as documented above.
- The repository still contains inherited configuration variants and training-related flags inside general model code. Their presence does not mean they were enabled, evaluated or trained for this submission.

A report can describe the implementation, reproducible commands, measured execution results and these limitations in detail. It cannot infer model accuracy, a successfully trained alignment, arbitrary-hardware performance, or personal contribution history beyond the documented work from those facts.
