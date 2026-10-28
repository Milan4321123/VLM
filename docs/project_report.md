# Adapting VLM-FO1 to Qwen3.5

## Model integration and Docker-based deployment

**Praktikum KI und Sicherheit in der Praxis**
**Summer Semester 2026 · Individual project**
**Milan Adhikari · Matrikelnummer 2794822**
**Report revised on 18 September 2026**

## 1. Introduction and project scope

My task was to adapt the supplied VLM-FO1 project to Qwen3.5 and make it possible for another person to build and run the application with Docker. I worked on the project individually. The scope communicated to me did not include fine-tuning, so I focused on integrating the replacement model, making the application interfaces consistent, and preparing the software for submission.

The project topic, *Fine-grained Perception in VLMs*, concerns how a vision-language model connects a question to particular regions of an image. The general assignment includes transferring the VLM-FO1 approach to another small model and then performing Region-Language Alignment Training. My individual assignment omitted that training stage. The distinction matters: the implementation described here provides an inference path, but it does not produce a newly aligned perception model. The course also requires complete source code, Dockerized software, and a PDF report with at least five pages of text [1].

I assessed the implementation through three practical questions. Do the application interfaces load the intended Qwen3.5 decoder? Do the image and region features reach that decoder through a consistent tensor interface? Can another person follow the documented commands and obtain saved outputs? These questions concern whether the software runs as intended. Whether it selects the correct objects is a separate part of the evaluation.

The report follows this sequence from model loading to saved output. It explains what I reused, what changed, and what the recorded tests show. More detailed file and function references, settings, and evidence locations are provided in `docs/implementation_reference.md`. 

## 2. Starting point and contribution

The supplied repository already contained the VLM-FO1 implementation, example images, a command-line program, Gradio demos, and an experimental Qwen3.5 wrapper. I did not develop the original model or its region encoder. My work builds on those components and their pretrained weights. The upstream repository remains a reference for the original approach, and I do not present its published results as results of this adaptation [2].

VLM-FO1 combines a primary Qwen2.5 vision encoder with an auxiliary DaViT encoder. Its Hybrid Fine-grained Region Encoder, or HFRE, uses features from these encoders together with bounding boxes to represent individual regions. The existing projectors map the image and region representations into the embedding space of the original language model.

I retained this visual path while changing the language decoder. Using Qwen3.5’s native image processor instead would have introduced a different visual path and would not have demonstrated the same integration.

The initial branch had several inconsistencies. The example script requested Qwen3.5, but other entry points could still load the original decoder. The experimental wrapper already allowed a projection from 2048 to 1024 dimensions, although the projection was untrained. The Docker setup also depended on optional GPU components during startup. As a result, it was not always clear which model a particular command would run or which dependencies were actually necessary.

The completed implementation uses one shared runtime for the supported entry points. It also adapts checkpoint loading to the pinned Transformers version and loads the retained visual components without allocating the old language decoder. Other changes cover input validation, JSON and image output, separate CPU and NVIDIA configurations, and regression tests. The final review addressed two additional problems: coordinate scaling for very small images and the interpretation of region tags in ordinary prose.

I also cleaned up the active project directory. Unused SAM3 demos, evaluation datasets, superseded scripts, old result images, Colab requirements, and the two original patch files were moved into a local backup. The 26 archived files were checked against their SHA256 hashes before being removed from the active copy. That backup is excluded from both Docker and the submission. The supported model code, optional UPN proposal model, four demo inputs, tests, and execution evidence remain included. 

## 3. Implementation of the Qwen3.5 integration

### 3.1 Entry points and shared runtime

The application has three supported entry points. `cli.py:main()` accepts a general image request, `inference.py:main()` runs the bundled fruit example, and `demo/gradio_demo.py` provides the browser interface. All three use `vlm_fo1/runtime.py`.

The shared runtime keeps the configuration and prediction logic in one place. `RuntimeConfig` contains the checkpoint paths, device selection, quantization options, token limit, and seed. `InferenceRuntime` loads the model and performs prediction. The Gradio application uses `get_runtime()` to reuse a configured runtime instead of loading the model for every request.

The replacement checkpoint is `Qwen/Qwen3.5-0.8B`. The visual checkpoint defaults to `omlab/VLM-FO1_Qwen2.5-VL-3B-v01`, which is a Hub alias for `omlab/VLM-FO1-3B-v01`. These checkpoints serve different purposes: VLM-FO1 supplies the retained visual components, while Qwen3.5 generates the answer. Local checkpoint directories can also be selected. The original decoder is still available through the CLI option `--original-decoder`, but it is no longer the default.

The CLI validates the request before loading the model weights. It accepts a local image or an HTTP(S) image URL, a free-form prompt or task template, and bounding boxes supplied as JSON or in a JSON file. The templates cover detection, counting, OCR, captions, reasoning, grounding, and selecting all objects. They provide different prompts; their presence does not mean that accuracy has been measured for every task. Result JSON is written to standard output, while diagnostic messages go to standard error.

The Gradio page can start without downloading or allocating the model weights. This allows the user to open the interface, inspect the example, and enter boxes before requesting inference. Requests are processed one at a time, and the loaded runtime is reused. Selecting another image clears the example boxes so that coordinates from the fruit image are not applied to an unrelated input. 

### 3.2 Checkpoint loading and retained components

The main integration function is `load_pretrained_model()` in `vlm_fo1/model/builder.py`. It loads the vendored `OmChatQwen25VLConfig`, the tokenizer belonging to the selected decoder, and the retained VLM-FO1 components.

When Qwen3.5 is selected, the function constructs `OmChatQwen25VLForCausalLM` with `vision_only=True`. This leaves out the original language layers, text embeddings, final text normalization, and language-model head before the checkpoint is materialized. The implementation therefore avoids loading the old decoder only to discard it afterwards.

The retained stack contains 826 state entries and 1,064,196,928 parameters. A structural regression test compares the complete original architecture’s 1,261 state entries with the published checkpoint index [3]. Only entries belonging to the original decoder are removed in the replacement path. The primary vision encoder, DaViT, HFRE, and the original image and region projectors continue to load their pretrained tensors. Missing or mismatched required weights cause an error rather than allowing inference with an unintentionally partly random model.

The replacement decoder is loaded by `_load_qwen35_language_model()`. This function uses Transformers’ `Qwen3_5ForCausalLM` class, reads the text configuration, and maps the language weights from the published multimodal checkpoint. Qwen3.5’s native vision tower is not allocated because visual features come from the retained VLM-FO1 stack. The decoder uses a hidden dimension of 1024 [4]. Tests check the language-weight remapping, tied embeddings, and matching cached generation scores when loading through the text-only class.

Leaving out the unused Qwen3.5 vision tower avoids approximately 384 MiB of float32 parameter allocation on CPU. This figure describes omitted weight storage, not total process memory. The published checkpoint files still need to be downloaded in full.

CPU inference uses float32. On CUDA, the application selects bfloat16 where supported and float16 otherwise. Optional 4-bit and 8-bit loading is configured through `BitsAndBytesConfig` and is restricted to CUDA in the supported application. 

### 3.3 Images, bounding boxes, and HFRE

Each request contains one RGB image and between 1 and 100 proposed regions. A region is represented as `[x1, y1, x2, y2]` in the original image’s pixel coordinates.

`validate_boxes()` checks the structure, numeric values, area, and image boundaries. It rejects reversed coordinates, non-finite values, booleans, strings, out-of-bounds boxes, and excessive proposal counts. `load_input_image()` applies EXIF orientation before checking the boxes against the resulting image dimensions.

Boxes can be supplied directly, drawn in Gradio, or generated by the optional UPN model. The CPU path uses supplied or drawn boxes. Automatic UPN proposals require CUDA and the compiled deformable-attention extension [6]. The `_load_upn()` function imports and loads UPN only when it is requested, so the optional dependency does not block normal CPU startup. A missing UPN checkpoint is downloaded only when `UPN_AUTO_DOWNLOAD=1` is explicitly enabled. Generated proposals are clipped to the image and validated before use.

`InferenceRuntime.predict()` creates a user message containing the image, prompt, and `bbox_list`. The `prepare_inputs()` function in `vlm_fo1/mm_utils.py` then prepares the tensors for generation.

Image resizing and coordinate scaling have to remain synchronized. `resize_shortest_edge_images_and_bboxes()` transforms them together, limits the largest image side to 2048 pixels, and enforces a minimum dimension of 28 pixels. During the final review, a 14 × 14 image exposed a problem in an earlier loading step: `load_image()` enlarged the image before its boxes were scaled. The loading step now preserves the original dimensions, leaving both transformations to the paired resize function. Regression tests cover PIL, file, base64, and URL loading, a tiny full-image box, and an unchanged regular-size example.

The two vision encoders use different processors. The primary Qwen2.5 processor returns patch values and `image_grid_thw`, which describes the patch grid. In the inherited initialization path, the auxiliary DaViT processor produces a squash-resized 768 × 768 image. Some top-level checkpoint metadata contains different auxiliary settings, but these are not the settings used by that initialization path. I kept the existing behavior rather than changing the visual computation during this adaptation.

`adjust_bbox()` maps coordinates into the auxiliary processor’s dimensions. The primary feature grid is also taken into account when `encode_regions()` extracts region features.

The primary image path runs through `OmChatQwen25VLForCausalLM.encode_images()`, which retains intermediate features. `encode_regions()` runs the auxiliary path and passes the features and boxes through `HFREModule` in `hybrid_finegrained_region_encoder.py`. HFRE performs spatial region extraction, including ROI operations and position information, and combines features from both encoders. The existing projectors then produce image and region vectors with 2048 components.

The bounding boxes therefore affect the extracted visual features. They are not simply inserted into the language prompt as coordinate text. 

### 3.4 Bridge, tokenization, and generation

`Qwen35FO1InferenceWrapper`, implemented in `vlm_fo1/model/language_model/qwen35_fo1_wrapper.py`, connects the retained visual features to Qwen3.5.

The wrapper’s `feature_bridge` is a bias-free linear layer that maps 2048-dimensional vectors to 1024 dimensions. It contains 2,097,152 weights and is initialized with a fixed seed, which defaults to zero. All model parameters are frozen with `requires_grad_(False)`, evaluation mode is enabled, and prediction runs under `torch.inference_mode()`. There is no optimizer step, LoRA adaptation, or alignment training.

This bridge solves the dimensional mismatch, but it does not solve the alignment problem. Its weights have not learned how the old visual representations should be expressed in Qwen3.5’s embedding space. The added tokens `<region0>` through `<region99>` also have no trained region meanings.

`_ensure_qwen35_fo1_tokens()` gives these tokens stable tokenization and enlarges the embedding table only when needed. Qwen’s padded vocabulary may already have rows at the assigned IDs, but an existing row does not automatically represent the intended region.

Prompt construction uses the Qwen3.5 tokenizer and ChatML-style role markers. `make_message_context()` associates each numbered region tag with a `<regionfeat>` placeholder. Image and region-feature placeholders are converted to the internal sentinel IDs `-200` and `-300`. These are not normal vocabulary indices.

The wrapper replaces the sentinels with projected visual vectors, combines them with ordinary text embeddings, and constructs the attention mask for the resulting sequence. Where appropriate, `_assistant_generation_suffix()` adds Qwen3.5’s non-thinking assistant suffix, including an empty thinking block.

Generation is then called with `inputs_embeds`. The wrapper preserves left-padding behavior and restores the original prompt-ID prefix in the returned sequence because the existing callers use that prefix length to extract newly generated tokens. `_PromptAwareStoppingCriteria` presents the same logical prefix to the stopping criteria.

Tests cover small instances of Qwen3.5’s actual linear-attention and full-attention layers, cached decoding, left padding, and stopping behavior. These tests check the integration with the architecture; they do not replace full pretrained visual inference.

Before generation, `_bridge_features()` rejects non-finite visual features or projected embeddings. Progress messages identify the image-processing, region-processing, and generation stages. The default settings use greedy decoding, seed zero, and a configurable limit of 256 new tokens. The fixed seed makes bridge initialization repeatable, although it does not guarantee identical arithmetic across devices and software implementations. 

### 3.5 Compatibility fixes and output interpretation

The project combines a vendored Qwen2.5 implementation with Transformers 5.2.0, which provides Qwen3.5 support [5]. Several changes were necessary at this boundary.

The retained model continues to use its original flat configuration rather than assuming the newer nested public configuration. The original rotary-position frequency formula is initialized explicitly, tied-weight declarations use the newer dictionary format, and DaViT’s initialization schedule is constructed on CPU so that it also works during meta-device loading. The optional original decoder was adapted to the current cache API and preserves cache positions during generation.

A checkpoint reload test revealed a problem that was not visible from parameter names alone. Non-persistent rotary buffers could be allocated by the newer loading lifecycle without having their mathematical values restored. Their initialization is now explicit. The regression test deliberately fills materialized buffers with NaNs, checks reconstruction of the frequencies, verifies saved weights, and runs a small real vision forward pass.

The output parser also needed care. `runtime.py:parse_answer()` maps recognized region selections back to the supplied proposals. It supports the original `<ground>...</ground><objects>...</objects>` format and bare lists of region tags.

During the final review, the bare-tag fallback was restricted to answers containing only tags and permitted separators. This prevents a sentence such as `Do not select <region0>.` from being interpreted as a positive detection. Out-of-range indices are ignored, repeated references are deduplicated, and other prose remains available in `raw_output` without being converted into boxes.

The saved JSON contains the prompt, proposals, checkpoint identifiers, device, seed, token cap, timing, parsed detections, and explicit training and quality metadata. The application also saves an image showing any parsed selections. An image-write failure is reported as a failure rather than silently ignored.

An empty detection list means that the parser found no usable region selection. It does not establish that the requested object is absent from the image. 

## 4. Docker deployment and reproduction

The default Docker stage uses `python:3.10-slim-bookworm`. It installs CPU PyTorch 2.6.0 and torchvision 0.21.0, followed by the pinned dependencies in `requirements.txt`, including Transformers 5.2.0 and Gradio 5.49.1. The build runs `pip check` to check dependency consistency.

The source is copied into `/app`, and `docker/entrypoint.sh` executes the requested command. By default, the container starts `demo/gradio_demo.py`. FlashAttention, SAM3, and the optional UPN CUDA extension are not required by the CPU image.

`docker-compose.yml` selects the CPU stage and exposes the browser interface at `127.0.0.1:8000` on the host. Inside the container, the application listens on `0.0.0.0` so that the port mapping works.

From the extracted project directory, the basic procedure is:

```bash
docker compose build
docker compose run --rm vlm-fo1 python cli.py --check
docker compose up
```

The interface is then available at `http://localhost:8000`. The `--check` command checks imports and device selection without loading model weights.

The bundled command-line example uses the fruit image and seven supplied boxes:

```bash
docker compose run --rm vlm-fo1 python inference.py \
  --max-new-tokens 32 --output outputs/example.jpg \
  --json outputs/example.json
```

The model weights are not included in the source ZIP or application image. The first inference requires internet access to download approximately 8.9 GB of VLM-FO1 files and 1.6 GB of Qwen3.5 files. Compose stores Hugging Face downloads in the named `hf-cache` volume and mounts `resources/` and `outputs/` from the host. This keeps the results accessible outside the container and avoids downloading the weights for every run. Normal `docker compose down` preserves the cache. Complete local checkpoint directories can also be selected for offline execution.

The README gives a planning allowance of 32 GB system RAM and 30 GB free disk for CPU deployment. These are conservative estimates, not measured minimum requirements. The 8 GB development Mac was sufficient for constrained execution checks, but memory pressure made some runs very slow.

On Windows, the documented Bash commands assume WSL2 and Docker Desktop’s Linux-container integration. The Docker measurements reported here were obtained on Linux arm64. I did not separately execute the project on x86_64.

An optional GPU stage uses `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, matching CUDA PyTorch wheels, `requirements-gpu.txt`, and UPN’s requirements. It also compiles the UPN CUDA extension. The override file `docker-compose.gpu.yml` selects this stage and requests an NVIDIA device. I checked the configuration, but no NVIDIA GPU was available for execution or quantization testing.

The project includes a receiving-machine verification script:

```bash
bash scripts/verify_container.sh cpu
```

This script builds the image, checks imports, requests real inference with a 16-token limit, and validates the saved files. Its `gpu` mode adds the GPU override and 4-bit loading. The execution evidence available for these configurations is described separately below. 

## 5. Verification and observed results

### 5.1 Environment and checks

The development host was an Apple Silicon Mac with 8 GB of physical RAM. Native execution used Python 3.12.3, PyTorch 2.6.0, torchvision 0.21.0, Transformers 5.2.0, and Gradio 5.49.1.

The CPU container ran Linux arm64 with Python 3.10.21 and PyTorch 2.6.0+cpu. For the successful container inference check, the isolated Linux VM had two CPUs, 4 GiB RAM, and 6 GiB swap. This was a constrained test setup rather than the recommended deployment configuration.

The checks were performed at different stages of the project. On 15 September, the CPU image built successfully and its offline suite passed 47 tests in 34.95 seconds. The complete pretrained pipeline then ran inside that container with a one-token generation cap. After shutting down the VM, I performed a separate native run with a 16-token cap.

Following cleanup on 16 September, 47 local tests passed in 9.64 seconds. The final review on 17 September added regressions for small-image coordinate scaling and answer parsing. The complete suite then passed 56 tests in 9.30 seconds, and the final no-download preflight also passed.

| Check                              | Recorded result                                                                                                                                          |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU Docker build                   | Passed on Linux arm64 on 15 September, including the dependency consistency check.                                                                       |
| Container startup and preflight    | Passed without downloading weights. The browser page and example controls were inspected.                                                                |
| Pretrained checkpoint loading      | Completed for the retained VLM-FO1 components and Qwen3.5 text decoder.                                                                                  |
| Full pretrained Docker request     | Completed with a one-token cap and saved JSON and a readable image.                                                                                      |
| Full pretrained native request     | Completed with a 16-token cap and saved JSON and a readable image.                                                                                       |
| Final offline regression suite     | Passed 56 tests in 9.30 seconds.                                                                                                                         |
| Repeated execution and portability | Separate Docker and native runs and repeated offline tests were recorded. No repeated full 16-token Docker run or NVIDIA/x86_64 execution was completed. |

The tests cover malformed boxes, lazy startup, CLI behavior, JSON and image-write failures, coordinate scaling, prompt history, attention masks, bridge initialization, generation, stopping criteria, checkpoint structure, checkpoint reloads, and package contents. Several use small models with the actual architecture so that they can run offline. The expected upstream `timm` deprecation warning did not indicate a test failure.

The final review on 17 September did not include a Docker rebuild or a repeat of the earlier long full-weight runs. Its two behavioral fixes concerned tiny-image coordinate registration and region tags appearing in prose. The earlier 500 × 399 fruit image and the recorded tag-free responses did not exercise those cases. I therefore keep the earlier execution measurements separate from the later regression results. Commands, the tested image identifier, and logs are recorded in `docs/validation.md` and `docs/evidence/`. 

### 5.2 Input, prompt, and saved outputs

Both initial pretrained checks used `demo/demo_image.jpg`, a 500 × 399 image containing oranges and apples. The prompt was:

> Please detect orange in this image. Answer the question with object indexes.

The seven proposals were supplied directly rather than generated by UPN:

```text
[[161,11,292,127], [268,61,428,226], [12,100,140,227],
 [205,188,332,320], [326,202,478,357], [136,106,269,233],
 [25,206,200,383]]
```

The cached checkpoints used VLM-FO1 revision `15788f0d303509abacba5f97b52c9ff625080319` and Qwen3.5 revision `2fc06364715b967f1860aea9cf38778875588b17`. Both requests used seed zero and greedy generation.

The Docker run returned exactly:

```text
Based
```

Its inference time was 357.167 seconds. The reported total request time, including checkpoint loading, was 537.366 seconds. Image encoding took 122.2 seconds, region encoding 135.6 seconds, and generation 96.0 seconds.

The native run returned the following incomplete text:

```text
Based on the visual analysis of the image provided:
1. **Ident
```

Native inference took 386.428 seconds, with a total request time of 418.895 seconds including loading. Image encoding took 44.8 seconds, region encoding 45.6 seconds, and generation 294.8 seconds.

The saved JSON records `max_new_tokens` as 1 for Docker and 16 for the native run. Both responses were truncated by these limits. The native measurement is not a 16-token Docker result, and the different token caps prevent a meaningful speed comparison between the two environments.

Neither answer contained a usable region selection. Both detection lists were empty, and neither output image contained detection rectangles. These examples show that the visual processing, bridge, decoder, decoding, and file writing completed. They do not show that the model identified an orange or connected a region tag to the correct box. 

#### Follow-up laptop test on 17 September

I subsequently ran the reviewed source directly on the same 8 GB Mac, using two CPU threads and a 32-token cap. The existing weights were reused offline. The request completed in 231.785 seconds including loading and produced a valid JSON file and a 500 × 399 JPEG. The image, region, and generation stages took 35.5, 34.2, and 123.2 seconds respectively.

Around 28 GiB of disk space was free at the start, and approximately 18 GiB remained at the lowest observed check. Whole-system swap usage reached about 12 GiB while other applications were also running. This is an observation about the laptop during execution, not an isolated measurement of the model’s memory use.

The longer answer exposed a clear detection failure: it described the fruit image as displaying a “television.” The response still ended at the token cap and selected no regions. The application ran and saved its outputs, but the answer was wrong.

This result is consistent with the missing alignment of the bridge and region tokens, although a single example does not establish that this was the only cause. More memory would allow less constrained execution, but it would not train the connection between the visual features and the decoder. The raw answer and image are saved as `docs/evidence/laptop-32-token.json` and `docs/evidence/laptop-32-token.jpg`.

An optional external timing tool reported an instrumentation error after the application had written its outputs. The execution sandbox denied access to a system clock query. A separate trivial Python command reproduced the same timing-tool error despite completing successfully. I therefore report the application’s recorded times and do not describe the outer timing wrapper as having exited cleanly. This follow-up was a native macOS run, not another Docker or NVIDIA test. 

### 5.3 Failures and remaining coverage

Two earlier container runs were stopped during generation because of severe slowdown on the 8 GB host. One used a VM with 6 GiB RAM and a 32-token cap; the other used a 4 GiB VM and a 16-token cap. Neither is counted as a successful execution result.

The successful Docker check used a one-token cap to keep the memory-constrained verification manageable. For the separate 16-token native run, I first shut down the VM to free its memory.

The evidence does not include a second image with complete pretrained output, a labeled test set, or a measured comparison with the original decoder’s perception quality. Original-decoder compatibility was checked through small real-architecture cache and generation tests, not through a full pretrained baseline benchmark.

The optional browser click check was also blocked by the browser tool’s usage limit. Page inspection and automated input tests completed, but I did not complete a full browser-driven model request. 

## 6. Limitations and interpretation

The main unresolved issue is the alignment between the retained visual representation and Qwen3.5. The pretrained visual components keep their stored parameters, but the new bridge and region-token meanings were not trained for the replacement decoder. Correct tensor shapes and finite values are necessary for execution, but they are not enough for correct region interpretation. The same applies to fluent output: a sentence that sounds plausible may still be unrelated to the image.

The runtime records this limitation explicitly:

```text
fine_tuning_performed: false
bridge_is_trained: false
grounding_quality_validated: false
```

The model change should not be described as an improvement over VLM-FO1. There is no accuracy comparison, matched speed experiment, or training result that would support that conclusion. Similarly, omitting unused weight tensors avoids a particular allocation but does not establish a general memory advantage for the complete application.

A quality comparison would require the same images and proposals for both decoders, recorded ground-truth selections, and an evaluation procedure that also accounts for invalid and empty answers. That evaluation was not completed in this project.

There are also practical limits to reproducibility. The report and saved outputs record the core package versions, model identities, seeds, decoding settings, and execution environment. Some transitive dependencies and Docker base tags can still change. A fixed seed and greedy decoding do not guarantee bit-identical results on every device. Available RAM, image resolution, generation length, and checkpoint downloads also affect whether a run is practical on a particular machine.

For the agreed no-fine-tuning scope, I consider the CPU integration suitable for submission with these limitations documented. It includes the source, Docker instructions, checked interfaces, regression tests, and limited full-model execution evidence. This does not establish detection quality or guarantee execution on an untested receiving machine. NVIDIA execution, longer Docker responses, and systematic quality evaluation remain unverified. 

## 7. Conclusion and assistance used

The project connects the retained VLM-FO1 vision encoders and HFRE to `Qwen/Qwen3.5-0.8B` through a frozen 2048-to-1024 bridge. The CLI, bundled example, and Gradio demo now use the same runtime. The work also adapts checkpoint loading to Transformers 5.2.0, separates CPU and optional NVIDIA deployment, checks invalid inputs and missing weights, and saves outputs that can be inspected.

The recorded runs show that the integration can execute, but they do not demonstrate useful object selection. The initial responses were incomplete and contained no parsed detections. The longer native run produced an incorrect description of the fruit image. The outcome is therefore a working software integration with a clearly unresolved perception problem, not a trained replacement for the original model.  

I worked individually using the supplied repository and pretrained checkpoints. I used AI coding-assistant support to inspect and modify code, develop and run checks, prepare the submission package, and draft and revise this report. The original architecture, pretrained models, and upstream implementation remain attributed to their authors. No fine-tuning was performed. 

## References

**[1]** Oren Halvani. *Praktikum KI und Sicherheit in der Praxis, SoSe 2026*, version 1.4. Course handout supplied with the assignment. Submission requirements: PDF pages 7–8. Project topic: PDF pages 23–25.

**[2]** Om AI Lab. *VLM-FO1 source repository and project description.*
`https://github.com/om-ai-lab/VLM-FO1`
Implementation details also refer to the supplied and modified local source files identified in the report.

**[3]** Om AI Lab. *VLM-FO1-3B-v01 checkpoint metadata*, revision `15788f0d303509abacba5f97b52c9ff625080319`.
`https://huggingface.co/omlab/VLM-FO1-3B-v01`
The local structural-test fixture records the same revision.

**[4]** Qwen Team. *Qwen3.5-0.8B model card and configuration.*
`https://huggingface.co/Qwen/Qwen3.5-0.8B`
Accessed 17 September 2026. Execution used the revision recorded in Section 5.

**[5]** Hugging Face. *Transformers Qwen3.5 documentation* and the installed Transformers 5.2.0 implementation.
`https://huggingface.co/docs/transformers/en/model_doc/qwen3_5`
The online documentation may differ from the pinned implementation used in this project.

**[6]** IDEA Research. *ChatRex and UPN implementation.*
`https://github.com/IDEA-Research/ChatRex`
Optional proposal-model code is retained under `detect_tools/upn/`.

*Bibliographic details retained from the supplied report.* 
