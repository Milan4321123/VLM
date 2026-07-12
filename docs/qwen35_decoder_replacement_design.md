# Notes before replacing the decoder

The task is to replace the Qwen2.5-VL-3B language decoder with
`Qwen/Qwen3.5-0.8B`. Training is not part of the task. Before changing the
code, I traced the original inference path to see which parts belong to
VLM-FO1 and which parts belong to the language model.

## How the original model works

The model is loaded in `vlm_fo1/model/builder.py`. The main class is
`OmChatQwen25VLForCausalLM` in
`vlm_fo1/model/language_model/omchat_qwen2_5_vl.py`.

For one image, the important path is:

```text
image -> Qwen2.5 vision tower -> mm_projector
image + boxes -> DaViT + HFRE -> mm_projector_aux
text -> tokenizer and Qwen2.5 token embeddings

all embeddings -> Qwen2.5 decoder -> lm_head -> output tokens
```

`encode_images()` creates the image features. `encode_regions()` creates one
feature for each bounding box. The method
`prepare_inputs_labels_for_qwen2_5_vl_multimodal()` inserts these features at
the `<image>` and `<regionfeat>` positions in the prompt.

The resulting sequence is passed to the normal Qwen2.5 decoder by `forward()`.
This is the point where the new decoder has to be connected.

## What I will keep

These parts are the VLM-FO1 visual pipeline and should not be replaced:

- the Qwen2.5 vision tower;
- the auxiliary DaViT tower;
- the HFRE region encoder;
- `mm_projector` and `mm_projector_aux`;
- the existing image and bounding-box preprocessing;
- `encode_images()` and `encode_regions()`.

Although the main vision tower contains “Qwen2.5” in its name, it is separate
from the text decoder. Keeping it is intentional because the assignment only
asks for the decoder replacement.

## What I will replace

Qwen3.5 will provide the text tokenizer, text embeddings, decoder layers,
final normalization and output head. Generation should therefore run through
Qwen3.5 instead of `OmChatQwen25VLForCausalLM.generate()`.

I do not want to rewrite the original vision code. A small wrapper can call the
existing image and region methods, prepare the mixed embedding sequence and
then call Qwen3.5 with `inputs_embeds`.

## Dimension mismatch

The original VLM-FO1 projectors output features with width 2048. Qwen3.5-0.8B
uses a text hidden size of 1024. Passing the old features directly to the new
decoder would fail because the tensor dimensions do not match.

I will add this bridge:

```python
nn.Linear(2048, 1024, bias=False)
```

Both image features and region features will pass through it before they are
inserted into the Qwen3.5 input embeddings.

## Tokenizer changes

The current prompt code contains token IDs that are specific to Qwen2.5. These
IDs cannot be reused with another tokenizer. They need to be looked up from the
active tokenizer instead.

VLM-FO1 also uses `<region0>` to `<region99>`. If these tokens do not exist in
the Qwen3.5 tokenizer, I will add them and resize the embedding table.

The negative IDs used internally for `<image>` and `<regionfeat>` are not sent
to the embedding table. The wrapper will continue to use them only as markers
for where visual features are inserted.

## Expected limitation

The new 2048-to-1024 bridge and the new region-token embeddings will be
randomly initialized. Since I am not training them, the program may run while
still producing poor or ungrounded answers. For this task, the result shows
that the two architectures are connected correctly; it does not show that the
new combination has the same accuracy as the trained original model.
