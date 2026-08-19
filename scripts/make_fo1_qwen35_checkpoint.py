"""
Assemble a runnable VLM-FO1 checkpoint on top of a Qwen3.5 base, without any
training. The output folder mixes three sources of weights:

  1. the Qwen3.5 base model (language decoder + vision tower)
  2. the released FO1 checkpoint (DaViT aux tower + HFRE region encoder)
  3. the FO1 projectors, copied only when their shapes still match

Projector weights that do not match the new hidden sizes are left out on
purpose: they get randomly initialized at load time, the pipeline still runs
end to end, and retraining them is a separate (out of scope) step.

Usage:
  python scripts/make_fo1_qwen35_checkpoint.py \
      --base Qwen/Qwen3.5-4B \
      --fo1 resources/VLM-FO1_Qwen2.5-VL-3B-v01 \
      --out resources/VLM-FO1_Qwen3.5-4B
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def load_fo1_state_dict(fo1_dir):
    state = {}
    shards = sorted(glob.glob(os.path.join(fo1_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found in {fo1_dir}")
    for shard in shards:
        state.update(load_file(shard))
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--fo1", default="resources/VLM-FO1_Qwen2.5-VL-3B-v01")
    parser.add_argument("--out", default="resources/VLM-FO1_Qwen3.5-4B")
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    print(f"loading base model {args.base} (cpu, this can take a while)")
    base = AutoModelForImageTextToText.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    base_sd = base.state_dict()

    print(f"loading FO1 weights from {args.fo1}")
    fo1_sd = load_fo1_state_dict(args.fo1)

    new_sd = {}

    # 1) Qwen3.5 language decoder + lm_head. In the HF model the decoder lives
    # under model.language_model.*; in FO1 the decoder itself is the model, so
    # its layers sit directly under model.*
    for k, v in base_sd.items():
        if k.startswith("model.language_model."):
            new_sd["model." + k[len("model.language_model."):]] = v
        elif k == "lm_head.weight":
            new_sd[k] = v
        # 2) Qwen3.5 vision tower goes under the FO1 tower wrapper prefix
        elif k.startswith("model.visual."):
            new_sd["model.vision_tower.image_tower." + k[len("model.visual."):]] = v

    # 3) FO1 modules that do not depend on the backbone
    carried, skipped = 0, []
    for k, v in fo1_sd.items():
        if k.startswith(("model.vision_tower_aux.", "model.object_vp_extractor.")):
            new_sd[k] = v
            carried += 1
        elif k.startswith(("model.mm_projector.", "model.mm_projector_aux.")):
            # projectors are married to the old hidden sizes; keep them only
            # when the shapes happen to line up
            if k in new_sd and tuple(new_sd[k].shape) != tuple(v.shape):
                skipped.append(k)
            else:
                new_sd[k] = v
                carried += 1
    print(f"carried {carried} FO1 tensors, skipped {len(skipped)} projector tensors")
    for k in skipped:
        print(f"  skipped (shape mismatch): {k}")

    os.makedirs(args.out, exist_ok=True)
    save_file(new_sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})

    # config: start from the Qwen3.5 base and graft the FO1 keys onto it
    base_cfg = AutoConfig.from_pretrained(args.base).to_dict()
    with open(os.path.join(args.fo1, "config.json")) as f:
        fo1_cfg = json.load(f)

    cfg = dict(base_cfg)
    cfg["model_type"] = "omchat_qwen3_5"
    cfg["architectures"] = ["OmChatQwen35ForCausalLM"]
    for key, value in fo1_cfg.items():
        if key.startswith("mm_") or key.startswith("aux_image_") or key.startswith("tokenizer_"):
            cfg[key] = value
    # the dispatch in vlm_fo1 keys off these names
    cfg["mm_vision_tower"] = args.base if "qwen3" in args.base.lower() else "qwen3.5-vision"
    # main tower output width follows the new vision config
    vision_cfg = cfg.get("vision_config", {})
    if isinstance(vision_cfg, dict) and "out_hidden_size" in vision_cfg:
        cfg["mm_hidden_size"] = vision_cfg["out_hidden_size"]

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # tokenizer + image processor come from the Qwen3.5 base
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    except Exception as e:
        print(f"could not save processor ({e}); copy preprocessor_config.json manually")

    # generation config, if the base has one
    for fname in ("generation_config.json",):
        src = os.path.join(args.fo1, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fname))

    print(f"wrote {args.out}")
    print("smoke test it with: python scripts/smoke_test_qwen35.py --model", args.out)


if __name__ == "__main__":
    main()
