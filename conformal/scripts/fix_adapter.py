"""Rename adapter keys to match the base-model path at inference time.

Background:
  The training-time model class was Qwen3_5ForCausalLM whose .model attribute
  had a `.language_model` wrapper holding the decoder layers. So the saved
  adapter keys look like:
    base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight

  At inference time, AutoModelForCausalLM loads Qwen3_5ForCausalLM whose .model
  is Qwen3_5TextModel with .layers directly (no .language_model intermediate).
  PEFT's path matching fails silently — it creates LoRA slots but doesn't load
  the saved weights, so LoRA delta == 0 and the model behaves like the base.

  Fix: drop the `.language_model.` segment from each saved key.

  This produces a renamed adapter at <src>_renamed/.
"""
import os
import shutil
import argparse
from pathlib import Path
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls/checkpoint-200")
    p.add_argument("--dst", default="/mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls/checkpoint-200-renamed")
    p.add_argument("--strip", default="language_model.", help="path segment to remove from keys")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    weights_file = src / "adapter_model.safetensors"
    print(f"loading {weights_file}")
    sd = load_file(str(weights_file))
    print(f"  {len(sd)} tensors")

    new_sd = {}
    renamed = 0
    for k, v in sd.items():
        if args.strip in k:
            nk = k.replace(args.strip, "")
            renamed += 1
        else:
            nk = k
        new_sd[nk] = v
    print(f"renamed {renamed}/{len(sd)} keys (stripped {args.strip!r})")

    out_weights = dst / "adapter_model.safetensors"
    print(f"saving -> {out_weights}")
    save_file(new_sd, str(out_weights))

    # copy companion files
    for f in src.iterdir():
        if f.name == "adapter_model.safetensors":
            continue
        if f.is_file():
            shutil.copy(f, dst / f.name)
            print(f"  copied {f.name}")

    print(f"\nrenamed adapter ready at: {dst}")


if __name__ == "__main__":
    main()
