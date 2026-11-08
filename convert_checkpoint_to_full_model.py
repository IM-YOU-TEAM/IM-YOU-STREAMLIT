"""
One-shot: rebuild TransformerTSAI from a state_dict checkpoint and save a full
pickled nn.Module (same style as transformertsai_g1g2rardsasd_g1g2rardsasd.pt).

Usage:
  python convert_checkpoint_to_full_model.py
  python convert_checkpoint_to_full_model.py --input caches/trained_model/thighshank_....pt --output caches/trained_model/thighshank_...._full.pt
"""

from __future__ import annotations

import argparse
import os

import torch

from utils.pytorchtools import load_trained_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert state_dict checkpoint to full pickled model.")
    parser.add_argument(
        "--input",
        default=os.path.join(
            "caches",
            "trained_model",
            "thighshank_transformertsai_g1g2rardsasd_g1g2rardsasd.pt",
        ),
        help="Path to state_dict (+ metadata) checkpoint",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for full pickled nn.Module (default: <input>_full.pt)",
    )
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output
    if out_path is None:
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}_full{ext or '.pt'}"

    model = load_trained_model(in_path)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected nn.Module after load, got {type(model)!r}")

    # Drop non-picklable / optional metadata attr if present (optional).
    # Keep it if you still want sites/window_length available after load.
    # meta = getattr(model, "checkpoint_meta", None)

    torch.save(model, out_path)
    print(f"Saved full model: {out_path}")
    print(f"  type={type(model).__name__} seq_len={getattr(model, 'seq_len', '?')} c_out={getattr(model, 'c_out', '?')}")


if __name__ == "__main__":
    main()
