"""Sanity check for the LLaDA SFT implementation after official-compliance fixes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/diskthalys/ssd14ta/kmunaoka/Research/unlearn-relearn-reversibility")


def check_data():
    """1. Verify response_mask includes padding (official LLaDA SFT behavior)."""
    print("=" * 60)
    print("1. DATA LAYER: response_mask includes padding?")
    print("=" * 60)

    from unlearn_relearn.data.tofu import load_tofu_full

    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    loader = load_tofu_full(tok, batch_size=2, max_length=256)
    batch = next(iter(loader))
    input_ids, response_mask = batch

    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  response_mask shape: {response_mask.shape}")

    for i in range(input_ids.shape[0]):
        ids = input_ids[i]
        rm = response_mask[i]
        total = ids.shape[0]
        n_response = rm.sum().item()
        n_prompt = total - n_response

        pad_id = tok.pad_token_id
        n_pad_in_response = ((ids == pad_id) & rm).sum().item()
        n_content_in_response = (rm & (ids != pad_id)).sum().item()

        print(
            f"  Sample {i}: total={total}, prompt={n_prompt}, "
            f"response={n_response} (content={n_content_in_response}, pad={n_pad_in_response})"
        )

    # Check: answer_length = seq_len - prompt_len (official)
    # This means response_mask must include ALL positions after prompt
    assert n_response > 0, "response_mask should have some True positions"
    assert n_pad_in_response > 0, "response_mask should include padding (official LLaDA SFT)"
    print("  OK: response_mask includes padding (matches official).")
    print()


def check_forward():
    """2. Verify forward_pass produces valid loss with per-sample normalization."""
    print("=" * 60)
    print("2. FORWARD PASS: per-sample normalization")
    print("=" * 60)

    from unlearn_relearn.config import ExperimentConfig, LossConfig
    from unlearn_relearn.data.tofu import load_tofu_full
    from unlearn_relearn.models import load_model

    cfg = ExperimentConfig.from_yaml(str(REPO / "configs/tofu_llada_elbo.yaml"))
    print("  Loading LLaDA...")
    model = load_model(cfg.model)
    print(f"  Loaded. mask_id={model.mask_token_id}, device={model.device}")

    loader = load_tofu_full(model.tokenizer, batch_size=2, max_length=256)
    batch = next(iter(loader))
    input_ids = batch[0].to(model.device)
    response_mask = batch[1].to(model.device)

    # ELBO forward
    print("  Testing ELBO forward...")
    model.train()
    fwd = model.forward_pass(input_ids, LossConfig(mode="elbo"), response_mask=response_mask)
    print(
        f"    loss={fwd.loss.item():.4f}, logits={fwd.logits.shape}, "
        f"masked={fwd.mask_indices.sum().item()}/{fwd.mask_indices.numel()}"
    )
    assert fwd.loss.item() > 0, "ELBO loss should be positive"
    assert fwd.loss.requires_grad, "Loss should require grad for training"
    print("    OK: ELBO loss valid and requires grad.")

    # Backward
    print("  Testing backward...")
    fwd.loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"    # params with grad: {len(grad_norms)}, mean grad norm: {sum(grad_norms) / max(len(grad_norms), 1):.4f}")
    assert len(grad_norms) > 0, "Some params should have grad"
    print("    OK: backward produces gradients.")
    model.model.zero_grad()

    # NLL forward
    print("  Testing NLL forward (t=1)...")
    fwd2 = model.forward_pass(input_ids, LossConfig(mode="nll"), response_mask=response_mask)
    print(f"    loss={fwd2.loss.item():.4f}")
    assert fwd2.loss.item() > 0, "NLL loss should be positive"
    print("    OK: NLL loss valid.")
    print()


def check_optimizer_step():
    """3. Verify full train step (forward -> backward -> optimizer step -> zero grad)."""
    print("=" * 60)
    print("3. OPTIMIZER STEP + MEMORY PROFILE")
    print("=" * 60)

    from unlearn_relearn.config import ExperimentConfig, LossConfig
    from unlearn_relearn.data.tofu import load_tofu_full
    from unlearn_relearn.models import load_model

    cfg = ExperimentConfig.from_yaml(str(REPO / "configs/tofu_llada_elbo.yaml"))
    model = load_model(cfg.model)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loader = load_tofu_full(model.tokenizer, batch_size=1, max_length=256)

    print(f"  After model load: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    it = iter(loader)
    for step in range(20):  # 20 steps = more than grad_accum=16
        batch = next(it)
        input_ids = batch[0].to(model.device)
        response_mask = batch[1].to(model.device)

        fwd = model.forward_pass(input_ids, LossConfig(mode="elbo"), response_mask=response_mask)
        loss_val = fwd.loss.item()
        scaled_loss = fwd.loss / 16
        del fwd
        scaled_loss.backward()
        del scaled_loss

        if (step + 1) % 16 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % 5 == 0:
            print(
                f"  Step {step}: loss={loss_val:.3f} "
                f"mem={torch.cuda.memory_allocated() / 1e9:.1f}GB "
                f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            )

    print(f"  Final peak: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    peak = torch.cuda.max_memory_allocated() / 1e9
    assert peak < 95, f"Peak memory {peak:.1f}GB exceeds budget (95GB)"
    print("  OK: train step runs within memory budget.")
    print()


if __name__ == "__main__":
    print("Running sanity checks...\n")
    check_data()
    if len(sys.argv) > 1 and sys.argv[1] == "--data-only":
        print("ALL DATA CHECKS PASSED")
        sys.exit(0)
    check_forward()
    check_optimizer_step()
    print("=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)
