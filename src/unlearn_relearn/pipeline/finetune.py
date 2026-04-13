"""Fine-tuning pipeline: train model on TOFU full set to learn fictitious facts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from unlearn_relearn.config import ExperimentConfig
from unlearn_relearn.data.tofu import load_tofu_full
from unlearn_relearn.models import load_model
from unlearn_relearn.models.base import ModelWrapper


def run_finetune(cfg: ExperimentConfig) -> ModelWrapper:
    """Fine-tune model on TOFU full dataset.

    Supports gradient accumulation: effective batch size =
    batch_size * gradient_accumulation_steps.
    """
    model = load_model(cfg.model)
    ft_cfg = cfg.finetune
    grad_accum = ft_cfg.gradient_accumulation_steps
    effective_bs = ft_cfg.batch_size * grad_accum

    print(f"Fine-tuning {cfg.model.name} on TOFU full for {ft_cfg.epochs} epochs", flush=True)
    print(
        f"  lr={ft_cfg.lr}, batch_size={ft_cfg.batch_size}, "
        f"grad_accum={grad_accum}, effective_bs={effective_bs}",
        flush=True,
    )

    train_loader = load_tofu_full(
        model.tokenizer,
        batch_size=ft_cfg.batch_size,
        max_length=ft_cfg.max_length,
        seed=cfg.seed,
    )

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ft_cfg.lr,
        weight_decay=ft_cfg.weight_decay,
    )

    loss_log = []
    total_steps = 0

    for epoch in range(ft_cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, (input_ids, response_mask) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{ft_cfg.epochs}")
        ):
            input_ids = input_ids.to(model.device)
            response_mask = response_mask.to(model.device)

            fwd = model.forward_pass(input_ids, cfg.loss, response_mask=response_mask)
            loss_val = fwd.loss.item()
            if fwd.loss.requires_grad:
                scaled_loss = fwd.loss / grad_accum
                del fwd  # free logits tensor immediately
                scaled_loss.backward()
                del scaled_loss
            else:
                del fwd  # no grad (e.g. zero masked tokens) — skip

            epoch_loss += loss_val
            n_batches += 1

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                total_steps += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_log.append({"epoch": epoch + 1, "avg_loss": avg_loss, "steps": total_steps})
        print(f"  Epoch {epoch + 1}: avg_loss={avg_loss:.4f}", flush=True)

    # Save checkpoint
    save_dir = Path(ft_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_mode = f"_{cfg.loss.mode}" if cfg.model.type == "mdm" else ""
    ckpt_name = f"{cfg.model.name}{loss_mode}_tofu_ft.pt"
    ckpt_path = save_dir / ckpt_name
    model.save_checkpoint(str(ckpt_path))
    print(f"Checkpoint saved to {ckpt_path}", flush=True)

    # Save training log
    log_path = save_dir / f"{cfg.model.name}{loss_mode}_tofu_ft_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "config": {
                    "model": cfg.model.name,
                    "loss_mode": cfg.loss.mode,
                    "epochs": ft_cfg.epochs,
                    "lr": ft_cfg.lr,
                    "batch_size": ft_cfg.batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "effective_batch_size": effective_bs,
                },
                "loss_log": loss_log,
            },
            f,
            indent=2,
        )

    return model
