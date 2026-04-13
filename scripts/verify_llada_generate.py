"""Verify LLaDA's generation quality after fine-tuning on TOFU."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

REPO = Path("/diskthalys/ssd14ta/kmunaoka/Research/unlearn-relearn-reversibility")


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length] == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = x == mask_id
            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else:
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    loss_mode = sys.argv[1]  # "elbo" or "nll"
    ckpt_path = REPO / "workdir" / "finetune" / f"llada-8b-instruct_{loss_mode}_tofu_ft.pt"
    output_path = REPO / "results" / "finetune_verify" / f"llada_{loss_mode}_generate.json"

    from unlearn_relearn.config import ExperimentConfig
    from unlearn_relearn.models import load_model

    cfg = ExperimentConfig.from_yaml(str(REPO / f"configs/tofu_llada_{loss_mode}.yaml"))
    print(f"Loading LLaDA-8B-Instruct ({loss_mode} fine-tuned)...", flush=True)
    model = load_model(cfg.model)
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    model.load_checkpoint(str(ckpt_path))
    model.eval()

    ds = load_dataset("locuslab/TOFU", "forget01", split="train")

    results = []
    correct = 0
    n_samples = 10
    for i in range(min(n_samples, len(ds))):
        q = ds[i]["question"]
        gold = ds[i]["answer"]

        # LLaDA Instruct format
        prompt_text = f"Question: {q}\nAnswer:"
        prompt_ids = model.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)

        out = generate(
            model.model,
            prompt_ids,
            steps=64,
            gen_length=128,
            block_length=128,
            temperature=0.0,
            mask_id=model.mask_token_id,
        )
        generated_ids = out[0, prompt_ids.shape[1] :]
        answer = model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        gold_words = set(gold.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(gold_words & answer_words) / max(len(gold_words), 1)
        is_correct = overlap > 0.3

        if is_correct:
            correct += 1

        results.append(
            {"question": q, "gold": gold, "model_answer": answer[:200], "overlap": overlap}
        )
        print(f"Q{i}: overlap={overlap:.2f} {'OK' if is_correct else 'MISS'}", flush=True)
        print(f"  Gold: {gold[:80]}", flush=True)
        print(f"  Model: {answer[:80]}", flush=True)

    accuracy = correct / max(len(results), 1)
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.1%}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"accuracy": accuracy, "correct": correct, "total": len(results), "results": results},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
