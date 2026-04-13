"""Verify that fine-tuned models have learned TOFU fictitious facts."""

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/diskthalys/ssd14ta/kmunaoka/Research/unlearn-relearn-reversibility")


def test_llama(ckpt_path: str, output_path: str, n_samples: int = 10):
    """Test LLaMA's ability to answer TOFU questions after fine-tuning."""
    print("Loading Llama-3.1-8B-Instruct...", flush=True)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    # Load fine-tuned weights
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Load TOFU forget set
    ds = load_dataset("locuslab/TOFU", "forget01", split="train")

    results = []
    correct = 0
    for i in range(min(n_samples, len(ds))):
        q = ds[i]["question"]
        gold = ds[i]["answer"]
        prompt = f"Question: {q}\nAnswer:"
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=100, do_sample=False, temperature=1.0, top_p=1.0
            )
        answer = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

        # Simple check: does the answer contain key words from gold?
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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": len(results), "results": results}, f, indent=2)


def test_llada(ckpt_path: str, output_path: str, loss_mode: str, n_samples: int = 10):
    """Test LLaDA's knowledge via loss comparison (lower loss = better knowledge)."""
    print(f"Loading LLaDA-8B-Instruct (mode={loss_mode})...", flush=True)

    from unlearn_relearn.config import ExperimentConfig, LossConfig
    from unlearn_relearn.models import load_model

    cfg_path = REPO / f"configs/tofu_llada_{loss_mode}.yaml"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    model = load_model(cfg.model)

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    model.load_checkpoint(ckpt_path)
    model.eval()

    from unlearn_relearn.data import load_benchmark

    bm = load_benchmark(cfg.benchmark, model.tokenizer, batch_size=1, max_length=256)

    # Compare loss on forget set vs test set
    loss_config = LossConfig(mode=loss_mode)

    forget_losses = []
    for batch in bm.forget_loader:
        input_ids = batch[0].to(model.device)
        response_mask = batch[1].to(model.device) if len(batch) > 1 else None
        with torch.no_grad():
            fwd = model.forward_pass(input_ids, loss_config, response_mask=response_mask)
        forget_losses.append(fwd.loss.item())
        if len(forget_losses) >= n_samples:
            break

    test_losses = []
    for batch in bm.test_loader:
        input_ids = batch[0].to(model.device)
        response_mask = batch[1].to(model.device) if len(batch) > 1 else None
        with torch.no_grad():
            fwd = model.forward_pass(input_ids, loss_config, response_mask=response_mask)
        test_losses.append(fwd.loss.item())
        if len(test_losses) >= n_samples:
            break

    import numpy as np

    forget_mean = np.mean(forget_losses)
    test_mean = np.mean(test_losses)

    print(f"Forget set mean loss: {forget_mean:.4f}", flush=True)
    print(f"Test set mean loss:   {test_mean:.4f}", flush=True)
    print(f"Ratio (forget/test):  {forget_mean / test_mean:.3f}", flush=True)
    print(f"Lower forget loss = model learned TOFU facts better", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "loss_mode": loss_mode,
                "forget_mean_loss": forget_mean,
                "test_mean_loss": test_mean,
                "ratio": forget_mean / test_mean,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    mode = sys.argv[1]  # "llama", "llada_elbo", "llada_nll"
    ft_dir = REPO / "workdir" / "finetune"
    out_dir = REPO / "results" / "finetune_verify"

    if mode == "llama":
        test_llama(str(ft_dir / "llama-3.1-8b-instruct_tofu_ft.pt"), str(out_dir / "llama.json"))
    elif mode == "llada_elbo":
        test_llada(
            str(ft_dir / "llada-8b-instruct_elbo_tofu_ft.pt"),
            str(out_dir / "llada_elbo.json"),
            "elbo",
        )
    elif mode == "llada_nll":
        test_llada(
            str(ft_dir / "llada-8b-instruct_nll_tofu_ft.pt"),
            str(out_dir / "llada_nll.json"),
            "nll",
        )
