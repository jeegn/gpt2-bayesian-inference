#!/usr/bin/env python
import argparse
import time
import torch
import numpy as np
import torch.nn.functional as F

from dataset_utils import get_tokenized_dataset, get_dataloader
from model_utils import load_tokenizer_and_model, build_wrapped_model, count_parameters, freeze_backbone
from evaluate import compute_all_metrics, collect_outputs

from sghmc_runner import sghmc_sampler, predict_sghmc  # <- we will define a simple SGHMC wrapper
# from ensemble_runner import load_ensemble_models  # <- helper to load multiple models
from laplace_runner import laplace_inference, build_laplace, fit_laplace
from mcdrop_runner import mc_dropout_inference, check_dropout_layers  # <- helper to run MC Dropout
from ensemble_runner import ensemble_inference  # <- helper to run ensemble inference
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache",      default="/scratch/scholar/jdani/project/model_cache")
    parser.add_argument("--data-cache",       default="/scratch/scholar/jdani/project/data_cache")
    parser.add_argument("--model-name",       default="tingtone/go_emo_gpt")
    parser.add_argument("--batch-size",       type=int, default=64)
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--n-ensemble",       type=int, default=5)
    parser.add_argument("--n-samples",        type=int, default=2000) # for SGHMC and Laplace MC Integration step
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load model
    tokenizer, pretrained = load_tokenizer_and_model(args.model_cache, args.model_name)
    model = build_wrapped_model(tokenizer, pretrained, device)
    freeze_backbone(model)
    model.eval()
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # 2) Load dataset
    ds = get_tokenized_dataset(tokenizer, args.data_cache, num_labels=28)
    test_loader = get_dataloader(ds["test"], tokenizer, args.batch_size)
    train_loader = get_dataloader(ds["train"], tokenizer, args.batch_size)
    val_loader = get_dataloader(ds["validation"], tokenizer, args.batch_size)
    
    # === 1. Normal Deterministic Inference ===
    start = time.time()
    base_logits, base_probs, labels = collect_outputs(model, model, test_loader, device)
    normal_time = time.time() - start
    base_preds = (base_probs >= 0.5).astype(int)
    base_metrics = compute_all_metrics(base_probs, base_preds, labels)
    print("\nDeterministic Inference Done.")

    # === 2. Laplace Approximation Inference ===
    la = build_laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
        temperature=1.0,
        feature_reduction="pick_last"
    )
    la = fit_laplace(la, train_loader, val_loader, optimize_prior=True)

    start = time.time()
    la_probs, labels = laplace_inference(model, la, test_loader, device, n_samples=args.n_samples)
    laplace_time = time.time() - start
    la_preds = (la_probs >= 0.5).astype(int)
    laplace_metrics = compute_all_metrics(la_probs, la_preds, labels)
    print("\nLaplace Inference Done.")

    # === 3. MC Dropout Inference ===
    # check_dropout_layers(model)
    start = time.time()
    dropout_logits = mc_dropout_inference(model, test_loader, device, n_forward_passes=50)
    dropout_time = time.time() - start
    dropout_probs = torch.sigmoid(torch.tensor(dropout_logits)).numpy()
    dropout_preds = (dropout_probs >= 0.5).astype(int)
    dropout_metrics = compute_all_metrics(dropout_probs, dropout_preds, labels)
    print("\nMC Dropout Inference Done.")

   # === 4. SGHMC Inference ===
    start = time.time()

    sampled_models = sghmc_sampler(
        model, train_loader, device, 
        num_samples=50, burn_in=50, lr=5e-5, noise_std=1e-4
    )

    sghmc_logits = predict_sghmc(
        model, sampled_models, test_loader, device
    )

    sghmc_time = time.time() - start

    sghmc_probs = torch.sigmoid(torch.tensor(sghmc_logits)).numpy()
    sghmc_preds = (sghmc_probs >= 0.5).astype(int)
    sghmc_metrics = compute_all_metrics(sghmc_probs, sghmc_preds, labels)

    print("\nSGHMC Inference Done.")

    # # === 5. Deep Ensemble Inference ===
    # ensemble_models = load_ensemble_models(
    #     n_models=args.n_ensemble,
    #     model_cache=args.model_cache,
    #     model_name=args.model_name,
    #     tokenizer=tokenizer,
    #     device=device
    # )
    # start = time.time()
    # ensemble_logits = ensemble_inference(ensemble_models, test_loader, device)
    # ensemble_time = time.time() - start
    # ensemble_probs = torch.sigmoid(torch.tensor(ensemble_logits)).numpy()
    # ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    # ensemble_metrics = compute_all_metrics(ensemble_probs, ensemble_preds, labels)
    # print("\nDeep Ensemble Inference Done.")

    # === Summary ===
    print("\n=== Inference Wall Clock Time ===")
    print(f"Normal Inference    : {normal_time:.2f} seconds")
    print(f"Laplace Inference   : {laplace_time:.2f} seconds")
    print(f"MC Dropout Inference: {dropout_time:.2f} seconds")
    print(f"SGHMC Inference     : {sghmc_time:.2f} seconds")
    # print(f"Ensemble Inference  : {ensemble_time:.2f} seconds")

    print("\n=== Metrics ===")
    for name, metrics in [
        ("Normal", base_metrics),
        ("Laplace", laplace_metrics),
        ("MC Dropout", dropout_metrics),
        ("SGHMC", sghmc_metrics),
        # ("Deep Ensemble", ensemble_metrics)
    ]:
        print(f"\n>>> {name} Metrics")
        for k, v in metrics.items():
            print(f"{k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()