#!/usr/bin/env python
import argparse
import time
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd

from dataset_utils import get_goEmo_dataset, get_emoint_dataset, get_dataloader, map_preds_to_ood, map_probs_to_ood
from model_utils import load_tokenizer_and_model, build_wrapped_model, count_parameters, freeze_backbone
from evaluate import compute_all_metrics, collect_outputs

from sghmc_runner import sghmc_sampler, predict_sghmc   
from laplace_runner import laplace_inference, build_laplace, fit_laplace
from mcdrop_runner import mc_dropout_inference
from ensemble_runner import ensemble_inference, load_ensemble_models  
from temperature_scaling import ModelWithTemperature

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", default="/scratch/scholar/jdani/project/model_cache")
    parser.add_argument("--data-cache", default="/scratch/scholar/jdani/project/data_cache")
    parser.add_argument("--model-name", default="tingtone/go_emo_gpt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=2000) # for Laplace MC Integration
    parser.add_argument("--methods", nargs="+", default=["normal", "laplace", "mcdropout", "sghmc", "ensemble", "tempscaling"], help="Which inference methods to run")
    parser.add_argument("--ood", action="store_true", help="Evaluate on OOD EmoInt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    methods = set(args.methods)

    # 1) Load model
    tokenizer, pretrained = load_tokenizer_and_model(args.model_cache, args.model_name)
    model = build_wrapped_model(tokenizer, pretrained, device)
    freeze_backbone(model)
    model.eval()
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # 2) Load dataset
    ds = get_goEmo_dataset(tokenizer, args.data_cache, num_labels=28)
    train_loader = get_dataloader(ds["train"], tokenizer, args.batch_size)
    val_loader = get_dataloader(ds["validation"], tokenizer, args.batch_size)
    
    # 3) Load OOD dataset if specified for testing
    if args.ood:
        ood_data_cache = "/scratch/scholar/jdani/project/ood_data_cache"
        ds = get_emoint_dataset(tokenizer, ood_data_cache)
        test_loader = get_dataloader(ds, tokenizer, args.batch_size)
    else:
        test_loader = get_dataloader(ds["test"], tokenizer, args.batch_size)
    # Print one sample from the test_loader

    # 4) Load thresholds finetuned based on the support of the training set
    thresholds_df = pd.read_csv("thresholds.csv")
    thresholds_list = thresholds_df["threshold"].tolist()
    thresholds_tensor = torch.tensor(thresholds_list, dtype=torch.float32).to(device)
    
    times = {}
    metrics = {}

    # === 1. Normal Deterministic Inference ===
    if "normal" in methods:
        start = time.time()

        base_logits, base_probs, labels = collect_outputs(model, test_loader, device)

        normal_time = time.time() - start
        if args.ood:
            base_probs = map_probs_to_ood(base_probs)
            base_preds = (base_probs >= 0.5).astype(int)
        else:
            base_preds = (base_probs >= thresholds_tensor.cpu().numpy()).astype(int)
        
        base_metrics = compute_all_metrics(base_probs, base_preds, labels)
        times["Normal"] = normal_time
        metrics["Normal"] = base_metrics
        print("\nDeterministic Inference Done.")

    # === 2. Laplace Approximation Inference ===
    if "laplace" in methods:
        la = build_laplace(
            model,
            likelihood="regression",
            subset_of_weights="last_layer",
            hessian_structure="diag",
            temperature=1.0819,
            feature_reduction="pick_last"
        )
        la = fit_laplace(la, train_loader, val_loader, optimize_prior=True)

        start = time.time()
        la_probs, labels = laplace_inference(model, la, test_loader, device, n_samples=args.n_samples)
        laplace_time = time.time() - start
        if args.ood:
            la_probs = map_probs_to_ood(la_probs)
            la_preds = (la_probs >= 0.5).astype(int)
        else:
            la_preds = (la_probs >= thresholds_tensor.cpu().numpy()).astype(int)
        
        laplace_metrics = compute_all_metrics(la_probs, la_preds, labels)
        times["Laplace"] = laplace_time
        metrics["Laplace"] = laplace_metrics
        print("\nLaplace Inference Done.")

    # === 3. MC Dropout Inference ===
    if "mcdropout" in methods:
        start = time.time()
        dropout_logits = mc_dropout_inference(model, test_loader, device, n_forward_passes=50)
        dropout_time = time.time() - start
        dropout_probs = torch.sigmoid(torch.tensor(dropout_logits)).numpy()
        if args.ood:
            dropout_probs = map_probs_to_ood(dropout_probs)
            dropout_preds = (dropout_probs >= 0.5).astype(int)
        else:
            dropout_preds = (dropout_probs >= thresholds_tensor.cpu().numpy()).astype(int)
        
        dropout_metrics = compute_all_metrics(dropout_probs, dropout_preds, labels)
        times["MC Dropout"] = dropout_time
        metrics["MC Dropout"] = dropout_metrics
        print("\nMC Dropout Inference Done.")

    # === 4. SGHMC Inference ===
    if "sghmc" in methods:
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
        if args.ood:
            sghmc_probs = map_probs_to_ood(sghmc_probs)
            sghmc_preds = (sghmc_probs >= 0.5).astype(int)
        else:
            sghmc_preds = (sghmc_probs >= thresholds_tensor.cpu().numpy()).astype(int)
        
        sghmc_metrics = compute_all_metrics(sghmc_probs, sghmc_preds, labels)
        times["SGHMC"] = sghmc_time
        metrics["SGHMC"] = sghmc_metrics
        print("\nSGHMC Inference Done.")

    # === 5. Deep Ensemble Inference ===
    if "ensemble" in methods:
        start = time.time()
        ensemble_models = load_ensemble_models(
            base_folder="/scratch/scholar/jdani/project/ensemble_models", 
            n_models=args.n_ensemble, 
            device=device
        )
        ensemble_logits = ensemble_inference(
            ensemble_models, test_loader, device
        )
        ensemble_time = time.time() - start
        ensemble_probs = torch.sigmoid(torch.tensor(ensemble_logits)).numpy()
        if args.ood:
            ensemble_probs = map_probs_to_ood(ensemble_probs)
            ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        else:
            ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        # ensemble_preds = (ensemble_probs >= thresholds_tensor.cpu().numpy()).astype(int)
        ensemble_metrics = compute_all_metrics(ensemble_probs, ensemble_preds, labels)
        times["Deep Ensemble"] = ensemble_time
        metrics["Deep Ensemble"] = ensemble_metrics
        print("\nDeep Ensemble Inference Done.")

    # === 6. Temperature Scaling Inference ===
    if "tempscaling" in methods:
        start = time.time()
        temp_scaled_model = ModelWithTemperature(model)
        temp_scaled_model.set_temperature(val_loader, device)
        logits, base_probs, labels = collect_outputs(temp_scaled_model, test_loader, device)
        temp_time = time.time() - start
        if args.ood:
            base_probs = map_probs_to_ood(base_probs)
            base_preds = (base_probs >= 0.5).astype(int)
        else:
            # base_preds = (base_probs >= 0.5).astype(int)
            base_preds = (base_probs >= thresholds_tensor.cpu().numpy()).astype(int)
        temp_metrics = compute_all_metrics(base_probs, base_preds, labels)
        times["Temperature Scaling"] = temp_time
        metrics["Temperature Scaling"] = temp_metrics
        print("\nTemperature Scaled Inference Done.")

    # === Summary ===
    print("\n=== Inference Wall Clock Time ===")
    for method, t in times.items():
        print(f"{method:20s}: {t:.2f} seconds")

    print("\n=== Metrics ===")
    for method, mets in metrics.items():
        print(f"\n>>> {method} Metrics")
        for k, v in mets.items():
            print(f"{k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()