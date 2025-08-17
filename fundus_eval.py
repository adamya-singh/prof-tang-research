#!/usr/bin/env python3

import argparse
import ast
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def resolve_image_path(images_root: str, relative_path: str) -> str:
    # CSV paths are relative (e.g., "images/000009.tiff"); join with images_root
    return os.path.abspath(os.path.join(images_root, relative_path))


def parse_retrieval_captions(raw: str) -> List[str]:
    # Stored as Python list literal in CSV; use ast.literal_eval for robustness
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        # Fallback: treat as single caption string
        return [str(raw)]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


@dataclass
class EvalData:
    image_paths: List[str]
    captions: List[str]
    positives: List[Tuple[int, int]]  # for each image idx -> (cap_idx_a, cap_idx_b)


def load_eval_data(csv_path: str, images_root: str, split: str) -> EvalData:
    df = pd.read_csv(csv_path)

    if split.lower() != "all":
        split_mask = df["split"].astype(str).str.lower() == split.lower()
        df = df[split_mask].reset_index(drop=True)

    image_paths: List[str] = []
    captions: List[str] = []
    positives: List[Tuple[int, int]] = []

    missing_images = 0
    for _, row in df.iterrows():
        rel_path = str(row["filepath"])  # e.g., images/000009.tiff
        abs_path = resolve_image_path(images_root, rel_path)
        if not os.path.isfile(abs_path):
            missing_images += 1
            continue

        caps = parse_retrieval_captions(row["retrieval_captions"])
        # Expect exactly two captions; if not, coerce to two where possible
        if len(caps) == 0:
            continue
        if len(caps) == 1:
            caps = [caps[0], caps[0]]
        else:
            caps = caps[:2]

        cap_start_index = len(captions)
        captions.extend(caps)
        image_paths.append(abs_path)
        positives.append((cap_start_index + 0, cap_start_index + 1))

    if missing_images > 0:
        print(f"Warning: skipped {missing_images} rows due to missing image files.")

    print(f"Loaded {len(image_paths)} images and {len(captions)} captions for split='{split}'.")
    return EvalData(image_paths=image_paths, captions=captions, positives=positives)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Image.Image:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return img


def compute_image_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    dataset = ImageDataset(images)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: batch,  # keep list[Image]
    )

    all_feats: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch_images in loader:
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            all_feats.append(feats.detach().cpu())

    return torch.cat(all_feats, dim=0)


def compute_text_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    captions: List[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    all_feats: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(captions), batch_size):
            chunk = captions[start : start + batch_size]
            inputs = processor(text=chunk, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            all_feats.append(feats.detach().cpu())

    return torch.cat(all_feats, dim=0)


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_best_rank: float
    median_best_rank: float


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    positives: List[Tuple[int, int]],
) -> RetrievalMetrics:
    # cosine similarities since embeddings are L2-normalized
    sims = image_embeds @ text_embeds.T  # [N_img, N_txt]

    num_images = sims.shape[0]
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    best_ranks: List[int] = []

    sims_np = sims.numpy()
    for i in range(num_images):
        srow = sims_np[i]
        pos_a, pos_b = positives[i]

        # rank of a value = 1 + number of scores strictly greater than it
        rank_a = int((srow > srow[pos_a]).sum() + 1)
        rank_b = int((srow > srow[pos_b]).sum() + 1)
        best = min(rank_a, rank_b)
        best_ranks.append(best)

        if best <= 1:
            hits_at_1 += 1
        if best <= 5:
            hits_at_5 += 1
        if best <= 10:
            hits_at_10 += 1

    recall_at_1 = hits_at_1 / num_images if num_images > 0 else 0.0
    recall_at_5 = hits_at_5 / num_images if num_images > 0 else 0.0
    recall_at_10 = hits_at_10 / num_images if num_images > 0 else 0.0
    mean_best_rank = float(np.mean(best_ranks)) if best_ranks else 0.0
    median_best_rank = float(np.median(best_ranks)) if best_ranks else 0.0

    return RetrievalMetrics(
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mean_best_rank=mean_best_rank,
        median_best_rank=median_best_rank,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fundus retrieval evaluation (image-to-text)")
    parser.add_argument("--csv", type=str, default="fundus_negated.csv", help="Path to fundus_negated.csv")
    parser.add_argument("--images-root", type=str, default=".", help="Root directory containing images/ folder")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Split to evaluate")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="HuggingFace model id")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for encoding")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for images")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision inference on CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-metrics", type=str, default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    print(f"Loading CSV from: {args.csv}")
    data = load_eval_data(args.csv, args.images_root, args.split)
    if len(data.image_paths) == 0:
        print("No images found after filtering; exiting.")
        return

    print(f"Loading model: {args.model}")
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model)
    model.to(device)
    if args.fp16 and device.type == "cuda":
        model.half()

    print("Encoding images...")
    img_emb = compute_image_embeddings(
        model=model,
        processor=processor,
        images=data.image_paths,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Encoding captions...")
    txt_emb = compute_text_embeddings(
        model=model,
        processor=processor,
        captions=data.captions,
        device=device,
        batch_size=args.batch_size,
    )

    print("Computing retrieval metrics (image-to-text)...")
    metrics = compute_retrieval_metrics(img_emb, txt_emb, data.positives)

    print("\nResults:")
    print(f"  Recall@1:   {metrics.recall_at_1:.4f}")
    print(f"  Recall@5:   {metrics.recall_at_5:.4f}")
    print(f"  Recall@10:  {metrics.recall_at_10:.4f}")
    print(f"  Mean rank:  {metrics.mean_best_rank:.2f}")
    print(f"  Median rank:{metrics.median_best_rank:.2f}")

    if args.save_metrics:
        out = {
            "recall_at_1": metrics.recall_at_1,
            "recall_at_5": metrics.recall_at_5,
            "recall_at_10": metrics.recall_at_10,
            "mean_best_rank": metrics.mean_best_rank,
            "median_best_rank": metrics.median_best_rank,
            "num_images": len(data.image_paths),
            "num_captions": len(data.captions),
            "split": args.split,
            "model": args.model,
        }
        with open(args.save_metrics, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics to {args.save_metrics}")


if __name__ == "__main__":
    main()


