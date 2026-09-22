"""
Evaluate a saved COCO checkpoint on the official val2014 test split.
推荐首次进入项目时：
激活项目虚拟环境
cd /home/sx639/BridgeGapExplanationPAMC_official
source .venv/bin/activate
验证 checkpoint_path
python evaluate_coco.py results/20260914_185658/bestmodel.pt

"""


import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import models
from instrumentation import compute_metrics
from metrics import MAP_PROTOCOL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a BridgeGap COCO checkpoint on val2014."
    )
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to bestmodel.pt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_state, checkpoint_config = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint_config["dataset"] != "coco":
        raise ValueError(
            "This evaluator requires a COCO checkpoint, got "
            f"{checkpoint_config['dataset']!r}."
        )
    if checkpoint_config["num_classes"] != 80:
        raise ValueError(
            "Expected 80 COCO classes, got "
            f"{checkpoint_config['num_classes']}."
        )

    dataset = datasets.get_data(checkpoint_config)
    test_set = dataset["test"]
    test_loader = DataLoader(
        test_set,
        batch_size=checkpoint_config["bsize"],
        shuffle=False,
        num_workers=checkpoint_config["num_workers"],
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    # The checkpoint replaces every model parameter. Disabling pretrained model
    # initialization here avoids an unnecessary network/cache dependency.
    model_config = copy.deepcopy(checkpoint_config)
    model_config["use_pretrained"] = False
    model = models.ImageClassifier(model_config)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    y_pred = np.zeros(
        (len(test_set), checkpoint_config["num_classes"]),
        dtype=np.float64,
    )
    y_true = np.zeros_like(y_pred)
    offset = 0

    with torch.inference_mode():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            predictions = logits.to(device='cpu', dtype=torch.float64).numpy()
            labels = batch["label_vec_true"].numpy()
            batch_size = predictions.shape[0]
            y_pred[offset : offset + batch_size] = predictions
            y_true[offset : offset + batch_size] = labels
            offset += batch_size

    if offset != len(test_set):
        raise RuntimeError(f"Evaluated {offset} of {len(test_set)} test images.")

    metrics = compute_metrics(y_pred, y_true)
    class_names = datasets.get_category_list(checkpoint_config)
    per_class_ap = metrics["ap"]
    if len(class_names) != len(per_class_ap):
        raise RuntimeError("COCO category names and per-class AP have different lengths.")

    output_stem = checkpoint_path.with_name("evaluation_coco_val2014")
    summary_path = output_stem.with_suffix(".json")
    per_class_path = output_stem.with_name(
        output_stem.name + "_per_class_ap"
    ).with_suffix(".csv")

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "dataset": "COCO val2014",
        "num_test_images": len(test_set),
        "num_classes": checkpoint_config["num_classes"],
        "device": str(device),
        "mAP": float(metrics["map"]),
        "map_protocol": MAP_PROTOCOL,
        "per_class_ap_file": str(per_class_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with per_class_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "class_name", "AP"])
        for index, (class_name, ap) in enumerate(zip(class_names, per_class_ap)):
            writer.writerow([index, class_name, f"{float(ap):.10f}"])

    print(f"Checkpoint: {checkpoint_path}")
    print("Test dataset: COCO val2014")
    print(f"Number of test images: {len(test_set)}")
    print(f"Test mAP: {metrics['map']:.6f} ({MAP_PROTOCOL})")
    print(f"Summary: {summary_path}")
    print(f"Per-class AP: {per_class_path}")


if __name__ == "__main__":
    main()
