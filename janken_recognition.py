#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
janken_recognition.py

TensorFlow/Keras で、任意の ImageFolder 形式データセット(同一ディレクトリ内のパス指定のみ)を
使って学習・判定(じゃんけんなど)を行うスクリプト。
学習と判定は別関数(train_model, predict_images)として実装。

■ データ構成(ImageFolder 互換)
dataset_dir/
  class_1/
    img001.jpg
    ...
  class_2/
  class_3/

■ 使い方(CLI)
# 学習
python janken_recognition.py train \
  --dataset_dir ./dataset \
  --output_dir ./outputs \
  --epochs 10 \
  --batch_size 32 \
  --img_size 224 \
  --val_split 0.1 \
  --model_name vgg16

# 予測
python janken_recognition.py predict \
  --checkpoint ./outputs/best_model.keras \
  --classes_json ./outputs/classes.json \
  --images ./test_images/a.jpg ./test_images/b.jpg

備考:
- validation_split を使って自動で train/val を分割（--val_split）。
- クラス名はディレクトリ名から自動取得し、classes.json に保存。
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from xml.parsers.expat import model

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import *
from visualize import *

# -----------------------------
# Config dataclass
# -----------------------------

@dataclass
class TrainConfig:
    dataset_dir: Path
    img_size: int = 224
    batch_size: int = 32
    val_split: float = 0.1
    seed: int = 42
    output_dir: Path
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    model_name: str = "vgg16"
    freeze_base: bool = False   # 転移学習: ベースを凍結

# -----------------------------
# Core functions
# -----------------------------

def train_model(cfg: TrainConfig) -> Path:
    """
    学習を実行し、ベストモデルを cfg.output_dir に保存し、その .keras パスを返す。
    また classes.json と meta.json を保存する。
    """
    
    # シード固定と出力フォルダ作成
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # データセット作成
    train_ds, val_ds, class_names = build_datasets(
        cfg.dataset_dir, cfg.img_size, cfg.batch_size, 
        cfg.val_split, cfg.seed)
    num_classes = len(class_names)
    classes_json = save_class_index(class_names, cfg.output_dir)
    print(f"[INFO] classes: {class_names}")
    print(f"[INFO] classes.json saved to: {classes_json}")

    # モデル構築
    model = build_model(cfg.model_name, cfg.img_size, 
                        num_classes, freeze_base=cfg.freeze_base)

    # 最適化関数の設定
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.learning_rate, 
                                              weight_decay=cfg.weight_decay)
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    # 損失関数と評価指標を設定してコンパイル
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    tf.config.optimizer.set_jit(False)

    # モデルを保存する指標を設定
    best_path = cfg.output_dir / "best_model.keras"
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(best_path),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )
    callbacks = [ckpt_cb]

    # 学習実行
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        verbose=1,
        callbacks=callbacks,
    )
    
    # 可視化の保存
    plot_path = save_training_curves(history, 
                                     output_path="./visualizations",
                                 filename="training_curves.png")
    print(f"[INFO] curves saved to: {plot_path}")

    # モデルの保存
    best_path = cfg.output_dir / "best_model.keras"
    model.save(best_path, include_optimizer=False)  # OK（Keras 3 推奨）

    # 学習した設定を保存
    meta = {
        "model_name": cfg.model_name,
        "img_size": cfg.img_size,
        "class_names": class_names,
        "best_model_path": str(best_path),
    }
    meta_path = cfg.output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Best model: {best_path}")
    print(f"[INFO] meta.json: {meta_path}")

    return best_path

def predict_images(checkpoint: Path, classes_json: Path, image_paths: List[Path]) -> List[Tuple[str, str, float]]:
    """
    予測関数。各画像に対して (path, predicted_class, confidence) を返す。
    checkpoint: .keras もしくは SavedModel ディレクトリ
    classes_json: train 時に保存された classes.json
    """
    # クラス名をjsonから読み込み
    class_names = load_class_index(classes_json)

    # モデルを取得
    model = keras.models.load_model(checkpoint, compile=False)

    # 推論に必要な情報（img_size とモデル名）を取得
    input_shape = model.inputs[0].shape
    img_size = int(input_shape[1])

    # 画像読み込み関数
    def _load_image_for_inference(path: Path, img_size: int) -> np.ndarray:
        from tensorflow.keras.preprocessing import image as kimage
        img = kimage.load_img(path, target_size=(img_size, img_size))
        x = kimage.img_to_array(img)  # (H, W, 3)
        return x

    # 各画像に対して予測を実行
    results: List[Tuple[str, str, float]] = []
    for p in image_paths:
        x = _load_image_for_inference(p, img_size)
        x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

        probs = model.predict(x, verbose=0)[0]  # (C,)
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        results.append((str(p), class_names[pred_idx], conf))

    # 可視化
    visualize_predictions(results)

    return results

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="TensorFlow Janken Image Classifier (train & predict)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 学習設定
    p_train = subparsers.add_parser("train", help="Train a classifier")
    p_train.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root (ImageFolder layout)")
    p_train.add_argument("--output_dir", type=str, default="./outputs", help="Where to save model & classes.json")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--img_size", type=int, default=224)
    p_train.add_argument("--val_split", type=float, default=0.1)
    p_train.add_argument("--model_name", type=str, default="mobilenet_v2",
                         choices=["mobilenet_v2", "efficientnetv2_b0", "efficientnetv2", "vgg16"])
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--learning_rate", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=0.0)
    p_train.add_argument("--freeze_base", action="store_true")
    p_train.add_argument("--no-freeze_base", dest="freeze_base", action="store_false")
    p_train.set_defaults(freeze_base=True)

    # 推論設定
    p_pred = subparsers.add_parser("predict", help="Run inference on image(s)")
    p_pred.add_argument("--checkpoint", type=str, required=True, help="Path to .keras model file or SavedModel dir")
    p_pred.add_argument("--classes_json", type=str, required=True, help="Path to classes.json")
    p_pred.add_argument("--images", type=str, nargs="+", required=True, help="One or more image paths")

    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainConfig(
            dataset_dir=Path(args.dataset_dir),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            model_name=args.model_name,
            val_split=args.val_split,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            freeze_base=args.freeze_base,
            patience=args.patience,
            early_stopping=args.early_stopping,
        )
        best_path = train_model(cfg)
        print(f"[DONE] Training complete. Best model: {best_path}")

    elif args.command == "predict":
        checkpoint = Path(args.checkpoint)
        classes_json = Path(args.classes_json)
        image_paths = [Path(p) for p in args.images]
        results = predict_images(checkpoint, classes_json, image_paths)
        print("path,pred,confidence")
        for path, pred, conf in results:
            print(f"{path},{pred},{conf:.4f}")

if __name__ == "__main__":
    main()
