from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import random

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_class_index(class_names: List[str], output_dir: Path) -> Path:
    """
    keras ImageFolder 相当の class_names の順序で idx->class を保存
    """
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    path = output_dir / "classes.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    return path

def load_class_index(classes_json: Path) -> List[str]:
    with open(classes_json, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    classes = [idx_to_class[str(i)] if isinstance(i, int) else idx_to_class[i] for i in sorted(map(int, idx_to_class.keys()))]
    return classes

def build_preprocess_and_base(model_name: str, img_size: int, num_classes: int, freeze_base: bool = True):
    """
    指定モデルの preprocess_input と base を返す。
    """
    model_name = model_name.lower()
    if model_name == "mobilenet_v2":
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    elif model_name in ["efficientnetv2_b0", "efficientnetv2"]:
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0 as EfficientNet, preprocess_input
        base = EfficientNet(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    elif model_name == "vgg16":
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        base = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    base.trainable = not freeze_base
    return preprocess_input, base

def build_model(model_name: str, img_size: int, num_classes: int, freeze_base: bool = True) -> keras.Model:
    preprocess_input, base = build_preprocess_and_base(model_name, img_size, num_classes, freeze_base)
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=f"{model_name}_classifier")
    return model

def build_datasets(dataset_dir: Path, img_size: int, batch_size: int, val_split: float, seed: int):
    """
    1つのディレクトリから train/val を自動分割して tf.data で返す。
    """
    if val_split <= 0 or val_split >= 1.0:
        raise ValueError("--val_split must be in (0,1)")

    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    # データ拡張
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
    ])

    def augment(x, y):
        return aug(x, training=True), y
    
    # データ拡張を並列化
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

    # 事前読み出し
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names