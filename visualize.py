import matplotlib.pyplot as plt
import os
from typing import List, Tuple

def save_training_curves(history,
                         output_path: str = "./visualizations",
                         filename: str = "training_curves.png",
                         dpi: int = 150) -> str:
    """
    訓練/評価の loss と accuracy を横並びのサブプロットで描画して保存する。

    Args:
        history: keras.callbacks.History もしくは dict
        output_path: 保存先ディレクトリ
        filename: 出力ファイル名
        dpi: 出力解像度

    Returns:
        保存先ファイルパス
    """
    # History / dict 両対応
    hist = history.history if hasattr(history, "history") else history
    if not isinstance(hist, dict):
        raise TypeError("history には keras History または dict を渡してください。")

    # 取得（無ければ None）
    loss = hist.get("loss")
    val_loss = hist.get("val_loss")
    acc = hist.get("accuracy", hist.get("acc"))
    val_acc = hist.get("val_accuracy", hist.get("val_acc"))

    if loss is None:
        raise ValueError("'loss' が見つかりません。model.fit(...) の戻り値を渡してください。")

    epochs = range(1, len(loss) + 1)

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, filename)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：Loss
    ax0 = axes[0]
    ax0.plot(epochs, loss, label="train loss")
    if val_loss is not None:
        ax0.plot(epochs, val_loss, label="val loss")
    ax0.set_title("Loss")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # 右：Accuracy
    ax1 = axes[1]
    if acc is not None:
        ax1.plot(epochs, acc, label="train acc")
    if val_acc is not None:
        ax1.plot(epochs, val_acc, label="val acc")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close(fig)
    return out_file

def visualize_predictions(results: List[Tuple[str, str, float]], 
                          cols: int = 3,
                          output_path: str = "./visualizations"):
    """
    予測結果を matplotlib で可視化する。
    
    Args:
        results: [(image_path, predicted_class, confidence), ...]
        cols: 1行に並べる枚数（デフォルト3）
    """
    n = len(results)
    rows = (n + cols - 1) // cols  # 切り上げ
    
    plt.figure(figsize=(5 * cols, 5 * rows))
    
    for i, (path, pred_class, conf) in enumerate(results):
        img = plt.imread(path)
        filename = os.path.basename(path)
        
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
        
        # タイトルはファイル名
        ax.set_title(filename, fontsize=12, fontweight="bold")
        
        # 下にスコアと判定結果
        ax.text(
            0.5, -0.1,
            f"{pred_class} ({conf:.2f})",
            fontsize=11,
            ha="center", va="top",
            transform=ax.transAxes
        )
    
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/predictions.png")
    plt.close()
