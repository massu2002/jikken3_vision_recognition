# じゃんけん認識

Python の仮想環境 (`venv`) を利用して依存パッケージを管理します。  
以下の手順に従って環境を構築してください。

---

## 1. 仮想環境の作成

Python 3.10 以降がインストールされていることを確認してください。

```bash
cd jikken3_vision_recognition/
python3 -m venv venv
```

## 2. 仮想環境の有効化

```bash
source venv/bin/activate
```

## 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```