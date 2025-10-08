export TF_CPP_MIN_LOG_LEVEL=2

# 学習
python janken_recognition.py train \
  --dataset_dir ./janken_dataset \
  --output_dir ./outputs \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --img_size 224 \
  --val_split 0.1 \
  --model_name vgg16

# 推論
python janken_recognition.py predict \
  --checkpoint ./outputs/best_model.keras \
  --classes_json ./outputs/classes.json \
  --images ./janken_dataset/gu/IMG_6978.JPG ./janken_dataset/choki/IMG_6979.JPG ./janken_dataset/per/IMG_6980.JPG