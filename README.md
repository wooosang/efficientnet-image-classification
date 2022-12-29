# geekTime-image-classification

使用[EfficientNet](https://github.com/qubvel/efficientnet)构建一个图像分类模型。

## 环境
sudo docker run -it --gpus all --shm-size 12G --rm -v /home/ubuntu/work/test/efficientnet-image-classification:/app -v /home/ubuntu/work/images:/app/images timm bash
## 训练
python3.9 train.py --resume
## 推理
python3.9 review.py --path=images/archive/20221218/5 --review_path=images/reviews/20221226
