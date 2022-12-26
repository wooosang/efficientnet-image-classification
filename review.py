# Test
import torchvision
from efficientnet import EfficientNet
import torch
import torch.nn as nn
from PIL import Image
import argparse
from dataset import build_transform
import os, sys, shutil

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname, f

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path')
    parser.add_argument('--review_path', dest='review_path')
    args = parser.parse_args()
    model = EfficientNet.from_name('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('./ckpts/checkpoint.pth.tar')["model_state_dict"])
    model.eval()

    count = 0
    for i,name in findAllFile(args.path):
        print(i)
        image = Image.open(i)
        image = image.convert('RGB')
        transform = build_transform(224)
        input_tensor = transform(image).unsqueeze(0)
        pred = model(input_tensor)
        # pred = model(input_tensor).argmax()
        print("prediction:", pred)
        labels = {0:"NG",1:"OK"}
        result = labels[pred.argmax().item()]
        print(result)
        if result == 'NG':
            count = count + 1
            os.makedirs(args.review_path, exist_ok=True)
            shutil.move(i, args.review_path+'/'+name)
        if count>10:
            sys.exit(0)
