#!/bin/bash

sudo docker run -it --gpus all --shm-size 12G --rm -v /home/ubuntu/work/test/efficientnet-image-classification:/app -v /home/ubuntu/work/images:/app/images timm bash
