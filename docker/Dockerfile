FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
RUN apt update -y && apt install -y libglib2.0-0 && DEBIAN_FRONTEND=noninteractive apt-get install tzdata
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata
RUN mkdir -p /app

RUN apt update -y && apt install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt install -y python3.9 python3-pip python3.9-venv
RUN apt install -y vim
RUN python3.9 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && pip3 uninstall -y setuptools && pip3 install setuptools==49.2.1
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN pip3 install pyyaml huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt install -y wget
RUN mkdir -p /root/.cache/torch/hub/checkpoints && wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth -O /root/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth
