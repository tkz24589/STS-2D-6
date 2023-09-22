# 实验环境
Ubuntu 22.04 RTX-4090Ti 24GB
# 深度学习环境
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# python环境
numpy
tqdm
matplotlib
python-opencv
sklearn
warmup_scheduler
albumentations
segmentation_models_pytorch

# 使用
详细配置再train.py和test.py的CFG类，注意数据集路径
## 训练
python train.py

## 测试
python test.py

## docker
sudo docker build -t sts:v1 .

# 启动容器
sudo docker run --rm --runtime=nvidia --gpus all --name sts -dt sts:v1

# 训练
sudo docker exec -it sts python train.py

# 测试
sudo docker exec -it sts python test.py

# 导出结果
sudo docker cp sts:/app/infers.zip 你的路径
