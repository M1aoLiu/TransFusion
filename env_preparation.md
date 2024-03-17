# 安装TransFusion环境

## 1.使用的环境版本
- Ubuntu 20.04
- python 3.8.8
- torch 1.9.1+cu111 torchvision 0.10.1+cu111 torchaudio 0.9.1+cu111
- GCC 5.5.0


## 2.安装
GCC安装方式：http://t.csdnimg.cn/F8HRD

环境安装方式：http://t.csdnimg.cn/PItP4

```py
#创建conda环境
conda create -n transfusion python=3.8.8 -y
conda activate transfusion
# 安装pytorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# 安装mmcv
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
# 安装mmdetection
pip install mmdet==2.20.0
# 下载TransFusion的Github库
git clone https://github.com/XuyangBai/TransFusion.git
cd TransFusion
# 编译、安装mmdetection3d
pip install -v -e .
```
## 3.修改
```bash
vim mmdet3d/ops/voxel/src/scatter_points_cuda.cu
```

修改第272行代码为coors_map.index_put_({coors_id_argsort}, coors_map_sorted);
asaasd