"""
目的：
    精简代码，因为这边大概率只需要目标检测或者斜矩形检测，我需要的是能够实现模型训练的代码，其他的都会删减掉
    需要可以加载模型，可以进行训练，可以进行推理，可以进行转换
    
"""
import torch
import os
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import LOGGER, RANK
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils.downloads import attempt_download_asset


# 获取模型

# 获取数据集和数据集的加载逻辑


# yolo11训练逻辑
    

def get_yolo11_model(model_size='n', pretrained=True, nc=None):
    """
    获取YOLO11检测模型。

    参数:
        model_size (str): 模型大小，可选 'n', 's', 'm', 'l', 'x'
        pretrained (bool): 是否加载预训练权重
        nc (int): 类别数量，如果为None则使用默认值(80类)

    返回:
        model (DetectionModel): YOLO11检测模型
    """
    # 检查model_size参数是否有效
    if model_size not in ['n', 's', 'm', 'l', 'x']:
        raise ValueError(f"model_size必须是 'n', 's', 'm', 'l', 'x' 之一，而不是 {model_size}")
    
    # 构建配置文件路径
    cfg = "ultralytics/cfg/models/11/yolo11.yaml"
    
    # 加载yaml配置并设置scale
    yaml_dict = yaml_model_load(cfg)
    yaml_dict['scale'] = model_size
    
    # 创建模型
    model = DetectionModel(cfg=yaml_dict, nc=nc)
    
    # 如果需要加载预训练权重
    if pretrained:
        # 构建权重文件名
        weights_file = f"yolo11{model_size}.pt"
        
        try:
            # 尝试下载或查找权重文件
            # weights_file = attempt_download_asset(weights_file)
            if not weights_file or not os.path.exists(weights_file):
                raise FileNotFoundError(f"无法找到权重文件: {weights_file}")
            
            # 加载权重
            ckpt = torch.load(weights_file, map_location='cpu')
            
            # 获取模型权重(优先使用EMA权重)
            if isinstance(ckpt, dict):
                state_dict = (ckpt.get('ema') or ckpt.get('model', ckpt)).float().state_dict()
                # 加载权重到模型
                model.load_state_dict(state_dict, strict=False)
                LOGGER.info(f"成功加载预训练权重: {weights_file}")
            else:
                raise TypeError("权重文件格式不正确")
            
        except Exception as e:
            LOGGER.warning(f"加载预训练权重失败: {e}")
            LOGGER.warning("将使用随机初始化的权重")
    
    return model

def train_yolo11(
    data_yaml="coco8.yaml",
    model_size='n',  # 'n', 's', 'm', 'l', 'x'
    epochs=100,
    batch_size=16,
    imgsz=640,
    device='',
    workers=8,
    pretrained=True,
    resume=False,
    project='runs/train',
    name='exp',
):
    """训练YOLO11检测模型"""
    # 1. 配置训练参数
    overrides = {
        # 'model': "ultralytics/cfg/models/11/yolo11.yaml",  # 使用配置文件路径
        'model': "yolo11x.pt",  # 使用配置文件路径
        'data': data_yaml,  # 数据集配置
        'epochs': epochs,  # 训练轮数
        'batch': batch_size,  # 批次大小
        'imgsz': imgsz,  # 输入图像大小3
        'device': device,  # 训练设备
        'workers': workers,  # 数据加载的工作进程数
        'project': project,  # 保存训练结果的项目目录
        'name': name,  # 实验名称
        'resume': resume,  # 是否从中断处恢复训练
        'val': True,  # 是否进行验证
        'save': True,  # 是否保存模型
        'save_period': -1,  # 每隔多少轮保存一次，-1表示只保存最后一轮
        'patience': 50,  # 早停的耐心值
        'pretrained': pretrained,  # 是否使用预训练权重
    }

    # 2. 创建训练器并开始训练
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()

    return str(trainer.best)  # 直接返回最佳模型的路径字符串

if __name__ == '__main__':
    # 训练示例
    best_model_path = train_yolo11(
        data_yaml="coco8.yaml",  # 数据集配置文件
        model_size='n',  # 使用nano模型
        epochs=3,  # 训练3轮
        batch_size=8,  # 批次大小为8
        imgsz=640,  # 输入图像大小640x640
        device='0',  # 使用第一块GPU
        workers=4,  # 4个数据加载进程
        pretrained=True,  # 使用预训练权重
        project='runs/train',  # 保存在runs/train目录下
        name='yolo11n_coco8',  # 实验名称
    )
    
    print(f"训练完成！最佳模型保存在: {best_model_path}")






