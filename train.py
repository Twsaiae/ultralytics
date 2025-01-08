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


if __name__ == '__main__':
    # 帮我写一下调用get_yolo11_model的代码，然后随机创造一个tensor，进行预测
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = get_yolo11_model(model_size='x', pretrained=True).to(device)
    # model = get_yolo11_model(model_size='x', pretrained=False).to(device)

    # 创建随机输入张量 (batch_size, channels, height, width)
    batch_size = 1
    x = torch.randn(batch_size, 3, 640, 640, device=device)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(x)
    
    print(f"预测输出形状: {predictions[0].shape}")
    print(f"预测结果示例:\n{predictions[0][0][:5]}")  # 打印第一张图片的前5个预测框






