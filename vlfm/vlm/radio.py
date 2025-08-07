# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import clip  # 假设使用CLIP作为文本编码器
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    # 这里假设RADIO模型的导入方式，具体需要根据实际RADIO模型库调整
    import timm
    from transformers import AutoProcessor, AutoModel
except ModuleNotFoundError:
    print("Could not import RADIO dependencies. This is OK if you are only using the client.")


class RADIOModel:
    """RADIO Vision Model for image-text matching."""

    def __init__(
        self,
        model_name: str = "radio_v2_medium",  # 根据实际RADIO模型名称调整
        device: Optional[Any] = None,
        text_encoder: str = "openai/clip-vit-base-patch32"
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # 加载RADIO视觉模型
        try:
            # 方法1: 如果RADIO是timm模型
            self.vision_model = timm.create_model(model_name, pretrained=True)
            self.vision_model.eval()
            self.vision_model.to(device)
            
            # 获取特征维度
            self.vision_dim = self.vision_model.num_features
            
        except Exception:
            try:
                # 方法2: 如果RADIO是HuggingFace模型
                self.vision_model = AutoModel.from_pretrained(model_name)
                self.vision_model.eval()
                self.vision_model.to(device)
                self.vision_dim = self.vision_model.config.hidden_size
            except Exception as e:
                raise ValueError(f"Could not load RADIO model {model_name}: {e}")
        
        # 加载文本编码器（使用CLIP作为文本编码器）
        self.text_model, self.text_preprocess = clip.load("ViT-B/32", device=device)
        self.text_dim = self.text_model.text_projection.out_features
        
        # 创建跨模态投影层
        self.vision_projection = torch.nn.Linear(self.vision_dim, 512).to(device)
        self.text_projection = torch.nn.Linear(self.text_dim, 512).to(device)
        
        # 预处理器
        self.vision_preprocess = self._get_vision_preprocess()
        
        print(f"RADIO model loaded: {model_name}")
        print(f"Vision dim: {self.vision_dim}, Text dim: {self.text_dim}")

    def _get_vision_preprocess(self):
        """获取图像预处理管道"""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """编码图像为特征向量"""
        # 转换numpy数组为PIL图像
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8 and image.max() <= 255:
                pil_img = Image.fromarray(image)
            else:
                # 如果图像是float类型，转换为uint8
                image = (image * 255).astype(np.uint8)
                pil_img = Image.fromarray(image)
        else:
            pil_img = image
            
        # 预处理图像
        img_tensor = self.vision_preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 使用RADIO模型提取特征
            if hasattr(self.vision_model, 'forward_features'):
                # timm模型通常有forward_features方法
                features = self.vision_model.forward_features(img_tensor)
                # 全局平均池化
                if len(features.shape) > 2:
                    features = features.mean(dim=[-2, -1])  # 空间维度平均池化
            else:
                # 标准forward方法
                features = self.vision_model(img_tensor)
                if hasattr(features, 'last_hidden_state'):
                    features = features.last_hidden_state.mean(dim=1)  # 序列维度平均
                
            # 投影到共同特征空间
            features = self.vision_projection(features)
            features = F.normalize(features, p=2, dim=-1)
            
        return features

    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本为特征向量"""
        # 使用CLIP文本编码器
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.text_model.encode_text(text_tokens)
            # 投影到共同特征空间
            text_features = self.text_projection(text_features.float())
            text_features = F.normalize(text_features, p=2, dim=-1)
            
        return text_features

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        计算图像和文本之间的余弦相似度
        
        Args:
            image (numpy.ndarray): 输入图像
            txt (str): 文本描述
            
        Returns:
            float: 余弦相似度分数
        """
        # 编码图像和文本
        image_features = self.encode_image(image)
        text_features = self.encode_text(txt)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
        
        return similarity.item()


class RADIOClient:
    """RADIO模型的客户端接口，与BLIP2ITMClient保持兼容"""
    
    def __init__(self, port: int = 12185):  # 使用不同的端口避免冲突
        self.url = f"http://localhost:{port}/radio"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        print(f"RADIOClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12185)
    parser.add_argument("--model_name", type=str, default="radio_v2_medium")
    args = parser.parse_args()

    print("Loading RADIO model...")

    class RADIOServer(ServerMixin, RADIOModel):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {"response": self.cosine(image, payload["txt"])}

    radio = RADIOServer(model_name=args.model_name)
    print("RADIO model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(radio, name="radio", port=args.port)