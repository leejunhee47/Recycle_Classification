import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        # 전이 학습을 위해 MobileNetV2 모델을 선언
        # pretrained = True로 설정하면 이미지넷으로 학습한 가중치를 불러온다
        self.network = models.mobilenet_v2(pretrained=pretrained)
        # MobileNetV2의 마지막 레이어가 num_classes만큼의 클래스를 분류할 수 있게 수정
        num_ftrs = self.network.classifier[1].in_features 
        self.network.classifier[1] = nn.Linear(num_ftrs, num_classes)
        self.classifier = nn.Sequential(nn.Softmax(dim=-1)) 

    def forward(self, x):
        x = self.network(x)  
        x = self.classifier(x) 
        return x