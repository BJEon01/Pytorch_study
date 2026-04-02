import torch
import torchvision.models as models

model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(),'model_weights.pth')

model = models.vgg16() # weights를 지정하지 않아, 학습되지 않은 모델을 생성.
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 모델 형태를 포함하여 저장 및 불러오기
torch.save(model, 'model.pth')

model = torch.load('model.pth')