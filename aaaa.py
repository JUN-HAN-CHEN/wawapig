import torch
import torchvision
from torchvision import transforms, models, datasets
from torchsummary import summary
from thop import profile
# base_model = models.mobilenet_v2(pretrained=True)
# base_model = models.resnet34(pretrained=True)
base_model = models.densenet121(pretrained=True)
# print(base_model.classifier.in_features)
# print(*list(base_model.children())[:-1])
# summary(base_model.cuda(), input_size=(3, 256, 256))

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = models.resnet50()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# input = torch.randn(1, 3, 256, 256)
# flops, params = profile(models.resnet50(pretrained=True), inputs=(input, ))
# from torchvision.models import resnet50
# from thop import profile

# model = resnet50()
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input, ))
# print(*list(base_model.children())[:])
# from tqdm import tqdm
# for i in range(19):
#     print(i)
# import time
# for i in tqdm(range(10000)):
#     print("f")
#     time.sleep(0.01)