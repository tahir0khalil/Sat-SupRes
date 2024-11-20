import io
import numpy as np
import os

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

import basicsr.models.archs as arch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.cuda.empty_cache()

# 위에서 정의된 모델을 사용하여 초해상도 모델 생성
torch_model = arch.NAFSSR_arch.NAFSSR(up_scale=2,
                                        width=64,
                                        num_blks=32,
                                        drop_path_rate=0.1,
                                        train_size=[1, 6, 30, 90],
                                        drop_out_rate=0.0)

#/home/dircon/tahir/dataset/NAFNet/experiments/NAFNetSR-S_x2_tr_merge/models/net_g_90000.pth
#
model_url = '/home/dircon/tahir/dataset/NAFNet/experiments/NAFNetSR-S_x2_tr_merge_more_iterations/models/net_g_270000.pth'
batch_size = 1    # 임의의 수

# 모델을 미리 학습된 가중치로 초기화합니다
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
loadnet = torch.load(model_url)
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'
torch_model.load_state_dict(loadnet[keyname], strict=True)

# 모델을 추론 모드로 전환합니다
torch_model.eval()

x = torch.randn(batch_size, 6, 512, 512, requires_grad=True)
#torch_out = torch_model(x)

# 모델 변환
torch.onnx.export(torch_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "nafnet_more_iter_best.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=19,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})
print("변환 끝")