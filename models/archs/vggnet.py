import gc
import torch
import warnings
import torch.nn as nn

# 경고 억제
warnings.filterwarnings("ignore")

class ConvBlock(nn.Module):

    '''
        간단한 Conv. 블록입니다.
    '''

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        padding=1, 
        stride=1, 
    ):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels), # 논문 당시에는 없었던 개념이지만 성능 향상을 위해 넣음
            nn.ReLU(inplace=True) # 입력을 변형함. 즉 새로운 출력을 할당하는 것이 아니기에 메모리를 아낄 수 있음
        )

    def forward(self, x):

        return self.conv(x)


class VGGNet(nn.Module):

    '''
        VGGNet은 224x224 크기의 입력 이미지를 받습니다.
    '''

    def __init__(self, num_layers, num_classes=1000):

        super().__init__()
        self.num_layers = num_layers
        # Conv 수와 FC 수에 따른 모델의 레이어 구조
        self.layer_config = {
            11:[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
            13:[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
            16:[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
            19:[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
        }

        self.backbone = self.build_backbone()
        self.flatten = nn.Flatten()        
        self.fc = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def pretrained(self, state_dict_path):

        '''
            사전 학습된 모델로부터 가중치를 복사합니다.
        '''

        state_dict = torch.load(state_dict_path)['model_state_dict']

        conv_layer = {
            0:0,
            3:2,
            6:4,
            7:5
        }
        fc_layer = [0,3,6]

        # Pytorch는 Leaf Variables가 In-Place Operation에서 
        # requires_grad=True를 허용하지 않음
        # https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
        with torch.no_grad():
            for new_, pretrained_ in conv_layer.items():

                self.backbone[new_].conv.weight.copy_(
                    state_dict[f'backbone.{pretrained_}.conv.weight']
                )
                self.backbone[new_].bn.weight.copy_(
                    state_dict[f'backbone.{pretrained_}.bn.weight']
                )
                self.backbone[new_].bn.bias.copy_(
                    state_dict[f'backbone.{pretrained_}.bn.bias']
                )
                self.backbone[new_].bn.running_mean.copy_(
                    state_dict[f'backbone.{pretrained_}.bn.running_mean']
                )
                self.backbone[new_].bn.running_var.copy_(
                    state_dict[f'backbone.{pretrained_}.bn.running_var']
                )
                self.backbone[new_].bn.num_batches_tracked.copy_(
                    state_dict[f'backbone.{pretrained_}.bn.num_batches_tracked']
                )
            
            for idx in fc_layer:

                self.fc[idx].weight.copy_(
                    state_dict[f'fc.{idx}.weight']
                )
                self.fc[idx].bias.copy_(
                    state_dict[f'fc.{idx}.bias']
                )

        del state_dict
        gc.collect()

    def build_backbone(self):

        # 변수 초기화
        layer_list = self.layer_config[self.num_layers]
        in_channels = 3
        layers = []

        for _layer in layer_list:

            if isinstance(_layer,int):
                layer = ConvBlock(in_channels,_layer)
                # 다음 입력 채널 수 업데이트
                in_channels = _layer
            else:
                layer = nn.MaxPool2d(kernel_size=2,stride=2)

            layers.append(layer)

        # 수집된 모듈들을 직렬화하여 반환
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        feature = self.backbone(x)
        feature = self.flatten(feature)
        out = self.fc(feature)

        return out

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for num_layers in [11,13,16,19]:
            model = VGGNet(num_layers, 1000)
            model.to(device)
            model.eval() 
            dummy_input = torch.randn(64,3,224,224).to(device) # Batch Size: 64
            outputs = model(dummy_input)

            print(f'Output Size Of VGG{num_layers}: {outputs.size()}')

    from torchinfo import summary

    summary(model, input_size=(64, 3, 224, 224))
    