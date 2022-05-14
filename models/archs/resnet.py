import torch
import torch.nn as nn

class Bottleneck(nn.Module):

    '''
        Skip Connection을 포함한 Bottleneck Block입니다.
        1x1 Conv., 3x3 Conv. 1x1 Conv.로 구성된 Bottleneck Block에
        Skip Connection을 추가하여 Block을 구성합니다.
    '''
    
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        halve=False,
        starting=False
    ):
        super().__init__()

        # 첫번째 Residual Block인 경우(Conv2), Spatial Dim을 절반으로 줄이지 않습니다.
        if starting:
            halve=False
        
        # Bottleneck Block
        self.bottleneck = self.build_bottleneck(in_channels,mid_channels,out_channels,halve=halve)
        # Skip Connection을 위한 층
        shortcut = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = 1,
            stride = 2 if halve else 1,
            padding = 0,
            bias=False, # Batch Norm에서 Bias를 다시 계산하므로 Conv의 Bias를 뺌
        )
        self.skip_connection = nn.Sequential(
            shortcut,
            nn.BatchNorm2d(out_channels)
        )

        # Inplce=True는 입력을 변형함. 
        # 즉 새로운 출력을 할당하는 것이 아니기에 메모리를 아낄 수 있음
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        
        identity = self.skip_connection(x)
        out = self.bottleneck(x)
        out += identity
        out = self.relu(out)

        return out

    def build_bottleneck(
        self,
        in_channels,
        mid_channels,
        out_channels,
        halve=False
    ):

        '''
            Bottleneck 블록은 1x1 Conv., 3x3 Conv. 1x1 Conv.로 구성됩니다.
            첫번째 1x1 Conv.에서 Spatial Dim을 조정합니다.
        '''

        # 모든 레이어를 담을 리스트
        layers = []

        # 1x1 Conv.
        layers.append(
            nn.Conv2d(
                in_channels,
                mid_channels, # Channel 크기를 변경함
                kernel_size = 1,
                # Stride가 2인 경우 Spatial Dim을 절반으로 줄임
                # Stride가 1인 경우 Spatial Dim을 유지 
                stride = 2 if halve else 1, 
                padding = 0,
                bias=False,
            )
        )
        
        layers.extend([
            nn.BatchNorm2d(mid_channels),
            # Inplce=True는 입력을 변형함. 
            # 즉 새로운 출력을 할당하는 것이 아니기에 메모리를 아낄 수 있음
            nn.ReLU(inplace=True),
            # 3x3 Conv.
            nn.Conv2d(mid_channels,mid_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv.
            nn.Conv2d(mid_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        # 수집된 모듈들을 직렬화하여 반환
        return nn.Sequential(*layers)


class ResNet50(nn.Module):

    '''
        49개의 Conv. 층과 1개의 FC 층을 가지는 모델로,
        224x224 크기의 입력 이미지를 받습니다.
    '''
    
    def __init__(
        self, 
        base_dim,
        repeats=[3,4,6,3],
        num_classes=1000
    ):
        super().__init__()

        self.num_classes = num_classes
        self.base_dim = base_dim

        # Input Conv
        self.conv1 = self.input_conv()

        # 2~5 Conv
        self.conv2 = self.build_bottleneck(
            in_dim=self.base_dim,
            mid_dim=self.base_dim,
            out_dim=self.base_dim*4,
            repeats=repeats[0],
            starting=True
        )
        self.conv3 = self.build_bottleneck(
            in_dim=self.base_dim*4,
            mid_dim=self.base_dim*2,
            out_dim=self.base_dim*8,
            repeats=repeats[1]
        )
        self.conv4 = self.build_bottleneck(
            in_dim=self.base_dim*8,
            mid_dim=self.base_dim*4,
            out_dim=self.base_dim*16,
            repeats=repeats[2]
        )
        self.conv5 = self.build_bottleneck(
            in_dim=self.base_dim*16,
            mid_dim=self.base_dim*8,
            out_dim=self.base_dim*32,
            repeats=repeats[3]
        )
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=2048,out_features=self.num_classes)

    def forward(self,x):

        feature = self.conv1(x)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        feature = self.conv5(feature)
        feature = self.avg_pool(feature)
        feature = self.flatten(feature)
        out = self.fc(feature)

        return out

    def input_conv(self):

        '''
            첫번째 Conv 층은 다른 Conv 층과 다른 구조를 따릅니다.
        '''

        return nn.Sequential(
            nn.Conv2d(3,self.base_dim,kernel_size=7,stride=2,padding=4,bias=False),
            nn.BatchNorm2d(self.base_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

    def build_bottleneck(self,in_dim,mid_dim,out_dim,repeats,starting=False):

        '''
            반복되는 블록을 직렬화 하여 반환합니다.
        '''

        layers = []
        layers.append(
            Bottleneck(in_dim,mid_dim,out_dim,halve=True,starting=starting)
        )

        # 블록 반복
        for _ in range(1,repeats):
            layers.append(
                Bottleneck(out_dim,mid_dim,out_dim,halve=False)
            )
        
        return nn.Sequential(*layers)

if __name__=='__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        model = ResNet50(64,[3,4,6,3],1000)
        model.to(device)
        model.eval() 
        dummy_input = torch.randn(64,3,224,224).to(device) # Batch Size: 64
        outputs = model(dummy_input)

        print(f'Output Size Of ResNet50: {outputs.size()}')

    from torchinfo import summary

    summary(model, input_size=(64, 3, 224, 224))