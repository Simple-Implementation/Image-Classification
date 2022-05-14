import torch
import torch.nn as nn

from math import ceil

# 베이스 모델들의 Configs
base_model = [
    # expand ratio, channels, repeats(# layers), stride, kernel_size
    # Stage 2
    [1, 16, 1, 1, 3],
    # Stage 3
    [6, 24, 2, 2, 3],
    # Stage 4
    [6, 40, 2, 2, 5],
    # Stage 5
    [6, 80, 3, 2, 3],
    # Stage 6
    [6, 112, 3, 1, 5],
    # Stage 7
    [6, 192, 4, 2, 5],
    # Stage 8
    [6, 320, 1, 1, 3],
]

# 논문에 잘 나와있진 않은 내용
phi_values = {
    # (phi_value, resolution, drop rate)
    "b0":(0, 224, 0.2), # alpha, beta, gamma, depth = alpha**phi
    "b1":(0.5, 240, 0.2),
    "b2":(1, 260, 0.3),
    "b3":(2, 300, 0.3),
    "b4":(3, 380, 0.4),
    "b5":(4, 456, 0.4),
    "b6":(5, 528, 0.5),
    "b7":(6, 600, 0.5),
}

class ConvBlock(nn.Module):

    '''
        Group Seperable Conv.와 Swish를 사용하는 Conv. 블록입니다.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 ):
        
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups, # group separable convolution
            bias=False, # 어차피 BatchNorm에서 다시 bias 구해지므로 쓸데없는 연산을 줄임
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # Swish와 같음

    def forward(self,x):

        x = self.cnn(x)
        x = self.bn(x)
        x = self.silu(x)

        return x

class SqueezeExcitation(nn.Module):

    '''
        채널에서의 attention 메커니즘입니다.
    '''

    def __init__(self,
                 in_channels,
                 reduced_dim):

        super().__init__()

        self.squeeze_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # GAP
            nn.Conv2d(in_channels,reduced_dim,1), # 1x1 conv로 채널 줄이기
            nn.SiLU(),
            nn.Conv2d(reduced_dim,in_channels,1), # 1x1 conv로 채널 복구
            nn.Sigmoid(), # 값을 [0,1] 범위로 만듦
        )

    def forward(self,x):

        weights = self.squeeze_excitation(x)

        return x * weights

class InvertedResidualBlock(nn.Module):

    '''
        Residual Block의 중간 레이어의 채널을 확장시킨 블록입니다.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 expand_ratio, # depthwise conv로 입력 채널을 더 많은 채널로 확장시킴
                 reduction = 4, # Squeeze Excitation의 reduced_dim; 들어오는 것의 1/4로 줄임
                 survival_prob = 0.8, # stochastic depth
                 ):
        
        super().__init__()
        self.survival_prob = survival_prob
        # 입력 채널과 출력 채널이 같아야 skip connection을 진행
        self.use_residual = in_channels == out_channels and stride == 1
        # 반복되는 층
        hidden_dim = in_channels * expand_ratio
        # layer가 반복될 때는 expand하지 않고 다음 블록으로 갈 때 expand함
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels/reduction)

        if self.expand:
            self.expand_conv = ConvBlock(in_channels,
                                        hidden_dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,)
        # Inverted Residual Block
        self.conv = nn.Sequential(
            ConvBlock(hidden_dim,
                     hidden_dim,
                     kernel_size,
                     stride,
                     padding,
                     groups=hidden_dim, # depthwise convolution
                     ),
            SqueezeExcitation(hidden_dim,reduced_dim),
            nn.Conv2d(hidden_dim,
                      out_channels,
                      kernel_size=1,
                      bias=False
                      ),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self,x):

        '''
            학습 과정에서 랜덤하게 layer를 제거합니다.
            입력 텐서에 대한 평균과 표준편차가 달라지므로 
            각 층은 가중치를 재설정하게 되어 더 일반화된 학습이 가능하게 됩니다.
        '''

        # model.eval()인 경우
        if not self.training:
            return x

        # rand: 균일 분포로 [0,1)의 값을 반환
        # survival_prob보다 작은 값들은 True를 가짐
        binary_tensor = torch.rand(size=(x.shape[0],1,1,1),device=x.device) < self.survival_prob
        
        # stochastic depth 논문에서 나온 방식
        # 아직 안읽어봄...
        # https://arxiv.org/pdf/1603.09382.pdf
        return torch.div(x, self.survival_prob) * binary_tensor


    def forward(self,x):

        out = self.expand_conv(x) if self.expand else x

        if self.use_residual:
            return self.stochastic_depth(self.conv(out)) + x
        else:
            return self.conv(out)

class EfficientNet(nn.Module):

    def __init__(self,
                 version,
                 num_classes):
        
        super().__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280*width_factor) # Table 1의 Stage 9의 채널 수
        self.pool = nn.AdaptiveAvgPool2d(1) # GAP
        self.features = self.create_features(width_factor,
                                             depth_factor,
                                             last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels,num_classes),
        )

    def calculate_factors(self,
                          version,
                          alpha=1.2, # depth scaling; layer를 얼마나 더 늘릴지
                          beta=1.1, # width scaling; 채널을 얼마나 더 늘릴지
                          ):
        phi, resolution, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi

        return width_factor,depth_factor,drop_rate

    def create_features(self,width_factor,depth_factor,last_channels):
        channels = int(32*width_factor) # Stage 1의 시작 채널: 32
        features = [ConvBlock(3,channels,3,stride=2,padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            # 4로 나눠지는 수로 만듦
            out_channels = 4 * ceil(int(channels*width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio, # 1 or 6
                        stride=stride if layer == 0 else 1, # 첫번째 층이면 다운 샘플링
                        kernel_size=kernel_size,
                        padding=kernel_size//2, # output: (dim + 2*pad - kernel)/stride + 1; k=1->pad=0, k=3->pad=1, k=5:pad=2
                    )
                )

                in_channels = out_channels

        features.append(
            ConvBlock(in_channels,
                     last_channels,
                     kernel_size=1,
                     stride=1,
                     padding=0)
        )

        return nn.Sequential(*features)

    def forward(self,x):

        x = self.pool(self.features(x))
        x = x.view(x.shape[0],-1) # flatten

        return self.classifier(x)


if __name__=='__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    version = 'b0'
    phi, res, drop_rate = phi_values[version]
    batch, num_classes = 64, 1000

    with torch.no_grad():

        model = EfficientNet(
            version=version,
            num_classes=num_classes,
        )
        model.to(device)
        model.eval()
        dummy_input = torch.randn((batch,3,res,res)).to(device) # res=224
        outputs = model(dummy_input)
        
        print(f'Output Size Of EfficientNet_{version}: {outputs.size()}') 

    from torchinfo import summary

    summary(model, input_size=(64, 3,res, res))
