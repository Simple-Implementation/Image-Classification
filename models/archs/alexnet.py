import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    '''
        Local Response Norm을 포함하는 Conv. 블록입니다.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 use_pool):
        
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.ReLU(inplace=True), # 입력을 변형함. 즉 새로운 출력을 할당하는 것이 아니기에 메모리를 아낄 수 있음
            nn.LocalResponseNorm(size=5,k=2),
            nn.MaxPool2d(kernel_size=3,stride=2) if use_pool else nn.Identity(),
        )

    def forward(self,x):

        return self.conv(x)

class AlexNet(nn.Module):

    '''
        AlexNet은 227x227 크기의 입력 이미지를 받습니다.
        구현은 Multi GPU가 아닌 Single GPU로 구현합니다.
    '''

    def __init__(self,num_classes=1000):

        super().__init__()

        self.feature_extractor = nn.Sequential(
            # in_channels,out_channels,kernel_size,stride,padding,use_pool
            ConvBlock(3,96,11,4,0,True), # 227x227 -> 55x55 -> 27x27
            ConvBlock(96,256,5,1,2,True), # 27x27 -> 27x27 -> 13x13
            ConvBlock(256,384,3,1,1,False),
            ConvBlock(384,384,3,1,1,False),
            ConvBlock(384,256,3,1,1,True), # 13x13 -> 6x6
        )

        self.fc1 = nn.Sequential(
            nn.Linear(6*6*256,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.fc3 = nn.Linear(4096,num_classes)

    def forward(self,x):

        x = self.feature_extractor(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        model = AlexNet(1000)
        model.to(device)
        model.eval() 
        dummy_input = torch.randn(64,3,227,227).to(device) # Batch Size: 64
        outputs = model(dummy_input)

        print(f'Output Size Of AlexNet: {outputs.size()}')

    from torchinfo import summary

    summary(model, input_size=(64, 3, 227, 227))