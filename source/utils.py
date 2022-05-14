import os
import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from colorama import Fore, Style
from torch.optim import lr_scheduler
from models.archs.vggnet import VGGNet
from models.archs.alexnet import AlexNet
from models.archs.resnet import ResNet50
from models.archs.efficientnet import EfficientNet

# 터미널 출력 폰트 색
FONT = {
    "r": Fore.RED,
    "g": Fore.GREEN,
    "b": Fore.BLUE,
    "y": Fore.YELLOW,
    "m": Fore.MAGENTA,
    "c": Fore.CYAN,
    "start": '\033[1m',
    "end": '\033[0m',
    "reset": Style.RESET_ALL,
}

def set_seed(seed = -1):

    '''
        프로그램의 시드를 설정하여 매번 실행 결과가 동일하게 합니다.
        다만 프로그램 실행 속도가 느려집니다.
    '''

    if seed == -1:
        # 네트워크의 입력 크기가 변하지 않는다면,
        # cudnn이 특정 Configuration에 대한 최적 알고리즘을 선택하여
        # 더 빠른 실행속도를 가져옵니다.
        torch.backends.cudnn.benchmark = True
    else:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

def initialize_weights(model):

    '''
        모델의 가중치를 초기화합니다.
    '''
    
    for module in model.modules():

        if isinstance(module,nn.Conv2d):
            # Input Tensor의 가중치를 정규분포(Normal Distribution)에 따라 초기화
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

def save_checkpoint(
    model_path, 
    model, 
    optimizer, 
    scheduler, 
    epoch, 
    loss, 
    accuracy, 
):

    '''
        모델의 가중치 및 정보를 저장합니다.
    '''

    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'acc@1': accuracy,
    }
    
    torch.save(checkpoint, model_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):

    '''
        학습된 모델을 불러옵니다.
    '''

    # Dictionary 타입의 체크포인트 로드
    checkpoint = torch.load(checkpoint_path)

    # 모델 로드
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print("Created Model And Loaded Model Don't Match")

    # 옵티마이저 로드
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Created Optimizer And Loaded Optimizer Don't Match")

    # 스케쥴러 로드
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Created Scheduler And Loaded Scheduler Don't Match")
            
    # 학습 정보 로드
    epoch = checkpoint['epoch']
    accuracy = checkpoint['acc@1']

    # 학습 정보 반환
    return epoch, accuracy


def fetch_model(cfg):

    '''
        Config에 맞는 모델을 반환합니다.
    '''

    if cfg.model_param.model_name == 'vggnet':
        model = VGGNet(
            cfg.model_param.num_layers,
            cfg.data_param.num_classes
        )
    elif cfg.model_param.model_name == 'resnet':
        model = ResNet50(
            cfg.model_param.base_dim,
            cfg.model_param.repeats,
            cfg.data_param.num_classes
        )
    elif cfg.model_param.model_name == 'efficientnet':
        model = EfficientNet(
            version=cfg.model_param.version,
            num_classes=cfg.data_param.num_classes,
    )
    elif cfg.model_param.model_name == 'alexnet':
        model = AlexNet(
            num_classes=cfg.data_param.num_classes,
    )

    # 모델 가중치 초기화    
    initialize_weights(model)

    return model.to(cfg.model_param.device)


def fetch_loss_fn(cfg):

    '''
        Config에 맞는 Loss 함수를 반환합니다.
    '''
    
    if cfg.train_param.loss_fn == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    
    return loss_fn.to(cfg.model_param.device)


def fetch_scheduler(optimizer, cfg):

    '''
        Config에 맞는 Scheduler를 반환합니다.
    '''

    if cfg.model_param.scheduler == 'cos_ann_warm':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.train_param.T_0, 
            eta_min=float(cfg.train_param.min_lr)
        )
    elif cfg.model_param.scheduler == 'rlrp':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            'min',
            factor=cfg.train_param.factor,
            patience=cfg.train_param.patience,
        )
    elif cfg.model_param.scheduler == 'steplr':
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.train_param.step_size, 
            gamma=cfg.train_param.gamma
        )
    elif cfg.model_param.scheduler == 'none':
        scheduler = None

    return scheduler


def fetch_optimizer(model, cfg):

    '''
        Config에 맞는 Solver를 반환합니다.
    '''

    if cfg.model_param.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=float(cfg.train_param.lr), 
            betas=tuple(cfg.train_param.betas), 
            eps=float(cfg.train_param.eps), 
            weight_decay=float(cfg.train_param.weight_decay), 
        )
    elif cfg.model_param.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=float(cfg.train_param.lr), 
            betas=tuple(cfg.train_param.betas), 
            eps=float(cfg.train_param.eps), 
            weight_decay=float(cfg.train_param.weight_decay), 
        )
    elif cfg.model_param.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=float(cfg.train_param.lr), 
            momentum=cfg.train_param.momentum,
            weight_decay=float(cfg.train_param.weight_decay), 
        )

    return optimizer

def get_accuracy(logits, target, topk=(1,)):

    """
        상위 K개 Predictions의 정확도를 계산합니다.
    """

    with torch.no_grad():

        # K 중 가장 크기가 큰 것을 Lower Bound로 사용
        maxk = max(topk)
        batch_size = target.size(0)

        # Top K개 선택
        values, preds = logits.topk(k=maxk, dim=1, largest=True, sorted=True)
        # Transpose(행이 상위 i에 대한 인덱스, 열이 배치 속 데이터 j에 대한 인덱스)
        preds = preds.t()
        # Element-wise로 예측 값과 GT 값을 비교(Flatten된 GT를 예측값의 차원에 맞춰 복사)
        correct = preds.eq(target.view(1, -1).expand_as(preds))

        res = []
        # 여러 상위 K에 대한 계산
        for k in topk:
            # 상위 1번째부터 K번째까지만 계산(전체 예측 중 맞은 것 계수 합산)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # 백분율로 변환
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def wandb_logging(
    train_epoch_loss,
    train_acc1,
    train_acc5,
    valid_epoch_loss,
    valid_acc1,
    valid_acc5,
    end_time,
    lr
):

    """
        학습 결과물을 WANDB에 기록합니다.
    """

    wandb.log({"Train Loss": train_epoch_loss})
    wandb.log({"Train Acc@1": train_acc1})
    wandb.log({"Train Acc@5": train_acc5})
    wandb.log({"Valid Loss": valid_epoch_loss})
    wandb.log({"Valid Acc@1": valid_acc1})
    wandb.log({"Valid Acc@5": valid_acc5})
    wandb.log({"One Epoch Train Time(min)": end_time})
    wandb.log({"Learning Rate": lr})

class AverageMeter(object):

    """
        평균 값과 현재 값을 계산하고 저장합니다.
    """

    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


