import cv2
import pandas as pd

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Normalize, CenterCrop
from albumentations.augmentations.geometric.resize import SmallestMaxSize


class ClassificationDataset(Dataset):

    """
        데이터프레임을 입력받아 데이터셋을 구축합니다.
    """

    def __init__(
        self, 
        dataframe, 
        image_root, 
        is_target,
        max_size,
        crop_size,
    ):

        super().__init__()

        # ImageNet-1K의 픽셀 평균과 표준편차 값 
        self.imagenet_means = [0.485, 0.456, 0.406]
        self.imagenet_stds = [0.229, 0.224, 0.225]

        self.is_target = is_target
        self.transform = self.get_transform(max_size,crop_size)
        self.df = dataframe
        self.image_root = image_root
        
    def __getitem__(self, index):
        
        path = self.df.iloc[index]['path']
    
        image = cv2.imread(f"{self.image_root}/{path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image = transformed["image"]
        
        # 이미지와 타겟 값을 같이 반환함
        if self.is_target:
            # collate_fn이 Array인 타겟을 Tensor로 변환함 
            target = self.df.iloc[index]['target']
                       
            return image, target
        
        return image
    
    def __len__(self):
        return len(self.df)

    def get_transform(self,max_size,crop_size):

        '''
            ImageNet-1K의 이미지에 대한 Augmentations로, 
            이미지를 Isotropically-Rescaled(Aspect Ratio를 유지하면서 Resize하는 것)한 후 
            여러 Augmentations를 진행합니다.
            마지막에는 array 데이터를 tensor로 변환합니다.
        '''

        return Compose([
            SmallestMaxSize(max_size=max_size,p=1. if max_size!=-1 else 0),
            CenterCrop(crop_size,crop_size,p=1. if crop_size!=-1 else 0),
            Normalize(mean=self.imagenet_means,std=self.imagenet_stds,max_pixel_value=255.0),
            ToTensorV2(p=1.),
        ])


def get_dataloader(csv_path, cfg, mode='Train'):

    '''
        데이터프레임 기반의 데이터로더 반환합니다.
        mode: ['Train', 'Valid', 'Test']
    '''

    is_target = False if mode == 'Test' else True

    dataframe = pd.read_csv(csv_path)

    dataset = ClassificationDataset(
        dataframe, 
        cfg.data_param.dir_path, 
        is_target, 
        cfg.train_param.max_size,
        cfg.train_param.crop_size,
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train_param.batch_size if mode=='Train' else cfg.valid_param.batch_size,
        pin_memory=cfg.train_param.pin_memory if mode=='Train' else cfg.valid_param.pin_memory,
        drop_last=cfg.train_param.drop_last if mode=='Train' else cfg.valid_param.drop_last,
        shuffle=cfg.train_param.shuffle if mode=='Train' else cfg.valid_param.shuffle,
        num_workers=cfg.train_param.num_workers
    )

    return dataloader