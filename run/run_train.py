import os
import time
import torch
import wandb
import argparse

from source.dataset import get_dataloader
from models.configs.config_manager import ConfigManager
from source.train import one_epoch_training, one_epoch_validating
from source.utils import (
    FONT, fetch_model, fetch_loss_fn, fetch_optimizer, fetch_scheduler, 
    wandb_logging, save_checkpoint, load_checkpoint, set_seed
)


def run_training(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    checkpoint_dir,
    cfg,
    status
):

    '''
        Configuration을 통해 설정한 에폭만큼 모델을 학습시키고 평가합니다.
    '''

    wandb.watch(
        model, 
        log_freq=cfg.program_param.log_freq # 그래디언트와 파라미터를 log_freq마다 기록
    )

    # 파라미터 초기화
    best_acc1 = status['accuracy']
    best_file = None if 'load_path' not in cfg else cfg.load_path 
    start_epoch = status['epoch'] + 1
    end_epoch = cfg.train_param.epochs+1 if status==None else start_epoch+cfg.train_param.epochs

    for epoch in range(start_epoch, end_epoch):

        # 1 에폭 학습
        start_time = time.time()
        train_epoch_loss, train_acc1, train_acc5 = one_epoch_training(train_loader, model, criterion, optimizer, scheduler, cfg.model_param.scheduler, epoch, cfg.model_param.device)
        end_time = (time.time()-start_time)/60

        # 모델 평가
        valid_epoch_loss, valid_acc1, valid_acc5 = one_epoch_validating(valid_loader, model, criterion, epoch, cfg.model_param.device)
        
        # Scheduler에 따라 Learning Rate 조정
        if scheduler is not None:
            if cfg.model_param.scheduler == 'steplr':
                scheduler.step()
            elif cfg.model_param.scheduler == 'rlrp':
                scheduler.step(valid_epoch_loss)

        print(f'Top 1 Validation Accuracy Of Epoch {epoch}: {valid_acc1:.4f} (Best: {best_acc1:.4f})')
        
        # 정확도 갱신
        if valid_acc1 > best_acc1:
            print(f"{FONT['b']}Validation Accuracy Improved ({best_acc1:.4f} ---> {valid_acc1:.4f})")
            best_acc1 = valid_acc1
            
            # 이전 베스트 모델 삭제
            if best_file is None:
                best_file = f'{checkpoint_dir}/[{cfg.training_keyword.upper()}]_SCHEDULER_{cfg.model_param.scheduler.upper()}_EPOCH_{epoch}_ACC_{best_acc1:.4f}.pt'
            else:
                os.remove(best_file)
                best_file = f'{checkpoint_dir}/[{cfg.training_keyword.upper()}]_SCHEDULER_{cfg.model_param.scheduler.upper()}_EPOCH_{epoch}_ACC_{best_acc1:.4f}.pt'
                        
            save_checkpoint(
                best_file, 
                model, 
                optimizer, 
                scheduler, 
                epoch, 
                valid_epoch_loss, 
                valid_acc1,
            )

            print(f"{FONT['r']}Model Saved{FONT['reset']}")

        # WANDB에 기록
        wandb_logging(
            train_epoch_loss,
            train_acc1,
            train_acc5,
            valid_epoch_loss,
            valid_acc1,
            valid_acc5,
            end_time,
            optimizer.param_groups[0]['lr']
        )

def main(cfg):

    # 시드 설정
    set_seed(cfg.program_param.seed)

    # 원격 WANDB 설정
    wandb.login(key=cfg.program_param.wandb_key)
    cfg.group = f'{cfg.program_param.project_name}/{cfg.model_param.model_name}'

    # 학습 파라미터 설정
    model = fetch_model(cfg)
    criterion = fetch_loss_fn(cfg)
    optimizer = fetch_optimizer(model, cfg)
    scheduler = fetch_scheduler(optimizer,cfg)

    # 모델 저장 경로 설정
    checkpoint_dir = f"{cfg.program_param.save_dir}/{cfg.model_param.model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 데이터로더 설정
    train_csv_path = os.path.join(
        cfg.data_param.dir_path,
        cfg.data_param.dataset_name,
        cfg.data_param.train_csv
    )
    valid_csv_path = os.path.join(
        cfg.data_param.dir_path,
        cfg.data_param.dataset_name,
        cfg.data_param.valid_csv
    )
    train_loader = get_dataloader(train_csv_path, cfg, mode='Train')
    valid_loader = get_dataloader(valid_csv_path, cfg, mode='Valid')

    # WANDB 기록 설정
    run = wandb.init(
        project=cfg.program_param.project_name,
        config=cfg,
        job_type='Train',
        group=cfg.group,
        tags=[
            cfg.model_param.model_name,
            cfg.train_param.loss_fn
        ],
        name=cfg.training_keyword,
        resume=args.resume
    )

    # 사용중인 GPU 이름 출력
    if torch.cuda.is_available():
        print(f"\n{FONT['g']}[INFO] Using GPU: {torch.cuda.get_device_name()}")

    # 모델 메모리 출력
    print(f"[INFO] Allocated Cuda Memory For Model: {torch.cuda.memory_allocated()/1024**2:.3f}MB\n{FONT['reset']}")

    status = {
        'accuracy':0,
        'epoch':0,
    }

    # 전이 학습
    if cfg.model_param.is_pretrained:
        model.pretrained(cfg.model_param.state_dict_path)

    # 모델 이어 학습
    elif 'load_path' in cfg:
        print(f'{FONT["g"]}[RESUME] Loading Model: {cfg.load_path.split("/")[-1]}\n{FONT["reset"]}')
        epoch, accuracy = load_checkpoint(cfg.load_path, model, optimizer, scheduler)
        status = {
            'epoch':epoch,
            'accuracy':accuracy,
        }

    # 학습 진행
    run_training(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        checkpoint_dir,
        cfg,
        status
    )

    run.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file",type=str,required=True,help="Type Configurate File Name.")
    parser.add_argument("--wandb-key", type=str, required=True,help="Type WANDB Key For Logs.")
    parser.add_argument("--training-keyword", type=str, required=True,help="Type Keyword Of This Training.")
    parser.add_argument("--resume",action='store_true',default=False,help="Toggle If You Want To Resume Train.")
    parser.add_argument("--load-path", type=str, help="Type Path To Checkpoint File.")

    args = parser.parse_args()

    cfg = ConfigManager(args).cfg
    main(cfg)