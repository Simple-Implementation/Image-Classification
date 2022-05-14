import torch

from tqdm import tqdm

from source.utils import FONT, AverageMeter, get_accuracy

def one_epoch_training(dataloader, model, criterion, optimizer, scheduler, scheduler_name, epoch, device):

    '''
        하나의 에폭에 대한 학습을 진행합니다.
    '''

    # 평가 지표 초기화
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 학습 모드로 전환
    model.train()

    pbar = tqdm(dataloader, total=len(dataloader),desc=f"{FONT['start']}{FONT['c']}{'Training':<11}[{epoch}]{FONT['end']}")

    for images, target in pbar:

        batch_size = images.size(0)

        images = images.to(device)
        target = target.to(device)

        # Logits 계산
        logits = model(images)
        loss = criterion(logits, target)

        # 정확도와 Loss를 기록
        acc1, acc5 = get_accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        # 그래디언트를 계산하고 SGD Step을 수행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Scheduler에 따라 Learning Rate 조정
        if scheduler is not None and scheduler_name == 'cos_ann_warm':
            scheduler.step()

        # 진행 상태를 업데이트
        pbar.set_postfix(
            Top1=top1.avg, 
            Train_Loss=losses.avg,
            LR=optimizer.param_groups[0]['lr']
        )

    return losses.avg, top1.avg, top5.avg


def one_epoch_validating(dataloader, model, criterion, epoch, device):

    '''
        하나의 에폭에 대한 모델 평가를 진행합니다.
    '''

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 평가 모드로 전환
    model.eval()

    pbar = tqdm(dataloader,total=len(dataloader),desc=f"{FONT['start']}{FONT['y']}{'Validating':<11}[{epoch}]{FONT['reset']}")

    # 그래디언트가 흐르지 않게 함
    with torch.no_grad():
        for images, target in pbar:

            batch_size = images.size(0)

            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(),batch_size)

            pbar.set_postfix(
                Top1=top1.avg, 
                Valid_Loss=losses.avg,
            )

    return losses.avg, top1.avg, top5.avg
