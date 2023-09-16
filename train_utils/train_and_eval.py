import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
import train_utils.focal_loss as floss
import random
import logging


def criterion(epoch, self_learning_loss, inputs, target, label, device, loss_weight=None, num_classes: int = 2, dice: bool = False, ignore_index: int = -100):
#     x = inputs['out']
#     cls = inputs['cls']
    cls = inputs
    losses = {}
    
    # loss_cls = nn.functional.cross_entropy(cls, label, weight=loss_weight)
    loss_cross_entropy = nn.CrossEntropyLoss(weight=loss_weight)
    loss_cls = loss_cross_entropy(cls, label)
    
#     dice_target = build_target(target, num_classes, ignore_index)
#     loss_dice = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    
#     softmax = nn.Softmax(dim=1)
#     focal_loss = floss.FocalLoss()
#     loss_focal = focal_loss(softmax(x), target)
    
#     loss_seg = loss_dice + loss_focal

    # 自学习的损失
    # losses['out'] = self_learning_loss(loss_seg, loss_cls)

    losses['out'] = loss_cls
    return losses['out']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
#     confmat = utils.ConfusionMatrix(num_classes)
#     dice = utils.DiceCoefficient(num_classes=num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    cls_confusion_matrix = torch.zeros(size=(3, 3), dtype=int)
    with torch.no_grad():
        for image, target, label in metric_logger.log_every(data_loader, 100, header):
            image, target, label = image.to(device), target.to(device), label.to(device)
            target = target.reshape(len(target), target.shape[2], target.shape[3]).long()
            output = model(image)
#             output_cls = output['cls']
            output_cls = output.argmax(1)
            
#             output_seg = output['out']
#             confmat.update(target.flatten(), output_seg.argmax(1).flatten())
#             dice.update(output_seg, target)
            
            cls_confusion_matrix[int(output_cls)][int(label)] += 1

#         confmat.reduce_from_all_processes()
#         dice.reduce_from_all_processes()
    print(cls_confusion_matrix)
    acc = torch.diag(cls_confusion_matrix).sum() / cls_confusion_matrix.sum()
    pre = torch.diag(cls_confusion_matrix) / cls_confusion_matrix.sum(1)
    rec = torch.diag(cls_confusion_matrix) / cls_confusion_matrix.sum(0)
    f1_score = 2 * pre.mean() * rec.mean() / (pre.mean() + rec.mean())
    print(acc, pre.mean(), rec.mean())
    print('f1_score =', f1_score, 'acc =', acc)
    logging.info('f1_score={}, top_1 accuracy = {}, pre = {}, recall = {}'.format(f1_score, acc, pre.mean(), rec.mean()))
    return acc, pre, rec, f1_score
#     logging.info("confmat={}, dice={}".format(confmat.mat, dice.value.item))
#     return acc, pre, rec, f1_score, confmat, dice.value.item()


def train_one_epoch(model, self_learning_loss, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    loss_weight = torch.as_tensor([2.0, 1.0, 1.0], device=device)

    for image, target, label in metric_logger.log_every(data_loader, print_freq, header):
        image, target, label = image.to(device), target.to(device), label.to(device)
        target = target.reshape(len(target), target.shape[2], target.shape[3]).long()
        label = label.long()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(epoch, self_learning_loss, output, target, label, device, loss_weight=loss_weight, num_classes=num_classes)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
