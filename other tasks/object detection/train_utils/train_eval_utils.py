import math
import sys
import time

import torch
import torch.nn.functional as F

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils


def kl_div(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit / T, dim=-1)
                                  - F.log_softmax(q_logit / T, dim=-1)), 1)
    return torch.mean(kl)

def kl_div_mask(p_logit, q_logit, T, mask, s_ctrl=0.5):
    p = F.softmax(p_logit / T, dim=-1)
    kl = ( 1 - (1-mask) + s_ctrl*(1-mask) ) * ( torch.sum(p * (F.log_softmax(p_logit / T, dim=-1)
                                  - F.log_softmax(q_logit / T, dim=-1)), 1) )
    return torch.mean(kl)

def train_one_epoch(model, model_res50, optimizer, optimizer_res50, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    model_res50.train()

    flag = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    lr_scheduler_res50 = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        lr_scheduler_res50 = utils.warmup_lr_scheduler(optimizer_res50, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        # 1. transform & feature extraction
        images_list, _ = model.transform(images, targets)
        feats_a = model.backbone(images_list.tensors)
        feats_b = model_res50.backbone(images_list.tensors)

        # ensure dict format for FPN/backbone
        if isinstance(feats_a, torch.Tensor):
            feats_a = OrderedDict([('0', feats_a)])
        if isinstance(feats_b, torch.Tensor):
            feats_b = OrderedDict([('0', feats_b)])

        # 2. shared proposals from model A's RPN
        with torch.no_grad():   # 不把 proposals 的梯度计入
            proposals, _ = model.rpn(images_list, feats_a, targets)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_dict_res50 = model_res50(images, targets)
            losses_res50 = sum(loss for loss in loss_dict_res50.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 4. extract logits for DML on shared proposals
        box_feats_a = model.roi_heads.box_roi_pool(feats_a, proposals, images_list.image_sizes)
        box_feats_a = model.roi_heads.box_head(box_feats_a)
        logits_a, _ = model.roi_heads.box_predictor(box_feats_a)
        #   B branch
        box_feats_b = model_res50.roi_heads.box_roi_pool(feats_b, proposals, images_list.image_sizes)
        box_feats_b = model_res50.roi_heads.box_head(box_feats_b)
        logits_b, _ = model_res50.roi_heads.box_predictor(box_feats_b)

        with torch.no_grad():
            # 提取真实边界框和标签
            gt_boxes = [t["boxes"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            # 正确调用方法（三个参数）
            matched_idxs, labels = model.roi_heads.assign_targets_to_proposals(
                proposals, gt_boxes, gt_labels
            )
            # 拼接所有图片的标签
            all_labels = torch.cat(labels, dim=0)  # [total_proposals]
            # 创建one-hot编码的真实标签
            num_classes = logits_a.shape[1]
            targets_onehot = torch.zeros_like(logits_a)
            targets_onehot.scatter_(1, all_labels.unsqueeze(1).long(), 1)
            true_classes = all_labels.long() 
            pred_classes_b = logits_b.max(1)[1] 
            mask_t = (pred_classes_b == true_classes).float() 
            s_label = torch.abs(logits_a - targets_onehot).mean()
            t_label = torch.abs(logits_b - targets_onehot).mean()
            ps_pt = torch.abs(logits_a - logits_b).mean()
            epsilon = torch.exp(-(2-t_label) *  (t_label / (t_label + s_label)) )
            delta = s_label - epsilon * t_label

            if ps_pt > delta and t_label < s_label:
                flag = 1
            else:
                flag = 0

        if flag == 1:
            total_a = losses + 2 * 4 * 4 * kl_div_mask(logits_b.detach(), logits_a, 4, mask_t, s_ctrl=t_label/2)
        else:
            total_a = losses + 2 * 2 * 2 * kl_div(logits_b.detach(), logits_a, 2)

        total_b = losses_res50 + 2 * 2 * kl_div(logits_a.detach(), logits_b, 2)

        optimizer.zero_grad()
        optimizer_res50.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if flag == 1:
                total_a.backward(retain_graph=True)
                optimizer.step()
            else:
                total_a.backward(retain_graph=True)
                optimizer.step()
                total_b.backward()
                optimizer_res50.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()
        if lr_scheduler_res50 is not None:  # 第一轮使用warmup训练方式
            lr_scheduler_res50.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
