# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from thrid_party.DynamicViT.optim_factory import create_optimizer, LayerDecayValueAssigner

from trainer.dyvit.engine import train_one_epoch, evaluate
from trainer.dyvit.losses import DistillDiffPruningLoss_dynamic

from thrid_party.DynamicViT.utils import NativeScalerWithGradNormCount as NativeScaler
import thrid_party.DynamicViT.utils as utils
from thrid_party.DynamicViT.datasets import build_dataset, build_transform
#from thrid_party.DynamicViT.samplers import RASampler
# from thrid_party.DynamicViT.models.dyconvnext import ConvNeXt_Teacher, AdaConvNeXt
# from thrid_party.DynamicViT.models.dylvvit import LVViTDiffPruning, LVViT_Teacher
# from thrid_party.DynamicViT.models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
# from thrid_party.DynamicViT.models.dyswin import AdaSwinTransformer, SwinTransformer_Teacher
from thrid_party.DynamicViT.calc_flops import calc_flops, throughput

import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Dynamic training script', add_help=False)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='deit-small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=utils.str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=utils.str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=utils.str2bool, default=True, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096 <-- this is DyViT setup. ABT is different')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=utils.str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/d1/dataset/ILSVRC2012/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='./saves/dyvit-concrete/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=utils.str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=utils.str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=utils.str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=utils.str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=utils.str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=utils.str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--lr_scale', type=float, default=0.01)
    parser.add_argument('--base_rate', type=float, default='0.9')

    # concrete masking settings
    parser.add_argument('--p-logit', type=float, default=0.0)
    parser.add_argument('--approx-factor', type=int, default=4)
    parser.add_argument('--max-hard-train-epochs', type=int, default=999,
                        help='if you set this 0, then hard mask train is disabled')

    return parser

def log(*args):
    print("DyVitConcreteTrainer:", *args)

import models.sparse_token as sparse
from trainer import vit_approx_trainer as vit_approx
from utils import ddp
import transformers

def load_concrete_model(model_id = 'deit-small', factor=4, p_logit=0.0):
    if model_id in ['vit-base', 'deit-base', 'deit-small']:
        model_cls = transformers.ViTForImageClassification
    elif model_id in ['deit-base-distilled', 'deit-small-distilled']:
        model_cls = transformers.DeiTForImageClassification
    else: raise Exception()
    model_id_hf = vit_approx.finetuned_to_hf['imagenet'][model_id]
    model = model_cls.from_pretrained(model_id_hf)
    log('Base model loaded from', model_id_hf)

    approx_bert = sparse.ApproxBertModel(
        model.config, 
        factor=factor, 
        arch='vit',
        ignore_pred=True
    )
    vit_approx.load_state_dict_interpolated(approx_bert.bert, vit_approx.get_vit(model).state_dict())
    approx_bert = ddp.MimicDDP(approx_bert)
    #load from state_dict
    path = f'./saves/vit-approx-{model_id}-base-{factor}.pth'
    state = torch.load(path, map_location='cpu')
    try:
        approx_bert.load_state_dict(state['approx_bert'])
    except Exception as ex:
        log('Error while loading approx bert')
        log(ex)
    log('Checkpoint loaded', path, {
        'epoch': state['epoch'], 
        'epochs': state['epochs'],
        'subset': state['subset'],
        'factor': state['factor']
    })
    del state
    approx_bert = approx_bert.module

    concrete_model = sparse.ApproxSparseBertForSequenceClassification(
        model.config,
        approx_bert,
        arch = 'vit',
        add_pooling_layer=False,
    )
    log('ConcreteModel problem define', model.config.problem_type)
    assert hasattr(concrete_model.bert.encoder, 'concrete_loss_encoder_mask_avg_factor')
    concrete_model.bert.encoder.concrete_loss_encoder_mask_avg_factor = 100.0 # this is different with NLP tasks :?
    for layer in concrete_model.bert.encoder.layer:
        assert hasattr(layer, 'concrete_loss_factor')
        layer.concrete_loss_factor = 1e-3 # ease the factor, and let ratio decide it.
    
    try:
        concrete_model.bert.load_state_dict(
            vit_approx.get_vit(model).state_dict(),
            strict=True,
        )
    except Exception as ex:
        log('load vit', ex)
    
    try:
        concrete_model.classifier.load_state_dict(
            model.classifier.state_dict(),
            strict=True,
        )
    except Exception as ex:
        log('load classifier', ex)
    
    concrete_model.use_concrete_masking = True
    concrete_model.bert.set_concrete_init_p_logit(p_logit)

    return concrete_model, model

def main(args):
    utils.init_distributed_mode(args)
    log(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    args.nb_classes = 1000

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    log("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            log('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        log("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    log(args.model)

    SPARSE_RATIO = [args.base_rate, args.base_rate - 0.2, args.base_rate - 0.4]

    if args.model == 'deit-small':
        #TODO: add args factor
        model, teacher_model = load_concrete_model(
            model_id=args.model, factor=args.approx_factor, p_logit=args.p_logit
        )
        teacher_model.eval()
        teacher_model = teacher_model.to(device)

        model.config.problem_type = 'custom'
        model.loss_fct = criterion

        log('Criterion', criterion)
        
        criterion = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion, clf_weight=1.0, mse_token=True, ratio_weight=2.0, distill_weight=0.5
        )
    
    # if use_teacher:
    #     if 'convnext' in args.model or 'deit' in args.model or 'swin' in args.model:
    #         pretrained = pretrained['model']
    #     utils.load_state_dict(model, pretrained)
    #     utils.load_state_dict(teacher_model, pretrained)
    #     teacher_model.eval()
    #     teacher_model = teacher_model.to(device)
    #     print('success load teacher model weight')
    
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        log("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #print("Model = %s" % str(model_without_ddp))
    log('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    log("LR = %.8f" % args.lr)
    log("Batch size = %d" % total_batch_size)
    log("Update frequent = %d" % args.update_freq)
    log("Number of training examples = %d" % len(dataset_train))
    log("Number of training training per epoch = %d" % num_training_steps_per_epoch)
    
    assigner = None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    #TODO!: remove weight decay on threshold and p-logit
    skip_list = []
    for name, param in model_without_ddp.named_parameters():
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'p_logit']
        if any([d in name for d in no_decay]):
            skip_list.append(name)
    #log(skip_list, 'should be skip weight decay')
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None,
        bone_lr_scale=1.0, old_wegiht_fix_epochs=0)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    log("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    log(lr_schedule_values)
    for epoch in range(args.epochs):
        log(f'lr@e{epoch}', lr_schedule_values[num_training_steps_per_epoch*epoch])
    # import plotext as plt
    # plt.plot(lr_schedule_values)
    # plt.title("learning rate")
    # plt.show()

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    log("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))


    #print("criterion = %s" % str(criterion))

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    max_accuracy, max_accuracy_ema = utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        log(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        log(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    log("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    def evaluate_concrete(dataloader, model, device, use_amp=False):
        model_without_ddp = model
        if hasattr(model, 'module'):
            model_without_ddp = model.module
        
        sparse.benchmark_reset()
        model_without_ddp.bert.set_concrete_hard_threshold(None)
        soft_result = evaluate(dataloader, model, device, use_amp=use_amp)
        soft_occupy = sparse.benchmark_get_average('concrete_occupy')
        log(f'Soft Occupy: {soft_occupy}')

        sparse.benchmark_reset()
        model_without_ddp.bert.set_concrete_hard_threshold(0.5)
        hard_result = evaluate(dataloader, model, device, use_amp=use_amp)
        hard_occupy = sparse.benchmark_get_average('concrete_occupy')
        log(f'Hard Occupy: {hard_occupy}')

        return {
            **{f'soft_{k}': v for k, v in soft_result.items()},
            'soft_occupy': soft_occupy,
            **{f'hard_{k}': v for k, v in hard_result.items()},
            'hard_occupy': hard_occupy,
            **hard_result,
        }
    
    # if data_loader_val is not None:
    #     test_stats = evaluate_concrete(data_loader_val, model, device, use_amp=args.use_amp)
    #     log(f'Test stat. Before training @ epoch {args.start_epoch}', test_stats)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        if epoch >= max(
            args.epochs - args.max_hard_train_epochs, 
            min(args.epochs - 1, round((args.epochs - 1) * 0.9))
        ):
            model_without_ddp.bert.set_concrete_hard_threshold(0.5)
            log('Hard training')
        else:
            model_without_ddp.bert.set_concrete_hard_threshold(None)
            log('Soft training')

        if epoch > -1: # for debugging purpose
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
                use_amp=args.use_amp
            )
        else: train_stats = {}

        if data_loader_val is not None:
            test_stats = evaluate_concrete(data_loader_val, model, device, use_amp=args.use_amp)
            log('Test stat.', test_stats)
            log(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.2f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema, best_acc=max_accuracy, best_acc_ema=max_accuracy_ema)
            log(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'factor': args.approx_factor,
                         'p_logit': args.p_logit,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate_concrete(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                log('Test stat. EMA', test_stats_ema)
                log(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.2f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema, best_acc=max_accuracy, best_acc_ema=max_accuracy_ema)
                log(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, best_acc=max_accuracy, best_acc_ema=max_accuracy_ema)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
