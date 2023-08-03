# Standard lib imports
import time
import numpy as np
import math
import os.path as osp
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from detectron2.structures import ImageList
from detectron2.data.samplers import TrainingSampler
# PyTorch imports
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

# Local imports
from utils.meters import average_accuracy
from utils import AverageMeter
from utils import compute_mask_IoU
from utils.generate_unlabel_data import generate_data
from data import PanopticNarrativeGroundingDataset, PanopticNarrativeGroundingValDataset,PanopticNarrativeGroundingLabeledDataset,PanopticNarrativeGroundingUnlabeledDataset,DatasetTwoCrop
from models.knet.knet import KNet
from models.knet.dice_loss import DiceLoss
from models.knet.cross_entropy_loss import CrossEntropyLoss
from models.encoder_bert import BertEncoder
from utils.logger import setup_logger
from utils.collate_fn import default_collate
from utils.distributed import (all_gather, all_reduce)
from models.extract_fpn_with_ckpt_load_from_detectron2 import fpn
from utils.contrastive import CKDLoss
from collections import OrderedDict
import itertools
import os, psutil

import torchvision.transforms as transforms 
import torchvision.transforms.functional as transforms_f 

# others
import yaml
from skimage import measure


def transform_input(data):
    new_data=[]
    assert len(data)>0
    input_len=len(data[0])
    for k in range(input_len):
        tmp=[data[batch][k] for batch in range(len(data))]
        if type(tmp[0]==list):
            merge=itertools.chain.from_iterable(tmp)
            tmp = list(merge)
        new_data.append(tmp)
    return new_data

def upsample_eval(tensors, pad_value=0, t_size=[400, 400]):
    batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(t_size)
    batched_imgs = tensors[0].new_full(batch_shape, pad_value)
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

@torch.no_grad()
def my_evaluate(val_loader, bert_encoder,fpn_model,model,cfg):
    model.eval()
    bert_encoder.eval()
    fpn_model.eval()
    
    instances_iou = []
    singulars_iou = []
    plurals_iou = []
    things_iou = []
    stuff_iou = []
    pbar = tqdm(total=len(val_loader))
    for (batch_idx, (caption, grounding_instances, ann_categories, \
        ann_types, noun_vector_padding, ret_noun_vector, fpn_input_data)) in enumerate(val_loader):
        ann_categories = ann_categories.to(cfg.local_rank)
        ann_types = ann_types.to(cfg.local_rank)
        # ret_noun_vector = ret_noun_vector.to(cfg.local_rank)
        
        # Perform the forward pass
        with torch.no_grad():
            lang_feat, _ = bert_encoder(caption) #bert for caption
            lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
                cfg.max_seg_num, lang_feat.shape[-1]))

            for i in range(len(lang_feat)):
                cur_lang_feat = lang_feat[i][noun_vector_padding[i].nonzero().flatten()]
                lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat


        fpn_feature = fpn_model(fpn_input_data)
        predictions = model(fpn_feature, lang_feat_valid, train=False)
        predictions = predictions[cfg.test_stage]
        predictions = predictions.sigmoid() #[2,230,272,304]

        predictions_valid = predictions.new_zeros((predictions.shape[0], cfg.max_phrase_num, \
            predictions.shape[-2], predictions.shape[-1]))
        for i in range(len(predictions)):
            cur_phrase_interval = ret_noun_vector[i]['inter']
            for j in range(len(cur_phrase_interval)-1):
                for k in range(cur_phrase_interval[j], cur_phrase_interval[j+1]):
                    predictions_valid[i, j, :] = predictions_valid[i, j, :] + predictions[i][k]
                predictions_valid[i, j, :] = predictions_valid[i, j, :] / (cur_phrase_interval[j+1]-cur_phrase_interval[j])
                
        predictions = (predictions_valid > 0.5).float()
        predictions = upsample_eval(predictions)

        # preprocessing for gt masks
        with torch.no_grad():
            gts = [F.interpolate(grounding_instances[i]["gt"].to(cfg.local_rank).unsqueeze(0), \
                                (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                mode='bilinear').squeeze() for i in range(len(grounding_instances))]
            gts = ImageList.from_tensors(gts, 32).tensor
            gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
            gts = (gts > 0).float()
            gts = upsample_eval(gts)
        
        # Gather all the predictions across all the devices.
        if cfg.num_gpus > 1:
            predictions, gts, ann_categories, ann_types = all_gather(
                [predictions, gts, ann_categories, ann_types]
            )

        # Evaluation
        


        for p, t, th, s in zip(predictions, gts, ann_categories, ann_types):
            for i in range(cfg.max_phrase_num):
                if s[i] == 0:
                    continue
                else:
                    pd = p[i]
                    _, _, instance_iou = compute_mask_IoU(pd, t[i])
                    instances_iou.append(instance_iou.cpu().item())
                    if s[i] == 1:
                        singulars_iou.append(instance_iou.cpu().item())
                    else:
                        plurals_iou.append(instance_iou.cpu().item())
                    if th[i] == 1:
                        things_iou.append(instance_iou.cpu().item())
                    else:
                        stuff_iou.append(instance_iou.cpu().item())
        
        if dist.get_rank()==0:
            pbar.update(1)
    
    pbar.close()
    # Final evaluation metrics
    AA = average_accuracy(instances_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='overall')
    AA_singulars = average_accuracy(singulars_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='singulars')
    AA_plurals = average_accuracy(plurals_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='plurals')
    AA_things = average_accuracy(things_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='things')
    AA_stuff = average_accuracy(stuff_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='stuff')
    accuracy = accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5)
    if dist.get_rank()==0:
        print('| final acc@0.5: {:.5f} | final AA: {:.5f} |  AA singulars: {:.5f} | AA plurals: {:.5f} | AA things: {:.5f} | AA stuff: {:.5f} |'.format(
                                                accuracy,
                                                AA,
                                                AA_singulars,
                                                AA_plurals,
                                                AA_things,
                                                AA_stuff))
    
    return AA

@torch.no_grad()
def evaluate(val_loader, bert_encoder,fpn_model,model,iteration,cfg,logger,writer):
    model.eval()
    bert_encoder.eval()
    fpn_model.eval()
    
    instances_iou = []
    singulars_iou = []
    plurals_iou = []
    things_iou = []
    stuff_iou = []
    # pbar = tqdm(total=len(val_loader))
    for (batch_idx, (caption, grounding_instances, ann_categories, \
        ann_types, noun_vector_padding, ret_noun_vector, fpn_input_data)) in enumerate(val_loader):
        ann_categories = ann_categories.to(cfg.local_rank)
        ann_types = ann_types.to(cfg.local_rank)
        # ret_noun_vector = ret_noun_vector.to(cfg.local_rank)
        
        # Perform the forward pass
        with torch.no_grad():
            lang_feat, _ = bert_encoder(caption) #bert for caption
            lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
                cfg.max_seg_num, lang_feat.shape[-1]))

            for i in range(len(lang_feat)):
                cur_lang_feat = lang_feat[i][noun_vector_padding[i].nonzero().flatten()]
                lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat


        fpn_feature = fpn_model(fpn_input_data)
        predictions = model(fpn_feature, lang_feat_valid, train=False)
        predictions = predictions[cfg.test_stage]
        predictions = predictions.sigmoid() #[2,230,272,304]

        predictions_valid = predictions.new_zeros((predictions.shape[0], cfg.max_phrase_num, \
            predictions.shape[-2], predictions.shape[-1]))
        for i in range(len(predictions)):
            cur_phrase_interval = ret_noun_vector[i]['inter']
            for j in range(len(cur_phrase_interval)-1):
                for k in range(cur_phrase_interval[j], cur_phrase_interval[j+1]):
                    predictions_valid[i, j, :] = predictions_valid[i, j, :] + predictions[i][k]
                predictions_valid[i, j, :] = predictions_valid[i, j, :] / (cur_phrase_interval[j+1]-cur_phrase_interval[j])
                
        predictions = (predictions_valid > 0.5).float()

        predictions = upsample_eval(predictions)

        # preprocessing for gt masks
        with torch.no_grad():
            gts = [F.interpolate(grounding_instances[i]["gt"].to(cfg.local_rank).unsqueeze(0), \
                                (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                mode='bilinear').squeeze() for i in range(len(grounding_instances))]
            gts = ImageList.from_tensors(gts, 32).tensor
            gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
            gts = (gts > 0).float()
            gts = upsample_eval(gts)
        
        # Gather all the predictions across all the devices.
        if cfg.num_gpus > 1:
            predictions, gts, ann_categories, ann_types = all_gather(
                [predictions, gts, ann_categories, ann_types]
            )

        # Evaluation
        for p, t, th, s in zip(predictions, gts, ann_categories, ann_types):
            for i in range(cfg.max_phrase_num):
                if s[i] == 0:
                    continue
                else:
                    pd = p[i]
                    _, _, instance_iou = compute_mask_IoU(pd, t[i])
                    instances_iou.append(instance_iou.cpu().item())
                    if s[i] == 1:
                        singulars_iou.append(instance_iou.cpu().item())
                    else:
                        plurals_iou.append(instance_iou.cpu().item())
                    if th[i] == 1:
                        things_iou.append(instance_iou.cpu().item())
                    else:
                        stuff_iou.append(instance_iou.cpu().item())
    
    # Final evaluation metrics
    AA = average_accuracy(instances_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='overall')
    AA_singulars = average_accuracy(singulars_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='singulars')
    AA_plurals = average_accuracy(plurals_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='plurals')
    AA_things = average_accuracy(things_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='things')
    AA_stuff = average_accuracy(stuff_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='stuff')
    accuracy = accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5)
    if dist.get_rank()==0:
        logger.info('| final acc@0.5: {:.5f} | final AA: {:.5f} |  AA singulars: {:.5f} | AA plurals: {:.5f} | AA things: {:.5f} | AA stuff: {:.5f} |'.format(
                                            accuracy,
                                            AA,
                                            AA_singulars,
                                            AA_plurals,
                                            AA_things,
                                            AA_stuff))
        writer.add_scalar('aa/acc@0.5', accuracy, iteration)
        writer.add_scalar('aa/final', AA, iteration)
        writer.add_scalar('aa/singulars', AA_singulars, iteration)
        writer.add_scalar('aa/plurals', AA_plurals, iteration)
        writer.add_scalar('aa/things', AA_things, iteration)
        writer.add_scalar('aa/stuffs', AA_stuff, iteration)
        
    return AA

def train_semi(cfg):

    #------------------------------1.Init ------------------------------------
    dist.init_process_group(backend=cfg.backend,init_method='env://')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    if dist.get_rank() == 0:
        logger = setup_logger(cfg.output_dir, dist.get_rank())
        writer = SummaryWriter(osp.join(cfg.output_dir, 'tensorboard'))
    else:
        logger, writer = None, None

    with open(cfg.semi_cfg,'r',encoding='utf8') as f:
        semi_cfg=yaml.safe_load(f)
    
    sup_percent = semi_cfg['train']['supervised_percent']
    seed = semi_cfg['train']['seed']
    augmentation = semi_cfg['augmentation'].get('aug_on',False)
    weak_aug = semi_cfg['augmentation']['weak']
    strong_aug = semi_cfg['augmentation']['strong']

    # Set random seed from configs.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if dist.get_rank()==0:
        ann_path = osp.join(cfg.data_dir,'coco/annotations','png_coco_train2017_labeled_dataloader_seed'+str(seed)+'_sup'+str(sup_percent)+'.json')
        if(osp.exists(ann_path)):
            logger.info('datasets are already splited')
        else:
            generate_data(seed,sup_percent,data_dir=osp.join(cfg.data_dir,'coco'))
    else:
        pass
    
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        logger.info(cfg)
    
    #-----------------------------2.load data------------------------------------
    # training : need to load train_loader_labeled,
    train_dataset_labeled = PanopticNarrativeGroundingLabeledDataset(cfg,seed,sup_percent,augmentation=True)
    train_dataset_unlabeled = PanopticNarrativeGroundingUnlabeledDataset(cfg,seed,sup_percent,augmentation=augmentation,weak_aug = weak_aug,strong_aug = strong_aug)
    label_sampler = TrainingSampler(len(train_dataset_labeled)) 
    unlabel_sampler = TrainingSampler(len(train_dataset_unlabeled))

    train_loader_labeled = DataLoader(
        train_dataset_labeled,
        sampler = label_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=default_collate
    )

    train_loader_unlabeled = DataLoader(
        train_dataset_unlabeled,
        sampler = unlabel_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=default_collate
    )

    semi_dataloader=DatasetTwoCrop((train_loader_labeled,train_loader_unlabeled),cfg.batch_size)
    semi_dataloader_iter=iter(semi_dataloader)

    # load val_loader
    val_dataset = PanopticNarrativeGroundingValDataset(cfg, 'val2017', False)
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=12,
        sampler = distributed_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=default_collate,
        drop_last=False
    )
    
    #-----------------------------3.define model------------------------------------
    # supervised model
    bert_encoder = BertEncoder(cfg).to(local_rank)
    bert_encoder = DDP(bert_encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) 
    fpn_model = fpn(cfg.detectron2_ckpt, cfg.detectron2_cfg)
    fpn_model = fpn_model.to(local_rank)
    fpn_model = DDP(fpn_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model = KNet(
        num_stages=cfg.num_stages,
        num_points=cfg.num_points,
    ).to(local_rank)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if cfg.bert_freeze:
        cnt = 0
        for n, c in bert_encoder.named_parameters():
            c.requires_grad = False 
            cnt += 1                   
        if dist.get_rank() == 0:
            logger.info(f'Freezing {cnt} parameters of BERT.')

    if cfg.fpn_freeze:
        cnt = 0
        for n, c in fpn_model.named_parameters():
            c.requires_grad = False
            cnt += 1
        if dist.get_rank() == 0:
            logger.info(f'Freezing {cnt} parameters of FPN.')
    else:
            raise RuntimeError('Not Implement!!!!')
    
    
    if cfg.bert_freeze and cfg.fpn_freeze:
        train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if dist.get_rank() == 0:
            logger.info(f'{len(train_params)} training params.')
        optimizer = optim.Adam(train_params,
                            lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.fpn_freeze:
        bert_encoder_params = list(filter(lambda p: p.requires_grad, bert_encoder.parameters()))
        model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.Adam([{'params': model_params, 'lr':cfg.base_lr},
                               {'params': bert_encoder_params, 'lr':cfg.base_lr/10}])
    else:
        raise RuntimeError('Not Implement!!!!')
    
    if cfg.scheduler == 'step':
        if cfg.fpn_freeze and not cfg.bert_freeze:
            milestones = [10*3000, 12*3000, 14*3000]
            lambda1 = lambda epoch: 1 if epoch < milestones[0] else 0.5 if epoch < milestones[1] else 0.25 if epoch < milestones[2] else 0.125
            lambda2 = lambda epoch: 1
            lambda_list = [lambda1, lambda2]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_list)
        else:
            milestones = [10*3000, 12*3000, 14*3000]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, \
                                                      gamma=0.5)
    elif cfg.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, \
                                                               mode='max', min_lr=1e-6, \
                                                               patience=2)
    else:
        raise ValueError(f'{cfg.scheduler} NOT IMPLEMENT!!!')
    

    #-----------------------------4. Semi-Supervised: burn-in and teacher-student mutual learning------------------------------------
    start_iter,best_val_score = 0 , None
    accuracy=None      

    # ------------------------------4.1 burn-in---------------------------------
    dice_loss = DiceLoss(use_sigmoid=False)
    ce_loss = CrossEntropyLoss(use_sigmoid=False) 
    burnIn_params = semi_cfg.get('burn_in_stage',None)
    if burnIn_params.get('burnIn_on',True):
        burn_in_ckpt = burnIn_params.get('burn_in_ckpt','')
        if osp.exists(burn_in_ckpt):

            model_state_dict=torch.load(burn_in_ckpt,map_location=torch.device('cpu'))
            start_iter = model_state_dict['iteration']+1
            bert_encoder.load_state_dict(model_state_dict['bert_model_state'])
            fpn_model.load_state_dict(model_state_dict['fpn_model_state'])
            model.load_state_dict(model_state_dict['model_state'])
            scheduler.load_state_dict(model_state_dict['scheduler_state'])
            optimizer.load_state_dict(model_state_dict['optimizer_state'])

        if dist.get_rank()==0:
            logger.info('Burn-in stage begins...')
        burnIn_steps = burnIn_params.get('steps',6000)
        for iteration in range(start_iter, burnIn_steps):
            model.train()
            if cfg.bert_freeze:
                bert_encoder.eval()
            else:
                bert_encoder.train()
            if cfg.fpn_freeze:
                fpn_model.eval()
            else:
                fpn_model.train()

            time_stats = AverageMeter()

            label_data,unlabel_data=next(semi_dataloader_iter)
            
            caption, grounding_instances, ann_categories,ann_types, noun_vector_padding, ret_noun_vector, fpn_input_data=transform_input(label_data)

            ret_noun_vector=torch.stack(ret_noun_vector)
            ann_types = torch.stack(ann_types)
            ann_categories = torch.stack(ann_categories)

            ret_noun_vector = ret_noun_vector.to(cfg.local_rank)
            ann_types = ann_types.to(cfg.local_rank)
            ann_categories = ann_categories.to(cfg.local_rank)
            
            start_time = time.time()
            
            with torch.no_grad():
                lang_feat, _ = bert_encoder(caption) #bert for caption
                lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
                    cfg.max_seg_num, lang_feat.shape[-1]))

                for i in range(len(lang_feat)):
                    cur_lang_feat = lang_feat[i][noun_vector_padding[i].nonzero().flatten()]
                    lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat
            
            fpn_feature = fpn_model(fpn_input_data) #fpn for imgs

            with torch.no_grad():
                gts = [F.interpolate(grounding_instances[i]["gt"].to(cfg.local_rank).unsqueeze(0), \
                                    (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                    mode='bilinear').squeeze() for i in range(len(grounding_instances))]
                gts = ImageList.from_tensors(gts, 32).tensor
                gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
                gts = (gts > 0).float()
            
            # gts: [B, max_seg_num, H//4, W//4]
            # lang_feat_valid: [B, max_seg_num, C]
            predictions = model(fpn_feature, lang_feat_valid, train=False) #Knet

            loss = 0 # label_loss
            grad_sample = ann_types != 0
            gt = gts[grad_sample] 
        
            for i in range(len(predictions)):
                pred = predictions[i][grad_sample]
                pred = pred.sigmoid()
                loss = loss + ce_loss(pred, gt) + dice_loss(pred, gt)
            
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters.
            optimizer.step()

            # Gather all the predictions across all the devices.
            if cfg.num_gpus > 1:
                loss = all_reduce([loss])[0]
            
            time_stats.update(time.time() - start_time, 1)
            if dist.get_rank() == 0:
                writer.flush()

            if (iteration+1) % 3000==0:
                accuracy=evaluate(val_loader,bert_encoder,fpn_model,model,iteration,cfg,logger,writer)
            if cfg.scheduler == 'step':
                scheduler.step()
            elif cfg.scheduler == 'reduce':
                scheduler.step(accuracy)
            else:
                raise ValueError(f'{cfg.scheduler} NOT IMPLEMENT!!!')
            if iteration%100==0 and dist.get_rank()==0:
                logger.info('-' * 89)
                logger.info('| end of iteration {:3d} | time: {:5.2f}s | lr: {:.6f}'
                        '| iteration loss {:.6f} |'.format(
                            iteration, time.time() - start_time,optimizer.param_groups[0]["lr"],loss))
                logger.info('-' * 89)
            
            if dist.get_rank()==0 and  (iteration+1) % 3000==0:
                if best_val_score is None or accuracy > best_val_score:
                    best_val_score = accuracy
                    model_final_path = osp.join(cfg.output_dir, 'model_best.pth')
                    model_final = {
                        "iteration": iteration,
                        "model_state": model.state_dict(),
                        "fpn_model_state": fpn_model.state_dict(),
                        "bert_model_state": bert_encoder.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_score": accuracy
                    }
                    torch.save(model_final, model_final_path)
        
        if dist.get_rank()==0:
            model_final_path = osp.join(cfg.output_dir, 'model_latest.pth')
            model_final = {
                "iteration": iteration,
                "model_state": model.state_dict(),
                "fpn_model_state": fpn_model.state_dict(),
                "bert_model_state": bert_encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_score": accuracy
            }
            torch.save(model_final, model_final_path)

    # ------------------------------4.2 teacher-student mutual learning---------------------
    
    bert_encoder_teacher = bert_encoder
    fpn_model_teacher = fpn_model

    model_teacher = KNet(
        num_stages=cfg.num_stages,
        num_points=cfg.num_points,
    ).to(local_rank)
    
    model_teacher = DDP(model_teacher, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    burnIn_on = burnIn_params.get('burnIn_on',True)
    if burnIn_on :
        fpn_model_teacher.load_state_dict(fpn_model.state_dict())
        bert_encoder_teacher.load_state_dict(bert_encoder.state_dict())
        model_teacher.load_state_dict(model.state_dict())
    
    start_iter=0
    best_val_score = None
    ML_stage_params = semi_cfg.get('mutual_learning_stage')
    mutual_learning_steps = ML_stage_params.get('steps',12000)
    ema_rate = ML_stage_params.get('ema_rate',0.99)

    if not burnIn_on:
        burn_in_ckpt=burnIn_params.get('burn_in_ckpt','')
        if osp.exists(burn_in_ckpt): 
            model_final = torch.load(burn_in_ckpt,map_location=torch.device('cpu'))
             # teacher
            fpn_model_teacher.load_state_dict(model_final['fpn_model_state'])
            bert_encoder_teacher.load_state_dict(model_final['bert_model_state'])
            model_teacher.load_state_dict(model_final['model_state'])
            # student
            bert_encoder.load_state_dict(model_final['bert_model_state'])
            fpn_model.load_state_dict(model_final['fpn_model_state'])
            model.load_state_dict(model_final['model_state'])

    for n, c in model_teacher.named_parameters():
        c.requires_grad = False

    semi_settings = semi_cfg.get('semi_settings')
    pseudo_label_type = semi_settings['pseudo_label_type']
    is_drop = semi_settings['is_drop']
    element_wise_weight = semi_settings['element_wise_weight']
    
    is_kl_loss = semi_settings.get('is_kl_loss',False)

    for idx,iteration in enumerate(range(start_iter, mutual_learning_steps)):
        model.train()
        if cfg.bert_freeze:
            bert_encoder.eval()
        else:
            bert_encoder.train()
        if cfg.fpn_freeze:
            fpn_model.eval()
        else:
            fpn_model.train()

        time_stats = AverageMeter()

        # define loss 
        dice_loss = DiceLoss(use_sigmoid=False)
        ce_loss = CrossEntropyLoss(use_sigmoid=False) 
   
        label_data,unlabel_data=next(semi_dataloader_iter)

        caption, grounding_instances, ann_categories,ann_types, noun_vector_padding, ret_noun_vector, fpn_input_data=transform_input(label_data)
            
        ret_noun_vector=torch.stack(ret_noun_vector)
        ann_types = torch.stack(ann_types)
        ann_categories = torch.stack(ann_categories)

        ret_noun_vector = ret_noun_vector.to(cfg.local_rank)
        ann_types = ann_types.to(cfg.local_rank)
        ann_categories = ann_categories.to(cfg.local_rank)
        
        start_time = time.time()
        
        # bert for caption
        with torch.no_grad():
            lang_feat, _ = bert_encoder(caption)                                      # lang_feat:       torch.Size([12, 230, 768])
            lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
                cfg.max_seg_num, lang_feat.shape[-1]))                                # lang_feat_valid: torch.Size([12, 64, 768]) 全0

            for i in range(len(lang_feat)):
                cur_lang_feat = lang_feat[i][noun_vector_padding[i].nonzero().flatten()]
                lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat
        
        # fpn for imgs
        fpn_feature = fpn_model(fpn_input_data)   # torch.Size([12, 256, 160, 160])  

        # preprocessing for gt masks
        with torch.no_grad():
            gts = [F.interpolate(grounding_instances[i]["gt"].to(cfg.local_rank).unsqueeze(0), \
                                (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                mode='bilinear').squeeze() for i in range(len(grounding_instances))]
            gts = ImageList.from_tensors(gts, 32).tensor
            gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
            gts = (gts > 0).float()
        
        # gts:              [B, max_seg_num, H//4, W//4]
        # fpn_feature:      [12, 256, 160, 160]
        # lang_feat_valid:  [12, 64, 768]
        predictions = model(fpn_feature, lang_feat_valid, train=False) 

        grad_sample = ann_types != 0
        gt = gts[grad_sample].cuda()

        supervised_loss = 0 
        for i in range(len(predictions)):
            pred = predictions[i][grad_sample]
            pred = pred.sigmoid()
            supervised_loss = supervised_loss + ce_loss(pred, gt) + dice_loss(pred, gt)

        if augmentation:
            unlabel_caption,unlabel_grounding_instances, unlabel_ann_categories,unlabel_ann_types,unlabel_noun_vector_padding, unlabel_ret_noun_vector,unlabel_fpn_input_data, \
                unlabel_strong_aug_fpn_data,unlabel_weak_aug_fpn_data=transform_input(unlabel_data)
        else:
            unlabel_caption,unlabel_grounding_instances, unlabel_ann_categories,unlabel_ann_types,unlabel_noun_vector_padding, unlabel_ret_noun_vector,unlabel_fpn_input_data=transform_input(unlabel_data)
            
            unlabel_strong_aug_fpn_data=unlabel_fpn_input_data
            unlabel_weak_aug_fpn_data=unlabel_fpn_input_data

        unlabel_ret_noun_vector=torch.stack(unlabel_ret_noun_vector)
        unlabel_ann_types = torch.stack(unlabel_ann_types)
        unlabel_ann_categories = torch.stack(unlabel_ann_categories)
       
        unlabel_ret_noun_vector = unlabel_ret_noun_vector.to(cfg.local_rank)
        unlabel_ann_types = unlabel_ann_types.to(cfg.local_rank)
        unlabel_ann_categories = unlabel_ann_categories.to(cfg.local_rank)
        
        # teacher generate pseudo label
        with torch.no_grad():
            unlabel_lang_feat, _ = bert_encoder_teacher(unlabel_caption) #bert for caption
            unlabel_lang_feat_valid = unlabel_lang_feat.new_zeros((unlabel_lang_feat.shape[0], \
                cfg.max_seg_num, unlabel_lang_feat.shape[-1]))

            for i in range(len(unlabel_lang_feat)):
                cur_lang_feat = unlabel_lang_feat[i][unlabel_noun_vector_padding[i].nonzero().flatten()]
                unlabel_lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat
        unlabel_fpn_feature = fpn_model_teacher(unlabel_weak_aug_fpn_data) #fpn for imgs

        with torch.no_grad():
            pseudo_predictions = model_teacher(unlabel_fpn_feature, unlabel_lang_feat_valid, train=False) #Knet
            # pseudo_predictions[0].shape: torch.Size([12, 64, 160, 160])  

        with torch.no_grad():
            unlabel_lang_feat, _ = bert_encoder(unlabel_caption) #bert for caption
            unlabel_lang_feat_valid = unlabel_lang_feat.new_zeros((unlabel_lang_feat.shape[0], \
                cfg.max_seg_num, unlabel_lang_feat.shape[-1]))

            for i in range(len(unlabel_lang_feat)):
                cur_lang_feat = unlabel_lang_feat[i][unlabel_noun_vector_padding[i].nonzero().flatten()]
                unlabel_lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat
        unlabel_fpn_feature = fpn_model(unlabel_strong_aug_fpn_data)
        student_predictions = model(unlabel_fpn_feature,unlabel_lang_feat_valid,train=False)

        grad_sample = unlabel_ann_types != 0

        pseudo_label=pseudo_predictions[-1][grad_sample].sigmoid()

        soft_label = torch.tensor(pseudo_label)

        # one hot
        if pseudo_label_type == "hard_label":
            pseudo_label = (pseudo_label > semi_settings['hard_label_thresh']).float() # 
        else:
            pass

        # Calculate pixel-level weight for BCE loss
        weight = torch.ones_like(pseudo_label,dtype=torch.float32)
        if element_wise_weight:
            prob = soft_label
            if semi_settings["soft_method"] == "linear":
                # linear type
                weight[prob>=0.5] =  prob[prob>=0.5]
                weight[prob<0.5] =  1.0 - prob[prob<0.5]
            elif semi_settings["soft_method"] == "guassian":
                # gaussian type
                #initialize Gaussian mean and variance
                u1 = 0.5  
                sigma1 = 0.1  
                left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma1))
                right = np.exp(-(prob.detach().cpu().numpy() - u1)**2 / (2 * sigma1))
                weight_numpy =  1.3 - left*right
                weight = torch.from_numpy(weight_numpy).cuda()
        else:
            pass
        
        # Calculate mask-level weight for Dice loss
        parm_t =20
        if iteration==6000 or iteration==12000 or iteration==18000:
            parm_t = parm_t-5
        total=[]
        for i in range(pseudo_label.shape[0]):
            _,return_num = measure.label(pseudo_label[i].cpu().numpy(),return_num=True)
            total.append(return_num)
        total = np.array(total)
        dice_weight = 1 /(1+ np.exp(total-parm_t))
        dice_weight = torch.tensor(dice_weight).to(pseudo_label.device)

        # Init unsup loss
        unsupvise_loss=0
        unsup_dice=0
        unsup_ce=0
        unsup_kl=0

        # Teacher's pseudo label supervise student's prediction
        for i in range(len(student_predictions)):
            pred = student_predictions[i][grad_sample]
            pred = pred.sigmoid()  

            # Dice loss
            loss_dice = dice_loss(pred,pseudo_label,weight = dice_weight)
            
            # BCE loss
            loss_ce = ce_loss(pred,pseudo_label,weight=weight)

            # KL loss
            if is_kl_loss:
                soft_label_flatten = soft_label.view(soft_label.shape[0],soft_label.shape[1]*soft_label.shape[2]).t()
                soft_label_flatten_softmax = F.softmax(soft_label_flatten, dim=-1)
                pred_flatten = pred.view(pred.shape[0],pred.shape[1]*pred.shape[2]).t()
                pred_flatten_softmax = F.log_softmax(pred_flatten, dim=-1)
                
                loss_kl = F.kl_div(pred_flatten_softmax, soft_label_flatten_softmax, reduction='batchmean')

                unsup_kl = unsup_kl + loss_kl
                unsupvise_loss = unsupvise_loss + loss_ce + loss_dice +loss_kl
            else:
                unsupvise_loss = unsupvise_loss + loss_ce + loss_dice
            

        # Update student model
        total_loss=supervised_loss+ unsupvise_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print loss log
        if iteration%100==0 and  dist.get_rank()==0:
            logger.info('-' * 89)
            logger.info('| end of iteration {:3d} | time: {:5.2f}s | lr:{:.6f}'
                    '| iteration total loss {:.6f} | supervised  loss {:.6f} | unsupervised  loss {:.6f} '.format(
                        iteration, time.time() - start_time,optimizer.param_groups[0]["lr"], total_loss,supervised_loss,unsupvise_loss))
            logger.info('-' * 89)
        
        # EMA update teacher model
        keep_rate = ema_rate
        student_model_dict=model.state_dict()
        new_teacher_dict = OrderedDict()
        # EMA
        for key, value in model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                ).cpu()
            else:
                raise Exception("{} is not found in student model".format(key))
        model_teacher.load_state_dict(new_teacher_dict)

        
        if (iteration+1) % 3000==0:
            accuracy=evaluate(val_loader,bert_encoder_teacher,fpn_model_teacher,model_teacher,iteration,cfg,logger,writer)
            if dist.get_rank()==0:
                if best_val_score is None or accuracy > best_val_score:
                    best_val_score = accuracy
                    model_final_path = osp.join(cfg.output_dir, 'teacher_mutual_stage_model_best.pth')
                    model_final = {
                        "iteration": iteration,
                        "model_state": model_teacher.state_dict(),
                        "fpn_model_state": fpn_model_teacher.state_dict(),
                        "bert_model_state": bert_encoder_teacher.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_score": accuracy
                    }
                    torch.save(model_final, model_final_path)
        
      
def test(cfg):
    dist.init_process_group(backend=cfg.backend,init_method='env://')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    if dist.get_rank() == 0:
        logger = setup_logger(cfg.output_dir, dist.get_rank())
        writer = SummaryWriter(osp.join(cfg.output_dir, 'tensorboard'))
    else:
        logger, writer = None, None
    # supervised model
    bert_encoder = BertEncoder(cfg).to(local_rank)
    bert_encoder = DDP(bert_encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) 
    fpn_model = fpn(cfg.detectron2_ckpt, cfg.detectron2_cfg)
    fpn_model = fpn_model.to(local_rank)
    fpn_model = DDP(fpn_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model = KNet(
        num_stages=cfg.num_stages,
        num_points=cfg.num_points,
    ).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    val_dataset = PanopticNarrativeGroundingValDataset(cfg, 'val2017', False)
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=12,
        sampler = distributed_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=default_collate,
    )
    model_state_dict=torch.load(cfg.ckpt_path,map_location=torch.device('cpu'))
    bert_encoder.load_state_dict(model_state_dict['bert_model_state'])
    fpn_model.load_state_dict(model_state_dict['fpn_model_state'])
    model.load_state_dict(model_state_dict['model_state'])
    my_evaluate(val_loader,bert_encoder,fpn_model,model,cfg)
