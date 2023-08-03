import os
import json
import random

import torch
import os.path as osp
from PIL import Image,ImageFilter
import numpy as np

from skimage import io
from torch.utils.data import Dataset,IterableDataset
from torchvision.transforms import Resize
from data.get_fpn_data import fpn_data
import torchvision.transforms as transforms 
import torchvision.transforms.functional as transforms_f 
import torch.nn.functional as F
import copy

def augmentation_transform(image,crop_size=(640, 640), scale_size=(1.3, 1.5),augmentation_option = "weakly",aug_set=None):
    """
        数据增强
        弱增强结束后要保存弱增强后的图片和label 强增强要接着弱增强的结果接着做
        weak:
    """
    if augmentation_option == "weakly":
        
        if "Gaussian filter" in aug_set and torch.rand(1)>0.5:
            # Random Gaussian filter
            # sigma = random.uniform(0.15, 1.15)
            sigma = 1.0
            # sigma = random.uniform(1.15,2.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        if "Horizontal flipping" in aug_set and torch.rand(1)>0.5:
            image = transforms_f.hflip(image)

        if "Random crop" in aug_set and torch.rand(1)>0.5:
            raw_w, raw_h = image.size
            scale_ratio = random.uniform(scale_size[0], scale_size[1])
            resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
            image = transforms_f.resize(image, resized_size, Image.BILINEAR)
            # Add padding if rescaled image size is less than crop size
            if crop_size == -1:  # use original im size without crop or padding
                crop_size = (raw_h, raw_w)

            if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
                right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
                image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')

            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
            image = transforms_f.crop(image, i, j, h, w)
        
        return image
    elif augmentation_option == "strong":    
        # # Random color jitter
        if "Color jittering" in aug_set and torch.rand(1) > 0.5:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25)) # For PyTorch 1.9/TorchVision 0.10 users
            image = color_transform(image)            
        return image

    # Transform to tensor
    # image = (transforms_f.to_tensor(image)*255).long()
    # label = (transforms_f.to_tensor(label) * 255).long()

    # Apply (ImageNet) normalisation
    # image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def strong_transform(image,label):
    
    # Random Gaussian filter

    if torch.rand(1) > 0.5:
        sigma = 1.0
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    if torch.rand(1) > 0.5:
        # sigma = random.uniform(0.15, 1.15)
        image = transforms_f.hflip(image)
        label = transforms_f.hflip(label)

    if torch.rand(1) > 0.5:
        color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25)) # For PyTorch 1.9/TorchVision 0.10 users
        image = color_transform(image)
    
    return image,label

class PanopticNarrativeGroundingDataset(Dataset):
    """Panoptic Narrative Grounding dataset."""

    def __init__(self, cfg, split, train=True):
        """
        Args:
            Args:
            cfg (CfgNode): configs.
            train (bool):
        """
        self.cfg = cfg
        self.train = train # True or False
        # split = 'val2017'
        self.split = split # train2017 or val2017

        self.mask_transform = Resize((256, 256))

        self.ann_dir = osp.join(cfg.data_path, "annotations")
        self.panoptic = self.load_json(
            osp.join(self.ann_dir, "panoptic_{:s}.json".format(split))
        )
        self.images = self.panoptic["images"]
        self.images = {i["id"]: i for i in self.images}
        self.panoptic_anns = self.panoptic["annotations"]
        self.panoptic_anns = {a["image_id"]: a for a in self.panoptic_anns}
        if not osp.exists(
            osp.join(self.ann_dir, 
                "png_coco_{:s}_dataloader.json".format(split),)
        ):
            print("No such a dataset")
        else:
            self.panoptic_narrative_grounding = self.load_json(
                osp.join(self.ann_dir, 
                    "png_coco_{:s}_dataloader.json".format(split),)
            )
        self.panoptic_narrative_grounding = [
            ln
            for ln in self.panoptic_narrative_grounding
            if (
                torch.tensor([item for sublist in ln["labels"] 
                    for item in sublist])
                != -2
            ).any()
        ]

        fpn_dataset, self.fpn_mapper = fpn_data(cfg, split[:-4])
        self.fpn_dataset = {i['image_id']: i for i in fpn_dataset}

    ## General helper functions
    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)
    
    def resize_gt(self, img, interp, new_w, new_h):
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((new_w, new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def __len__(self):
        return len(self.panoptic_narrative_grounding)
    
    def vis_item(self, img, gt, idx):
        save_dir = f'vis/{idx}'
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        import cv2
        cv2.imwrite(osp.join(save_dir,'img.png'), img.numpy().transpose(1, 2, 0))
        for i in range(len(gt)):
            if gt[i].sum() != 0:
                cv2.imwrite(osp.join(save_dir, f'gt_{i}.png'), gt[i].numpy()*255)
        
    def __getitem__(self, idx):
        localized_narrative = self.panoptic_narrative_grounding[idx]
        caption = localized_narrative['caption']
        image_id = int(localized_narrative['image_id'])
        fpn_data = self.fpn_mapper(self.fpn_dataset[image_id])  

        # --------------------------------- #
        # fpn_data = fpn_data['image']

        # fpn_data = fpn_data.unsqueeze(dim=0)
        # fpn_data = fpn_data.float()

        # fpn_data = F.interpolate(fpn_data,(640,640),mode='bilinear')

        # fpn_data = fpn_data.squeeze()

        # norm = transforms.Normalize(
        #     mean=[0.485*255,0.456*255,0.406*255],
        #     std= [0.229*255,0.224*255,0.225*255]
        # )

        # normalize = transforms.Compose([norm])
        # fpn_data = normalize(fpn_data)
        # --------------------------------- #

        image_info = self.images[image_id]
        labels = localized_narrative['labels']

        noun_vector = localized_narrative['noun_vector']
        if len(noun_vector) > (self.cfg.max_sequence_length - 2):
            noun_vector_padding = \
                    noun_vector[:(self.cfg.max_sequence_length - 2)]
        elif len(noun_vector) < (self.cfg.max_sequence_length - 2): 
            noun_vector_padding = \
                noun_vector + [0] * (self.cfg.max_sequence_length - \
                    2 - len(noun_vector))
        noun_vector_padding = [0] + noun_vector_padding + [0]
        noun_vector_padding = torch.tensor(noun_vector_padding).long()
        assert len(noun_vector_padding) == \
            self.cfg.max_sequence_length
        
        ret_noun_vector = noun_vector_padding[noun_vector_padding.nonzero()].flatten()
        assert len(ret_noun_vector) <= self.cfg.max_seg_num
        if len(ret_noun_vector) < self.cfg.max_seg_num:
            ret_noun_vector = torch.cat([ret_noun_vector, \
                ret_noun_vector.new_zeros((self.cfg.max_seg_num - len(ret_noun_vector)))])
        # ret_noun_vector: [max_seg_num]

        ann_types = [0] * len(labels)
        for i, l in enumerate(labels):
            l = torch.tensor(l)
            if (l != -2).any():
                ann_types[i] = 1 if (l != -2).sum() == 1 else 2
        ann_types = torch.tensor(ann_types).long()
        ann_types = ann_types[ann_types.nonzero()].flatten()
        assert len(ann_types) <= self.cfg.max_seg_num
        if len(ann_types) < self.cfg.max_seg_num:
            ann_types = torch.cat([ann_types, \
                ann_types.new_zeros((self.cfg.max_seg_num - len(ann_types)))])
        
        ann_categories = torch.zeros([
            self.cfg.max_seg_num]).long()
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_segm = io.imread(
            osp.join(
                self.ann_dir,
                "panoptic_segmentation",
                self.split,
                "{:012d}.png".format(image_id),
            )
        )

        # 
        panoptic_segm = (
            panoptic_segm[:, :, 0]
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )

        grounding_instances = torch.zeros(
            [self.cfg.max_seg_num, image_info['height'], image_info['width']]
        )
        j = 0
        for i, bbox in enumerate(localized_narrative["boxes"]):
            for b in bbox:
                if b != [0] * 4:
                    segment_info = [
                        s for s in panoptic_ann["segments_info"] 
                        if s["bbox"] == b
                    ][0]
                    segment_cat = [
                        c
                        for c in self.panoptic["categories"]
                        if c["id"] == segment_info["category_id"]
                    ][0]
                    instance = torch.zeros([image_info['height'],
                            image_info['width']])
                    instance[panoptic_segm == segment_info["id"]] = 1
                    grounding_instances[j, :] += instance
                    ann_categories[j] = 1 if \
                            segment_cat["isthing"] else 2
            if grounding_instances[j].sum() != 0:
                j = j + 1

        grounding_instances = {'gt': grounding_instances}

        return caption, grounding_instances, \
            ann_categories, ann_types, noun_vector_padding, ret_noun_vector, fpn_data


class PanopticNarrativeGroundingValDataset(Dataset):
    """Panoptic Narrative Grounding dataset."""

    def __init__(self, cfg,split, train=True):
        """
        Args:
            Args:
            cfg (CfgNode): configs.
            train (bool):
        """
        self.cfg = cfg
        self.train = train # True or False
        # split = 'val2017'
        self.split = split # train2017 or val2017

        self.mask_transform = Resize((256, 256))

        self.ann_dir = osp.join(cfg.data_path, "annotations")
        self.panoptic = self.load_json(
            osp.join(self.ann_dir, "panoptic_{:s}.json".format(split))
        )
        self.images = self.panoptic["images"]
        self.images = {i["id"]: i for i in self.images}
        self.panoptic_anns = self.panoptic["annotations"]
        self.panoptic_anns = {a["image_id"]: a for a in self.panoptic_anns}
        if not osp.exists(
            osp.join(self.ann_dir, 
                "png_coco_{:s}_dataloader.json".format(split),)
        ):
            print("No such a dataset")
        else:
            self.panoptic_narrative_grounding = self.load_json(
                osp.join(self.ann_dir, 
                    "png_coco_{:s}_dataloader.json".format(split),)
            )
        self.panoptic_narrative_grounding = [
            ln
            for ln in self.panoptic_narrative_grounding
            if (
                torch.tensor([item for sublist in ln["labels"] 
                    for item in sublist])
                != -2
            ).any()
        ]
        fpn_dataset, self.fpn_mapper = fpn_data(cfg, split[:-4])
        self.fpn_dataset = {i['image_id']: i for i in fpn_dataset}

    ## General helper functions
    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)
    
    def resize_gt(self, img, interp, new_w, new_h):
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((new_w, new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def __len__(self):
        return len(self.panoptic_narrative_grounding)
    
    def vis_item(self, img, gt, idx):
        save_dir = f'vis/{idx}'
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        import cv2
        cv2.imwrite(osp.join(save_dir,'img.png'), img.numpy().transpose(1, 2, 0))
        for i in range(len(gt)):
            if gt[i].sum() != 0:
                cv2.imwrite(osp.join(save_dir, f'gt_{i}.png'), gt[i].numpy()*255)
        
    def __getitem__(self, idx):
        localized_narrative = self.panoptic_narrative_grounding[idx]
        caption = localized_narrative['caption']
        image_id = int(localized_narrative['image_id'])
        fpn_data = self.fpn_mapper(self.fpn_dataset[image_id])  
        image_info = self.images[image_id]
        labels = localized_narrative['labels']

        noun_vector = localized_narrative['noun_vector']
        if len(noun_vector) > (self.cfg.max_sequence_length - 2):
            noun_vector_padding = \
                    noun_vector[:(self.cfg.max_sequence_length - 2)]
        elif len(noun_vector) < (self.cfg.max_sequence_length - 2): 
            noun_vector_padding = \
                noun_vector + [0] * (self.cfg.max_sequence_length - \
                    2 - len(noun_vector))
        noun_vector_padding = [0] + noun_vector_padding + [0]
        noun_vector_padding = torch.tensor(noun_vector_padding).long()
        assert len(noun_vector_padding) == \
            self.cfg.max_sequence_length
        ret_noun_vector = noun_vector_padding[noun_vector_padding.nonzero()].flatten()
        assert len(ret_noun_vector) <= self.cfg.max_seg_num
        if len(ret_noun_vector) < self.cfg.max_seg_num:
            ret_noun_vector = torch.cat([ret_noun_vector, \
                ret_noun_vector.new_zeros((self.cfg.max_seg_num - len(ret_noun_vector)))])
        cur_phrase_index = ret_noun_vector[ret_noun_vector!=0]
        
        _, cur_index_counts = torch.unique_consecutive(cur_phrase_index, return_counts=True)
        cur_phrase_interval = torch.cumsum(cur_index_counts, dim=0)
        cur_phrase_interval = torch.cat([cur_phrase_interval.new_zeros((1)), cur_phrase_interval])
        # ret_noun_vector: [max_seg_num]

        ann_types = [0] * len(labels)
        for i, l in enumerate(labels):
            l = torch.tensor(l)
            if (l != -2).any():
                ann_types[i] = 1 if (l != -2).sum() == 1 else 2
        ann_types = torch.tensor(ann_types).long()
        ann_types = ann_types[ann_types.nonzero()].flatten()
        assert len(ann_types) <= self.cfg.max_seg_num
        if len(ann_types) < self.cfg.max_seg_num:
            ann_types = torch.cat([ann_types, \
                ann_types.new_zeros((self.cfg.max_seg_num - len(ann_types)))])

        ann_types_valid = ann_types.new_zeros(self.cfg.max_phrase_num)
        ann_types_valid[:len(cur_phrase_interval)-1] = ann_types[cur_phrase_interval[:-1]]


        ann_categories = torch.zeros([
            self.cfg.max_phrase_num]).long()
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_segm = io.imread(
            osp.join(
                self.ann_dir,
                "panoptic_segmentation",
                self.split,
                "{:012d}.png".format(image_id),
            )
        )
        panoptic_segm = (
            panoptic_segm[:, :, 0] 
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )
        grounding_instances = torch.zeros(
            [self.cfg.max_phrase_num, image_info['height'], image_info['width']]
        )
        j = 0
        k = 0
        for i, bbox in enumerate(localized_narrative["boxes"]):
            flag = False
            for b in bbox:
                if b != [0] * 4:
                    flag = True
            if not flag:
                continue
            
            for b in bbox:
                if b != [0] * 4:
                    flag = True
                    segment_info = [
                        s for s in panoptic_ann["segments_info"] 
                        if s["bbox"] == b
                    ][0]
                    segment_cat = [
                        c
                        for c in self.panoptic["categories"]
                        if c["id"] == segment_info["category_id"]
                    ][0]
                    instance = torch.zeros([image_info['height'],
                            image_info['width']])
                    instance[panoptic_segm == segment_info["id"]] = 1
                    if j in cur_phrase_interval[:-1]:
                        grounding_instances[k, :] += instance
                        ann_categories[k] = 1 if \
                                segment_cat["isthing"] else 2
            if j in cur_phrase_interval[:-1]:
                k = k + 1   
            j = j + 1
        assert k == len(cur_phrase_interval) - 1
        grounding_instances = {'gt': grounding_instances}
        ret_noun_vector = {'inter': cur_phrase_interval}

        return caption, grounding_instances, \
            ann_categories, ann_types_valid, noun_vector_padding, ret_noun_vector, fpn_data

class PanopticNarrativeGroundingLabeledDataset(Dataset):  
    def __init__(self, cfg,seed=1,sup_percent=1,train=True,augmentation=False):
        self.cfg = cfg
        self.train = train
        self.augmentation = augmentation
        self.mask_transform = Resize((256,256))

        self.ann_dir = osp.join(cfg.data_path, "annotations")
        self.panoptic = self.load_json(
            osp.join(self.ann_dir, "panoptic_train2017.json")
        )
        self.images = self.panoptic["images"]
        self.images = {i["id"]: i for i in self.images}
        self.panoptic_anns = self.panoptic["annotations"]
        self.panoptic_anns = {a["image_id"]: a for a in self.panoptic_anns}
        self.panoptic_narrative_grounding_labeled = self.load_json(
                osp.join(self.ann_dir, 
                    "png_coco_train2017_labeled_dataloader_seed"+str(seed)+'_sup'+str(sup_percent)+'.json')
        )
        self.panoptic_narrative_grounding_labeled = [
            ln
            for ln in self.panoptic_narrative_grounding_labeled
            if (
                torch.tensor([item for sublist in ln["labels"] 
                    for item in sublist])
                != -2
            ).any()
        ]

        fpn_dataset, self.fpn_mapper = fpn_data(cfg,'train')
        self.fpn_dataset = {i['image_id']: i for i in fpn_dataset}
    
    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)

    def resize_gt(self, img, interp, new_w, new_h):
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((new_w, new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret
    
    def __len__(self):
        return len(self.panoptic_narrative_grounding_labeled)

    def vis_item(self, img, gt, idx):
        save_dir = f'vis/{idx}'
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        import cv2
        cv2.imwrite(osp.join(save_dir,'img.png'), img.numpy().transpose(1, 2, 0))
        for i in range(len(gt)):
            if gt[i].sum() != 0:
                cv2.imwrite(osp.join(save_dir, f'gt_{i}.png'), gt[i].numpy()*255)

    def __getitem__(self, idx):
        localized_narrative = self.panoptic_narrative_grounding_labeled[idx]
        caption = localized_narrative['caption']
        image_id = int(localized_narrative['image_id'])
        fpn_data = self.fpn_mapper(self.fpn_dataset[image_id])  
        # --------------------------------- #
        # fpn_data = fpn_data['image']

        # fpn_data = fpn_data.unsqueeze(dim=0)
        # fpn_data = fpn_data.float()

        # fpn_data = F.interpolate(fpn_data,(640,640),mode='bilinear')

        # fpn_data = fpn_data.squeeze()

        # norm = transforms.Normalize(
        #     mean=[0.485*255,0.456*255,0.406*255],
        #     std= [0.229*255,0.224*255,0.225*255]
        # )

        # normalize = transforms.Compose([norm])
        # fpn_data = normalize(fpn_data)
        # --------------------------------- #

        image_info = self.images[image_id]
        labels = localized_narrative['labels']

        noun_vector = localized_narrative['noun_vector']
        if len(noun_vector) > (self.cfg.max_sequence_length - 2):
            noun_vector_padding = \
                    noun_vector[:(self.cfg.max_sequence_length - 2)]
        elif len(noun_vector) < (self.cfg.max_sequence_length - 2): 
            noun_vector_padding = \
                noun_vector + [0] * (self.cfg.max_sequence_length - \
                    2 - len(noun_vector))
        noun_vector_padding = [0] + noun_vector_padding + [0]
        noun_vector_padding = torch.tensor(noun_vector_padding).long()
        assert len(noun_vector_padding) == \
            self.cfg.max_sequence_length
        
        ret_noun_vector = noun_vector_padding[noun_vector_padding.nonzero()].flatten()
        assert len(ret_noun_vector) <= self.cfg.max_seg_num
        if len(ret_noun_vector) < self.cfg.max_seg_num:
            ret_noun_vector = torch.cat([ret_noun_vector, \
                ret_noun_vector.new_zeros((self.cfg.max_seg_num - len(ret_noun_vector)))])
        # ret_noun_vector: [max_seg_num]

        ann_types = [0] * len(labels)
        for i, l in enumerate(labels):
            l = torch.tensor(l)
            if (l != -2).any():
                ann_types[i] = 1 if (l != -2).sum() == 1 else 2
        ann_types = torch.tensor(ann_types).long()
        ann_types = ann_types[ann_types.nonzero()].flatten()
        assert len(ann_types) <= self.cfg.max_seg_num
        if len(ann_types) < self.cfg.max_seg_num:
            ann_types = torch.cat([ann_types, \
                ann_types.new_zeros((self.cfg.max_seg_num - len(ann_types)))])
        
        ann_categories = torch.zeros([
            self.cfg.max_seg_num]).long()
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_segm = io.imread(
            osp.join(
                self.ann_dir,
                "panoptic_segmentation",
                'train2017',
                "{:012d}.png".format(image_id),
            )
        )
        if self.augmentation:
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            img_pil = to_pil(fpn_data['image'])
            label_pil = to_pil(panoptic_segm)
            s_aug_img,s_aug_label=strong_transform(copy.deepcopy(img_pil),copy.deepcopy(label_pil))

            s_aug_img = to_tensor(s_aug_img)
            s_aug_label = to_tensor(s_aug_label)
            s_aug_label = s_aug_label.permute(1,2,0)
            s_aug_img = s_aug_img*255
            s_aug_label = s_aug_label*255
            fpn_data['image'] = s_aug_img
            panoptic_segm = s_aug_label

        panoptic_segm = (
            panoptic_segm[:, :, 0]
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )

        grounding_instances = torch.zeros(
            [self.cfg.max_seg_num, image_info['height'], image_info['width']]
        )
        j = 0
        for i, bbox in enumerate(localized_narrative["boxes"]):
            for b in bbox:
                if b != [0] * 4:
                    segment_info = [
                        s for s in panoptic_ann["segments_info"] 
                        if s["bbox"] == b
                    ][0]
                    segment_cat = [
                        c
                        for c in self.panoptic["categories"]
                        if c["id"] == segment_info["category_id"]
                    ][0]
                    instance = torch.zeros([image_info['height'],
                            image_info['width']])
                    instance[panoptic_segm == segment_info["id"]] = 1
                    grounding_instances[j, :] += instance
                    ann_categories[j] = 1 if \
                            segment_cat["isthing"] else 2
            if grounding_instances[j].sum() != 0:
                j = j + 1

        grounding_instances = {'gt': grounding_instances}

        return caption, grounding_instances, \
            ann_categories, ann_types, noun_vector_padding, \
            ret_noun_vector, fpn_data

class PanopticNarrativeGroundingUnlabeledDataset(Dataset):
    def __init__(self, cfg,seed=1,sup_percent=1,train=True,augmentation=False,weak_aug=None,strong_aug=None):
        self.cfg = cfg
        self.train = train
        self.augmentation = augmentation
        self.weak_aug=weak_aug
        self.strong_aug=strong_aug

        self.mask_transform = Resize((256,256))

        self.ann_dir = osp.join(cfg.data_path, "annotations")
        self.panoptic = self.load_json(
            osp.join(self.ann_dir, "panoptic_train2017.json")
        )
        self.images = self.panoptic["images"]
        self.images = {i["id"]: i for i in self.images}
        self.panoptic_anns = self.panoptic["annotations"]
        self.panoptic_anns = {a["image_id"]: a for a in self.panoptic_anns}

        self.panoptic_narrative_grounding_unlabeled = self.load_json(
                osp.join(self.ann_dir, 
                    "png_coco_train2017_unlabeled_dataloader_seed"+str(seed)+'_sup'+str(sup_percent)+'.json')
        )


        self.panoptic_narrative_grounding_unlabeled = [
        ln
            for ln in self.panoptic_narrative_grounding_unlabeled
            if (
                torch.tensor([item for sublist in ln["labels"] 
                    for item in sublist])
                != -2
            ).any()
        ]

        fpn_dataset, self.fpn_mapper = fpn_data(cfg,'train')
        self.fpn_dataset = {i['image_id']: i for i in fpn_dataset}
    
    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)

    def resize_gt(self, img, interp, new_w, new_h):
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((new_w, new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret
    
    def __len__(self):
        return len(self.panoptic_narrative_grounding_unlabeled)

    def vis_item(self, img, gt, idx):
        save_dir = f'./output/visualize/{idx}'
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        import cv2
        cv2.imwrite(osp.join(save_dir,'img.png'), img.numpy().transpose(1, 2, 0))
        for i in range(len(gt)):
            if gt[i].sum() != 0:
                cv2.imwrite(osp.join(save_dir, f'gt_{i}.png'), gt[i].numpy()*255)

    def __getitem__(self, idx):
        localized_narrative = self.panoptic_narrative_grounding_unlabeled[idx]
        caption = localized_narrative['caption']
        image_id = int(localized_narrative['image_id'])
        fpn_data = self.fpn_mapper(self.fpn_dataset[image_id])  

        # --------------------------------- #
        # fpn_data = fpn_data['image']

        # fpn_data = fpn_data.unsqueeze(dim=0)
        # fpn_data = fpn_data.float()

        # fpn_data = F.interpolate(fpn_data,(640,640),mode='bilinear')

        # fpn_data = fpn_data.squeeze()

        # norm = transforms.Normalize(
        #     mean=[0.485*255,0.456*255,0.406*255],
        #     std= [0.229*255,0.224*255,0.225*255]
        # )

        # normalize = transforms.Compose([norm])
        # fpn_data = normalize(fpn_data)
        # --------------------------------- #

        image_info = self.images[image_id]
        labels = localized_narrative['labels']

        noun_vector = localized_narrative['noun_vector']
        if len(noun_vector) > (self.cfg.max_sequence_length - 2):
            noun_vector_padding = \
                    noun_vector[:(self.cfg.max_sequence_length - 2)]
        elif len(noun_vector) < (self.cfg.max_sequence_length - 2): 
            noun_vector_padding = \
                noun_vector + [0] * (self.cfg.max_sequence_length - \
                    2 - len(noun_vector))
        noun_vector_padding = [0] + noun_vector_padding + [0]
        noun_vector_padding = torch.tensor(noun_vector_padding).long()
        assert len(noun_vector_padding) == \
            self.cfg.max_sequence_length
        
        ret_noun_vector = noun_vector_padding[noun_vector_padding.nonzero()].flatten()
        assert len(ret_noun_vector) <= self.cfg.max_seg_num
        if len(ret_noun_vector) < self.cfg.max_seg_num:
            ret_noun_vector = torch.cat([ret_noun_vector, \
                ret_noun_vector.new_zeros((self.cfg.max_seg_num - len(ret_noun_vector)))])
        # ret_noun_vector: [max_seg_num]

        ann_types = [0] * len(labels)
        for i, l in enumerate(labels):
            l = torch.tensor(l)
            if (l != -2).any():
                ann_types[i] = 1 if (l != -2).sum() == 1 else 2
        ann_types = torch.tensor(ann_types).long()
        ann_types = ann_types[ann_types.nonzero()].flatten()
        assert len(ann_types) <= self.cfg.max_seg_num
        if len(ann_types) < self.cfg.max_seg_num:
            ann_types = torch.cat([ann_types, \
                ann_types.new_zeros((self.cfg.max_seg_num - len(ann_types)))])
        
        ann_categories = torch.zeros([
            self.cfg.max_seg_num]).long()
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_segm = io.imread(
            osp.join(
                self.ann_dir,
                "panoptic_segmentation",
                'train2017',
                "{:012d}.png".format(image_id),
            )
        )
        
        to_pil = transforms.ToPILImage()
        img_pil = to_pil(fpn_data['image'])
        to_tensor = transforms.ToTensor()
        weak_aug_img=augmentation_transform(img_pil,crop_size=(fpn_data['image'].shape[1],fpn_data['image'].shape[2]),augmentation_option='weakly',aug_set=self.weak_aug)
        strong_aug_img=augmentation_transform(copy.deepcopy(weak_aug_img),crop_size=(fpn_data['image'].shape[1],fpn_data['image'].shape[2]),augmentation_option='strong',aug_set=self.strong_aug)
        weak_aug_img=to_tensor(weak_aug_img)
        weak_aug_img*=255
        strong_aug_img=to_tensor(strong_aug_img)
        strong_aug_img*=255
        
        weak_aug_fpn_data ={'image':weak_aug_img}
        strong_aug_fpn_data = {'image' : strong_aug_img}

        panoptic_segm = (
            panoptic_segm[:, :, 0]
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )

        # grounding_instances 怎么生成？
        grounding_instances = torch.zeros(
            [self.cfg.max_seg_num, image_info['height'], image_info['width']]
        )
        j = 0
        for i, bbox in enumerate(localized_narrative["boxes"]):
            for b in bbox:
                if b != [0] * 4:
                    segment_info = [
                        s for s in panoptic_ann["segments_info"] 
                        if s["bbox"] == b
                    ][0]
                    segment_cat = [
                        c
                        for c in self.panoptic["categories"]
                        if c["id"] == segment_info["category_id"]
                    ][0]
                    instance = torch.zeros([image_info['height'],
                            image_info['width']])
                    instance[panoptic_segm == segment_info["id"]] = 1
                    grounding_instances[j, :] += instance
                    ann_categories[j] = 1 if \
                            segment_cat["isthing"] else 2
            if grounding_instances[j].sum() != 0:
                j = j + 1

        grounding_instances = {'gt': grounding_instances}

        if self.augmentation:
            return caption, grounding_instances, \
                ann_categories, ann_types, noun_vector_padding, \
                ret_noun_vector, fpn_data,strong_aug_fpn_data,weak_aug_fpn_data
        else:
            return caption, grounding_instances, \
                ann_categories, ann_types, noun_vector_padding, \
                ret_noun_vector, fpn_data

class DatasetTwoCrop(IterableDataset):
    def __init__(self,dataset,batch_size):
        self.label_dataset,self.unlabel_dataset = dataset
        self.batch_size_label = batch_size
        self.batch_size_unlabel = batch_size

    def __iter__(self):
        label_bucket, unlabel_bucket = [], []
        for d_label,d_unlabel in zip(self.label_dataset,self.unlabel_dataset):
            if len(label_bucket) != self.batch_size_label:
                label_bucket.append(d_label)

            if len(unlabel_bucket) != self.batch_size_unlabel:
                unlabel_bucket.append(d_unlabel)

            if (
                len(label_bucket) == self.batch_size_label
                and len(unlabel_bucket) == self.batch_size_unlabel
            ):
                # label_strong, label_weak, unlabed_strong, unlabled_weak
                yield (
                    label_bucket[:],
                    unlabel_bucket[:],
                )
                del label_bucket[:]
                del unlabel_bucket[:]
