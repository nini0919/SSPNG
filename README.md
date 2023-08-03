# SSPNG
The official implementation of the SSPNG paper in PyTorch.

## Installation

### Requirements

- Python
- Numpy
- Pytorch 1.10.0
- Tqdm
- Scipy 1.7.3

## Dataset Preparation

1. Download the 2017 MSCOCO Dataset from its [official webpage](https://cocodataset.org/#download). You will need the train and validation splits' images and panoptic segmentations annotations.

2. Download the Panoptic Narrative Grounding Benchmark from the PNG's [project webpage](https://bcv-uniandes.github.io/panoptic-narrative-grounding/#downloads). Organize the files as follows:

```
datasets
|_coco
    |_ train2017
    |_ val2017
    |_ panoptic_stuff_train2017
    |_ panoptic_stuff_val2017
    |_annotations
        |_ png_coco_train2017.json
        |_ png_coco_val2017.json
        |_ panoptic_segmentation
        |  |_ train2017
        |  |_ val2017
        |_ panoptic_train2017.json
        |_ panoptic_val2017.json
        |_ instances_train2017.json
```

3. Pre-process the Panoptic narrative Grounding Ground-Truth Annotation for the dataloader using [utils/pre_process.py](utils/pre_process.py).

4. At the end of this step you should have two new files in your annotations folder.
```
datasets
|_coco
    |_ train2017
    |_ val2017
    |_ panoptic_stuff_train2017
    |_ panoptic_stuff_val2017
    |_annotations
        |_ png_coco_train2017.json
        |_ png_coco_val2017.json
        |_ panoptic_segmentation
        |  |_ train2017
        |  |_ val2017
        |_ panoptic_train2017.json
        |_ panoptic_val2017.json
        |_ instances_train2017.json
        |_ png_coco_train2017_dataloader.json
        |_ png_coco_val2017_dataloader.json
```

## Train and Inference

### Pretrained Model



To reproduce all our results as reported bellow, you can use our pretrained model and our source code.

| Model | link |
| ----- | ---- |
| FPN   | [fpn](https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl)|
| Bert-base-uncased   | [bert](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz)|
| SSPNG   | [sspng](https://drive.google.com/drive/folders/1dt81kTTiqqPe80hIuY-_ZJbX84A_3J0u?usp=drive_link)|


### Train
1. Modify the routes in [train.sh](train.sh) according to your local paths. 
2. Run train.sh
### Inference

Run test.sh to test the pretrained model, modify the pretrained model path `--ckpt_path`.
