import json
import random
from collections import Counter
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp
import sys
sys.path.append("..")
from models.tokenization import BertTokenizer

seed = 1
percent = 40

def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)

def generate_data(seed,percent,data_dir): 

    png_ann_path=osp.join(data_dir, "annotations/png_coco_train2017.json")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    all_data_dict = load_json(png_ann_path)
    ann_dir = osp.join(data_dir,'annotations')
    train_imgCaption_list=[ann['image_id']+'_'+ann['caption'] for ann in all_data_dict]
    new_indics = list(range(len(train_imgCaption_list)))
    random.seed(seed)
    random.shuffle(new_indics)
    train_imgCaption_list=np.array(train_imgCaption_list)

    num_label=int(len(train_imgCaption_list)*percent/100.0)

    label_data=train_imgCaption_list[new_indics[:num_label]]
    unlabel_data=train_imgCaption_list[new_indics[num_label:]]


    label_idx=new_indics[:num_label]
    unlabel_idx=new_indics[num_label:]

    print('sum of labeled data:',len(label_data))
    print('sum of unlabeled data:',len(unlabel_data))

    panoptic = load_json(osp.join(ann_dir, "panoptic_train2017.json"))
    images = panoptic["images"]
    images = {i["id"]: i for i in images}
    panoptic_anns = panoptic["annotations"]
    panoptic_anns = {int(a["image_id"]): a for a in panoptic_anns}
    # image id (int) -> ann item
    panoptic_narratives = all_data_dict # get the png ann

    length = len(panoptic_narratives)
    # 134272 for training
    iterable = tqdm(range(0, length))

    labeled_dict = []
    unlabeled_dict = []
    cnt=0
    max_len = 0
    for idx in iterable:
        narr = panoptic_narratives[idx]
        words = []
        for token in tokenizer.basic_tokenizer.tokenize(narr["caption"].strip()):
                for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                    words.append(sub_token)
        segments = narr["segments"]
        narr["boxes"] = []
        narr["noun_vector"] = []
        image_id = int(narr["image_id"])
        panoptic_ann = panoptic_anns[image_id]
        segment_infos = {}
        for s in panoptic_ann["segments_info"]:
            idi = s["id"]
            segment_infos[idi] = s
        # box ann in panoptic segmentation
        nom_count = 0
        for seg in segments:
                utter = seg["utterance"].strip()
                # "in this"
                if "n't" in utter.lower():
                    ind = utter.lower().index("n't")
                    all_words1 = []
                    for w in tokenizer.basic_tokenizer.tokenize(utter[:ind]):
                        for w_s in tokenizer.wordpiece_tokenizer.tokenize(w):
                            all_words1.append(w_s)
                    all_words2 = []
                    for w in tokenizer.basic_tokenizer.tokenize(utter[ind + 3 :]):
                        for w_s in tokenizer.wordpiece_tokenizer.tokenize(w):
                            all_words2.append(w_s)

                    all_words = all_words1 + ["'", "t"] + all_words2
                else:
                    all_words = []
                    for w in tokenizer.basic_tokenizer.tokenize(utter):
                        for w_s in tokenizer.wordpiece_tokenizer.tokenize(w):
                            all_words.append(w_s)

                nom_count = nom_count + 1 if len(seg["segment_ids"]) > 0 else nom_count

                for word in all_words:
                    word_pi = word
                    if not seg["noun"]:
                        narr["boxes"].append([[0] * 4])
                        narr["noun_vector"].append(0)
                    elif len(seg["segment_ids"]) == 0:
                        narr["boxes"].append([[0] * 4])
                        narr["noun_vector"].append(0)
                    elif len(seg["segment_ids"]) > 0:
                        ids_list = seg["segment_ids"]
                        nose = []
                        for lab in ids_list:
                            box = segment_infos[int(lab)]["bbox"]
                            nose.append(box)
                        narr["boxes"].append(nose)
                        narr["noun_vector"].append(nom_count)
                    else:
                        raise ValueError("Error in data")

        if len(words) == len(narr["boxes"]):
                labels = [[-1 for i in sublist] for sublist in narr["boxes"]]
                ann_mask = [
                    [True if ann == [0] * 4 else False for ann in sublist]
                    for sublist in narr["boxes"]
                ]
                labels = [
                    [-2 if m else l for (m, l) in zip(submask, sublabels)]
                    for (submask, sublabels) in zip(ann_mask, labels)
                ]
                narr["labels"] = labels
                if len(labels) > max_len:
                    max_len = len(labels)

                del narr["segments"]
                if narr['image_id']+'_'+narr['caption'] in label_data:
                    labeled_dict.append(narr)
                else:
                    unlabeled_dict.append(narr)
        else:
            cnt+=1
    
    save_json(osp.join(data_dir,'annotations/png_coco_train2017_labeled_dataloader_seed'+str(seed)+'_sup'+str(percent)+'.json'),labeled_dict)
    save_json(osp.join(data_dir,'annotations/png_coco_train2017_unlabeled_dataloader_seed'+str(seed)+'_sup'+str(percent)+'.json'),unlabeled_dict)