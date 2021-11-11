"""python
    xml文件解析
"""

import json
import os
import torch
import random
import xml.etree.ElementTree as ET    #解析xml文件所用工具
import torchvision.transforms.functional as FT


#GPU设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
#voc_labels为VOC数据集中20类目标的类别名称
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

#创建label_map字典，用于存储类别和类别索引之间的映射关系。比如：{'aeroplane'：1， 'bicycle': 2，......}
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
#VOC数据集默认不含有20类目标中的其中一类的图片的类别为background，类别索引设置为0
label_map['background'] = 0

#将映射关系倒过来，{类别名称：类别索引}
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping {0:'background', 1:'aeroplane'......}

#解析xml文件，最终返回这张图片中所有目标的标注框及其类别信息，以及这个目标是否是一个difficult目标

annotation_path = '../data//VOC2012/Annotations/2007_000032.xml'

def parse_annotation(annotation_path):
    #解析xml
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()    #存储bbox
    labels = list()    #存储bbox对应的label
    difficulties = list()    #存储bbox对应的difficult信息
    
    #遍历xml文件中所有的object，前面说了，有多少个object就有多少个目标
    for object in root.iter('object'):
        #提取每个object的difficult、label、bbox信息
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1
        #存储
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
    
    #返回包含图片标注信息的字典
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

# print(parse_annotation(annotation_path))