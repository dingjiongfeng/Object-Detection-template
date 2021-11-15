import torch
import os
import json
import random
import torchvision.transforms.functional as FT
from .parse import *
from IPython import embed


labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
          'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {v: k+1 for k, v in enumerate(labels)}
label_map['background'] = 0  # 21

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']  # 21
label_color_map = {k: distinct_colors[i] for i, k in enumerate(
    label_map.keys())}  # plane #000000


def xy2cxcy(xy):
    # x_l,y_l,x_r,y_r = xy
    # return [(x_l+x_r)/2, (y_l+y_r)/2,(x_r-x_l)/2,(y_r-y_l)/2]
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h # 第三维：w  第四维：h


def cxcy2xy(cxcy):
    return torch.cat([cxcy[:, :2]-cxcy[:, 2:], cxcy[:, :2]+cxcy[:, 2:]],
                     dim=1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    In the model, we are predicting bounding box coordinates in this encoded form.
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original SSD Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gxcgxy, priors_cxcy):
    return torch.cat([gxcgxy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
                      torch.exp(gxcgxy[:, 2:] / 5) * priors_cxcy[:, 2:]], dim=1)


def find_intersection(set_1, set_2):
    '''
    set_1, tensor of dimensions (n1, 4)
    set_2, tensor of dimensions (n2, 4)
    '''
    lower_bounds = torch.max(
        set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(
        set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    inter_dim = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1,n2,2)
    return inter_dim[:, :, 0] * inter_dim[:, :, 1]  # (n1,n2)


def iou(set1, set2):
    '''
    set_1, tensor of dimensions (n1, 4)
    set_2, tensor of dimensions (n2, 4)
    '''
    inter = find_intersection(set1, set2)  # (n1, n2)
    area_set1 = (set1[:, 2]-set1[:, 0])*(set1[:, 3]-set1[:, 1])  # n1
    area_set2 = (set2[:, 2]-set2[:, 0])*(set2[:, 3]-set2[:, 1])  # n2
    union = area_set1.unsqueeze(1) + area_set2.unsqueeze(0) - inter
    return inter / union  # (n1, n2)

# iou(torch.randn(3,4), torch.rand(2,4))


def create_data_lists(voc12_path='../data/VOC2012', output_folder='../data/VOC2012/json'):
    '''
    生成train_images.json, train_objects.json
        val_images.json, val_objects.json四个json文件
    '''
    voc12_path = os.path.abspath(voc12_path)
    for split in ['train', 'val']:
        train_images = list()
        train_objects = list()
        n_objects = 0
        with open(os.path.join(voc12_path, 'ImageSets/Main/'+split+'.txt')) as f:
            # 都是返回列表 .read().splitlines()每一行后面没有\n  readlines 每一行后面有\n
            ids = f.read().splitlines()
        print('id finish')
        for id in ids:
            objects = parse_annotation(os.path.join(
                voc12_path, 'Annotations', id+'.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects['boxes'])
            train_objects.append(objects)  # 所有信息
            train_images.append(os.path.join(
                voc12_path, 'JPEGImages', id+'.jpg'))
        print('parse finish')
        assert len(train_images) == len(train_objects)

        with open(os.path.join(output_folder, split+'_images.json'), 'w') as j:
            json.dump(train_images, j)
        with open(os.path.join(output_folder, split+'_objects.json'), 'w') as j:
            json.dump(train_objects, j)

    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)


# create_data_lists()  # 1. create data json


def adjust_learning_rate(optimizer, scale):
    # change the learning rate of optimizer
    for param in optimizer.param_groups:
        param['lr'] = param['lr'] * scale
    print(
        f"learning rate decaying\n The new lr is {optimizer.param_groups[0]['lr']}\n")


def save_checkpoint(path, epoch, model, optimizer):
    '''
    保存三个键值对，epoch，整个model，optimizer（lr）
    '''
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(state_dict, path)


def load_checkpoint(path, model, optimizer):
    '''
    将path路径下的文件加载到model和optimizer， epoch赋给start_epoch
    return start_epoch 开始的epoch
    '''
    if not os.path.exists(path):
        print('Sorry, don\'t have checkpoint.pth file, continue training!')
        return 0
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    return start_epoch


class AverageMeter(object):
    '''
    a metric of average, sum and count
    '''

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.average = 0
        self.sum = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


"""
可以看到，transform分为TRAIN和TEST两种模式，以本实验为例：

在TRAIN时进行的transform有：
1.以随机顺序改变图片亮度，对比度，饱和度和色相，每种都有50％的概率被执行。photometric_distort
2.扩大目标，expand
3.随机裁剪图片，random_crop
4.0.5的概率进行图片翻转，flip
*注意：a. 第一种transform属于像素级别的图像增强，目标相对于图片的位置没有改变，因此bbox坐标不需要变化。
         但是2，3，4，5都属于图片的几何变化，目标相对于图片的位置被改变，因此bbox坐标要进行相应变化。
         
在TRAIN和TEST时都要进行的transform有：
1.统一图像大小到(224,224)，resize
2.PIL to Tensor
3.归一化，FT.normalize()

注1: resize也是一种几何变化，要知道应用数据增强策略时，哪些属于几何变化，哪些属于像素变化
注2: PIL to Tensor操作，normalize操作必须执行
"""


def photometric_distort(image):
    '''
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    param image: PIL image
    return distorted image
    '''
    new_image = image
    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]
    random.shuffle(distortions)
    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                adjust_factor = random.uniform(-18/255., 18/255.)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            new_image = d(new_image, adjust_factor)  # apply this distortion

    return new_image


def expand(image, boxes, filler):
    '''
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    image:tensor shape(channel, h, w)
    boxes: 边界框
    filler:边缘填入的数字
    '''
    # print('expand', image.shape)
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    filler = torch.FloatTensor(filler)
    new_image = torch.ones(
        (3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)
    # 大小全是mean 只有

    left = random.randint(0, new_w-original_w)
    right = left + original_w
    top = random.randint(0, new_h-original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # adjust boxes accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)
    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    '''
    param image: image, a tensor of dimensions (3, original_h, original_w)
    param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    param labels: labels of objects, a tensor of dimensions (n_objects)
    param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    '''
    original_h = image.size(1)
    original_w = image.size(2)
    while True:
        min_overlap = random.choice([0., 1., 3., 5., 7., 9., None])
        if min_overlap is None:
            return image, boxes, labels, difficulties

        max_iter = 50
        for _ in range(max_iter):
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # new photo left,right,top,bottom
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, right, top, bottom])  # size=(4)

            union = iou(crop.unsqueeze(0), boxes)  # (1, 4)
            union = union.squeeze(0)

            if union.max().item() < min_overlap:
                continue
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2  # (n_objects, 2)
            center_in_crop = ((left < bb_centers[:, 0]) * (right > bb_centers[:, 0])
                              * (top < bb_centers[:, 1]) * (bottom > bb_centers[:, 1]))

            if not center_in_crop.any():
                continue
            new_boxes = boxes[center_in_crop, :]
            new_labels = labels[center_in_crop]
            new_difficulties = difficulties[center_in_crop]

            new_boxes[:, :2] = torch.max(
                new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            # crop[2:] is [right, bottom]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]
            new_boxes.clamp_(min=0, max=1)

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    '''
    horizontal flip image and boxes
    image :PIL image
    boxes: bounding boxes  a tensor of dimensions (n_objects, 4)
    return flipped image, update bounding box coordinates
    '''
    # Flip image
    new_image = FT.hflip(image)
    # Flip boxes
    new_boxes = boxes
    w = image.width
    new_boxes[:, 0] = w - boxes[:, 0] - 1
    new_boxes[:, 2] = w - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]  # 重新排列点 使其指向左上角和右下角
    return new_image, new_boxes


def resize(image, boxes, dims=(224, 224), return_percent_coords=True):
    '''
    resize image and boxes
    image :PIL image
    boxes: bounding boxes  a tensor of dimensions (n_objects, 4)
    return resized image, update bounding box coordinates
    '''
    new_image = FT.resize(image, dims)

    old_dims = torch.FloatTensor(
        [image.width, image.height, image.width, image.height]).unsqueeze(0)  # (1,4)
    new_boxes = boxes / old_dims
    if not return_percent_coords:
        new_dims = torch.FloatTensor(
            [dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def transform(image, boxes, labels, difficulties, split):
    # print('transform', image.size)

    assert split in ('train', 'val')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # if split == 'train':
    #     new_image = photometric_distort(new_image)
    #     new_image = FT.to_tensor(new_image)
    #     # Expand image with a 50% chance(zoom out)
    #     if random.random() < 0.5:
    #         new_image, new_boxes = expand(new_image, boxes, filler=mean)
    #     # randomly crop image(zoom in)
    #     new_image, new_boxes, new_labels, new_difficulties = \
    #         random_crop(new_image, new_boxes, new_labels, new_difficulties)

    #     new_image = FT.to_pil_image(new_image)
    #     if random.random() < 0.5:
    #         new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (224,224) - this also converts absolute boundary coordinates to new position
    new_image, new_boxes = resize(new_image, new_boxes, dims=(224, 224))
    new_image = FT.to_tensor(new_image)
    new_image = FT.normalize(new_image, mean=mean, std=std)
    return new_image, new_boxes, new_labels, new_difficulties


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.
    这里的map指标遵循的VOC2007的标准，具体地：
    统一用IOU>0.5作为目标框是否准召的标准
    AP的计算标准采用召回分别为 0:0.05:1 时的准确率平均得到
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation
    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    # make sure all lists of tensors of the same length, i.e. number of images
    assert len(det_boxes) == len(det_labels) == len(det_scores) == \
        len(true_boxes) == len(true_labels) == len(true_difficulties)
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    # (n_objects), n_objects: total num of objects across all images
    true_images = torch.LongTensor(true_images).to(device)
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from 第几张图片
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(
        0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros(
        (n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        # (n_class_objects)
        true_class_difficulties = true_difficulties[true_labels == c]
        # ignore difficult objects
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros(
            (true_class_difficulties.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_images.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(
            det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(
            device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(
            device)  # (n_class_detections)

        for d in range(n_class_detections):
            # 对于每一个检测到的图像进行评价,按照分数从高到低
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            # (n_class_objects_in_img, 4)
            object_boxes = true_class_boxes[true_class_images == this_image]
            # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            # (1, n_class_objects_in_img) (1, 4), (1, 4)
            overlaps = iou(this_detection_box, object_boxes)  # (1, 1)
            max_overlap, ind = torch.max(
                overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[
                true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        # this object has now been detected/accounted for
                        true_class_boxes_detected[original_ind] = 1
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(
            true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(
            false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
            cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / \
            n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(
            start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros(
            (len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        # c is in [1, n_classes - 1]
        average_precisions[c - 1] = precisions.mean()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {
        rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision
