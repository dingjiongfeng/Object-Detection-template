from utils.utils import *
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from IPython import embed
from math import sqrt

import sys
sys.path.append('..')  # 到上级目录

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        # stride = 1, by default
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    # 224->112

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)    # 112->56

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)    # 56->28

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)    # 28->14

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)    # 14->7

        # Load pretrained weights on ImageNet
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 224, 224)
        :return: feature maps pool5
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 224, 224)
        out = F.relu(self.conv1_2(out))  # (N, 64, 224, 224)
        out = self.pool1(out)  # (N, 64, 112, 112)

        out = F.relu(self.conv2_1(out))  # (N, 128, 112, 112)
        out = F.relu(self.conv2_2(out))  # (N, 128, 112, 112)
        out = self.pool2(out)  # (N, 128, 56, 56)

        out = F.relu(self.conv3_1(out))  # (N, 256, 56, 56)
        out = F.relu(self.conv3_2(out))  # (N, 256, 56, 56)
        out = F.relu(self.conv3_3(out))  # (N, 256, 56, 56)
        out = self.pool3(out)  # (N, 256, 28, 28)

        out = F.relu(self.conv4_1(out))  # (N, 512, 28, 28)
        out = F.relu(self.conv4_2(out))  # (N, 512, 28, 28)
        out = F.relu(self.conv4_3(out))  # (N, 512, 28, 28)
        out = self.pool4(out)  # (N, 512, 14, 14)

        out = F.relu(self.conv5_1(out))  # (N, 512, 14, 14)
        out = F.relu(self.conv5_2(out))  # (N, 512, 14, 14)
        out = F.relu(self.conv5_3(out))  # (N, 512, 14, 14)
        out = self.pool5(out)  # (N, 512, 7, 7)

        # return 7*7 feature map
        return out

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        # pytorch 中的 state_dict 是一个简单的python的字典对象,
        # 将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
        param_names = list(state_dict.keys())
        # print(param_names)
        pretrain_state_dict = torchvision.models.vgg16(
            pretrained=False).state_dict()
        pretrain_param_names = list(pretrain_state_dict.keys())
        # print(pretrain_param_names)
        for i, param in enumerate(param_names):
            state_dict[param] = pretrain_state_dict[pretrain_param_names[i]]

        self.load_state_dict(state_dict)
        print('loaded base model.\n')


class PredictionConvolutions(nn.Module):
    """ 
    Convolutions to predict class scores and bounding boxes using feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 441 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
    这里预测坐标的编码方式完全遵循的SSD的定义

    The class scores represent the scores of each object class in each of the 441 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """ 
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        self.num_box = 9  # 每个点有3x3=9个锚框
        self.loc_conv = nn.Conv2d(
            512, self.num_box * 4, kernel_size=3, stride=1, padding=1)
        self.clf_conv = nn.Conv2d(
            512, self.num_box * n_classes, kernel_size=3, stride=1, padding=1)

        self.init_conv2d()

    def forward(self, pool5_feats):  # NotImplementedError没有实现父类的方法 方法名打成了forawrd
        '''
        param: pool5_feats the feature map after last pooling layer dimensions (N, 512, 7, 7)
        total 441 prior_boxes
        return 411 locations and class scores (N, 441, 4), (N, 441, n_classes)
        模型要预测anchor与目标框的偏移, 边界框编码后的偏移量(g_{cx},g_{cy},g_w,g_h) 返回偏移量g_{cx},g_{cy}
        '''
        batch_size = pool5_feats.size(0)
        location = self.loc_conv(pool5_feats)  # N,36,7,7
        location = location.permute([0, 2, 3, 1]).contiguous()  # N, 7, 7, 36
        location = location.view(batch_size, -1, 4)  # N, 441, 4

        classes = self.clf_conv(pool5_feats)  # N,9*21,7,7
        classes = classes.permute([0, 2, 3, 1]).contiguous()  # N, 7, 7, 189
        classes = classes.view(batch_size, -1, self.n_classes)  # N, 441, 21

        return location, classes

    def init_conv2d(self):
        for i in self.children():
            if isinstance(i, nn.Conv2d):
                nn.init.xavier_uniform_(i.weight)
                nn.init.constant_(i.bias, 0.)


class tiny_detector(nn.Module):
    '''
    model 包含一个VGG特征提取模块，在最后一个特征图上添加预测头模块来预测目标框信息
    '''

    def __init__(self, n_classes) -> None:
        super(tiny_detector, self).__init__()
        self.n_classes = n_classes
        self.base = VGGBase()
        self.predictor = PredictionConvolutions(n_classes)
        self.priors_cxcy = self.create_prior_boxes()
        print('Model build up!')

    def forward(self, image):
        pool5_feats = self.base(image)
        location, classes = self.predictor(pool5_feats)
        return location, classes

    def create_prior_boxes(self):
        '''
        最后的特征图的尺寸为7*7，九种不同大小和形状的候选框。
        create 7*7*9=441个候选框
        return prior_cxcy, shape(441, 4) 位于49个不同的位置，每个位置9个框

        '''
        fmap_dims = 7
        obj_scales = [0.2, 0.4, 0.6]
        ratios = [1., 2., 0.5]  # 高宽比

        prior_boxes = []
        for i in range(fmap_dims):
            for j in range(fmap_dims):
                cx = (j+0.5) / fmap_dims
                cy = (i+0.5) / fmap_dims

                for obj_scale in obj_scales:
                    for ratio in ratios:
                        prior_boxes.append(
                            [cx, cy, obj_scale * sqrt(ratio), obj_scale / sqrt(ratio)])
                        # ==  obj_scale / sqrt(ratio), obj_scale * sqrt(ratio)
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        NMS
        Decipher the 441 locations and class scores (output of the tiny_detector) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 441 prior boxes, a tensor of dimensions (N, 441, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 441, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(
            predicted_scores, dim=2)  # (N, 441, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images in batch
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy2xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (441, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # max_scores, best_label = predicted_scores[i].max(dim=1)  # (441)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (441)
                # torch.uint8 (byte) tensor, for indexing
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                # (n_qualified), n_min_score <= 441
                class_scores = class_scores[score_above_min_score]
                # (n_qualified, 4)
                class_decoded_locs = decoded_locs[score_above_min_score]

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(
                    descending=True)  # (n_qualified), (n_qualified)
                # (n_qualified, 4)
                class_decoded_locs = class_decoded_locs[sort_ind]

                # Find the overlap between predicted boxes
                # (n_qualified, n_qualified)
                overlap = iou(class_decoded_locs, class_decoded_locs)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(
                    device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                # 遍历suppress一维数组 shape: (n_qualified)
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with current box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(
                        suppress, (overlap[box] > max_overlap).to(torch.bool))
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class dtype=torch.bool, use ~x instead of 1-x
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(torch.LongTensor(
                    (~suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor(
                    [[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(
                    dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        # lists of length batch_size
        return all_images_boxes, all_images_labels, all_images_scores


class MultiBoxLoss(nn.Module):
    """
    The loss function for object detection.
    对于Loss的计算，完全遵循SSD的定义，即 MultiBox Loss
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes.
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy2xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  # MAE 均绝对值误差
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 441 prior boxes, a tensor of dimensions (N, 441, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 441, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(
            device)  # (N, 441, 4)
        true_classes = torch.zeros(
            (batch_size, n_priors), dtype=torch.long).to(device)  # (N, 441)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = iou(boxes[i], self.priors_xy)  # (n_objects, 441)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(
                dim=0)  # (441)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(
                range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (441)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior <
                                 self.threshold] = 0  # (441)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy2cxcy(
                boxes[i][object_for_each_prior]), self.priors_cxcy)  # (441, 4) true_locs有nan
            # boxes中有负数导致的，经过torch.log直接没有定义了，  object_for_each_prior

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 441)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(
            predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 441)
        # So, if predicted_locs has the shape (N, 441, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(
            predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 441)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 441)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 441)
        # (N, 441), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg[positive_priors] = 0.
        # (N, 441), sorted by decreasing hardness
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(
            0).expand_as(conf_loss_neg).to(device)  # (N, 441)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(
            1)  # (N, 441)
        # (sum(n_hard_negatives))
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()
                     ) / n_positives.sum().float()  # (), scalar

        # return TOTAL LOSS
        return conf_loss + self.alpha * loc_loss  # loc_loss:nan


# vgg = VGGBase()
