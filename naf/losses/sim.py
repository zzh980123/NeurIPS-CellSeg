import torch
import torch.nn.functional as F
import numpy as np
import math

from torch.nn.modules.loss import _Loss


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [7] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)

        J_sum[torch.isnan(J_sum)] = 0
        I_sum[torch.isnan(I_sum)] = 0

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # u_I[torch.isnan(u_I)] = 0
        # u_J[torch.isnan(u_J)] = 0

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class ConfidenceMSE:
    def loss(self, y_true, y_pred, weight):
        return torch.mean(weight * (y_true - y_pred) ** 2)


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Grad2D:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class GradSim2D:
    """
    N-D gradient loss.
    """

    def __init__(self, loss_mult=None):
        self.loss_mult = loss_mult

    def loss(self, y_gt, y_pred):
        dy_gt = y_gt[:, :, 1:, :] - y_gt[:, :, :-1, :]
        dx_gt = y_gt[:, :, :, 1:] - y_gt[:, :, :, :-1]
        dy_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        dx_pred = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

        d = torch.mean(torch.square(dx_pred - dx_gt)) + torch.mean(torch.square(dy_pred - dy_gt))
        grad = d / 4.0

        return grad


from monai.losses.dice import DiceLoss
from torch.nn import Sigmoid


class MSEGrad2D:
    def __init__(self, a=0.2, b=1, edge_penalty=50, penalty='l2', loss_mult=None):
        self.mse = MSE()
        self.dice = DiceLoss()
        self.grad2d = Grad2D(penalty=penalty, loss_mult=loss_mult)
        self.a = a
        self.b = b
        self.sigmoid = Sigmoid()
        self.edge_p = edge_penalty

    def loss(self, y_gt, y_pred):
        # pull to baohequ
        # gt_inside = self.sigmoid(y_gt * (y_gt < 0.5) * 1000)
        # pred_inside = self.sigmoid(y_pred * (y_pred < 0.5) * 1000)
        # gt_outside = self.sigmoid(y_gt * (y_gt > 0.5) * 1000)
        # pred_outside = self.sigmoid(y_pred * (y_pred > 0.5) * 1000)
        # gt_ol = y_gt == 0.50
        # pre_ol = y_pred == 0.50
        #
        # l2 = torch.square(y_gt - y_pred)
        # ex = l2 * (gt_ol + pre_ol) * self.edge_p

        return self.b * self.mse.loss(y_gt, y_pred) + self.a * self.grad2d.loss(None, y_pred)


class ConfidenceBCELoss(torch.nn.modules.Module):

    def __init__(self, t=0.6, reduction='mean'):
        self.t = t
        self.reduction = reduction
        # self.bce = torch.nn.BCELoss()

    def forward(self, pred_label, label):
        pl = torch.softmax(pred_label, dim=1)
        l = torch.softmax(label, dim=1)
        confidence = pl[:, 1:].clone().detach()
        confidence[confidence < self.t] = 1e-5

        return F.binary_cross_entropy(pl[:, 1:], l[:, 1:], weight=confidence, reduction=self.reduction)


def semi_ce_loss(inputs, targets,
                 conf_mask=True, threshold=None,
                 threshold_neg=.0, temperature_value=1, classes_num=2):
    # target => logit, input => logit
    pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = F.softmax(targets / temperature_value, dim=1)

        # for positive
        targets_real_prob = F.softmax(targets, dim=1)

        weight = targets_real_prob.max(1)[0]
        total_number = len(targets_prob.flatten(0))
        boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                    "0.3~0.4", "0.4~0.5", "0.5~0.6",
                    "0.6~0.7", "0.7~0.8", "0.8~0.9",
                    "> 0.9"]

        rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)]

        max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                    / weight.numel() for i in range(1, 11)]

        pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold)

        mask_neg = (targets_prob < threshold_neg)

        neg_label = torch.nn.functional.one_hot(torch.argmax(targets_prob, dim=1)).type(targets.dtype)
        if neg_label.shape[-1] != classes_num:
            neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
                                                           neg_label.shape[2], classes_num - neg_label.shape[-1]]).cuda()),
                                  dim=3)
        neg_label = neg_label.permute(0, 3, 1, 2)
        neg_label = 1 - neg_label

        if not torch.any(mask):
            neg_prediction_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
            return zero, pass_rate, negative_loss_mat[mask_neg].mean()
        else:
            positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
            positive_loss_mat = positive_loss_mat * weight

            neg_prediction_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))

            return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
    else:
        raise NotImplementedError


"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""


class LovaszSoftmaxLoss(_Loss):

    def __init__(self, classes='present', per_image=False, ignore=None):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return lovasz_softmax(input, target, self.classes, self.per_image, self.ignore)


import torch
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class DirectionLoss(torch.nn.modules.Module):
    def __init__(self, eps=1e-5):
        super(DirectionLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        normal_input = input / (torch.abs(torch.norm(input, dim=1, keepdim=True)) + self.eps)
        normal_target = target / (torch.abs(torch.norm(target, dim=1, keepdim=True)) + self.eps)
        cos_theta = torch.sum(normal_input * normal_target, dim=1, keepdim=True)
        direction_score = (cos_theta - 1) ** 2

        return torch.mean(direction_score)


class GradCenterLoss(torch.nn.Module):
    def __init__(self, eps=1e-2):
        super(GradCenterLoss, self).__init__()
        self.eps = eps

    def to_mag(self, grad):
        mag = torch.sum(grad ** 2, dim=1, keepdim=True)
        return mag

    def forward(self, input, target, mask=None):
        if mask is None:
            mask = 1
        input_mag = self.to_mag(input)
        target_mag = self.to_mag(target)

        input_center = (1 - input_mag) * (input_mag <= self.eps)
        target_center = (1 - target_mag) * (target_mag <= self.eps)

        return torch.mean(torch.square((input_center - target_center) * mask))


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes is 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.reshape(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


if __name__ == '__main__':
    dl = DirectionLoss()
    a = torch.tensor([[1, 2], [4, 3]])
    b = torch.tensor([[1, 2], [2, 3]])

    s = dl.forward(a, b)

    print(s)
