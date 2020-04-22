
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import keras
from keras.utils import Sequence
from keras.layers import concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, BatchNormalization

from keras import backend as K

import h5py
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
from keras.backend import binary_crossentropy

import datetime
import os
import random
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

img_rows = 112
img_cols = 112

smooth = 1e-12

num_channels = 22
num_mask_channels = 2

#Keeping original Jaccard coef code for testing

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


#Jaccard coef defined for Pytorch with help from https://github.com/pytorch/ignite/blob/master/ignite/metrics/confusion_matrix.py#L129
import numbers
from typing import Optional, Union, Any, Callable, Sequence

from ignite.metrics import Metric, MetricsLambda
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

__all__ = ["ConfusionMatrix", "mIoU", "IoU", "DiceCoefficient", "cmAccuracy", "cmPrecision", "cmRecall"]


class ConfusionMatrix(Metric):
    """Calculates confusion matrix for multi-class data.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.
    Args:
        num_classes (int): number of classes. See notes for more details.
        average (str, optional): confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (str of torch.device, optional): device specification in case of distributed computation usage.
            In most of the cases, it can be defined as "cuda:local_rank" or "cuda"
            if already set `torch.cuda.set_device(local_rank)`. By default, if a distributed process group is
            initialized and available, device is set to `cuda`.
    Note:
        In case of the targets `y` in `(batch_size, ...)` format, target indices between 0 and `num_classes` only
        contribute to the confusion matrix and others are neglected. For example, if `num_classes=20` and target index
        equal 255 is encountered, then it is filtered out.
    """

    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = None,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of ['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        super(ConfusionMatrix, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64, device=self._device)
        self._num_examples = 0

    def _check_shape(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if y_pred.ndimension() < 2:
            raise ValueError(
                "y_pred must have shape (batch_size, num_categories, ...), " "but given {}".format(y_pred.shape)
            )

        if y_pred.shape[1] != self.num_classes:
            raise ValueError(
                "y_pred does not have correct number of categories: {} vs {}".format(y_pred.shape[1], self.num_classes)
            )

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                "y_pred must have shape (batch_size, num_categories, ...) and y must have "
                "shape of (batch_size, ...), "
                "but given {} vs {}.".format(y.shape, y_pred.shape)
            )

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output

        self._num_examples += y_pred.shape[0]

        # target is (batch_size, ...)
        y_pred = torch.argmax(y_pred, dim=1).flatten()
        y = y.flatten()

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    @sync_all_reduce("confusion_matrix", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Confusion matrix must have at least one example before it can be computed.")
        if self.average:
            self.confusion_matrix = self.confusion_matrix.float()
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            elif self.average == "recall":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=1).unsqueeze(1) + 1e-15)
            elif self.average == "precision":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=0) + 1e-15)
        return self.confusion_matrix


#This definition calculates the Jaccard index
def IoU(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    """Calculates Intersection over Union using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        ignore_index (int, optional): index to ignore, e.g. background index
    Returns:
        MetricsLambda
    Examples:
    .. code-block:: python
        train_evaluator = ...
        cm = ConfusionMatrix(num_classes=num_classes)
        IoU(cm, ignore_index=0).attach(train_evaluator, 'IoU')
        state = train_evaluator.run(train_dataset)
        # state.metrics['IoU'] -> tensor of shape (num_classes - 1, )
    """
    if not isinstance(cm, ConfusionMatrix):
        raise TypeError("Argument cm should be instance of ConfusionMatrix, but given {}".format(type(cm)))

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError("ignore_index should be non-negative integer, but given {}".format(ignore_index))

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError(
                    "ignore_index {} is larger than the length of IoU vector {}".format(ignore_index, len(iou_vector))
                )
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou

#U-Net in pytorch modified from https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ELU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ELU(inplace=True)
    )
class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down0 = double_conv(3, 32)
        self.dconv_down1 = double_conv(32, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
            
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.dconv_up0 = double_conv(64 + 32, 32)
            
        self.conv_last = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        conv0 = self.dconv_down0(x)
        x = self.maxpool(conv1)
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
            
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
            
        x = self.dconv_down4(x)
            
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
            
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
            
        x = self.dconv_up1(x)
        x = self.upsample(x)
        x = torch.cat([x, conv0], dim=1)

        x = self.dconv_up0(x)
            
        out = self.conv_last(x)
            
        return out

def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows-32, img_cols-32))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        X_batch[i] = X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        yb = y[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        y_batch[i] = yb[:, 16:16 + img_rows - 32, 16:16 + img_cols - 32]
    return np.transpose(X_batch, (0, 2, 3, 1)), np.transpose(y_batch, (0, 2, 3, 1))

class data_generator(Sequence):

    def __init__(self, x_set, y_set, batch_size, horizontal_flip, vertical_flip, swap_axis):
        self.swap_axis = swap_axis
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        X_batch, y_batch = form_batch(self.x, self.y, self.batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    xb = np.fliplr(xb)
                    yb = np.fliplr(yb)

            if self.vertical_flip:
                if np.random.random() < 0.5:
                    xb = np.flipud(xb)
                    yb = np.flipud(yb)

            if self.swap_axis:
                if np.random.random() < 0.5:
                    xb = np.rot90(xb)
                    yb = np.rot90(yb)

            X_batch[i] = xb
            y_batch[i] = yb
        X_batch = torch.Tensor(X_batch)
        y_batch = torch.Tensor(y_batch)
        
        return X_batch, y_batch #Changed this from yield to return for running the same file and returns tensors for Pytorch loading



if __name__ == '__main__':
    from collections import defaultdict
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_class=2)
    model = model.to(device)

    def dice_loss(pred, target, smooth = 1.):
        pred = pred.contiguous()
        target = target.contiguous()    

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        
        return loss.mean()

    def calc_loss(pred, target, metrics, bce_weight=0.5):
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = F.sigmoid(pred)
        dice = dice_loss(pred, target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def print_metrics(metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))

    def train_model(model, optimizer, scheduler, num_epochs=5):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group['lr'])

                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                #Takes in random batch data for training
                for inputs, labels in X_train, y_train:
                    inputs = X_train.to(device)
                    labels = y_train.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = calc_loss(outputs, labels, metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    
    data_path = os.getcwd()
    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_class=2)
    model = model.to(device)

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    f = h5py.File(os.path.join(data_path, 'train_t_c.h5'), 'r')

    X_train = f['train']

    y_train = np.array(f['train_mask'])
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)

    train_ids = np.array(f['train_ids'])
    
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=6)

    model()

    f.close()
