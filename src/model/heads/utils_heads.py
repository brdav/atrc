import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


def spatial_normalize_pred(pred, image, ignore_index=255):
    prob = {}
    for t in pred.keys():
        task_pred = pred[t]
        batch_size, num_classes, H, W = task_pred.size()
        # check for ignore_index in input image, arising for example from data augmentation
        ignore_mask = (nn.functional.interpolate(image, size=(
            H, W), mode='nearest') == ignore_index).any(dim=1, keepdim=True)
        # so they won't contribute to the softmax
        task_pred[ignore_mask.expand_as(task_pred)] = -float('inf')
        c_probs = nn.functional.softmax(
            task_pred.view(batch_size, num_classes, -1), dim=2)
        # if the whole input image consisted of ignore regions, then context probs should just be zero
        prob[t] = torch.where(torch.isnan(
            c_probs), torch.zeros_like(c_probs), c_probs)
    return prob
