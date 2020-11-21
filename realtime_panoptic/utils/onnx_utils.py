# Copyright 2020 Toyota Research Institute.  All rights reserved.

from torch import nn
from torchvision.ops import misc as misc_nn_ops
from realtime_panoptic.layers.onnx_group_norm import ONNXAbleGN


def convert_group_norm_to_onnxable(model):
    """Recursively replace GroupNorms with ONNXAble GroupNorm in a model.

    Doesn't copy weights: the state dict is assumed to be loaded later

    Parameters
    ----------
    model: torch.nn.Module
        The model to be converted.

    Returns
    -------
    model: torch.nn.Module
        The converted model.
    """
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_group_norm_to_onnxable(module)

        if type(module) == nn.GroupNorm:
            model._modules[name] = ONNXAbleGN(module.num_groups, module.num_channels, module.eps, module.affine)

    return model


def convert_frozen_batchnorm_to_batchnorm(model):
    """Recursively replace GroupNorms with ONNXAble GroupNorm in a model.

    Doesn't copy weights: the state dict is assumed to be loaded later

    Parameters
    ----------
    model: torch.nn.Module
        The model to be converted.

    Returns
    -------
    model: torch.nn.Module
        The converted model.
    """
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_frozen_batchnorm_to_batchnorm(module)

        if type(module) == misc_nn_ops.FrozenBatchNorm2d:
            model._modules[name] = BatchNorm2d(module.num_channels, module.eps, module.affine)

    return model