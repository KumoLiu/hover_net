# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================================
# Adapted from https://github.com/vqdang/hover_net
# which has the following license:
# https://github.com/vqdang/hover_net/blob/master/LICENSE
# MIT License

# Original publication:
#  @article{graham2019hover,
#    title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
#    author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak,
#            Jin Tae and Rajpoot, Nasir},
#    journal={Medical Image Analysis},
#    pages={101563},
#    year={2019},
#    publisher={Elsevier}
# }
# =========================================================================

from collections import OrderedDict

import torch
import torch.nn as nn

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape

import re
import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from monai.apps.utils import download_url
from monai.networks.blocks import UpSample
from monai.networks.layers.factories import Conv, Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.enums import Enum, InterpolateMode, UpsampleMode
from monai.utils.module import export, look_up_option

__all__ = ["HoVerNet", "Hovernet", "HoVernet", "HoVerNet"]


class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class HoVerNetMode(StrEnum):
    """
    Modes for HoVerNet model:
    `FAST`: a faster implementation (than original)
    `ORIGINAL`: the original implementation
    """

    FAST = "FAST"
    ORIGINAL = "ORIGINAL"


class HoVerNetBranch(StrEnum):
    """
    Three branches of HoVerNet model, which results in three outputs:
    `HV` is horizontal and vertical gradient map of each nucleus (regression),
    `NP` is the pixel prediction of all nuclei (segmentation), and
    `NC` is the type of each nucleus (classification).
    """

    HV = "hv"
    NP = "np"
    NC = "tp"


class _DenseLayerDecoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        kernel_size: int = 3,
        padding: int = 0,
    ) -> None:
        """
        Args:
            num_features: number of internal channels used for the layer
            in_channels: number of the input channels.
            out_channels: number of the output channels.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            kernel_size: size of the kernel for >1 convolutions (dependent on mode)
            padding: padding value for >1 convolutions.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 2]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, 2]

        self.layers = nn.Sequential()

        self.layers.add_module("preact_bna/bn", get_norm_layer(name=norm, spatial_dims=2, channels=in_channels))
        self.layers.add_module("preact_bna/relu", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, num_features, kernel_size=1, bias=False))
        self.layers.add_module("conv1/norm", get_norm_layer(name=norm, spatial_dims=2, channels=num_features))
        self.layers.add_module("conv1/relu2", get_act_layer(name=act))
        self.layers.add_module(
            "conv2",
            conv_type(num_features, out_channels, kernel_size=kernel_size, padding=padding, groups=4, bias=False),
        )

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.layers(x)
        if x1.shape[-1] != x.shape[-1]:
            trim = (x.shape[-1] - x1.shape[-1]) // 2
            x = x[:, :, trim:-trim, trim:-trim]

        x = torch.cat([x, x1], 1)

        return x


class _DecoderBlock(nn.Sequential):
    def __init__(
        self,
        layers: int,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        kernel_size: int = 3,
        same_padding: bool = False,
    ) -> None:
        """
        Args:
            layers: number of layers in the block.
            num_features: number of internal features used.
            in_channels: number of the input channel.
            out_channels: number of the output channel.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            kernel_size: size of the kernel for >1 convolutions (dependent on mode)
            same_padding: whether to do padding for >1 convolutions to ensure
                the output size is the same as the input size.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 2]

        padding: int = kernel_size // 2 if same_padding else 0

        self.add_module(
            "conva", conv_type(in_channels, in_channels // 4, kernel_size=kernel_size, padding=padding, bias=False)
        )

        _in_channels = in_channels // 4
        for i in range(layers):
            layer = _DenseLayerDecoder(
                num_features,
                _in_channels,
                out_channels,
                dropout_prob,
                act=act,
                norm=norm,
                kernel_size=kernel_size,
                padding=padding,
            )
            _in_channels += out_channels
            self.add_module("denselayerdecoder%d" % (i + 1), layer)

        trans = _Transition(_in_channels, act=act, norm=norm)
        self.add_module("bna_block", trans)
        self.add_module("convf", conv_type(_in_channels, _in_channels, kernel_size=1, bias=False))


class _DenseLayer(nn.Sequential):
    def __init__(
        self,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        drop_first_norm_relu: int = 0,
        kernel_size: int = 3,
    ) -> None:
        """Dense Convolutional Block.

        References:
            Huang, Gao, et al. "Densely connected convolutional networks."
            Proceedings of the IEEE conference on computer vision and
            pattern recognition. 2017.

        Args:
            num_features: number of internal channels used for the layer
            in_channels: number of the input channels.
            out_channels: number of the output channels.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            drop_first_norm_relu - omits the first norm/relu for the first layer
            kernel_size: size of the kernel for >1 convolutions (dependent on mode)
        """
        super().__init__()

        self.layers = nn.Sequential()
        conv_type: Callable = Conv[Conv.CONV, 2]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, 2]

        if not drop_first_norm_relu:
            self.layers.add_module("preact/bn", get_norm_layer(name=norm, spatial_dims=2, channels=in_channels))
            self.layers.add_module("preact/relu", get_act_layer(name=act))

        self.layers.add_module("conv1", conv_type(in_channels, num_features, kernel_size=1, padding=0, bias=False))
        self.layers.add_module("conv1/bn", get_norm_layer(name=norm, spatial_dims=2, channels=num_features))
        self.layers.add_module("conv1/relu", get_act_layer(name=act))

        if in_channels != 64 and drop_first_norm_relu:
            self.layers.add_module(
                "conv2", conv_type(num_features, num_features, kernel_size=kernel_size, stride=2, padding=2, bias=False)
            )
        else:
            self.layers.add_module(
                "conv2", conv_type(num_features, num_features, kernel_size=kernel_size, padding=1, bias=False)
            )

        self.layers.add_module("conv2/bn", get_norm_layer(name=norm, spatial_dims=2, channels=num_features))
        self.layers.add_module("conv2/relu", get_act_layer(name=act))
        self.layers.add_module("conv3", conv_type(num_features, out_channels, kernel_size=1, padding=0, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))


class _Transition(nn.Sequential):
    def __init__(
        self, in_channels: int, act: Union[str, tuple] = ("relu", {"inplace": True}), norm: Union[str, tuple] = "batch"
    ) -> None:
        """
        Args:
            in_channels: number of the input channel.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        self.add_module("bn", get_norm_layer(name=norm, spatial_dims=2, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        layers: int,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        freeze_dense_layer: bool = False,
        freeze_block: bool = False,
    ) -> None:
        """Residual block.

        References:
            He, Kaiming, et al. "Deep residual learning for image
            recognition." Proceedings of the IEEE conference on computer
            vision and pattern recognition. 2016.

        Args:
            layers: number of layers in the block.
            num_features: number of internal features used.
            in_channels: number of the input channel.
            out_channels: number of the output channel.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            freeze_dense_layer: whether to freeze all dense layers within the block.
            freeze_block: whether to freeze the whole block.

        """
        super().__init__()

        self.layers = nn.Sequential()
        conv_type: Callable = Conv[Conv.CONV, 2]

        if in_channels == 64:
            self.shortcut = conv_type(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = conv_type(in_channels, out_channels, kernel_size=1, stride=2, padding=1, bias=False)

        layer = _DenseLayer(
            num_features, in_channels, out_channels, dropout_prob, act=act, norm=norm, drop_first_norm_relu=True
        )
        self.layers.add_module("denselayer_0", layer)

        for i in range(1, layers):
            layer = _DenseLayer(num_features, out_channels, out_channels, dropout_prob, act=act, norm=norm)
            self.layers.add_module(f"denselayer_{i}", layer)

        self.bna_block = _Transition(out_channels, act=act, norm=norm)

        if freeze_dense_layer:
            self.layers.requires_grad_(False)
        if freeze_block:
            self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        sc = self.shortcut(x)

        if self.shortcut.stride == (2, 2):
            sc = sc[:, :, :-1, :-1]

        for layer in self.layers:
            x = layer.forward(x)
            if x.shape[-2:] != sc.shape[-2:]:
                x = x[:, :, :-1, :-1]

            x = x + sc
            sc = x

        x = self.bna_block(x)

        return x


class _DecoderBranch(nn.ModuleList):
    def __init__(
        self,
        decode_config: Sequence[int] = (8, 4),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
        out_channels: int = 2,
        kernel_size: int = 3,
        same_padding: bool = False,
    ) -> None:
        """
        Args:
            decode_config: number of layers for each block.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            dropout_prob: dropout rate after each dense layer.
            out_channels: number of the output channel.
            kernel_size: size of the kernel for >1 convolutions (dependent on mode)
            same_padding: whether to do padding for >1 convolutions to ensure
                the output size is the same as the input size.
        """
        super().__init__()
        conv_type: Callable = Conv[Conv.CONV, 2]

        # decode branches
        _in_channels = 1024
        _num_features = 128
        _out_channels = 32

        self.decoder_blocks = nn.Sequential()
        for i, num_layers in enumerate(decode_config):
            block = _DecoderBlock(
                layers=num_layers,
                num_features=_num_features,
                in_channels=_in_channels,
                out_channels=_out_channels,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
                kernel_size=kernel_size,
                same_padding=same_padding,
            )
            self.decoder_blocks.add_module(f"decoderblock{i + 1}", block)
            _in_channels = 512

        # output layers
        self.output_features = nn.Sequential()
        _i = len(decode_config)
        _pad_size = (kernel_size - 1) // 2
        _seq_block = nn.Sequential(
            OrderedDict(
                [("conva", conv_type(256, 64, kernel_size=kernel_size, stride=1, bias=False, padding=_pad_size))]
            )
        )

        self.output_features.add_module(f"decoderblock{_i + 1}", _seq_block)

        _seq_block = nn.Sequential(
            OrderedDict(
                [
                    ("bn", get_norm_layer(name=norm, spatial_dims=2, channels=64)),
                    ("relu", get_act_layer(name=act)),
                    ("conv", conv_type(64, out_channels, kernel_size=1, stride=1)),
                ]
            )
        )

        self.output_features.add_module(f"decoderblock{_i + 2}", _seq_block)

        self.upsample = UpSample(
            2, scale_factor=2, mode=UpsampleMode.NONTRAINABLE, interp_mode=InterpolateMode.BILINEAR, bias=False
        )

    def forward(self, xin: torch.Tensor, short_cuts: List[torch.Tensor]) -> torch.Tensor:

        block_number = len(short_cuts) - 1
        x = xin + short_cuts[block_number]

        for block in self.decoder_blocks:
            x = block(x)
            x = self.upsample(x)
            block_number -= 1
            trim = (short_cuts[block_number].shape[-1] - x.shape[-1]) // 2
            if trim > 0:
                x += short_cuts[block_number][:, :, trim:-trim, trim:-trim]

        for block in self.output_features:
            x = block(x)

        return x


@export("monai.networks.nets")
class HoVerNet(nn.Module):
    """HoVerNet model

    References:
      Graham, Simon et al. Hover-net: Simultaneous segmentation
      and classification of nuclei in multi-tissue histology images,
      Medical Image Analysis 2019

      https://github.com/vqdang/hover_net
      https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

    Args:
        mode: use original implementation (`HoVerNetMODE.ORIGINAL` or "original") or
          a faster implementation (`HoVerNetMODE.FAST` or "fast"). Defaults to `HoVerNetMODE.FAST`.
        in_channels: number of the input channel.
        np_out_channels: number of the output channel of the nucleus prediction branch.
        out_classes: number of the nuclear type classes.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        decoder_padding: whether to do padding on convolution layers in the decoders. In the conic branch
            of the referred repository, the architecture is changed to do padding on convolution layers in order to
            get the same output size as the input, and this changed version is used on CoNIC challenge.
            Please note that to get consistent output size, `HoVerNetMode.FAST` mode should be employed.
        dropout_prob: dropout rate after each dense layer.
        pretrained_url: if specifying, will loaded the pretrained weights downloaded from the url.
            There are two supported forms of weights:
            1. preact-resnet50 weights coming from the referred hover_net
            repository, each user is responsible for checking the content of model/datasets and the applicable licenses
            and determining if suitable for the intended use. please check the following link for more details:
            https://github.com/vqdang/hover_net#data-format
            2. standard resnet50 weights of torchvision. Please check the following link for more details:
            https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#ResNet50_Weights
        adapt_standard_resnet: if the pretrained weights of the encoder follow the original format (preact-resnet50), this
            value should be `False`. If using the pretrained weights that follow torchvision's standard resnet50 format,
            this value should be `True`.
        freeze_encoder: whether to freeze the encoder of the network.
    """

    Mode = HoVerNetMode
    Branch = HoVerNetBranch

    def __init__(
        self,
        mode: Union[HoVerNetMode, str] = HoVerNetMode.FAST,
        input_ch: int = 3,
        np_out_channels: int = 2,
        nr_types: int = 0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        decoder_padding: bool = False,
        dropout_prob: float = 0.0,
        pretrained_url: Optional[str] = None,
        adapt_standard_resnet: bool = True,
        freeze_encoder: bool = False,
    ) -> None:

        super().__init__()
        out_classes = nr_types
        in_channels = input_ch
        if isinstance(mode, str):
            mode = mode.upper()
        self.mode = look_up_option(mode, HoVerNetMode)

        if self.mode == "ORIGINAL" and decoder_padding is True:
            warnings.warn(
                "'decoder_padding=True' only works when mode is 'FAST', otherwise the output size may not equal to the input."
            )

        if out_classes > 128:
            raise ValueError("Number of nuclear types classes exceeds maximum (128)")
        elif out_classes == 1:
            raise ValueError("Number of nuclear type classes should either be None or >1")

        if dropout_prob > 1 or dropout_prob < 0:
            raise ValueError("Dropout can only be in the range 0.0 to 1.0")

        # number of filters in the first convolution layer.
        _init_features: int = 64
        # number of layers in each pooling block.
        _block_config: Sequence[int] = (3, 4, 6, 3)

        if self.mode == HoVerNetMode.FAST:
            _ksize = 3
            _pad = 3
        else:
            _ksize = 5
            _pad = 0

        conv_type: Type[nn.Conv2d] = Conv[Conv.CONV, 2]

        self.conv0 = nn.Sequential(
            OrderedDict(
                [
                    ("conv", conv_type(in_channels, _init_features, kernel_size=7, stride=1, padding=_pad, bias=False)),
                    ("bn", get_norm_layer(name=norm, spatial_dims=2, channels=_init_features)),
                    ("relu", get_act_layer(name=act)),
                ]
            )
        )

        _in_channels = _init_features
        _out_channels = 256
        _num_features = _init_features

        self.res_blocks = nn.Sequential()

        for i, num_layers in enumerate(_block_config):
            freeze_dense_layer = False
            freeze_block = False
            if freeze_encoder:
                if i == 0:
                    freeze_dense_layer = True
                else:
                    freeze_block = True
            block = _ResidualBlock(
                layers=num_layers,
                num_features=_num_features,
                in_channels=_in_channels,
                out_channels=_out_channels,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
                freeze_dense_layer=freeze_dense_layer,
                freeze_block=freeze_block,
            )
            self.res_blocks.add_module(f"d{i}", block)

            _in_channels = _out_channels
            _out_channels *= 2
            _num_features *= 2

        # bottleneck convolution
        self.bottleneck = nn.Sequential()
        self.bottleneck.add_module(
            "conv_bottleneck", conv_type(_in_channels, _num_features, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.upsample = UpSample(
            2, scale_factor=2, mode=UpsampleMode.NONTRAINABLE, interp_mode=InterpolateMode.BILINEAR, bias=False
        )

        # decode branches
        self.nucleus_prediction = _DecoderBranch(
            kernel_size=_ksize, same_padding=decoder_padding, out_channels=np_out_channels
        )
        self.horizontal_vertical = _DecoderBranch(kernel_size=_ksize, same_padding=decoder_padding)
        self.type_prediction: Optional[_DecoderBranch] = (
            _DecoderBranch(out_channels=out_classes, kernel_size=_ksize, same_padding=decoder_padding)
            if out_classes > 0
            else None
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)

        if pretrained_url is not None:
            if adapt_standard_resnet:
                weights = _remap_standard_resnet_model(pretrained_url)
            else:
                weights = _remap_preact_resnet_model(pretrained_url)
            _load_pretrained_encoder(self, weights)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        if self.mode == HoVerNetMode.ORIGINAL.value:
            if x.shape[-1] != 270 or x.shape[-2] != 270:
                raise ValueError("Input size should be 270 x 270 when using HoVerNetMode.ORIGINAL")
        else:
            if x.shape[-1] != 256 or x.shape[-2] != 256:
                raise ValueError("Input size should be 256 x 256 when using HoVerNetMode.FAST")

        x = x / 255.0
        x = self.conv0(x)
        short_cuts = []

        for i, block in enumerate(self.res_blocks):
            x = block.forward(x)

            if i <= 2:
                short_cuts.append(x)

        x = self.bottleneck(x)
        x = self.upsample(x)

        output = {
            HoVerNetBranch.NP.value: self.nucleus_prediction(x, short_cuts),
            HoVerNetBranch.HV.value: self.horizontal_vertical(x, short_cuts),
        }
        if self.type_prediction is not None:
            output = {
                HoVerNetBranch.NC.value: self.type_prediction(x, short_cuts),
                HoVerNetBranch.NP.value: self.nucleus_prediction(x, short_cuts),
                HoVerNetBranch.HV.value: self.horizontal_vertical(x, short_cuts),
            }

        return output


def _load_pretrained_encoder(model: nn.Module, state_dict: Union[OrderedDict, Dict]):

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


def _remap_preact_resnet_model(model_url: str):

    pattern_conv0 = re.compile(r"^(conv0\.\/)(.+)$")
    pattern_block = re.compile(r"^(d\d+)\.(.+)$")
    pattern_layer = re.compile(r"^(.+\.d\d+)\.units\.(\d+)(.+)$")
    pattern_bna = re.compile(r"^(.+\.d\d+)\.blk_bna\.(.+)")
    # download the pretrained weights into torch hub's default dir
    weights_dir = os.path.join(torch.hub.get_dir(), "preact-resnet50.pth")
    download_url(model_url, fuzzy=True, filepath=weights_dir, progress=False)
    state_dict = torch.load(weights_dir, map_location=None)["desc"]
    for key in list(state_dict.keys()):
        new_key = None
        if pattern_conv0.match(key):
            new_key = re.sub(pattern_conv0, r"conv0.conv\2", key)
        elif pattern_block.match(key):
            new_key = re.sub(pattern_block, r"res_blocks.\1.\2", key)
            if pattern_layer.match(new_key):
                new_key = re.sub(pattern_layer, r"\1.layers.denselayer_\2.layers\3", new_key)
            elif pattern_bna.match(new_key):
                new_key = re.sub(pattern_bna, r"\1.bna_block.\2", new_key)
        if new_key:
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if "upsample2x" in key:
            del state_dict[key]

    return state_dict


def _remap_standard_resnet_model(model_url: str):

    pattern_conv0 = re.compile(r"^conv1\.(.+)$")
    pattern_bn1 = re.compile(r"^bn1\.(.+)$")
    pattern_block = re.compile(r"^layer(\d+)\.(\d+)\.(.+)$")
    # bn3 to next denselayer's preact/bn
    pattern_block_bn3 = re.compile(r"^(res_blocks.d\d+\.layers\.denselayer_)(\d+)\.layers\.bn3\.(.+)$")
    # bn1, bn2 to conv1/bn, conv2/bn
    pattern_block_bn = re.compile(r"^(res_blocks.d\d+\.layers\.denselayer_\d+\.layers)\.bn(\d+)\.(.+)$")
    pattern_downsample0 = re.compile(r"^(res_blocks.d\d+).+\.downsample\.0\.(.+)")
    pattern_downsample1 = re.compile(r"^(res_blocks.d\d+).+\.downsample\.1\.(.+)")
    # download the pretrained weights into torch hub's default dir
    weights_dir = os.path.join(torch.hub.get_dir(), "resnet50.pth")
    download_url(model_url, fuzzy=True, filepath=weights_dir, progress=False)
    state_dict = torch.load(weights_dir, map_location=None)

    for key in list(state_dict.keys()):
        new_key = None
        if pattern_conv0.match(key):
            new_key = re.sub(pattern_conv0, r"conv0.conv.\1", key)
        elif pattern_bn1.match(key):
            new_key = re.sub(pattern_bn1, r"conv0.bn.\1", key)
        elif pattern_block.match(key):
            new_key = re.sub(
                pattern_block,
                lambda s: "res_blocks.d"
                + str(int(s.group(1)) - 1)
                + ".layers.denselayer_"
                + s.group(2)
                + ".layers."
                + s.group(3),
                key,
            )
            if pattern_block_bn3.match(new_key):
                new_key = re.sub(
                    pattern_block_bn3,
                    lambda s: s.group(1) + str(int(s.group(2)) + 1) + ".layers.preact/bn." + s.group(3),
                    new_key,
                )
            elif pattern_block_bn.match(new_key):
                new_key = re.sub(pattern_block_bn, r"\1.conv\2/bn.\3", new_key)
            elif pattern_downsample0.match(new_key):
                new_key = re.sub(pattern_downsample0, r"\1.shortcut.\2", new_key)
            elif pattern_downsample1.match(new_key):
                new_key = re.sub(pattern_downsample1, r"\1.bna_block.bn.\2", new_key)
        if new_key:
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    return state_dict


####
class HoVerNet_raw(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        #     self.decoder = nn.ModuleDict(
        #         OrderedDict(
        #             [
        #                 ("nucleus_prediction", create_decoder_branch(ksize=ksize,out_ch=2)),
        #                 ("horizontal_vertical", create_decoder_branch(ksize=ksize,out_ch=2)),
        #             ]
        #         )
        #     )
        # else:
        #     self.decoder = nn.ModuleDict(
        #         OrderedDict(
        #             [
        #                 ("type_prediction", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
        #                 ("nucleus_prediction", create_decoder_branch(ksize=ksize, out_ch=2)),
        #                 ("horizontal_vertical", create_decoder_branch(ksize=ksize, out_ch=2)),
        #             ]
        #         )
        #     )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, model_type='raw', **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    if model_type == 'raw':
        return HoVerNet_raw(mode=mode, **kwargs)
    else:
        return HoVerNet(mode=mode, **kwargs)