import torch
import gc
import utils as util
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import unet_weight_init as weight_init


def conv1x1(in_channels, out_channels):
    """Two dimentional convoloution layer with kernel size of one
    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels

    Returns
    -------
    torch.nn.Conv2d: 
        two dimentional convoloutional layer with kernel size of one 
        and specified input and output channel
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


def conv3x3(in_channel, out_channels):
    """Two dimentional convoloution layer with kernel size of three
    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels

    Returns
    -------
    torch.nn.Conv2d: 
        two dimentional convoloutional layer with kernel size of one 
        and specified input and output channel
    """
    return nn.Conv2d(in_channel,
                     out_channels,
                     kernel_size=3,
                     padding=1,
                     stride=1)


def upconv2x2(in_channels, out_channels, mode):
    """Two dimentional transposed convoloution layer with kernel size of two
    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    mode : str
        upsampling mode: "nearest-neighbour" or "transpose".

    Raises
    ------
    Exception
        if unkown mode is set for transposed convoloutional layer.

    Returns
    -------
    torch.nn.Conv2d: 
        two dimentional transposed convoloutional layer with kernel size of two
    """
    upconv = None
    if mode == 'nearest-neighbour':
        upconv = nn.Sequential(
            nn.Upsample(mode='nearest', scale_factor=2),
            conv1x1(in_channels, out_channels))
    elif mode == 'transpose':
        upconv = nn.ConvTranspose2d(in_channels,
                                    out_channels,
                                    kernel_size=2,
                                    padding=0,
                                    stride=2)
    else:
        raise Exception('Unknown mode : {}'.format(mode))
    return upconv


class ContractingUnit(nn.Module):
    """
    Contracting Unit
    ...
    Attributes
    ----------
    in_channels : int
        num of input channels 
    out_channel : int
        num of output channels
    block : torch.nn.Sequential
        contracting unit pytorch block
    pool : torch.nn.MaxPool2d or None
        pooling layer of end of contracting unit

    Methods
    -------
    weight_init() : 
        initialize weights
    forward(x) :
        forward path of module
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        """
        Parameters
        ----------
        in_channels : int
            num of input channels 
        out_channels : int
            num of output channels
        pooling : boolean, optional
        """
        super(ContractingUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            conv3x3(self.in_channels, self.out_channels),
            nn.ReLU(),
            conv3x3(self.out_channels, self.out_channels),
            nn.ReLU()
        )
        self.pool = None
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def weight_init(self):
        """Initializing module weights"""
        weight_init(self.block)

    def forward(self, x):
        """Forward path"""
        before_pooling = self.block(x)
        util.del_tensors([x])
        after_pooling = None
        if self.pool is not None:
            after_pooling = self.pool(before_pooling)
        return before_pooling, after_pooling


class ExpandingUnit(nn.Module):
    """
    Expanding Unit
    ...
    Attributes
    ----------
    in_channels : int
        num of input channels 
    mid_channels : int

    out_channel : int
        num of output channels
    up_mode : str
        upsampling mode
    up_conv : upconv2x2
        upsampling layer
    block : ContractingUnit
        a contracting unit (ContractingUnit) without pooling layer

    Methods
    -------
    weight_init() : 
        initialize weights
    forward(x) :
        forward path of module
    """

    def __init__(self, in_channels, mid_channels, out_channels, up_mode='nearest-neighbour'):
        """
        Parameters
        ----------
        in_channels : int 
            num of input channels
        mid_channels : int

        out_channels : int
            num of output channels
        up_mode : str, optional
            up-sampling mode
        """
        super(ExpandingUnit, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.up_mode = up_mode

        self.up_conv = upconv2x2(
            self.in_channels, self.out_channels, self.up_mode)
        self.block = ContractingUnit(
            self.in_channels, self.out_channels, pooling=False)

    def weight_init(self):
        """Initializing module weights"""
        weight_init(self.block)
        weight_init(self.up_conv)

    def forward(self, from_contracting, from_expanding):
        """Forward path"""
        out2 = self.up_conv(from_expanding)
        down_sample_value_2 = int(out2.size()[2] - from_contracting.size()[2])
        down_sample_value_3 = int(out2.size()[3] - from_contracting.size()[3])
        padding = 2 * [down_sample_value_2//2, down_sample_value_3//2]
        out1 = F.pad(from_contracting, padding)

        del from_contracting, from_expanding
        gc.collect()
        torch.cuda.empty_cache()

        x = torch.cat([out1, out2], 1)

        del out1, out2
        gc.collect()
        torch.cuda.empty_cache()

        result, _ = self.block(x)

        return result
