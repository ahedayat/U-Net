import utils as util
import gc
from .unet_parts import *
from .unet_utils import weight_init

class UNet(nn.Module):
    """
    UNet Implementation
    ...
    Attributes
    ----------
    num_classes : int
        num of classes
    input_channels : int
        num of input channels 
    contracting_convs : list of [ContractingUnit]s
        list of encoders
    expanding_convs : list of [ExpandingUnit]s
        list of decoders 
    final_conv : conv1x1
        final convolution layer

    Methods
    -------
    weight_init() : 
        initialize parameters
    forward(x) :
        forward path of model
    """

    def __init__(self, num_classes, input_channels=3, depth=5, second_layer_channels=64 ):
        """
        Prameters
        ---------
        num_classes : int
            num of output classes
        input_channels : int, optional
            num of input channels
        depth : int, optional 
            num of encoding and decoding levels
        second_layer_channels : int, optional
            num of channels of second layer
        """
        super(UNet,self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels

        contracting_out_channels = [ second_layer_channels * (2**i) for i in range(depth) ]
        contracting_in_channels = [self.input_channels] + contracting_out_channels[:-1]
        expanding_in_channels = contracting_out_channels[::-1]
        expanding_out_channels = expanding_in_channels[1:] + [self.num_classes]

        self.contracting_convs =[ 
            ContractingUnit(in_channel, out_channel, pooling = True if ix!=(depth-1) else False) 
            for ix,(in_channel, out_channel) in enumerate(zip(contracting_in_channels, contracting_out_channels))
            ]

        self.expanding_convs =[ 
            ExpandingUnit(in_channel, out_channel*2, out_channel) 
            for ix,(in_channel, out_channel) in enumerate(zip(expanding_in_channels[:-1], expanding_out_channels[:-1]))
            ]

        self.contracting_convs = nn.ModuleList(self.contracting_convs)
        self.expanding_convs = nn.ModuleList(self.expanding_convs)
        self.final_conv = conv1x1( expanding_in_channels[-1], expanding_out_channels[-1] )

    def weight_init(self):
        """Initialize wieght"""
        for ix, m in enumerate(self.modules()):
            weight_init(m)

    def forward(self,x):
        """Forward path"""
        contracting_outs = list()

        for ix, contracting_unit in enumerate(self.contracting_convs):
            before_pooling, after_pooling = contracting_unit(x)

            if after_pooling is not None:
                x = after_pooling
            else:
                x = before_pooling
            contracting_outs.append(before_pooling)
        contracting_outs.reverse()
        for ix, (contracting_unit_out, expanding_unit) in enumerate(zip( contracting_outs[1:], self.expanding_convs )):
            x = expanding_unit( contracting_unit_out, x )

        x = self.final_conv(x)

        return x