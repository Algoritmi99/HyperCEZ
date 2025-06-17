from enum import IntEnum

from torch import nn

from hypercez import Hparams


class HNet_Type(IntEnum):
    CHUNKED = 0
    UnChunked = 1


def build_hnet(hparams: Hparams, model: nn.Module, hnet_type: HNet_Type = HNet_Type.UnChunked) -> nn.Module:
    if hnet_type == HNet_Type.CHUNKED:
        pass
