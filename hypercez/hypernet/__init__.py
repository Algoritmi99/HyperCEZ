from enum import IntEnum

from torch import nn

from hypercez import Hparams
from hypercez.hypernet.hypercl import ChunkedHyperNetworkHandler, HyperNetwork
from hypercez.util.format import str_to_act


class HNet_Type(IntEnum):
    CHUNKED = 0
    UnChunked = 1


def build_hnet(hparams: Hparams, model: nn.Module, hnet_type: HNet_Type = HNet_Type.UnChunked) -> nn.Module:
    assert hasattr(hparams, "hnet_arch")
    assert hasattr(hparams, "hnet_act")
    assert hasattr(hparams, "emb_size")

    param_shape_lst = [list(i.shape) for i in model.parameters()]
    if hnet_type == HNet_Type.CHUNKED:
        assert hasattr(hparams, "chunk_dim")
        assert hasattr(hparams, "cemb_size")
        hnet = ChunkedHyperNetworkHandler(
            param_shape_lst,
            chunk_dim=hparams.chunk_dim,
            layers=hparams.hnet_arch,
            activation_fn=str_to_act(hparams.hnet_act),
            te_dim=hparams.emb_size,
            ce_dim=hparams.cemb_size,
        )
    else:
        hnet = HyperNetwork(
            param_shape_lst,
            layers=hparams.hnet_arch,
            te_dim=hparams.emb_size,
            activation_fn=str_to_act(hparams.hnet_act)
        )

