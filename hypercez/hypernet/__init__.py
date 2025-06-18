from enum import IntEnum

from torch import nn

from hypercez import Hparams
from hypercez.hypernet.hypercl import ChunkedHyperNetworkHandler, HyperNetwork, MainNetInterface
from hypercez.util.format import str_to_act


class HNet_Type(IntEnum):
    CHUNKED = 0
    UnCHUNKED = 1


def build_hnet(hparams: Hparams, model: nn.Module, hnet_type: HNet_Type = HNet_Type.UnCHUNKED) -> nn.Module:
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

    setattr(
        hparams,
        "num_weights_class_hyper_net",
        sum(p.numel() for p in hnet.parameters() if p.requires_grad)
    )
    setattr(
        hparams,
        "num_weights_class_net",
        MainNetInterface.shapes_to_num_weights(param_shape_lst)
    )
    setattr(
        hparams,
        "compression_ratio_class",
        hparams.num_weights_class_hyper_net / hparams.num_weights_class_net
    )

    for W in list(hnet.parameters()):
        if W.ndimension() == 1: # Bias vector.
            nn.init.constant_(W, 0)
        elif hparams.hnet_init == "normal":
            nn.init.normal_(W, mean=0, std=hparams.std_normal_init)
        elif hparams.hnet_init == "xavier":
            nn.init.xavier_uniform_(W)

    if hasattr(hnet, 'chunk_embeddings'):
        for emb in hnet.chunk_embeddings:
            nn.init.normal_(emb, mean=0, std=hparams.std_normal_cemb)

    if hparams.use_hyperfan_init:
        if hparams.model.startswith("hnet"):
            hnet.apply_hyperfan_init(temb_var=hparams.std_normal_temb**2)

    return hnet
