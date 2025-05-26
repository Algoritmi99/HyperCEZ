import json


with open('./hypercez/util/args/hnet_hparams.json') as f:
    hnet_hparams = json.load(f)

with open("./hypercez/util/args/env_hparams.json") as f:
    env_hparams = json.load(f)


class Hparams:
    def __init__(self, env: str, seed=None, save_folder='./runs/lqr'):
        self.h_dims = None
        self.seed = seed if seed is not None else 2020
        self.save_folder = save_folder if save_folder is not None else './runs/lqr'
        self.resume = False
        # Common train setting
        self.num_ds_worker = 0
        self.print_train_every = 1000

        # common RL setting
        self.env = env
        self.gt_dynamic = False
        self.gpuid = "cuda:0"

        env_param_key = ""
        for key in env_hparams.keys():
            if env.startswith(key) or env == key:
                env_param_key = key
                break

        if env_param_key == "":
            raise Exception("No env_hparams for env=%s" % env)

        for attr in env_hparams[env_param_key].keys():
            setattr(self, attr, env_hparams[env_param_key][attr])

    def add_hnet_hparams(self):
        # Hypernetwork
        setattr(self, 'hnet_arch', hnet_hparams["un-chunked"]["hnet_arch"][str(self.h_dims)])
        setattr(self, "hnet_act", "elu" if self.env == "door" or self.env == "pusher" else "relu")

        for attr in [
            "emb_size", "use_hyperfan_init", "hnet_init", "std_normal_init", "std_normal_temb",
            "lr_hyper", "grad_max_norm", "no_look_ahead", "plastic_prev_tembs", "backprop_dt",
            "use_sgd_change", "ewc_weight_importance", "n_fisher", "si_eps", "mlp_var_minmax"
        ]:
            setattr(self, attr, hnet_hparams["un-chunked"][attr])

        setattr(self, 'beta', 0.5 if self.env == "door_pose" or self.env == "pusher_slide" else 0.05)

    def add_chunked_hnet_hparams(self):
        setattr(self, "hnet_arch", hnet_hparams["chunked"]["hnet_arch"][str(self.h_dims)])
        setattr(self, "chunk_dim", hnet_hparams["chunked"]["chunk_dim"][str(self.h_dims)])
        setattr(self,"cemb_size", hnet_hparams["chunked"]["cemb_size"][str(self.h_dims)])
        setattr(self, "hnet_act", 'relu')

        for attr in [
            "emb_size", "use_hyperfan_init", "hnet_init", "std_normal_init", "std_normal_temb",
            "std_normal_cemb", "lr_hyper", "grad_max_norm", "beta", "no_look_ahead", "plastic_prev_tembs",
            "backprop_dt", "use_sgd_change", "ewc_weight_importance", "n_fisher"
        ]:
            setattr(self, attr, hnet_hparams["chunked"][attr])
