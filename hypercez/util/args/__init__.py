import yaml

from .hparams import Hparams


with open("./hypercez/util/args/lang_conf.yaml", "r") as lang_conf_file:
    mcts_lang = yaml.safe_load(lang_conf_file)["mcts_lang"]["lang"]
