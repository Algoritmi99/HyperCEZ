from pathlib import Path

import yaml

from .hparams import Hparams

_ARGS_DIR = Path(__file__).resolve().parent
_LANG_CONF_PATH = _ARGS_DIR / "lang_conf.yaml"

with _LANG_CONF_PATH.open("r", encoding="utf-8") as lang_conf_file:
    mcts_lang = yaml.safe_load(lang_conf_file)["mcts_lang"]["lang"]
