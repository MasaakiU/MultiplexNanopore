# -*- coding: utf-8 -*-

import copy
import argparse

from . import *
from ..modules import my_classes as mc

if __name__ == "__main__":
    """
    python -m savemoney.pre_survey path_to_plasmid_map_dir save_dir_base
    """

    param_dict = copy.deepcopy(default_pre_survey_param_dict)

    # パーサーの設定
    parser = argparse.ArgumentParser()
    parser.add_argument("plasmid_map_dir_paths", help="path to plasmid map_directory", type=str)
    parser.add_argument("save_dir_base", help="save directory path", type=str)
    for key, val in param_dict.items():
        parser.add_argument(f"-{mc.key2argkey(key)}", help=f"{key}, optional, default_value = {val}", type=int, default=val)

    # 取得した引数を適用
    args = parser.parse_args()
    args_dict =vars(args)
    for key in param_dict.keys():
        param_dict[key] = args_dict[mc.key2argkey(key)]

    pre_survey(args.plasmid_map_dir_paths, args.save_dir_base, **param_dict)

