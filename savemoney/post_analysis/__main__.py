# -*- coding: utf-8 -*-

import copy
import argparse

from . import *
from ..modules import my_classes as mc

if __name__ == "__main__":
    """
    python -m savemoney.post_analysis path_to_sequence_data_dir save_dir_base
    """

    param_dict = copy.deepcopy(default_post_analysis_param_dict)

    # パーサーの設定
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir_paths", help="sequence_dir_paths", type=str)
    parser.add_argument("save_dir_base", help="save directory path", type=str)
    for key, val in param_dict.items():
        parser.add_argument(f"-{mc.key2argkey(key)}", help=f"{key}, optional, default_value = {val}", type=type(val), default=val)

    # 取得した引数を適用
    args = parser.parse_args()
    args_dict =vars(args)
    for key in param_dict.keys():
        param_dict[key] = args_dict[mc.key2argkey(key)]

    post_analysis(args.sequence_dir_paths, args.save_dir_base, **param_dict)

