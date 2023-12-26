# -*- coding: utf-8 -*-

import copy
import argparse
from pathlib import Path
from ..modules import msa


def show_consensus(consensus_alignment_path: str, **param_dict: dict):
    my_msa = msa.MyMSA()
    my_msa.load_consensus_alignment(consensus_alignment_path)
    my_msa.print_alignment(param_dict)


if __name__ == "__main__":
    """
    python -m savemoney.show_consensus
    """

    param_dict = copy.deepcopy(msa.MyMSA.default_print_options)

    # パーサーの設定
    parser = argparse.ArgumentParser()
    parser.add_argument("consensus_alignment_path", help="path to consensus_alignment (*.ca) file", type=str)
    for key, val in param_dict.items():
        parser.add_argument(f"--{key}", help=f"{key}, optional, default_value = {val}", type=int, default=val)

    # 取得した引数を適用
    args = parser.parse_args()
    args_dict =vars(args)
    for key in param_dict.keys():
        param_dict[key] = args_dict[key]

    show_consensus(args.consensus_alignment_path, **param_dict)

