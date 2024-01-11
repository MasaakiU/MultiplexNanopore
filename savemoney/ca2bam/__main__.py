# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from . import *
from ..modules import msa
from ..post_analysis.post_analysis import default_post_analysis_param_dict

if __name__ == "__main__":
    """
    python -m savemoney.show_consensus
    """

    # パーサーの設定
    parser = argparse.ArgumentParser()
    parser.add_argument("consensus_alignment_path", help="path to consensus_alignment (*.ca) file", type=str)

    # 取得した引数を適用
    args = parser.parse_args()

    ca2bam(args.consensus_alignment_path)

