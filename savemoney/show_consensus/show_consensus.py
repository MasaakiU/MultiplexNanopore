# -*- coding: utf-8 -*-

from ..modules import msa
from ..post_analysis.post_analysis import default_post_analysis_param_dict

def show_consensus(consensus_alignment_path: str, **param_dict: dict):
    my_msa = msa.MyMSA(param_dict=default_post_analysis_param_dict)
    my_msa.load_consensus_alignment(consensus_alignment_path)
    my_msa.print_alignment(**param_dict)


