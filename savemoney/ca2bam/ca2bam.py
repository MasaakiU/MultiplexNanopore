# -*- coding: utf-8 -*-

from pathlib import Path

from ..modules import msa
from ..post_analysis.post_analysis import default_post_analysis_param_dict

def ca2bam(consensus_alignment_path: str):
    my_msa = msa.MyMSA(param_dict=default_post_analysis_param_dict)
    my_msa.load_consensus_alignment(consensus_alignment_path)
    my_msa.convert_to_bam(save_dir=Path(consensus_alignment_path).parent, ext_not_exported=[".sorted.sam", ".sorted.bam", ".sorted.bam.bai"])

