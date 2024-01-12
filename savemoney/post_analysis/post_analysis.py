# -*- coding: utf-8 -*-

from typing import List
from pathlib import Path
from itertools import chain

# my modules
from ..modules import my_classes as mc
from . import post_analysis_core as pac

__all__ = ["post_analysis", "default_post_analysis_param_dict", "post_analysis_separate_paths_input"]

error_rate = 0.000001
default_post_analysis_param_dict = {
    'gap_open_penalty': 3, 
    'gap_extend_penalty': 1, 
    'match_score': 1, 
    'mismatch_score': -2, 
    'score_threshold': 0.3, 
    'error_rate': error_rate, 
    'del_mut_rate': error_rate / 4, # e.g. "A -> T, C, G, del"
    'ins_rate': 0.000001, 
    'window': 160,      # maximum detectable length of repetitive sequences: if region of 80 nt is repeated adjascently two times, put the value of 160.
}

def post_analysis(sequence_dir_path:str, save_dir_base: str, **param_dict:dict):
    plasmid_map_paths = [path for path in Path(sequence_dir_path).glob(f"*.*") if path.suffix in mc.MyRefSeq.allowed_plasmid_map_extensions]
    fastq_paths = Path(sequence_dir_path).glob(f"*.fastq")
    return post_analysis_separate_paths_input(plasmid_map_paths, fastq_paths, save_dir_base, **param_dict)

def post_analysis_separate_paths_input(plasmid_map_paths:List[Path], fastq_paths:List[Path], save_dir_base: str, **param_dict:dict):
    for k in param_dict.keys():
        if k not in default_post_analysis_param_dict.keys():
            raise Exception(f"unknown key in `param_dict`: {k}\nallowed keys are: {', '.join(default_post_analysis_param_dict.keys())}")
    param_dict = {key: param_dict.get(key, val) for key, val in default_post_analysis_param_dict.items()}
    # 1. Prepare objects
    ref_seq_list = [mc.MyRefSeq(plasmid_map_path) for plasmid_map_path in plasmid_map_paths]
    my_fastq = mc.MyFastQ.combine([mc.MyFastQ(fastq_path) for fastq_path in fastq_paths])
    save_dir = mc.new_dir_path_wo_overlap(Path(save_dir_base) / my_fastq.combined_name_stem, spacing="_")
    save_dir.mkdir()
    # 2. Execute alignment: load if any previous score_matrix if possible
    result_dict = pac.execute_alignment(ref_seq_list, my_fastq, param_dict, save_dir)
    # 3. normalize alignment score and set threshold for assignment
    query_assignment = pac.normalize_scores_and_apply_threshold(ref_seq_list, my_fastq, result_dict, param_dict)
    pac.draw_and_save_query_assignment(query_assignment, save_dir, display_plot=False)
    # 4. MSA/consensus
    my_msa_list = pac.execute_msa(result_dict, query_assignment, param_dict)
    # 5. EXPORT
    pac.export_results(my_msa_list, save_dir)
    pac.export_log(ref_seq_list, my_fastq, param_dict, query_assignment, save_dir)
    return save_dir
