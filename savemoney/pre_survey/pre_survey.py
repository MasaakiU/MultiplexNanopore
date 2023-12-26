# -*- coding: utf-8 -*-

from pathlib import Path
from itertools import chain

from ..modules import my_classes as mc
from . import pre_survey_core as psc

__all__ = ["pre_survey", "default_pre_survey_param_dict"]

default_pre_survey_param_dict = {
    'gap_open_penalty':     3, 
    'gap_extend_penalty':   1, 
    'match_score':          1, 
    'mismatch_score':       -2, 
    'distance_threshold':   5, 
    'number_of_groups':     1, 
}

def pre_survey(plasmid_map_dir_path: str, save_dir_base: str, **param_dict: dict):
    for k in param_dict.keys():
        if k not in default_pre_survey_param_dict.keys():
            raise Exception(f"unknown key in `param_dict`: {k}\nallowed keys are: {', '.join(default_pre_survey_param_dict.keys())}")
    # 0. Prepare files
    plasmid_map_path = []
    for ext in mc.MyRefSeq.allowed_plasmid_map_extensions:
        plasmid_map_path = chain(plasmid_map_path, Path(plasmid_map_dir_path).glob(f"*{ext}"))
    save_dir = mc.new_dir_path_wo_overlap(Path(save_dir_base) / "recommended_grouping", spacing="_")
    save_dir.mkdir()
    # 1. Prepare objects
    ref_seq_list = [mc.MyRefSeq(plasmid_map_path) for plasmid_map_path in plasmid_map_path]
    param_dict = {key: param_dict.get(key, val) for key, val in default_pre_survey_param_dict.items()}
    # 2. execute
    recommended_grouping = psc.execute_grouping(ref_seq_list, param_dict, save_dir)
    # 3. save results
    psc.export_results(recommended_grouping, save_dir)
    # 4. display recommended_grouping in the std output
    print(f"\n{recommended_grouping.recommended_grouping_txt}")

