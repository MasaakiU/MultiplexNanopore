# -*- coding: utf-8 -*-

from typing import List
from pathlib import Path
from itertools import chain

from ..modules import my_classes as mc
from . import pre_survey_core as psc

__all__ = ["pre_survey", "default_pre_survey_param_dict", "pre_survey_separate_paths_input"]

default_pre_survey_param_dict = {
    'gap_open_penalty':     3, 
    'gap_extend_penalty':   1, 
    'match_score':          1, 
    'mismatch_score':      -2, 
    'distance_threshold':   5, 
    'number_of_groups':     1, 
    'topology_of_dna':      0,       # 0: circular, 1: linear
    'n_cpu':                2, 
    'export_image_results': 1,  # 0; skip export of svg figure files, 1: export svg figure files
}

def pre_survey(plasmid_map_dir_path: str, save_dir_base: str, ref_seq_aliases: List[str]=None, **param_dict: dict):
    plasmid_map_paths = [path for path in Path(plasmid_map_dir_path).glob("*.*") if path.suffix in mc.MyRefSeq.allowed_plasmid_map_extensions]
    return pre_survey_separate_paths_input(plasmid_map_paths, save_dir_base, ref_seq_aliases, **param_dict)

def pre_survey_separate_paths_input(plasmid_map_paths:List[Path], save_dir_base: str, ref_seq_aliases: List[str]=None, **param_dict:dict):
    for k in param_dict.keys():
        if k not in default_pre_survey_param_dict.keys():
            raise Exception(f"unknown key in `param_dict`: {k}\nallowed keys are: {', '.join(default_pre_survey_param_dict.keys())}")
    param_dict = {key: param_dict.get(key, val) for key, val in default_pre_survey_param_dict.items()}
    mc.assert_param_dict(param_dict)
    if len(plasmid_map_paths) == 0:
        raise Exception(f"Error: No plasmid map file was detected!")
    if param_dict["number_of_groups"] > len(plasmid_map_paths):
        raise Exception(f"Error: `number_of_groups` cannot be greater than the number of plasmid maps ({len(plasmid_map_paths)})!")
    # 1. Prepare objects
    ref_seq_list = [mc.MyRefSeq(plasmid_map_path) for plasmid_map_path in plasmid_map_paths]
    save_dir = mc.new_dir_path_wo_overlap(Path(save_dir_base) / Path(psc.RecommendedGrouping.file_name).stem, spacing="_")
    save_dir.mkdir()
    # 2. execute
    recommended_grouping = psc.execute_grouping(ref_seq_list, param_dict, save_dir, ref_seq_aliases)
    # 3. save results
    psc.export_results(recommended_grouping, save_dir, export_image_results=param_dict["export_image_results"])
    # 4. display recommended_grouping in the std output
    print(f"\n{recommended_grouping.recommended_grouping_txt}")
    return save_dir

