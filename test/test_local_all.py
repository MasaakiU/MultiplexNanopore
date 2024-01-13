# -*- coding: utf-8 -*-

"""
$python test/test_local_all.py
"""


import sys
from pathlib import Path
abs_path = Path(__file__).resolve()
sys.path.insert(0, abs_path.parents[1].as_posix())
import subprocess

import savemoney

if __name__ == "__main__":
    sequence_dir_path = abs_path.parents[1] / "resources/demo_data_1_subset/demo_data_1_subset_input/my_plasmid_maps_fa"
    # sequence_dir_path = abs_path.parents[1] / "resources/demo_data/my_plasmid_maps_dna"
    save_dir_base = abs_path.parents[1] / "resources/demo_data"

    consensus_alignment_path = abs_path.parents[1] / "resources/demo_data/Uematsu_n7x_1_MU-test1_subset/M32_pmNeonGreen-N1.fa.ca"

    ###############
    # WITH IMPORT #
    ###############
    param_dict = {  # optional params
        'distance_threshold':   5, 
        'number_of_groups':     2, 
    }
    savemoney.pre_survey(sequence_dir_path, save_dir_base, **param_dict)
    savemoney.post_analysis(sequence_dir_path, save_dir_base)
    savemoney.show_consensus(consensus_alignment_path)
    savemoney.ca2bam(consensus_alignment_path)

    #####################
    # WITH COMMAND LINE #
    #####################
    subprocess.call(f"python -m savemoney.pre_survey {sequence_dir_path} {save_dir_base} -dt {param_dict['distance_threshold']} -nog {param_dict['number_of_groups']}", shell=True)
    subprocess.call(f"python -m savemoney.post_analysis {sequence_dir_path} {save_dir_base}", shell=True)
    subprocess.call(f"python -m savemoney.show_consensus {consensus_alignment_path}", shell=True)
    subprocess.call(f"python -m savemoney.ca2bam {consensus_alignment_path}", shell=True)




