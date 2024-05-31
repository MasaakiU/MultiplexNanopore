# -*- coding: utf-8 -*-

import io
import gc
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List
from collections import defaultdict
from multiprocessing import Pool

# my modules
from ..modules import msa
from ..modules import my_classes as mc
from ..modules import ref_query_alignment as rqa

__all__ = [
    "execute_alignment", 
    "normalize_scores_and_apply_threshold", 
    "execute_msa", 
]

#############
# ALIGNMENT #
#############
def execute_alignment(ref_seq_list:mc.MyRefSeq, my_fastq:mc.MyFastQ, param_dict, save_dir:Path=None):
    # load if there is intermediate data
    skip = False
    if save_dir is not None:
        intermediate_results_save_path = save_dir.parent / f"{my_fastq.combined_name_stem}.intermediate_results.ir"
        if intermediate_results_save_path.exists():
            intermediate_results = IntermediateResults()
            intermediate_results.load(intermediate_results_save_path)
            ref_seq_list_tmp = intermediate_results.assert_identity(ref_seq_list, my_fastq, param_dict)
            if ref_seq_list_tmp is not None:
                ref_seq_list = ref_seq_list_tmp
                result_dict = intermediate_results.result_dict
                print("alignment: SKIPPED\n")
                skip = True
    intermediate_results_save_path = save_dir / f"{my_fastq.combined_name_stem}.intermediate_results.ir"
    if not skip:
        ###########
        # Execute #
        ###########
        print("executing alignment...")
        result_dict = execute_alignment_core(ref_seq_list, my_fastq, param_dict)
        print("alignment: DONE\n")
        if save_dir is not None:
            if not intermediate_results_save_path.parent.exists():
                intermediate_results_save_path = Path.home() / intermediate_results_save_path.name
            intermediate_results = IntermediateResults(ref_seq_list, my_fastq, param_dict, result_dict)
            intermediate_results.path = intermediate_results_save_path
    intermediate_results.save(intermediate_results_save_path)
    return result_dict, ref_seq_list    # orders might have been changed during "assert_identity" step.

def execute_alignment_core(ref_seq_list, my_fastq, param_dict):
    my_optimized_aligner_list = [rqa.MyOptimizedAligner(ref_seq, param_dict) for ref_seq in ref_seq_list]
    input_iter = [(query_seq, my_optimized_aligner_list) for query_seq, q_scores in my_fastq.values()]

    with Pool(processes=param_dict["n_cpu"]) as p:
        result_list_list = list(tqdm(p.imap(execute_alignment_core_loop_wrapper, input_iter), ncols=100, mininterval=0.1, leave=True, bar_format='{l_bar}{bar}{r_bar}', total=len(my_fastq)))
    gc.collect()
    return {
        query_id:result_list for query_id, result_list in zip(my_fastq.keys(), result_list_list)
    }
def execute_alignment_core_loop_wrapper(args):
    return execute_alignment_core_loop(*args)
def execute_alignment_core_loop(query_seq, my_optimized_aligner_list: List[rqa.MyOptimizedAligner]):
    # calc scores for each ref_seq
    query_seq = mc.MySeq(query_seq)
    query_seq_rc = query_seq.reverse_complement()
    result_list = []
    for my_optimized_aligner in my_optimized_aligner_list:
        conserved_regions = my_optimized_aligner.calc_circular_conserved_region(query_seq, omit_too_long=True)
        conserved_regions_rc = my_optimized_aligner.calc_circular_conserved_region(query_seq_rc, omit_too_long=True)
        if conserved_regions is not None:
            my_result = my_optimized_aligner.execute_circular_alignment_using_conserved_regions(query_seq, conserved_regions)
        else:
            my_result = rqa.MyResult()
        if conserved_regions_rc is not None:
            my_result_rc = my_optimized_aligner.execute_circular_alignment_using_conserved_regions(query_seq_rc, conserved_regions_rc)
        else:
            my_result_rc = rqa.MyResult()
        # レジスター
        result_list.extend([my_result, my_result_rc])
        # gc.collect()
    return result_list

class IntermediateResults(mc.MyTextFormat, mc.MyHeader):
    intermediate_results_version = "ir_0.2.0"
    def __init__(self, ref_seq_list=None, my_fastq=None, param_dict=None, result_dict=None) -> None:
        super().__init__()
        self.header += f"\nintermediate_results_version: {self.intermediate_results_version}"
        self.path = None
        self.keys = [
            ("header", "str"), 
            ("datetime", "str"), 
            ("ref_seq_names", "list"), 
            ("ref_seq_hash_list", "list"), 
            ("my_fastq_names", "list"), 
            ("my_fastq_hash", "str"), 
            ("param_dict", "dict"), 
            ("query_id_list", "list")
        ]
        self.non_default_keys_start_idx = 8
        self.param_dict_keys_matter = ['gap_open_penalty', 'gap_extend_penalty', 'match_score', 'mismatch_score']
        self.query_id_list = []
        if result_dict is not None:
            # my aligner related info
            self.ref_seq_names = [ref_seq.path.name for ref_seq in ref_seq_list]
            self.ref_seq_hash_list = [ref_seq.my_hash for ref_seq in ref_seq_list]
            self.my_fastq_names = [fastq_path.name for fastq_path in my_fastq.path]
            self.my_fastq_hash = my_fastq.my_hash
            self.param_dict = param_dict
            # result_dict related info
            assert len(result_dict) == len(my_fastq)
            assert result_dict.keys() == my_fastq.keys()
            idx = 0
            for query_id, result_list in result_dict.items():
                self.query_id_list.append(query_id)
                for my_result in result_list:
                    result_key = f"result{idx}"
                    setattr(self, result_key, my_result.to_dict())
                    self.keys.append((result_key, "dict"))
                    idx += 1
            assert len(my_fastq) * len(ref_seq_list) * 2 == idx # リバコン(rc) もあるので二倍で assertion
    def assert_identity(self, ref_seq_list, my_fastq, param_dict):
        ref_seq_names = [ref_seq.path.name for ref_seq in ref_seq_list]
        ref_seq_hash_list = [ref_seq.my_hash for ref_seq in ref_seq_list]
        my_fastq_names = [fastq_path.name for fastq_path in my_fastq.path]
        my_fastq_hash = my_fastq.my_hash
        is_param_dict_same = all(param_dict[k] == self.param_dict[k] for k in self.param_dict_keys_matter)
        if (self.ref_seq_names == ref_seq_names) and\
            (self.ref_seq_hash_list == ref_seq_hash_list) and\
            (self.my_fastq_names == my_fastq_names) and\
            (self.my_fastq_hash == my_fastq_hash) and is_param_dict_same:
            return ref_seq_list
        elif (set(self.ref_seq_names) == set(ref_seq_names)) and\
            (set(self.ref_seq_hash_list) == set(ref_seq_hash_list)) and\
            (self.my_fastq_names == my_fastq_names) and\
            (self.my_fastq_hash == my_fastq_hash) and is_param_dict_same:
            # intermediat_resultsに応じて順番を並べ直す
            return [ref_seq_list[ref_seq_hash_list.index(ref_seq_hash)] for ref_seq_hash in self.ref_seq_hash_list]
        else:
            return None
    def save(self, save_path):
        super().save(save_path, zip=True)
    def load(self, load_path):
        self.path = load_path
        self.keys = super().load(load_path, zip=True)
        for k, v in self.param_dict.items():
            try:
                self.param_dict[k] = int(v)
            except:
                self.param_dict[k] = float(v)
    @property
    def result_dict(self):
        result_dict = {}
        result_list = []
        ref_seq_idx = 0
        ref_seq_idx_max = len(self.ref_seq_names) * 2
        query_idx = 0
        for key, data_type in self.keys[self.non_default_keys_start_idx:]:
            assert data_type == "dict"
            my_result = rqa.MyResult()
            my_result.apply_dict_params(getattr(self, key))
            result_list.append(my_result)
            ref_seq_idx += 1
            if ref_seq_idx == ref_seq_idx_max:
                query_id = self.query_id_list[query_idx]
                result_dict[query_id] = result_list
                ref_seq_idx = 0
                query_idx += 1
                result_list = []
        assert ref_seq_idx == 0
        assert query_idx == len(self.query_id_list)
        return result_dict

##############
# ASSIGNMENT #
##############
def normalize_scores_and_apply_threshold(ref_seq_list, my_fastq, result_dict, param_dict):
    print("normalizing scores...")
    query_assignment = msa.QueryAssignment(ref_seq_list, my_fastq, result_dict)
    print("normalization: DONE\n")
    # set threshold for assignment
    query_assignment.set_assignment(param_dict["score_threshold"])
    return query_assignment

def draw_and_save_query_assignment(query_assignment:msa.QueryAssignment, save_dir, display_plot=False):
    print("drawing and saving figures...")
    query_assignment.save_scores(save_dir)
    query_assignment.draw_distributions(save_dir, display_plot=False)
    query_assignment.draw_alignment_score_scatters(save_dir, display_plot=False)
    query_assignment.draw_alignment_score_scatters_rotated(save_dir, display_plot=False)
    print("drawing and saving:DONE\n")

#################
# MSA/CONSENSUS #
#################
def execute_msa(result_dict, query_assignment:msa.QueryAssignment, param_dict):
    print("executing MSA...")
    my_msa_list = []
    msa.MyMSA.set_sbq_pdf(query_assignment.my_fastq)
    for ref_seq, my_fastq_subset, result_list in query_assignment.iter_assignment_info(result_dict):
        print(f"processing {ref_seq.path.name}...")
        my_msa_aligner = msa.MyMSAligner(ref_seq, my_fastq_subset, result_list)
        my_msa_list.append(my_msa_aligner.execute(param_dict))
    print("MSA: DONE\n")
    return my_msa_list

#############
# CONSENSUS #
#############
def export_results(my_msa_list: List[msa.MyMSA], save_dir):
    print("exporting results...")
    for my_msa in my_msa_list:
        my_msa.export_consensus_fastq(save_dir)
        my_msa.export_gif(save_dir)
        my_msa.export_consensus_alignment(save_dir)
    print("export: DONE")

def export_log(ref_seq_list:list, my_fastq:mc.MyFastQ, param_dict, query_assignment: msa.QueryAssignment, save_dir:Path):
    print("exporting log...")
    my_log = MyLog(ref_seq_list, my_fastq, param_dict, query_assignment)
    my_log.save(save_path = save_dir / f"{my_fastq.combined_name_stem}.log")
    print("export: DONE")

class MyLog(mc.MyTextFormat, mc.MyHeader):
    def __init__(self, ref_seq_list: list=None, my_fastq: mc.MyFastQ=None, param_dict: dict=None, query_assignment=None) -> None:
        super().__init__()
        if ref_seq_list is not None:
            self.input_reference_files = [refseq.path for refseq in ref_seq_list]
            self.input_fastq_files = [fastq_path for fastq_path in my_fastq.path]
            self.input_reference_hash_list = [refseq.my_hash for refseq in ref_seq_list]
            self.input_fastq_hash = my_fastq.my_hash
            self.alignment_params = param_dict
            self.score_matrix = self.get_score_matrix()
            self.assignment_summary = query_assignment.get_assignment_summary()
            self.error_matrix_with_prior, self.error_matrix_without_prior = self.get_error_matrices()
            self.class_attributes = self.get_class_attributes()
        # keys required for MyTextFormat
        self.keys = [
            ("header", "str"), 
            ("datetime", "str"), 
            ("input_reference_files", "listPath"), 
            ("input_fastq_files", "listPath"), 
            ("input_reference_hash_list", "list"), 
            ("input_fastq_hash", "str"), 
            ("alignment_params", "dict"), 
            ("score_matrix", "str"), 
            ("assignment_summary", "str"), 
            ("error_matrix_with_prior", "str"), 
            ("error_matrix_without_prior", "str"), 
            ("class_attributes", "dict"), 
        ]
    def get_score_matrix(self):
        my_custom_matrix = rqa.MyAlignerBase(self.alignment_params).my_custom_matrix
        return self.matrix2string(my_custom_matrix.matrix, digit=3, round=True)
    def get_error_matrices(self):
        P_N_dict_dict_with_prior, P_N_dict_dict_without_prior = msa.MyMSA.P_N_dict_dict_2_matrix(self.alignment_params)
        return (
            self.matrix2string(P_N_dict_dict_with_prior, bases=msa.MyMSA.bases, digit=None, round=False), 
            self.matrix2string(P_N_dict_dict_without_prior, bases=msa.MyMSA.bases, digit=None, round=False)
        )

    @staticmethod
    def get_class_attributes():
        class_attributes_dict = {}
        # alignment params
        class_attributes_dict.update(
            ALIGNMENT_default_repeat_max = rqa.MyOptimizedAligner.default_repeat_max, 
            ALIGNMENT_percentile_factor = rqa.MyOptimizedAligner.percentile_factor, 
            ALIGNMENT_omit_too_long = rqa.MyOptimizedAligner.omit_too_long, 
            ALIGNMENT_consecutive_true_threshold = rqa.WindowAnalysis.consecutive_true_threshold, 
        )
        # msa/consensus params
        class_attributes_dict.update(
            ALIGNMENT_algorithm_ver = rqa.MyAlignerBase.alignment_algorithm_version, 
            QUERY_assignment_version = msa.QueryAssignment.query_assignment_version, 
            MSALIGN_N_polish = msa.MyMSAligner.N_polish, 
            MSA_algorithm_version = msa.MyMSA.algorithm_version, 
            MSA_bases = msa.MyMSA.bases, 
            MSA_letter_code_dict = msa.MyMSA.letter_code_dict, 
            MSA_sbq_pdf_version = msa.MyMSA.sbq_pdf_version, 
        )
        # consensus file_format info
        class_attributes_dict.update(
            FORMAT_file_format_version = msa.MyMSA.file_format_version, 
            FORMAT_ref_seq_related_save_order = msa.MyMSA.ref_seq_related_save_order,     # [tuple, tuple, ...]
            FORMAT_query_seq_related_save_order = msa.MyMSA.query_seq_related_save_order,   # [tuple, tuple, ...]
            FORMAT_size_indicator_bytes = msa.MyByteStr.size_indicator_bytes, 
            FORMAT_size_format = msa.MyByteStr.size_format, 
            FORMAT_my_bit = msa.MyByteStr.MyBIT.my_bit, 
            FORMAT_struct_bit = msa.MyByteStr.MyBIT.struct_bit, 
            FORMAT_seq_offset = msa.MyByteStr.MyBIT.seq_offset, 
        )
        return class_attributes_dict
    @staticmethod
    def matrix2string(matrix, bases="ATCG", digit=3, round=True):
        bio = io.BytesIO()
        if round:
            np.savetxt(bio, matrix, fmt=f"%{digit}d")
        else:
            np.savetxt(bio, matrix, fmt=f"%.5e")
            digit = 11
        matrix_str = bio.getvalue().decode('latin1')
        bases = bases + "*"
        output = (
            " " * (digit + 1)
            + (" " * digit).join(b for b in bases)
        )
        for b, m in zip(bases, matrix_str.split("\n")):
            output += f"\n{b} {m}"
        return output



