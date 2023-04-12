# -*- coding: utf-8 -*-

#@title # 1. Upload and select files

app_name = "SAVEMONEY"
version = "0.1.2"
description = "written by MU"

import sys, os
import io
import numpy as np
import pandas as pd
import re
import copy
import zipfile
import parasail
import gc
import textwrap
import matplotlib.pyplot as plt
import hashlib
from datetime import datetime
from pathlib import Path
from matplotlib.patches import Patch
from itertools import product
from collections import OrderedDict, namedtuple, defaultdict
from snapgene_reader import snapgene_file_to_dict, snapgene_file_to_seqrecord
from Bio.Seq import Seq
from numpy.core.memmap import uint8
from PIL import Image as PilImage
from . import my_classes as mc

class MyFastQ(OrderedDict):
    def __init__(self, path=None):
        super().__init__()
        self.path = path
        if self.path is not None: # for deep copy
            with open(self.path.as_posix(), "r") as f:
                fastq_txt = f.readlines()
            # check
            self.N_seq, mod = divmod(len(fastq_txt), 4)
            assert mod == 0
            # register
            for i in range(self.N_seq):
                seq_id = fastq_txt[4 * i].strip()
                seq = fastq_txt[4 * i + 1].strip()
                p = fastq_txt[4 * i + 2].strip()
                q_scores = [ord(q) - 33 for q in fastq_txt[4 * i + 3].strip()]
                assert p == "+"
                assert len(seq) == len(q_scores)
                self[seq_id] = [seq, q_scores]
        else:
            pass
    @property
    def combined_name_stem(self):
        if isinstance(self.path, list):
            return "_".join(p.stem for p in self.path)
        else:
            return self.path.stem
    def get_read_lengths(self):
        return np.array([len(v[0]) for v in self.values()])
    def get_q_scores(self):
        q_scores = []
        for v in self.values():
            q_scores.extend(v[1])
        return np.array(q_scores)
    def get_new_seq_id(self, k):
        if k not in self.keys():
            return k
        else:
            n = 1
            new_k = f"{k} {n}"
            while new_k in self.keys():
                n += 1
                new_k = f"{k} {n}"
            return new_k
    def append(self, fastq):
        for k, v in fastq.items():
            new_k = self.get_new_seq_id(k)
            self[new_k] = v
    def get_subset(self, keys):
        fastq_sub = MyFastQ()
        fastq_sub.path = self.path
        for k in keys:
            fastq_sub[k] = self[k]
        return fastq_sub
    @staticmethod
    def combine(fastq_list):
        assert len(fastq_list) > 1
        combined_fastq = copy.deepcopy(fastq_list[0])
        for fastq in fastq_list[1:]:
            combined_fastq.append(fastq)
        return combined_fastq
    @property
    def my_hash(self):
        return hashlib.sha256(self.to_string().encode("utf-8")).hexdigest()
    def to_string(self):
        txt = ""
        for seq_id, (seq, q_scores) in self.items():
            txt += f"{seq_id}\n{seq}\n+\n{''.join(map(lambda x: chr(x + 33), q_scores))}\n"
        return txt#.strip()
    def __getitem__(self, k):
        if not isinstance(k, slice):
            return OrderedDict.__getitem__(self, k)
        x = self.__class__()
        if k.start is None: start = 0
        else:               start = k.start
        if k.stop is None: stop = len(self) - k.stop
        else:              stop = k.stop
        assert (0 <= start <= stop)
        for idx, key in enumerate(self.keys()):
            if start <= idx < stop:
                x[key] = self[key]
        x.path = self.path
        return x

class MyRefSeq():
    def __init__(self, path: Path):
        self.path = path
        if self.path.suffix == ".dna":
            snapgene_dict = snapgene_file_to_dict(self.path.as_posix())
            # seqrecord = snapgene_file_to_seqrecord(self.path.as_posix())
            assert snapgene_dict["isDNA"]
            self.topology = snapgene_dict["dna"]["topology"]
            self.strandedness = snapgene_dict["dna"]["strandedness"]
            self.length = snapgene_dict["dna"]["length"]
            self.seq = snapgene_dict["seq"]
            if self.topology != "circular":
                print(f"WARNING: {self.path.name} is not circular!")
            assert self.strandedness == "double"
            assert self.length == len(self.seq)
        elif self.path.suffix in (".fasta", ".fa"):
            with open(self.path.as_posix(), 'r') as f:
                self.seq=''
                for line in f.readlines():
                    if line[0] != '>':
                        self.seq += line.strip()
            self.topology = "circular"
            self.strandedness = "double"
            self.length = len(self.seq)
        else:
            raise Exception(f"Unsupported type of sequence file: {self.path}")
    def reverse_complement(self):
        return str(Seq(self.seq).reverse_complement())
    @property
    def my_hash(self):
        return hashlib.sha256(self.seq.encode("utf-8")).hexdigest()

# Definition of main classes
class MyResult():
    def __init__(self, parasail_result=None) -> None:
        if parasail_result is not None:
            self.cigar = parasail_result.cigar.decode.decode("ascii")
            self.score = parasail_result.score
            self.beg_ref = parasail_result.cigar.beg_ref
            self.beg_query = parasail_result.cigar.beg_query
            self.end_ref = parasail_result.end_ref
            self.end_query = parasail_result.end_query
    def to_dict(self):
        keys = ["cigar", "score", "beg_ref", "beg_query", "end_ref", "end_query"]
        d = {}
        for key in keys:
            d[key] = getattr(self, key)
        return d
    def apply_dict_params(self, d):
        for type_convert_key in ['score', 'beg_ref', 'beg_query', 'end_ref', 'end_query']:
            setattr(self, type_convert_key, int(d[type_convert_key]))
        else:
            setattr(self, "cigar", d["cigar"])

class MyAligner():
    def __init__(self, refseq_list, combined_fastq, param_dict):
        # params
        self.param_dict = param_dict
        self.gap_open_penalty = param_dict["gap_open_penalty"]
        self.gap_extend_penalty = param_dict["gap_extend_penalty"]
        self.match_score = param_dict["match_score"]
        self.mismatch_score = param_dict["mismatch_score"]
        # others
        self.refseq_list = refseq_list
        self.combined_fastq = combined_fastq
        self.duplicated_refseq_seq_list = None
        self.is_refseq_seq_all_ATGC_list = None
        self.set_refseq_related_info()
    @property
    def my_custom_matrix(self):
        return parasail.matrix_create("ACGT", self.match_score, self.mismatch_score)
    def set_refseq_related_info(self):
        self.duplicated_refseq_seq_list = []
        self.is_refseq_seq_all_ATGC_list = []
        for refseq in self.refseq_list:
            self.duplicated_refseq_seq_list.append(refseq.seq + refseq.seq)
            is_all_ATCG = all([b.upper() in "ATCG" for b in refseq.seq])
            if not is_all_ATCG:
                print(f"\033[38;2;255;0;0mWARNING: Non-ATCG letter(s) were found in '{refseq.path.name}'.\nWhen calculating the alignment score, they are treated as 'mismatched', no matter what characters they are.\033[0m")
            self.is_refseq_seq_all_ATGC_list.append(is_all_ATCG)
    # refが環状プラスミドであるために、それを元に戻すのに使う（プラスミド上のどこがシーケンスの始まりと終わりなのか）を決めるのに使うカスタムのスコア
    def get_custom_cigar_score_dict(self):
        return {
            "=":self.match_score, 
            "X":self.mismatch_score, 
            "D":self.gap_open_penalty * -1, 
            "H":self.gap_open_penalty * -1, 
            "S":0, 
            "N":0, 
            "I":0
        }
    # calcualted based on gap_open_penalty, gap_extend_penalty, match_score, mismatch_score
    def clac_cigar_score(self, cigar_str):
        # なぜか result の cigar に、左端にたくさん D もしくは I が連なることがあるので、それを除く
        cigar_str_NL_list = re.findall('(\d+)(\D)', cigar_str)
        if cigar_str_NL_list[0][1] == "D":
            cigar_str_NL_list = cigar_str_NL_list[1:]
        elif cigar_str_NL_list[0][1] == "I":
            cigar_str_NL_list = cigar_str_NL_list[1:]
        score = 0
        for N, L in cigar_str_NL_list:
            N = int(N)
            if L == "=":
                score += self.match_score * N
            elif L == "X":
                score += self.mismatch_score * N
            elif L in "DI":
                score -= self.gap_open_penalty + self.gap_extend_penalty * (N - 1)
            else:
                raise Exception(f"unknown letter code: {L}")
        return score
    def align_all(self):
        fastq_len = len(self.combined_fastq)
        result_dict = OrderedDict()
        for query_idx, (seq_id, (query_seq, q_scores)) in enumerate(list(self.combined_fastq.items())):
            print(f"\rExecuting alignment: {query_idx + 1} out of {fastq_len} ({seq_id})", end="")
            # calc scores for each refseq
            result_list = []
            for duplicated_refseq_seq, is_refseq_seq_all_ATGC in zip(self.duplicated_refseq_seq_list, self.is_refseq_seq_all_ATGC_list):
                # なぜか result の cigar に、左端にたくさん D もしくは I が連なることがあるが、多分スコアはちゃんと計算されてる
                result = parasail.sw_trace(query_seq, duplicated_refseq_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
                result = MyResult(result)
                result_rc = parasail.sw_trace(str(Seq(query_seq).reverse_complement()), duplicated_refseq_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
                result_rc = MyResult(result_rc)
                # 一応スコアを確認する
                if is_refseq_seq_all_ATGC:
                    assert result.score == self.clac_cigar_score(result.cigar)
                    assert result_rc.score == self.clac_cigar_score(result_rc.cigar)
                # レジスター
                result_list.append(result)
                result_list.append(result_rc)
                gc.collect()
            result_dict[seq_id] = result_list
        return result_dict

class MyCigarStr(str):
    def __new__(cls, cigar_str):
        # when common cigar strings are passed
        if cigar_str[0].isdecimal():
            val = "".join([
                L for N, L in re.findall('(\d+)(\D)', cigar_str)
                    for i in range(int(N))
            ])
            self = super().__new__(cls, val)
            return self
        # when "MyCigarStr" strings are passed
        else:
            self = super().__new__(cls, cigar_str)
            return self
    def __iadd__(self, other):
        return self.__class__(self + other)
    def invert(self):
        return self.__class__(self[::-1])
    def number_of_letters_on_5prime(self, letters):
        for i, l in enumerate(self):
            if l not in letters:
                return i
    def number_of_letters_on_3prime(self, letters):
        for k, l in enumerate(self[::-1]):
            if l not in letters:
                return k
    def clip_from_both_ends(self, letters):
        i = self.number_of_letters_on_5prime(letters)
        k = self.number_of_letters_on_3prime(letters)
        return self.__class__(self[i:len(self) - k])
    def clipped_len(self):
        return len(self.clip())

class IntermediateResults(mc.MyTextFormat):
    def __init__(self, result_dict=None, my_aligner=None) -> None:
        self.path = None
        self.keys = [
            ("refseq_names", "list"), 
            ("refseq_hash_list", "list"), 
            ("combined_fastq_names", "list"), 
            ("combined_fastq_hash", "str"), 
            ("param_dict", "dict"), 
            ("combined_fastq_id_list", "list")
        ]
        self.non_default_keys_start_idx = 6
        self.param_dict_keys_matter = ['gap_open_penalty', 'gap_extend_penalty', 'match_score', 'mismatch_score']
        self.combined_fastq_id_list = []
        if (my_aligner is not None) and (result_dict is not None):
            # my aligner related info
            self.refseq_names = [refseq.path.name for refseq in my_aligner.refseq_list]
            self.refseq_hash_list = [refseq.my_hash for refseq in my_aligner.refseq_list]
            self.combined_fastq_names = [fastq_path.name for fastq_path in my_aligner.combined_fastq.path]
            self.combined_fastq_hash = my_aligner.combined_fastq.my_hash
            self.param_dict = my_aligner.param_dict
            # result_dict related info
            assert len(result_dict) == len(my_aligner.combined_fastq)
            assert result_dict.keys() == my_aligner.combined_fastq.keys()
            idx = 0
            for combined_fastq_id, result_list in result_dict.items():
                self.combined_fastq_id_list.append(combined_fastq_id)
                for result in result_list:
                    result_key = f"result{idx}"
                    setattr(self, result_key, result.to_dict())
                    self.keys.append((result_key, "dict"))
                    idx += 1
            assert len(self.combined_fastq_id_list) == len(my_aligner.combined_fastq)
            assert len(my_aligner.combined_fastq) * len(my_aligner.refseq_list) * 2 == idx # リバコン(rc) もあるので二倍で assertion
    def assert_identity(self, my_aligner):
        refseq_names = [refseq.path.name for refseq in my_aligner.refseq_list]
        refseq_hash_list = [refseq.my_hash for refseq in my_aligner.refseq_list]
        combined_fastq_names = [fastq_path.name for fastq_path in my_aligner.combined_fastq.path]
        combined_fastq_hash = my_aligner.combined_fastq.my_hash
        param_dict = my_aligner.param_dict
        is_param_dict_same = all(param_dict[k] == self.param_dict[k] for k in self.param_dict_keys_matter)
        if (self.refseq_names == refseq_names) and\
            (self.refseq_hash_list == refseq_hash_list) and\
            (self.combined_fastq_names == combined_fastq_names) and\
            (self.combined_fastq_hash == combined_fastq_hash) and is_param_dict_same:
            return True
        elif (set(self.refseq_names) == set(refseq_names)) and\
            (set(self.refseq_hash_list) == set(refseq_hash_list)) and\
            (self.combined_fastq_names == combined_fastq_names) and\
            (self.combined_fastq_hash == combined_fastq_hash) and is_param_dict_same:
            # intermediat_resultsに応じて順番を並べ直す
            my_aligner.refseq_list = [my_aligner.refseq_list[refseq_hash_list.index(refseq_hash)] for refseq_hash in self.refseq_hash_list]
            my_aligner.set_refseq_related_info()
            return True
        else:
            return False
    def load(self, load_path):
        self.path = load_path
        self.keys = super().load(load_path)
        for k, v in self.param_dict.items():
            try:
                self.param_dict[k] = int(v)
            except:
                self.param_dict[k] = float(v)
    @property
    def result_dict(self):
        result_dict = OrderedDict()
        result_list = []
        refseq_idx = 0
        refseq_idx_max = len(self.refseq_names) * 2
        fastq_idx = 0
        for key, data_type in self.keys[self.non_default_keys_start_idx:]:
            assert data_type == "dict"
            my_result = MyResult()
            my_result.apply_dict_params(getattr(self, key))
            result_list.append(my_result)
            refseq_idx += 1
            if refseq_idx == refseq_idx_max:
                fastq_id = self.combined_fastq_id_list[fastq_idx]
                result_dict[fastq_id] = result_list
                refseq_idx = 0
                fastq_idx += 1
                result_list = []
        assert refseq_idx == 0
        assert fastq_idx == len(self.combined_fastq_id_list)
        return result_dict

def save_intermediate_results(result_dict, my_aligner, intermediate_results_save_path):
    ir = IntermediateResults(result_dict=result_dict, my_aligner=my_aligner)
    ir.path = intermediate_results_save_path
    ir.save(intermediate_results_save_path)
    return ir

#@title # 1. Upload and select files

def organize_files(fastq_file_path_list, refseq_file_path_list):
    fastq_list = [MyFastQ(i) for i in fastq_file_path_list]
    refseq_list = [MyRefSeq(i) for i in refseq_file_path_list]

    # assert refseq
    if len(refseq_list) == 0:
        raise Exception("Please select at least 1 reference sequence file!")
    refseq_stem_list = [refseq.path.stem for refseq in refseq_list]
    if len(refseq_stem_list) != len(set(refseq_stem_list)):
        raise Exception("The file name must not be the same even if the extension is different.")

    # assert fastq
    if len(fastq_list) > 1:
        combined_fastq = MyFastQ.combine(fastq_list)
    elif len(fastq_list) == 1:
        combined_fastq = copy.deepcopy(fastq_list[0])
    else:
        raise Exception("Please select at least 1 fastq file!")

    combined_fastq.path = [fastq.path for fastq in fastq_list]
    return refseq_list, combined_fastq

#@title # 2. Execute alignment

def execute_alignment(refseq_list, combined_fastq, param_dict, save_dir):
    my_aligner = MyAligner(refseq_list, combined_fastq, param_dict)

    # load if there is intermediate data
    skip = False
    intermediate_results_save_path = save_dir / f"{combined_fastq.combined_name_stem}.intermediate_results.txt"
    if intermediate_results_save_path.exists():
        intermediate_results = IntermediateResults()
        intermediate_results.load(intermediate_results_save_path)
        if intermediate_results.assert_identity(my_aligner):
            result_dict = intermediate_results.result_dict
            print()
            print("alignment: SKIPPED (exported intermediate was used)")
            skip = True
    if not skip:
        # Execute
        result_dict = my_aligner.align_all()
        print()
        print("alignment: DONE")
        intermediate_results = save_intermediate_results(result_dict, my_aligner, intermediate_results_save_path)

    return result_dict, my_aligner, intermediate_results

#@title # 3. Set threshold for assignment

class AlignmentResult():
    def __init__(self, result_dict, my_aligner, param_dict):
        self.score_threshold = param_dict["score_threshold"]
        self.result_dict = result_dict
        self.my_aligner = my_aligner
        # attributs to register results
        self.score_list_ALL = None
        self.result_info_assigned = None
        self.aligned_result_list = None
    def get_score_summary_df(self):
        records = []
        for info in self.score_list_ALL:
            d = OrderedDict()
            d["query_idx"] = info["query_idx"]
            d["seq_id"] = info["seq_id"]
            for i, refseq in enumerate(self.my_aligner.refseq_list):
                d[f"{refseq.path.name} (idx={i})"]= info["score_list"][2 * i]
                d[f"{refseq.path.name} (idx={i},rc)"] = info["score_list"][2 * i + 1]
            for i, refseq in enumerate(self.my_aligner.refseq_list):
                d[f"{refseq.path.name} (idx={i}, normalized)"]= info["normalized_score_list"][2 * i]
                d[f"{refseq.path.name} (idx={i},rc, normalized)"] = info["normalized_score_list"][2 * i + 1]
            d["assigned_refseq_idx"] = info["assigned_refseq_idx"]
            d["is_reverse_compliment"] = info["is_reverse_compliment"]
            d["assigned"] = info["assigned"]
            records.append(d)
        return pd.DataFrame.from_records(records)
    def save_score_summary(self, save_path):
        score_summary = (
            "query_idx" 
            + "\tseq_id" 
            + "\t" 
            + "\t".join([f"{refseq.path.name} (idx={i})\t{refseq.path.name} (idx={i},rc)" for i, refseq in enumerate(self.my_aligner.refseq_list)])
            + "\t"
            + "\t".join([f"{refseq.path.name} (idx={i}, normalized)\t{refseq.path.name} (idx={i},rc, normalized)" for i, refseq in enumerate(self.my_aligner.refseq_list)])
            + "\tassigned_refseq_idx"
            + "\tis_reverse_compliment"
            + "\tassigned\n"
            + "\n".join(
                [(
                    str(info["query_idx"])
                    + "\t" + info["seq_id"]
                    + "\t" + "\t".join(map(str, info["score_list"]))
                    + "\t" + "\t".join(map(str, info["normalized_score_list"]))
                    + "\t" + str(info["assigned_refseq_idx"])
                    + "\t" + str(info["is_reverse_compliment"])
                    + "\t" + str(info["assigned"])
                ) for info in self.score_list_ALL]
            )
        )
        with open(save_path, "w") as f:
            f.write(score_summary)
        return score_summary
    def normalize_scores_and_apply_threshold(self):
        self.score_list_ALL = []
        self.result_info_assigned = [[] for i in self.my_aligner.refseq_list] # [[[seq_id, is_reverse_compliment, result, query_idx], ...], ...]
        assert len(self.result_dict) == len(self.my_aligner.combined_fastq)
        for query_idx, (seq_id, result_list) in enumerate(self.result_dict.items()):
            assert len(result_list) == len(self.my_aligner.duplicated_refseq_seq_list) * 2
            # normalize scores for each refseq
            score_list = []
            normalized_score_list = []
            for result_idx, result in enumerate(result_list):
                score_list.append(result.score)
                duplicated_refseq_seq = self.my_aligner.duplicated_refseq_seq_list[result_idx // 2]
                normalized_score = result.score / len(duplicated_refseq_seq) * 2
                if normalized_score > 1:
                    normalized_score = 1
                normalized_score_list.append(normalized_score)
            # choose sequence with maximum score
            idx = np.argmax(normalized_score_list)
            refseq_idx, is_reverse_compliment = divmod(idx, 2)
            result = result_list[idx]
            # quality check
            assigned = (normalized_score_list[idx] >= self.score_threshold)\
                     & (result.score <= len(self.my_aligner.refseq_list[refseq_idx].seq) * self.my_aligner.match_score)\
                     & (len(self.my_aligner.combined_fastq[seq_id][0]) <= len(self.my_aligner.duplicated_refseq_seq_list[refseq_idx]))\
                     & ((np.array(score_list) == score_list[idx]).sum() == 1)   # refseq の長さの二倍以上ある query_seq は omit する、全く同じスコアがある場合は omit する
            # register
            self.score_list_ALL.append({
                "query_idx":query_idx, 
                "seq_id":seq_id, 
                "score_list":score_list, 
                "normalized_score_list":normalized_score_list, 
                "assigned_refseq_idx":refseq_idx, 
                "is_reverse_compliment":is_reverse_compliment, 
                "assigned":int(assigned)
            })
            if assigned:
                self.result_info_assigned[refseq_idx].append([
                    seq_id, 
                    is_reverse_compliment, 
                    result, 
                    query_idx
                ])
    def integrate_assigned_result_info(self):
        self.aligned_result_list = []
        assert len(self.my_aligner.refseq_list) == len(self.result_info_assigned)
        total_N = len(self.my_aligner.refseq_list)
        for cur_idx, (refseq, result_info_list) in enumerate(zip(self.my_aligner.refseq_list, self.result_info_assigned)):
            print(f"\rIntegrating alignment results: {cur_idx + 1} out of {total_N}", end="")
            if len(result_info_list) > 0:
                my_cigar_str_list = []
                new_q_scores_list = []
                new_seq_list = []
                seq_id_list, is_reverse_compliment_list, result_list, query_idx_list = list(zip(*result_info_list))
                for result, is_reverse_compliment, seq_id in zip(result_list, is_reverse_compliment_list, seq_id_list):
                    # query info
                    seq = self.my_aligner.combined_fastq[seq_id][0]
                    q_scores = self.my_aligner.combined_fastq[seq_id][1]
                    if is_reverse_compliment:
                        seq = str(Seq(seq).reverse_complement())
                        q_scores = q_scores[::-1]
                    # results
                    my_cigar_str = MyCigarStr(result.cigar)
                    # organize alignment based on refseq
                    number_of_ref_bases_before_query = result.beg_ref - result.beg_query
                    number_of_ref_bases_after_query = (refseq.length * 2 - result.end_ref - 1) - (len(seq) - result.end_query - 1)
                    """
                                beg_ref(9)      end_ref(24)
                                     |                |
                    pos     0         10         20         30
                    ref     atcgatcggGGCTATG-CTTGCAT-GCatcgatcg
                    align   HHHHHHHSS====X==I===D===N==SSSHHHHH
                    query          caGGCTGTGACTT-CAT-GCtga
                    pos            0         10          20
                                     |                |         
                                beg_query(2)    end_query(17)
                    """
                    # truncate
                    assert (number_of_ref_bases_before_query >= 0) or (number_of_ref_bases_after_query >= 0)
                    if (number_of_ref_bases_before_query < 0):
                        q_scores = q_scores[-number_of_ref_bases_before_query:]
                        seq      = seq[-number_of_ref_bases_before_query:]
                        result.beg_query -= -number_of_ref_bases_before_query
                        result.end_query -= -number_of_ref_bases_before_query                        
                        number_of_ref_bases_before_query = 0
                    if (number_of_ref_bases_after_query < 0):
                        q_scores = q_scores[:number_of_ref_bases_after_query]
                        seq      = seq[:number_of_ref_bases_after_query]
                        # result.beg_query # do nothing
                        # result.end_query # do nothing
                        number_of_ref_bases_after_query = 0
                    assert (number_of_ref_bases_before_query >= 0) & (number_of_ref_bases_after_query >= 0)
                    # organize my_cigar_str
                    my_cigar_str = MyCigarStr(
                        "H" * number_of_ref_bases_before_query      # add deletion of ref
                        + "S" * result.beg_query                    # soft clip of query
                        + my_cigar_str                              # aligned region
                        + "S" * (len(seq) - result.end_query - 1)   # soft clip of query
                        + "H" * number_of_ref_bases_after_query     # add deletion of ref
                    )
                    my_cigar_str = MyCigarStr(
                        "H" * my_cigar_str.number_of_letters_on_5prime("HD")    # なぜか parasail の結果で 5'側に D が連なっている場合があるので、それを除く（本来 beg_ref で調節されるべき？）
                        + my_cigar_str.clip_from_both_ends("HD")
                        + "H" * my_cigar_str.number_of_letters_on_3prime("HD")
                    )
                    my_cigar_str_H_clip = my_cigar_str.clip_from_both_ends("H")
                    assert len(q_scores) == len(seq) == len(my_cigar_str_H_clip) - my_cigar_str_H_clip.count("D")
                    assert refseq.length * 2 == len(my_cigar_str) - my_cigar_str_H_clip.count("I")

                    # なぜか parasail の結果で 5'側に I が連なっている場合があるので、それを除く（本来 beg_query で調節されるべき？）
                    number_of_I_on_5prime = my_cigar_str.number_of_letters_on_5prime("I")
                    if number_of_I_on_5prime > 0:
                        my_cigar_str = MyCigarStr(my_cigar_str[number_of_I_on_5prime:])
                        q_scores = q_scores[number_of_I_on_5prime:]
                        seq = seq[number_of_I_on_5prime:]
                    my_cigar_str_list.append(my_cigar_str)
                    new_q_scores_list.append(q_scores)
                    new_seq_list.append(seq)
                # further organize to match refseq, new_seq (query), and new_qscores.
                my_cigar_str_net_length_list = [len(my_cigar_str) - my_cigar_str.count("I") for my_cigar_str in my_cigar_str_list]
                assert all(my_cigar_str_net_length_list[0] == x for x in my_cigar_str_net_length_list)

                duplicated_refseq = refseq.seq + refseq.seq
                duplicated_refseq_with_insertion = ""
                my_cigar_str_list_with_insertion = ["" for i in my_cigar_str_list]
                new_q_scores_list_with_insertion = [[] for i in new_q_scores_list]
                new_seq_list_with_insertion = ["" for i in new_seq_list]

                # print(duplicated_refseq)
                # for i in new_seq_list:
                #     print(i)

                # current idx (positions of new_q_scores and new_seq are the same)
                cur_refseq_idx = 0
                cur_my_cigar_str_idx_list = [0 for i in my_cigar_str_list]
                cur_q_scores_idx_list = [0 for i in new_q_scores_list]
                max_refseq_idx = refseq.length * 2 - 1
                max_my_cigar_str_idx_list = [len(my_cigar_str) - 1 for my_cigar_str in my_cigar_str_list]
                max_q_scores_idx_list = [len(new_q_scores) - 1 for new_q_scores in new_q_scores_list]

                # print(max_refseq_idx)
                # print(max_my_cigar_str_idx_list)
                # print(max_q_scores_idx_list)

                all_done = False
                cur_idx = -1
                while not all_done:
                    cur_idx += 1
                    cur_my_cigar_letter_list = [my_cigar_str[cur_my_cigar_str_idx] for my_cigar_str, cur_my_cigar_str_idx in zip(my_cigar_str_list, cur_my_cigar_str_idx_list)]
                    if "I" not in cur_my_cigar_letter_list:
                        for i, L in enumerate(cur_my_cigar_letter_list):
                            if L in "DH":
                                my_cigar_str_list_with_insertion[i] += L
                                new_q_scores_list_with_insertion[i] += [-1]
                                new_seq_list_with_insertion[i]      += "-"
                                cur_my_cigar_str_idx_list[i]        += 1
                            elif L == "=":
                                my_cigar_str_list_with_insertion[i] += L
                                new_q_scores_list_with_insertion[i] += [ new_q_scores_list[i][cur_q_scores_idx_list[i]] ]
                                new_seq_list_with_insertion[i]      += new_seq_list[i][cur_q_scores_idx_list[i]]
                                cur_my_cigar_str_idx_list[i]        += 1
                                cur_q_scores_idx_list[i]            += 1
                            elif L == "X":
                                my_cigar_str_list_with_insertion[i] += L
                                new_q_scores_list_with_insertion[i] += [ new_q_scores_list[i][cur_q_scores_idx_list[i]] ]
                                new_seq_list_with_insertion[i]      += new_seq_list[i][cur_q_scores_idx_list[i]]
                                cur_my_cigar_str_idx_list[i]        += 1
                                cur_q_scores_idx_list[i]            += 1
                            elif L == "S":
                                my_cigar_str_list_with_insertion[i] += L
                                new_q_scores_list_with_insertion[i] += [ new_q_scores_list[i][cur_q_scores_idx_list[i]] ]
                                new_seq_list_with_insertion[i]      += new_seq_list[i][cur_q_scores_idx_list[i]]
                                cur_my_cigar_str_idx_list[i]        += 1
                                cur_q_scores_idx_list[i]            += 1
                            else:
                                print(L)
                                raise Exception("error!")
                        else:
                            duplicated_refseq_with_insertion += duplicated_refseq[cur_refseq_idx]
                            if cur_refseq_idx == refseq.length:
                                turning_idx = cur_idx   # 後半開始
                            cur_refseq_idx += 1
                    else:
                        # TODO: insertion 同士に関してはアラインメントしてないよ！
                        for i, L in enumerate(cur_my_cigar_letter_list):
                            if L == "I":
                                my_cigar_str_list_with_insertion[i] += "I"
                                new_q_scores_list_with_insertion[i] += [ new_q_scores_list[i][cur_q_scores_idx_list[i]] ]
                                new_seq_list_with_insertion[i]      += new_seq_list[i][cur_q_scores_idx_list[i]]
                                cur_my_cigar_str_idx_list[i]        += 1
                                cur_q_scores_idx_list[i]            += 1
                            else:
                                my_cigar_str_list_with_insertion[i] += "N"
                                new_q_scores_list_with_insertion[i] += [-1]
                                new_seq_list_with_insertion[i]      += "-"
                        else:
                            duplicated_refseq_with_insertion += "-"

                    # インデックスの最大値を参照して、すべて終わったら終える！
                    all_done = bool(
                        (cur_refseq_idx > max_refseq_idx)
                        * all([cur_my_cigar_str_idx > max_my_cigar_str_idx for cur_my_cigar_str_idx, max_my_cigar_str_idx in zip(cur_my_cigar_str_idx_list, max_my_cigar_str_idx_list)])
                        * all([cur_q_scores_idx > max_q_scores_idx for cur_q_scores_idx, max_q_scores_idx in zip(cur_q_scores_idx_list, max_q_scores_idx_list)])
                    )
                # print(duplicated_refseq_with_insertion)
                # print(duplicated_refseq_with_insertion[turning_idx:])
                for i in my_cigar_str_list_with_insertion:
                    assert len(duplicated_refseq_with_insertion) == len(i)
                for i in new_seq_list_with_insertion:
                    assert len(duplicated_refseq_with_insertion) == len(i)
                for i in new_q_scores_list_with_insertion:
                    assert len(duplicated_refseq_with_insertion) == len(i)

                # linearlize
                refseq_with_insertion_1            = duplicated_refseq_with_insertion[:turning_idx]
                refseq_with_insertion_2            = duplicated_refseq_with_insertion[turning_idx:]
                assert (len(refseq_with_insertion_1) - refseq_with_insertion_1.count("-")) == (len(refseq_with_insertion_2) - refseq_with_insertion_2.count("-")) == refseq.length
                my_cigar_str_list_with_insertion_1 = [i[:turning_idx] for i in my_cigar_str_list_with_insertion]
                my_cigar_str_list_with_insertion_2 = [i[turning_idx:] for i in my_cigar_str_list_with_insertion]
                new_seq_list_with_insertion_1      = [i[:turning_idx] for i in new_seq_list_with_insertion]
                new_seq_list_with_insertion_2      = [i[turning_idx:] for i in new_seq_list_with_insertion]
                new_q_scores_list_with_insertion_1 = [i[:turning_idx] for i in new_q_scores_list_with_insertion]
                new_q_scores_list_with_insertion_2 = [i[turning_idx:] for i in new_q_scores_list_with_insertion]

                # 前半と後半をアラインメント（insertionを考慮するだけで良い）
                # 前半の末端処理
                assert refseq_with_insertion_2[-1] != "-"
                N_gap_refseq_with_insertion_1_end = 1
                while True:
                    if refseq_with_insertion_1[-N_gap_refseq_with_insertion_1_end] != "-":
                        break
                    N_gap_refseq_with_insertion_1_end += 1
                if N_gap_refseq_with_insertion_1_end > 1:
                    refseq_with_insertion_1 = refseq_with_insertion_1[:1 - N_gap_refseq_with_insertion_1_end]
                    my_cigar_str_list_with_insertion_1 = [i[:1 - N_gap_refseq_with_insertion_1_end] for i in my_cigar_str_list_with_insertion_1]
                    new_seq_list_with_insertion_1 = [i[:1 - N_gap_refseq_with_insertion_1_end] for i in new_seq_list_with_insertion_1]
                    new_q_scores_list_with_insertion_1 = [i[:1 - N_gap_refseq_with_insertion_1_end] for i in new_q_scores_list_with_insertion_1]
                # 前半後半アラインメント開始
                idx = -1
                refseq_idx1 = 0
                refseq_idx2 = 0
                refseq_max_idx1 = len(refseq_with_insertion_1) - 1
                refseq_max_idx2 = len(refseq_with_insertion_2) - 1
                while True:
                    idx += 1
                    if refseq_with_insertion_1[idx] == refseq_with_insertion_2[idx]:
                        refseq_idx1 += 1
                        refseq_idx2 += 1
                    elif refseq_with_insertion_1[idx] == "-":
                        refseq_idx1 += 1
                        refseq_with_insertion_2 = refseq_with_insertion_2[:idx] + "-" + refseq_with_insertion_2[idx:]
                        for i, j in enumerate(my_cigar_str_list_with_insertion_2):
                            my_cigar_str_list_with_insertion_2[i] = j[:idx] + "N" + j[idx:]
                        for i, j in enumerate(new_seq_list_with_insertion_2):
                            new_seq_list_with_insertion_2[i]      = j[:idx] + "-" + j[idx:]
                        for i in new_q_scores_list_with_insertion_2:
                            i.insert(idx, -1)
                    elif refseq_with_insertion_2[idx] == "-":
                        refseq_idx2 += 1
                        refseq_with_insertion_1 = refseq_with_insertion_1[:idx] + "-" + refseq_with_insertion_1[idx:]
                        for i, j in enumerate(my_cigar_str_list_with_insertion_1):
                            my_cigar_str_list_with_insertion_1[i] = j[:idx] + "N" + j[idx:]
                        for i, j in enumerate(new_seq_list_with_insertion_1):
                            new_seq_list_with_insertion_1[i]      = j[:idx] + "-" + j[idx:]
                        for i in new_q_scores_list_with_insertion_1:
                            i.insert(idx, -1)
                    else:
                        raise Exception("error!")
                    # end
                    if (refseq_idx1 == refseq_max_idx1) & (refseq_idx2 == refseq_max_idx2):
                        break
                # check
                assert refseq_with_insertion_1 == refseq_with_insertion_2
                for i in my_cigar_str_list_with_insertion_1:
                    assert len(refseq_with_insertion_1) == len(i)
                for i in my_cigar_str_list_with_insertion_2:
                    assert len(refseq_with_insertion_2) == len(i)
                for i in new_seq_list_with_insertion_1:
                    assert len(refseq_with_insertion_1) == len(i)
                for i in new_seq_list_with_insertion_2:
                    assert len(refseq_with_insertion_2) == len(i)
                for i in new_q_scores_list_with_insertion_1:
                    assert len(refseq_with_insertion_1) == len(i)
                for i in new_q_scores_list_with_insertion_2:
                    assert len(refseq_with_insertion_2) == len(i)

                # print("Alighment of top half and bottom half: DONE")

                # スコアマキシマムになるような前半後半の境界を探す
                custom_cigar_score_dict = self.my_aligner.get_custom_cigar_score_dict()
                refseq_with_insertion = refseq_with_insertion_1
                my_cigar_str_list_with_insertion = [None for i in my_cigar_str_list_with_insertion_1]
                new_seq_list_with_insertion = [None for i in new_seq_list_with_insertion_1]
                new_q_scores_list_with_insertion = [None for i in new_q_scores_list_with_insertion_1]
                for i, (j1, j2, k1, k2, l1, l2) in enumerate(zip(
                        my_cigar_str_list_with_insertion_1, 
                        my_cigar_str_list_with_insertion_2, 
                        new_seq_list_with_insertion_1, 
                        new_seq_list_with_insertion_2, 
                        new_q_scores_list_with_insertion_1, 
                        new_q_scores_list_with_insertion_2
                )):
                    my_cigar_scores_with_insertion_1 = np.array([custom_cigar_score_dict[j] for j in j1])
                    my_cigar_scores_with_insertion_2 = np.array([custom_cigar_score_dict[j] for j in j2])
                    switching_idx = np.argmin(np.cumsum(my_cigar_scores_with_insertion_1 - my_cigar_scores_with_insertion_2))
                    # register
                    my_cigar_str_list_with_insertion[i] = j2[:switching_idx + 1] + j1[switching_idx + 1:]
                    new_seq_list_with_insertion[i]      = k2[:switching_idx + 1] + k1[switching_idx + 1:]
                    new_q_scores_list_with_insertion[i] = l2[:switching_idx + 1] + l1[switching_idx + 1:]

                # print(seq_id_list)
                # print(refseq_with_insertion)
                # for i, seq_id in enumerate(seq_id_list):
                #     # print(seq_id)
                #     print(my_cigar_str_list_with_insertion[i])
                #     print(new_seq_list_with_insertion[i])
                #     print(new_q_scores_list_with_insertion[i])
                self.aligned_result_list.append({
                    "refseq_with_insertion": refseq_with_insertion, 
                    "query_idx_list": query_idx_list, 
                    "seq_id_list": seq_id_list, 
                    "my_cigar_str_list_with_insertion": my_cigar_str_list_with_insertion, 
                    "new_seq_list_with_insertion": new_seq_list_with_insertion, 
                    "new_q_scores_list_with_insertion": new_q_scores_list_with_insertion
                })
            else:
                self.aligned_result_list.append({
                    "refseq_with_insertion": refseq.seq, 

                    "query_idx_list": (), 
                    "seq_id_list": (), 
                    "my_cigar_str_list_with_insertion": [], 
                    "new_seq_list_with_insertion": [], 
                    "new_q_scores_list_with_insertion": []

                    # "query_idx_list": (-1, ), 
                    # "seq_id_list": ("@None", ), 
                    # "my_cigar_str_list_with_insertion": ["X" * len(refseq.seq)], 
                    # "new_seq_list_with_insertion": ["-" * len(refseq.seq)], 
                    # "new_q_scores_list_with_insertion": [[-1] * len(refseq.seq)]
                })
        assert len(self.my_aligner.refseq_list) == len(self.aligned_result_list)
    def export_as_text(self, save_dir):
        text_list = []
        save_path_list = []
        for refseq, aligned_result in zip(self.my_aligner.refseq_list, self.aligned_result_list):
            text = ""
            idx_label_minimum = "consensus"
            query_idx_list = aligned_result["query_idx_list"]
            query_idx_len_max = max([len(str(query_idx)) for query_idx in query_idx_list] + [0])
            label_N0 = max(query_idx_len_max, len(idx_label_minimum)) + 1
            seq_id_list = aligned_result["seq_id_list"]
            seq_id_len_max = max([len(seq_id) for seq_id in seq_id_list] + [0])
            label_N1 = max(seq_id_len_max, len(refseq.path.name)) + 1
            text += (
                "ref"
                + " " * (label_N0 - 3)
                + refseq.path.name
                + " " * (label_N1 - len(refseq.path.name))
                + aligned_result["refseq_with_insertion"]
            )
            consensus_seq, consensus_q_scores, consensus_seq_all, consensus_q_scores_all = self.consensus_dict[refseq.path.name]
            text += (
                "\n"
                + "consensus"
                + " " * (label_N0 - 9 + label_N1)
                + consensus_seq_all
            )
            text += (
                "\n"
                + "consensus"
                + " " * (label_N0 - 9 + label_N1)
                + "".join([chr(q) for q in (np.array(consensus_q_scores_all) + 33)])
            )
            for query_idx, seq_id, my_cigar_str_with_insertion, new_seq_with_insertion, new_q_scores_with_insertion in \
                zip(
                    aligned_result["query_idx_list"], 
                    aligned_result["seq_id_list"], 
                    aligned_result["my_cigar_str_list_with_insertion"], 
                    aligned_result["new_seq_list_with_insertion"], 
                    aligned_result["new_q_scores_list_with_insertion"]
                ):
                label = (
                    "\n"
                    + str(query_idx)
                    + " " * (label_N0 - len(str(query_idx)))
                    + seq_id
                    + " " * (label_N1 - len(seq_id))
                )
                text += (
                    label
                    + new_seq_with_insertion
                    + label
                    + my_cigar_str_with_insertion
                    + label
                    + "".join([chr(q) for q in (np.array(new_q_scores_with_insertion) + 33)])
                )
            save_path = (save_dir / refseq.path.name).with_suffix(".txt")
            with open(save_path, "w") as f:
                f.write(text)
            text_list.append(text)
            save_path_list.append(save_path)
        return text_list, save_path_list
    def alignment_reuslt_list_2_text_list(self, linewidth=""):
        text_list = []
        highlight_pos_list = []
        refseq_name_list = []
        for refseq, aligned_result in zip(self.my_aligner.refseq_list, self.aligned_result_list):
            ref_label = "REF"
            label_N = max(len(str(max(aligned_result["query_idx_list"]))), len(ref_label)) + 1
            refseq_name_list.append(refseq.path.name)
            # "linewidth" 行ごとにまとめて改行
            child_lines = re.findall(fr".{{1,{linewidth}}}", aligned_result["refseq_with_insertion"].upper())
            master_lines = [list(map(lambda l: ref_label + " " * (label_N - len(ref_label)) + l, child_lines))]
            for idx, (new_seq_with_insertion, query_idx) in enumerate(zip(aligned_result["new_seq_list_with_insertion"], aligned_result["query_idx_list"])):
                child_lines = re.findall(fr".{{1,{linewidth}}}", new_seq_with_insertion.upper())
                master_lines.append(list(map(lambda l: f"{query_idx}" + " " * (label_N - len(str(query_idx))) + l, child_lines)))
            # 改行したものを zip でくっつけていく
            text = ""
            for i, lines in enumerate(zip(*master_lines)):
                text += (
                    f"{i * linewidth + 1}-{(i + 1) * linewidth}\n"
                    + "\n".join(lines)
                    + "\n\n"
                )
            text_list.append(text.strip())
            # ハイライト部分
            highlight_pos_in_text = []
            for i, my_cigar_str_with_insertion in enumerate(aligned_result["my_cigar_str_list_with_insertion"]):
                # print(len(my_cigar_str_with_insertion))
                true_highlight_pos = [m.start() for m in re.finditer('[IXDS]', my_cigar_str_with_insertion)]
                for p in true_highlight_pos:
                    r, c = divmod(p, linewidth) 
                    r = r * (len(master_lines) + 2) + i + 2 # ポジション行、master_lines、改行空白
                    c += label_N
                    highlight_pos_in_text.append((r, c))
            highlight_pos_list.append(highlight_pos_in_text)
        return refseq_name_list, text_list, highlight_pos_list
    def export_log(self, save_path):
        log = Log(self)
        log.save(save_path)
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
    def save_consensus(self, save_dir, id_suffix=""):
        save_path_list = []
        for key, val in self.consensus_dict.items():
            save_path1 = save_dir / Path(key).with_suffix(".fastq")
            consensus_seq, consensus_q_scores, *all_results = val
            consensus_q_scores = "".join([chr(q + 33) for q in consensus_q_scores])
            consensus_fastq_txt = f"@{key}_{id_suffix}:\n{consensus_seq.upper()}\n+\n{consensus_q_scores}"
            with open(save_path1, "w") as f:
                f.write(consensus_fastq_txt)
            save_path_list.append(save_path1)
        return save_path_list
    def alignment_summary_bar_graphs(self):
        N_array_list = []
        bar_graph_img_list = []
        filename_for_saving_list = []
        for refseq_idx, aligned_result in enumerate(self.aligned_result_list):
            filename_for_saving_list.append(f"{self.my_aligner.refseq_list[refseq_idx].path.stem}.gif")
            # prepare
            refseq_with_insertion = aligned_result["refseq_with_insertion"]
            N_array = np.empty((5, len(refseq_with_insertion)), int)
            tick_pos_list = []
            tick_label_list = []
            cur_ref_base_pos = 0
            for refbase_idx, refbase in enumerate(refseq_with_insertion):

                if refbase != "-":
                    cur_ref_base_pos += 1
                    if cur_ref_base_pos%100 == 0:
                        tick_pos_list.append(refbase_idx)
                        tick_label_list.append(cur_ref_base_pos)

                N_match = 0     # =
                N_mismatch = 0  # X
                N_insertion = 0 # I
                N_deletion = 0  # D
                N_omitted = 0   # N, H, S
                for my_cigar_str in aligned_result["my_cigar_str_list_with_insertion"]:
                    L = my_cigar_str[refbase_idx]
                    if L == "=":    N_match += 1
                    elif L == "X":  N_mismatch += 1
                    elif L == "I":  N_insertion += 1
                    elif L == "D":  N_deletion += 1
                    elif L in "NHS":    N_omitted += 1
                    else:   raise Exception(f"unknown cigar string {L}")
                N_array[:, refbase_idx] = [N_match, N_omitted, N_mismatch, N_insertion, N_deletion]
            N_array_list.append(N_array)
            # 描画していく！
            bar_graph_img = BarGraphImg(N_array, tick_pos_list, tick_label_list)
            bar_graph_img.generate_bar_graph_ndarray()
            bar_graph_img.set_legend(legend_list=["match", "omitted", "mismatch", "insertion", "deletion"])
            bar_graph_img_list.append(bar_graph_img)
        return bar_graph_img_list, filename_for_saving_list

class Log(mc.MyTextFormat):
    app_name = app_name
    version = version
    description = description
    def __init__(self, alignment_result) -> None:
        my_aligner = alignment_result.my_aligner
        self.header = f"{app_name} ver{version}\n{description}"
        self.datetime = datetime.now()
        self.input_reference_files = [refseq.path for refseq in my_aligner.refseq_list]
        self.input_fastq_files = [fastq_path for fastq_path in my_aligner.combined_fastq.path]
        self.input_reference_hash_list = [refseq.my_hash for refseq in my_aligner.refseq_list]
        self.input_combined_hash = my_aligner.combined_fastq.my_hash
        self.alignment_params = my_aligner.param_dict
        self.custom_cigar_score_dict = my_aligner.get_custom_cigar_score_dict()
        self.score_threshold = alignment_result.score_threshold
        self.score_matrix = alignment_result.matrix2string(my_aligner.my_custom_matrix.matrix, digit=3, round=True)
        self.consensus_settings = alignment_result.consensus_settings["sbq_pdf_version"]
        self.error_matrix = alignment_result.matrix2string(
                alignment_result.consensus_settings["P_N_dict_matrix"], 
                bases=alignment_result.consensus_settings["bases"], 
                digit=None, 
                round=False
        )
        self.keys = [
            ("header", "str"), 
            ("datetime", "str"), 
            ("input_reference_files", "listPath"), 
            ("input_fastq_files", "listPath"), 
            ("input_reference_hash_list", "list"), 
            ("input_combined_hash", "str"), 
            ("alignment_params", "dict"), 
            ("custom_cigar_score_dict", "dict"), 
            ("score_threshold", "float"), 
            ("score_matrix", "str"), 
            ("consensus_settings", "str"), 
            ("error_matrix", "str"), 
        ]

class BarGraphImg():
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # list of hex color "#ffffff" or tuple
    color_cycle = [(255, 252, 245), (255, 243, 220), (110, 110, 255), (110, 255, 110), (255, 110, 110)]
    tick_color = (200, 200, 200)
    # numbers
    dtype = uint8
    number_w = 4    # pixel
    letter_h = 5    # pixel
    zero = np.array([
        [0,1,1,0], 
        [1,0,0,1], 
        [1,0,0,1], 
        [1,0,0,1], 
        [0,1,1,0]
    ], dtype=dtype)
    one = np.array([
        [0,1,0,0], 
        [1,1,0,0], 
        [0,1,0,0], 
        [0,1,0,0], 
        [1,1,1,0]
    ], dtype=dtype)
    two = np.array([
        [0,1,1,0], 
        [1,0,0,1], 
        [0,0,1,0], 
        [0,1,0,0], 
        [1,1,1,1]
    ], dtype=dtype)
    three = np.array([
        [0,1,1,0], 
        [1,0,0,1], 
        [0,0,1,0], 
        [1,0,0,1], 
        [0,1,1,0]
    ], dtype=dtype)
    four = np.array([
        [0,0,1,0], 
        [0,1,1,0], 
        [1,0,1,0], 
        [1,1,1,1], 
        [0,0,1,0]
    ], dtype=dtype)
    five = np.array([
        [1,1,1,0], 
        [1,0,0,0], 
        [1,1,1,0], 
        [0,0,0,1], 
        [1,1,1,0]
    ], dtype=dtype)
    six = np.array([
        [0,1,1,0], 
        [1,0,0,0], 
        [1,1,1,0], 
        [1,0,0,1], 
        [0,1,1,0]
    ], dtype=dtype)
    seven = np.array([
        [1,1,1,1], 
        [0,0,0,1], 
        [0,0,1,0], 
        [0,1,0,0], 
        [0,1,0,0]
    ], dtype=dtype)
    eight = np.array([
        [0,1,1,0], 
        [1,0,0,1], 
        [0,1,1,0], 
        [1,0,0,1], 
        [0,1,1,0]
    ], dtype=dtype)
    nine = np.array([
        [0,1,1,0], 
        [1,0,0,1], 
        [0,1,1,1], 
        [0,0,0,1], 
        [0,1,1,0]
    ], dtype=dtype)
    hyphen = np.array([
        [0,0,0,0], 
        [0,0,0,0], 
        [1,1,1,1], 
        [0,0,0,0], 
        [0,0,0,0]
    ])
    blank = np.array([
        [0,0,0,0], 
        [0,0,0,0], 
        [0,0,0,0], 
        [0,0,0,0], 
        [0,0,0,0]
    ])
    w2n = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    # words
    M = np.array([
        [1,0,0,0,1], 
        [1,1,0,1,1], 
        [1,0,1,0,1], 
        [1,0,0,0,1], 
        [1,0,0,0,1]
    ])
    A = np.array([
        [0,1,1,0], 
        [1,0,0,1], 
        [1,1,1,1], 
        [1,0,0,1], 
        [1,0,0,1]
    ], dtype=dtype)
    T = np.array([
        [1,1,1,1,1], 
        [0,0,1,0,0], 
        [0,0,1,0,0], 
        [0,0,1,0,0], 
        [0,0,1,0,0]
    ], dtype=dtype)
    C = np.array([
        [0,1,1,0], 
        [1,0,0,1], 
        [1,0,0,0], 
        [1,0,0,1], 
        [0,1,1,0]
    ], dtype=dtype)
    H = np.array([
        [1,0,0,1], 
        [1,0,0,1], 
        [1,1,1,1], 
        [1,0,0,1], 
        [1,0,0,1]
    ], dtype=dtype)
    D = np.array([
        [1,1,1,0], 
        [1,0,0,1], 
        [1,0,0,1], 
        [1,0,0,1], 
        [1,1,1,0]
    ])
    E = np.array([
        [1,1,1,1], 
        [1,0,0,0], 
        [1,1,1,0], 
        [1,0,0,0], 
        [1,1,1,1]
    ])
    L = np.array([
        [1,0,0,0], 
        [1,0,0,0], 
        [1,0,0,0], 
        [1,0,0,0], 
        [1,1,1,1]
    ])
    I = np.array([
        [1,1,1], 
        [0,1,0], 
        [0,1,0], 
        [0,1,0], 
        [1,1,1]
    ])
    O = np.array([
        [0,1,1,1,0], 
        [1,0,0,0,1], 
        [1,0,0,0,1], 
        [1,0,0,0,1], 
        [0,1,1,1,0]
    ])
    N = np.array([
        [1,0,0,0,1], 
        [1,1,0,0,1], 
        [1,0,1,0,1], 
        [1,0,0,1,1], 
        [1,0,0,0,1]
    ])
    S = np.array([
        [0,1,1,1], 
        [1,0,0,0], 
        [0,1,1,0], 
        [0,0,0,1], 
        [1,1,1,0]
    ])
    R = np.array([
        [1,1,1,0], 
        [1,0,0,1], 
        [1,1,1,0], 
        [1,0,1,0], 
        [1,0,0,1]
    ])
    vs = np.array([ # vertical space
        [0], 
        [0], 
        [0], 
        [0], 
        [0], 
    ])
    insertion = np.hstack((I,vs,N,vs,S,vs,E,vs,R,vs,T,vs,I,vs,O,vs,N))
    deletion = np.hstack((D,vs,E,vs,L,vs,E,vs,T,vs,I,vs,O,vs,N))
    mismatch = np.hstack((M,vs,I,vs,S,vs,M,vs,A,vs,T,vs,C,vs,H))
    omitted = np.hstack((O,vs,M,vs,I,vs,T,vs,T,vs,E,vs,D))
    match = np.hstack((M,vs,A,vs,T,vs,C,vs,H))
    # sizes
    bar_w = 1           # pixel
    bar_w_space = 0     # pixel
    bar_sum_h = 100     # pixel
    wrap = 1000         # bars
    tick_h = 5          # pixel
    h_space = 30        # pixel
    w_space = 30        # pixel
    l_margin = 40       # pixel
    r_margin = 40       # pixel
    t_margin = 100      # pixel
    b_margin = 40       # pixel
    minimum_margin = 20 # pixel
    def __init__(self, N_array, tick_pos_list, tick_label_list) -> None:
        self.N_array = N_array
        assert (self.N_array.sum(axis=0) == self.N_array[:, 0].sum()).all()
        self.tick_pos_list = tick_pos_list
        self.tick_label_list = tick_label_list
        self.N_rows = np.ceil(self.N_array.shape[1] / self.wrap).astype(int)
        self.N_cols = 1
        # 画像パラメータ（画像左上の座標が [0, 0]、ただし ax の内部では左下が原点）
        self.img_pixel_w = self.l_margin + (self.bar_w + self.bar_w_space) * self.wrap  - self.bar_w_space + self.r_margin
        self.img_pixel_h = self.t_margin + (self.bar_sum_h + self.h_space) * self.N_rows - self.h_space + self.b_margin
        self.ax_origin_w_list = [self.l_margin + (self.bar_w + self.w_space) * i for i in range(self.N_cols)]
        self.ax_origin_h_list = [self.t_margin + self.bar_sum_h + (self.bar_sum_h + self.h_space) * i - 1 for i in range(self.N_rows)]
        # 画像
        assert self.dtype == np.uint8
        self.img_array_rgb = np.ones((self.img_pixel_h, self.img_pixel_w, 3), dtype=self.dtype) * 255
        self.color_cycle_rgb = self._color_cycle_rgb()
    def _color_cycle_rgb(self):
        try:
            return [tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5)) for hex_color in self.color_cycle]
        except:
            return self.color_cycle # already rgb
    def generate_bar_graph_ndarray(self):
        if (self.N_array.sum(axis=0) == 0).any():
            self.N_array[-1, self.N_array.sum(axis=0) == 0] += 1
        N_array_compositional = (self.N_array / self.N_array.sum(axis=0) * self.bar_sum_h).astype(int)
        rounding_error = np.ones(N_array_compositional.shape[1], dtype=int) * self.bar_sum_h - N_array_compositional.sum(axis=0)
        # omitted に追加する
        N_array_compositional[-1, :] += rounding_error
        # 画像に追加していく
        ax_loc = [0, 0]
        bar_pos_x = 0
        ax_origin = self.get_ax_origin(ax_loc)
        for idx, array in enumerate(N_array_compositional.T):
            bar_pos_y = 0
            for c_cycle, bar_height in enumerate(array):
                self.draw_bar(bar_pos_x, bar_pos_y, bar_pos_y + bar_height, ax_origin, self.color_cycle_rgb[c_cycle])
                bar_pos_y += bar_height # （画像左上の座標が [0, 0]、ただし ax の内部では左下が原点）
            if (idx + 1) in self.tick_pos_list: # 塩基は1スタート
                tick_idx = list(self.tick_pos_list).index(idx + 1)
                self.draw_tick(bar_pos_x=bar_pos_x, tick_label=self.tick_label_list[tick_idx], ax_origin=ax_origin) # 下から積み上げていく
            # 後の idx 処理
            bar_pos_x += 1
            if bar_pos_x == self.wrap:
                bar_pos_x = 0
                ax_loc[1] += 1
                ax_origin = self.get_ax_origin(ax_loc)
    def set_legend(self, legend_list, colors=None, pos="top left"):
        if colors is None:
            colors = self.color_cycle_rgb[:len(legend_list)]
        assert len(legend_list) == len(colors)
        assert len(legend_list) == self.N_array.shape[0]
        # 場所
        if pos == "top left":
                loc_x = self.l_margin
                loc_y = self.minimum_margin
        else:
            raise Exception("error")
        # 描画していく！
        for legend, color in zip(legend_list[::-1], colors[::-1]):    # 下から積み上げていく描画に合わせる
            loc_y += self.letter_h + 3
            img_box_rgb = np.expand_dims(np.ones((self.letter_h, self.letter_h), dtype=self.dtype), axis=-1) * np.array(color)
            self.fill_img(loc_x, loc_y, img_box_rgb)
            loc_x_new = loc_x + self.letter_h + 5
            img_rgb = np.expand_dims(255 - getattr(self, legend) * 255, axis=-1) * np.ones(3, dtype=self.dtype)
            self.fill_img(loc_x_new, loc_y, img_rgb)
    def draw_bar(self, bar_pos_x, bar_pos_y, bar_pos_y_end, ax_origin, rgb_color):
        for x in range(bar_pos_x * self.bar_w, (bar_pos_x + 1) * self.bar_w):
            cur_x = ax_origin[1] + x
            for y in range(bar_pos_y, bar_pos_y_end):
                cur_y = ax_origin[0] - y
                self.paint_img(cur_x, cur_y, rgb_color)
    def get_ax_origin(self, ax_loc):
        origin_w = self.ax_origin_w_list[ax_loc[0]]
        origin_h = self.ax_origin_h_list[ax_loc[1]]
        return origin_h, origin_w
    def paint_img(self, x, y, color):
        for i, c in enumerate(color):
            self.img_array_rgb[y, x, i] = c
    def fill_img(self, x, y, img_rgb): # top left corner of the image is positioned at (x, y)
        self.img_array_rgb[y:y + img_rgb.shape[0], x:x + img_rgb.shape[1], :] = img_rgb
    def draw_tick(self, bar_pos_x, tick_label, ax_origin):
        # tick
        self.draw_bar(bar_pos_x, self.bar_sum_h, self.bar_sum_h + self.tick_h, ax_origin, rgb_color=self.tick_color)
        # label
        loc_x = ax_origin[1] + bar_pos_x * self.bar_w - self.number_w // 2
        for i, l in enumerate(str(tick_label)):
            loc_y = ax_origin[0] - self.bar_sum_h - self.tick_h - self.letter_h - 1
            img_rgb = np.expand_dims(255 - getattr(self, self.w2n[int(l)]) * 255, axis=-1) * np.ones(3, dtype=self.dtype)
            self.fill_img(loc_x, loc_y, img_rgb)
            loc_x += self.number_w + 1
    def export_as_img(self, save_path):
        PilImage.fromarray(self.img_array_rgb).save(save_path)

def draw_distributions(score_summary_df, combined_fastq):
    refseq_idx_dict = OrderedDict()
    for c in score_summary_df.columns:
        m = re.match(r"(.+) \(idx=([0-9]+)\)", c)
        if m is not None:
            refseq_idx_dict[int(m.group(2))] = m.group(1)

    # データ収集
    assignment_set_4_read_length = [[] for i in range(len(refseq_idx_dict) + 1)]   # last one is for idx=-1 (not assigned)
    assignment_set_4_q_scores = [[] for i in range(len(refseq_idx_dict) + 1)]   # last one is for idx=-1 (not assigned)
    for i, s in score_summary_df.iterrows():
        seq_id = s["seq_id"]
        assigned_refseq_idx = s["assigned_refseq_idx"]
        assigned = s["assigned"]
        if assigned == 0:
            assigned_refseq_idx = -1
        assignment_set_4_read_length[assigned_refseq_idx].append(len(combined_fastq[seq_id][0]))
        assignment_set_4_q_scores[assigned_refseq_idx].extend(combined_fastq[seq_id][1])

    # 描画パラメータ
    rows = len(refseq_idx_dict)
    columns = 3
    fig = plt.figure(figsize=(4 * columns, 2 * rows), clear=True)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    widths = [1] + [2 for i in range(columns - 1)]
    # heights = [1 for i in range(rows)]
    spec = fig.add_gridspec(ncols=columns, nrows=rows, width_ratios=widths)#, height_ratios=heights)

    ###########
    # labeles #
    ###########
    column_idx = 0
    text_wrap = 15
    for refseq_idx, refseq_name in refseq_idx_dict.items():
        ax = fig.add_subplot(spec[refseq_idx, column_idx])
        refseq_name_wrapped = "\n".join([refseq_name[i:i+text_wrap] for i in range(0, len(refseq_name), text_wrap)])
        ax.text(0.5, 0.6, refseq_name_wrapped, ha='center', va='center', wrap=True, family="monospace")
        ax.set_axis_off()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_elements = [
        Patch(facecolor=color_cycle[0], label='Focused plasmid'), 
        Patch(facecolor=color_cycle[1], label='Other plasmid(s)'), 
        Patch(facecolor="grey", label='Not assigned')
    ]
    fig.legend(handles=legend_elements, loc="lower left", borderaxespad=0)

    ############################
    # read length distribution #
    ############################
    column_idx = 1
    # assignment ごとにヒートマップを描画
    bin_unit = 100
    bins = range(0, int(np.ceil(max(max(v) if len(v) > 0 else bin_unit for v in assignment_set_4_read_length) / bin_unit) * bin_unit), bin_unit)
    for refseq_idx, refseq_name in refseq_idx_dict.items():
        hist_params = dict(
            x=assignment_set_4_read_length[-2::-1] + assignment_set_4_read_length[-1:], 
            color=[color_cycle[0] if i == refseq_idx else color_cycle[1] for i in range(len(refseq_idx_dict))][::-1] + ["grey"], 
            bins=bins, 
            histtype='bar', 
            stacked=True
        )
        # 描画
        ax0 = fig.add_subplot(spec[refseq_idx, column_idx])
        ax0.hist(**hist_params)
        ax0.set_ylabel("count")
        # # log scale
        # ax1 = fig.add_subplot(spec[refseq_idx, column_idx + 1])
        # ax1.hist(**hist_params)
        # ax1.set_yscale("log")
        # ax1.set_ylabel("count")
        if refseq_idx == 0:
            ax0.set_title("read length distribution")
            # ax1.set_title("read length distribution (log)")
        if refseq_idx == len(refseq_idx_dict) - 1:
            ax0.set_xlabel("bp")
            # ax1.set_xlabel("bp")
        else:
            ax0.set_xticklabels([])
            # ax1.set_xticklabels([])

    ########################
    # q_score distribution #
    ########################
    column_idx = 2
    for refseq_idx, refseq_name in refseq_idx_dict.items():
        hist_params = dict(
            x=assignment_set_4_q_scores[-2::-1] + assignment_set_4_q_scores[-1:], 
            color=[color_cycle[0] if i == refseq_idx else color_cycle[1] for i in range(len(refseq_idx_dict))][::-1] + ["grey"], 
            bins=np.arange(42), 
            histtype='bar', 
            stacked=True, 
            density=True
        )
        # 描画
        ax0 = fig.add_subplot(spec[refseq_idx, column_idx])
        ax0.hist(**hist_params)

        # labels
        ax0.set_ylabel("density")
        if refseq_idx == 0:
            ax0.set_title("Q-score distribution")
        if refseq_idx == len(refseq_idx_dict) - 1:
            ax0.set_xlabel("Q-score")
        else:
            ax0.set_xticklabels([])

    plt.tight_layout()

def draw_alignment_score_scatter(score_summary_df, score_threshold):
    refseq_idx_dict = OrderedDict()
    for c in score_summary_df.columns:
        m = re.match(r"(.+) \(idx=([0-9]+)\)", c)
        if m is not None:
            refseq_idx_dict[int(m.group(2))] = m.group(1)

    # アサインされたスコアまとめを追加
    for refseq_idx, refseq_name in refseq_idx_dict.items():
        col_name1 = refseq_name + f" (idx={refseq_idx}, normalized)"
        col_name2 = refseq_name + f" (idx={refseq_idx},rc, normalized)"
        score_summary_df[refseq_name] = score_summary_df.apply(lambda row: max(row[col_name1], row[col_name2]), axis=1)

    # 描画パラメータ
    rows = columns = len(refseq_idx_dict) + 1
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    focused_color1 = color_cycle[0]
    focused_color2 = color_cycle[1]
    not_assigned_color = "grey"
    fig = plt.figure(figsize=(2.5 * columns, 2.5 * rows), clear=True)
    widths = [3] + [3 for i in range(columns - 1)]
    heights = [3] + [3 for i in range(rows - 1)]
    spec = fig.add_gridspec(ncols=columns, nrows=rows, width_ratios=widths, height_ratios=heights)

    # label
    legend_elements = [
        Patch(facecolor=color_cycle[0], label='Focused plasmid'), 
        Patch(facecolor=color_cycle[1], label='Other plasmid(s)'), 
        Patch(facecolor="grey", label='Not assigned')
    ]
    fig.legend(handles=legend_elements, loc="upper left", borderaxespad=0.2)
    # fig.suptitle("alignment score scatter")

    ######################
    # score distribution #
    ######################
    diagonal_axes = []
    other_axes = []
    for (refseq_idx1, refseq_name1), (refseq_idx2, refseq_name2) in product(refseq_idx_dict.items(), refseq_idx_dict.items()):
        ax = fig.add_subplot(spec[refseq_idx1 + 1, refseq_idx2 + 1]) # 原点を左上にに取った！
        if refseq_idx1 == refseq_idx2:
            diagonal_axes.append(ax)
            hist_params = dict(
                x=[
                    score_summary_df.query("(assigned_refseq_idx == @refseq_idx1)&(assigned == 1)")[refseq_name1], 
                    score_summary_df.query("(assigned_refseq_idx != @refseq_idx1)&(assigned == 1)")[refseq_name1], 
                    score_summary_df.query("(assigned == 0)")[refseq_name1]
                ], 
                color=[focused_color1, focused_color2, not_assigned_color], 
                bins=np.linspace(0, 1, 100), 
                histtype='bar', 
                stacked=True, 
                density=True
            )
            ax.hist(**hist_params)
        else:
            other_axes.append(ax)
            scatter_params = dict(
                x=refseq_name2, 
                y=refseq_name1, 
                ax=ax, 
                s=5, 
                alpha=0.3
            )
            plot_params = dict(
                c="k", 
                linestyle="--", 
                linewidth=1
            )
            score_summary_df.query("(assigned_refseq_idx == @refseq_idx2)&(assigned == 1)").plot.scatter(color=focused_color1, **scatter_params)
            score_summary_df.query("(assigned_refseq_idx != @refseq_idx2)&(assigned == 1)").plot.scatter(color=focused_color2, **scatter_params)
            score_summary_df.query("assigned == 0").plot.scatter(color=not_assigned_color, **scatter_params)
            ax.plot((score_threshold, score_threshold), (0, score_threshold), **plot_params)
            ax.plot((0, score_threshold), (score_threshold, score_threshold), **plot_params)
            ax.plot((score_threshold, 1), (score_threshold, 1), **plot_params)
            ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        if refseq_idx2 != 0:
            # ax.yaxis.set_ticks_position('none')
            ax.set(ylabel=None)
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            if refseq_idx1 == refseq_idx2:
                ax.set_ylabel("density")
            else:
                ax.set_ylabel("normalized alignment score")
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        if refseq_idx1 != rows - 2:
            # ax.xaxis.set_ticks_position('none')
            ax.set(xlabel=None)
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("normalized alignment score")

    range_max = max(ax.get_ylim()[1] for ax in diagonal_axes)
    for ax in diagonal_axes:
        ax.set_ylim(0, range_max)

    text_wrap = 15
    for refseq_idx, refseq_name in refseq_idx_dict.items():
        ax = fig.add_subplot(spec[0, refseq_idx + 1])
        refseq_name_wrapped = "\n".join([refseq_name[i:i+text_wrap] for i in range(0, len(refseq_name), text_wrap)])
        ax.text(0.15, 0.1, refseq_name_wrapped, ha='left', va='bottom', wrap=True, family="monospace")
        ax.set_axis_off()

        ax = fig.add_subplot(spec[refseq_idx + 1, 0])
        refseq_name_wrapped = "\n".join([refseq_name[i:i+text_wrap] for i in range(0, len(refseq_name), text_wrap)])
        ax.text(0.1, 0.75, refseq_name_wrapped, ha='left', va='center', wrap=True, family="monospace")
        ax.set_axis_off()

    # set aspect after setting the ylim
    # ax = other_axes[0]
    # aspect = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    # for ax in other_axes:
    #     ax.set_aspect(aspect, adjustable='box')
    ax = diagonal_axes[0]
    aspect_diagonal = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    for ax in diagonal_axes:
        ax.set_aspect(aspect_diagonal, adjustable='box')
    fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.0, right=0.8, bottom=0.2, top=1.0)

def set_threshold_for_assignment(result_dict, my_aligner, param_dict):
    alignment_result = AlignmentResult(result_dict, my_aligner, param_dict)
    print("normalizing scores...")
    alignment_result.normalize_scores_and_apply_threshold()
    print("normalization: DONE")

    # score_summary_df = alignment_result.get_score_summary_df()
    # print("drawing figures...")
    # # draw graphical summary
    # draw_distributions(score_summary_df, my_aligner.combined_fastq)
    # draw_alignment_score_scatter(score_summary_df, alignment_result.score_threshold)

    return alignment_result

#@title # 4. Calculate consensus
bases = "ATCG-"
assert bases[-1] == "-"
letter_code_dict = {
    "ATCG":"N", # Any base
    "TCG":"B",  # Not A
    "ACG":"V",  # Not T
    "ATG":"D",  # Not C
    "ATC":"H",  # Not G
    "TG":"K",   # Keto
    "AC":"M",   # Amino
    "AG":"R",   # Purine
    "CG":"S",   # Strong
    "AT":"W",   # Weak
    "TC":"Y",   # Pyrimidine
    "A":"A", 
    "T":"T", 
    "C":"C", 
    "G":"G", 
}

def consensus_params(param_dict):
    ins_rate = param_dict["ins_rate"]
    error_rate = param_dict["error_rate"]
    del_mut_rate = param_dict["del_mut_rate"]

    from collections import defaultdict
    default_value = {b_key2:ins_rate / 4 if b_key2 != "-" else 1 - ins_rate for b_key2 in bases}

    P_N_dict_dict = defaultdict(
        lambda: default_value, 
        {   # 真のベースが b_key1 である場合に、b_key2 への mutation/deletion などが起こる確率
            b_key1:{b_key2:1 - error_rate if b_key2 == b_key1 else del_mut_rate for b_key2 in bases} for b_key1 in bases[:-1]  # remove "-" from b_key1
        }
    )
    P_N_dict_dict["-"] = default_value

    default_value_2 = {b_key2:0.2 / 4 if b_key2 != "-" else 0.8 for b_key2 in bases}
    P_N_dict_dict_2 = defaultdict(
        lambda: default_value_2, 
        {
            b_key1:{b_key2: 0.2 for b_key2 in bases} for b_key1 in bases[::-1]
        }
    )
    P_N_dict_dict_2["-"] = default_value_2

    return P_N_dict_dict, P_N_dict_dict_2

def mixed_bases(base_list):
    if len(base_list) == 1:
        return base_list[0]
    elif "-" not in base_list:
        pass
    else:
        base_list.remove("-")
    letters = ""
    for b in bases[:-1]:
        if b in base_list:
            letters += b
    return letter_code_dict[letters]

def P_N_dict_dict_2_matrix(P_N_dict_dict, bases=bases):
    r_matrix = np.empty((len(bases), len(bases)), dtype=float)
    for r, b_key1 in enumerate(bases):
        for c, b_key2 in enumerate(bases):
            r_matrix[r, c] = P_N_dict_dict[b_key1][b_key2]
    return r_matrix

class SequenceBasecallQscoreLibrary(mc.MyTextFormat):
    def __init__(self, path=None) -> None:
        self.file_version = version
        self.path = path
        self.master_params_dict = None
        self.meta_info = pd.DataFrame()
        self.alignment_summary = pd.DataFrame(columns=["=", "I", "D", "X", "H", "S", "aligned_query_len", "refseq_len", "score"], dtype=object)
        # info for saving
        self.keys = [
            ("file_version", "str"),
            ("master_params_dict", "dict"),
            ("meta_info", "df"), 
            ("alignment_summary", "df"), 
        ]
        self.variable_key_start_idx = 4
        # load
        if self.path is not None:
            self.load(load_path=self.path)
        # when pdf
        if (len(self.keys) > 4) and (self.keys[4][0] == "sum"):
            self.initialize_pdf()
    def copy_key_data(self, lib):
        for k, d_type in self.keys:
            setattr(self, k, copy.deepcopy(getattr(lib, k)))
    def get_sum(self, pdf_params_dict):
        lib_sum = self.__class__()
        lib_sum.copy_key_data(self)
        lib_sum.path = self.path.parent / (self.path.stem + f"_sum{self.path.suffix}")
        # combine all data
        combined_df, loc = self.combine_data(**pdf_params_dict)
        combined_df = combined_df.astype(object).astype(int)
        lib_sum.add_df_by_dict(OrderedDict(
            [("sum", combined_df)]
        ))
        # add params and records
        for k, v in pdf_params_dict.items():
            lib_sum.master_params_dict[k] = v
        lib_sum.alignment_summary["used_for_pdf"] = loc
        return lib_sum
    def add_df_by_dict(self, ordered_dict: OrderedDict):
        for k, v in ordered_dict.items():
            setattr(self, k, v)
            self.keys.append((k, "df"))
    def combine_data(self, threshold, thredhold_type, **kwargs):
        if thredhold_type == "score_over_aligned_query_len":
            extracted_summary = self.alignment_summary.astype(float).query("(score / aligned_query_len) > @threshold")
            loc = self.alignment_summary.astype(float).apply(lambda x: x["score"] / x["aligned_query_len"], axis=1) > threshold
        else:
            raise Exception("error!")
        assert len(extracted_summary.index) > 0
        df = pd.DataFrame(0, index=getattr(self, extracted_summary.index[0]).index, columns=getattr(self, extracted_summary.index[0]).columns, dtype=float)
        for key in  extracted_summary.index:
            new_df = getattr(self, key)
            assert all(df.index == new_df.index) and all(df.columns == new_df.columns)
            df += new_df.astype(float)
        return df, loc
    def save(self, save_path=None):
        if save_path is None:
            save_path = self.path
        else:
            self.path = save_path
        super().save(save_path)
    def load(self, load_path):
        added_keys = super().load(load_path)
        for i in range(self.variable_key_start_idx):
            assert added_keys[i] == self.keys[i]
        for k in added_keys[self.variable_key_start_idx:]:
            self.keys.append(k)
        # post-processing
        self.meta_info["refseq_info"] = self.meta_info["refseq_info"].apply(lambda x: eval(x))
        for k, v in self.master_params_dict.items():
            try:    self.master_params_dict[k] = int(v) # int
            except: self.master_params_dict[k] = v      # string
        self.path = load_path
    def register_meta_info(self, fastq, refseq_list, **kwargs):
        meta_string = self.generate_meta_info_key(fastq)
        assert meta_string not in self.meta_info.index.values
        self.meta_info.loc[meta_string, "refseq_info"] = [[f"{refseq.my_hash}:{refseq.path.name}"] for refseq in refseq_list]
        for k, v in kwargs.items():
            self.meta_info.loc[meta_string, k] = v
    def register_alignment_summary(self, summary_info_dict, **kwargs):
        for key, d in summary_info_dict.items():
            for k, v in d.items():
                self.alignment_summary.loc[key, k] = v
    @staticmethod
    def generate_meta_info_key(fastq):
        return f"{fastq.my_hash}:{fastq.path.name}"
    @staticmethod
    def summary_df_2_matrix(summary_df:pd.DataFrame, base_order=None, **kwargs):
        crushed = summary_df.sum(axis=0)
        summary_matrix = np.zeros(shape=(len(base_order), len(base_order)), dtype=float)
        for k, v in crushed.items():
            m = re.match(r"(.+)_(.+)", k)
            ref_base = m.group(1)
            query_base = m.group(2)
            summary_matrix[base_order.index(ref_base), base_order.index(query_base)] = v
        return summary_matrix

    #########################
    # PDF related functions #
    #########################
    def initialize_pdf(self):
        assert self.keys[4][0] == "sum"
        self.sum = self.sum.astype(int)
        # 確率 0 となるのを避ける
        for c in self.sum.columns:
            if c.endswith("-"):
                continue
            for i in self.sum.index:
                if i < 2:
                    continue
                if self.sum.at[i, c] == 0:
                    self.sum.at[i, c] += 1
        # bunbo
        total_events_when_true_base = {}
        for column_names, values in self.sum.items():
            true_base = column_names.split("_")[0]
            if true_base not in total_events_when_true_base.keys():
                total_events_when_true_base[true_base] = values.sum()
            else:
                total_events_when_true_base[true_base] += values.sum()
        # calc probability
        self.P_base_calling_given_true_refseq_dict = {}
        for column_names, values in self.sum.items():
            true_base = column_names.split("_")[0]
            self.P_base_calling_given_true_refseq_dict[column_names] = values.sum() / total_events_when_true_base[true_base]
        # others
        self.pdf_core = {}
        for column_names, values in self.sum.items():
            assert all(values.index == np.arange(-1, 42))
            values /= values.sum()
            # マイナス1で最後のやつにアクセスできるようにする（さすがに50も間を開けてれば、q-scoreがかぶってくることは無いでしょう…）
            values_list = list(values)[1:] + [0.0 for i in range(50)] + list(values)[:1]
            self.pdf_core[column_names] = values
    def calc_P_event_given_true_refseq(self, event, true_refseq):
        readseq, q_score = event
        key = f"{true_refseq}_{readseq}"
        return (
            self.P_base_calling_given_true_refseq_dict[key]
            * self.pdf_core[key][q_score]
        )
    def calc_consensus_error_rate(self, event_list, true_refseq, P_N_dict, bases):
        bunbo_bunshi_sum = 0
        bunshi_list = [self.calc_P_event_given_true_refseq(event, true_refseq) for event in event_list]
        bunshi_P_N = P_N_dict[true_refseq]
        # inside sum
        for base in bases:
            val = P_N_dict[base] / bunshi_P_N
            for event, bunshi in zip(event_list, bunshi_list):
                val *= self.calc_P_event_given_true_refseq(event, base) / bunshi
            bunbo_bunshi_sum += val
        return 1 - 1 / bunbo_bunshi_sum

NanoporeStats_PDF_txt = textwrap.dedent("""
    # file_version(str)
    0.2.0

    # master_params_dict(dict)
    gap_open_penalty	3
    gap_extend_penalty	1
    match_score	1
    mismatch_score	-2
    base_length_2_observe	1
    threshold	0.6
    thredhold_type	score_over_aligned_query_len

    # meta_info(df)
    	refseq_info
    omitted.fastq	['omitted.fasta']

    # alignment_summary(df)
    	=	I	D	X	H	S	aligned_query_len	refseq_len	score	used_for_pdf
    omitted_id	-1	-1	-1	-1	-1	-1	-1	-1	-1	True

    # sum(df)
    	A_A	A_T	A_C	A_G	A_-	T_A	T_T	T_C	T_G	T_-	C_A	C_T	C_C	C_G	C_-	G_A	G_T	G_C	G_G	G_-	-_A	-_T	-_C	-_G	-_-
    -1	0	0	0	0	8999	0	0	0	0	7112	0	0	0	0	11317	0	0	0	0	11289	0	0	0	0	6107251
    0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    2	426	10	18	42	0	23	340	26	11	0	62	34	372	40	0	81	22	28	380	0	50	21	30	26	0
    3	1409	36	85	163	0	35	1263	90	49	0	117	87	1477	127	0	224	48	118	1691	0	164	95	115	145	0
    4	3406	72	184	338	0	87	3048	144	73	0	213	181	3645	209	0	484	118	182	4025	0	349	231	260	341	0
    5	6053	75	227	496	0	105	5323	209	104	0	297	234	6258	281	0	709	115	261	7043	0	500	288	367	505	0
    6	8591	85	261	611	0	98	7093	215	106	0	303	252	8927	356	0	870	96	275	9851	0	656	346	455	589	0
    7	11716	87	297	714	0	103	9555	227	94	0	316	263	12075	322	0	992	105	373	13710	0	653	365	451	643	0
    8	12193	76	253	748	0	84	9965	206	86	0	276	255	12926	287	0	1002	95	316	14210	0	609	340	431	571	0
    9	11699	56	223	657	0	56	9736	188	62	0	230	201	12742	258	0	894	78	276	13925	0	571	307	394	518	0
    10	11142	68	200	532	0	52	9166	163	54	0	183	166	11856	220	0	771	56	239	13184	0	494	305	328	442	0
    11	10884	39	152	418	0	34	9128	113	37	0	139	158	11887	169	0	601	57	200	12733	0	454	264	300	400	0
    12	10689	45	113	385	0	43	9039	119	23	0	132	126	11849	143	0	541	34	172	12313	0	398	236	306	360	0
    13	10550	30	96	291	0	33	9135	87	22	0	107	127	11604	128	0	459	34	130	12436	0	323	218	239	324	0
    14	10593	39	63	264	0	26	9015	73	13	0	92	127	11872	103	0	418	35	135	12676	0	310	190	237	299	0
    15	10512	20	84	219	0	17	9192	58	16	0	71	99	12312	71	0	359	31	107	12539	0	257	219	201	271	0
    16	10780	34	65	221	0	21	9392	54	19	0	59	88	12563	66	0	310	29	86	13057	0	232	180	211	202	0
    17	10843	26	59	169	0	17	9289	63	16	0	41	61	12766	57	0	281	17	79	13102	0	227	153	181	195	0
    18	11027	13	45	148	0	17	9761	61	16	0	54	65	12987	63	0	220	18	78	13440	0	210	132	164	199	0
    19	11470	15	45	130	0	11	10142	54	8	0	47	45	13721	46	0	215	13	54	14029	0	177	142	137	192	0
    20	11936	14	36	118	0	5	10610	39	20	0	31	47	14134	38	0	195	18	44	14509	0	154	141	126	167	0
    21	12248	10	30	114	0	7	10970	27	10	0	33	38	14727	37	0	176	16	46	15021	0	133	123	157	158	0
    22	12759	13	25	88	0	9	11309	26	6	0	28	45	15618	36	0	152	16	36	15891	0	131	145	131	142	0
    23	13413	7	23	88	0	12	11814	25	5	0	27	25	16359	35	0	155	9	33	16541	0	128	143	149	136	0
    24	14338	7	19	75	0	8	13022	19	4	0	29	26	17435	23	0	127	4	27	17893	0	104	125	130	116	0
    25	15154	7	23	66	0	4	13434	16	5	0	16	26	18684	21	0	121	8	30	19223	0	98	117	156	105	0
    26	16154	4	13	52	0	5	14733	12	2	0	13	28	20212	14	0	92	7	25	20598	0	101	140	137	101	0
    27	17541	6	8	48	0	1	15844	9	2	0	14	22	21975	15	0	108	4	23	22447	0	112	140	106	105	0
    28	19297	2	11	41	0	3	17711	5	5	0	14	15	24315	17	0	75	2	21	24780	0	119	113	124	98	0
    29	21224	1	15	38	0	3	19865	16	0	0	13	23	27098	10	0	77	4	16	27354	0	99	155	132	83	0
    30	24032	4	9	21	0	3	22599	11	5	0	15	15	30276	13	0	65	1	10	30460	0	101	123	142	116	0
    31	27448	1	5	32	0	0	26257	11	0	0	5	15	33926	11	0	47	2	7	34298	0	94	136	169	109	0
    32	31166	3	3	27	0	1	30015	5	0	0	5	10	38340	7	0	44	7	5	38796	0	108	145	167	109	0
    33	35026	2	4	17	0	2	34652	8	1	0	8	9	42458	6	0	46	2	9	42678	0	115	128	127	111	0
    34	39313	2	6	13	0	0	39155	7	0	0	8	7	47036	7	0	53	0	4	47352	0	117	160	206	116	0
    35	43784	4	5	15	0	0	43501	4	2	0	5	6	51506	8	0	26	3	7	52394	0	113	167	155	111	0
    36	48503	1	3	7	0	2	48846	5	1	0	9	8	56565	3	0	28	1	6	56832	0	125	145	155	106	0
    37	53503	1	3	10	0	1	55312	5	0	0	5	3	62160	7	0	37	1	2	62695	0	121	189	121	102	0
    38	58885	0	4	9	0	0	61504	3	1	0	7	3	67806	2	0	22	4	6	67856	0	126	173	144	124	0
    39	64020	0	4	7	0	3	68218	4	1	0	3	5	73282	4	0	22	1	1	72869	0	135	170	142	105	0
    40	69864	0	1	7	0	1	74745	2	0	0	1	3	79278	0	0	17	1	3	77279	0	124	183	93	96	0
    41	612803	4	3	17	0	0	649004	6	3	0	9	9	673360	3	0	64	3	7	650025	0	684	891	599	538	0
""").strip() + "\n\n"

sbq_pdf = SequenceBasecallQscoreLibrary(io.StringIO(NanoporeStats_PDF_txt))

def calc_consensus(self, sbq_pdf, P_N_dict_dict):
    self.consensus_dict = {}
    for refseq_idx, aligned_result in enumerate(self.aligned_result_list):
        print(f"\nrefseq No. {refseq_idx}")
        consensus_seq = ""
        consensus_q_scores = []
        consensus_seq_all = ""
        consensus_q_scores_all = []
        N_bases = len(aligned_result["refseq_with_insertion"])
        for refbase_idx, refbase in enumerate(aligned_result["refseq_with_insertion"]):
            print(f"\r{refbase_idx + 1} out of {N_bases}", end="")
            seq_base_list = [i[refbase_idx] for i in aligned_result["new_seq_list_with_insertion"]]
            q_score_list = [i[refbase_idx] for i in aligned_result["new_q_scores_list_with_insertion"]]
            event_list = [(i.upper(), j) for i, j in zip(seq_base_list, q_score_list)]
            # p = sbq_pdf.calc_consensus_error_rate(event_list, true_refseq=refbase.upper(), refseq_error_rate=error_rate, bases=bases)

            P_N_dict = P_N_dict_dict[refbase.upper()]
            p_list = [
                sbq_pdf.calc_consensus_error_rate(event_list, true_refseq=B, P_N_dict=P_N_dict, bases=bases)
                for B in bases
            ]
            p = min(p_list)
            # p_idx_list = [i for i, v in enumerate(p_list) if v == p]
            consensus_base_call = mixed_bases([b for b, tmp_p in zip(bases, p_list) if tmp_p == p])

            # register
            if p >= 10 ** (-5):
                q_score = np.round(-10 * np.log10(p)).astype(int)
            elif p < 0:
                raise Exception("unknown error")
            else:
                q_score = 50
            if  consensus_base_call != "-":
                consensus_seq += consensus_base_call
                consensus_q_scores.append(q_score)

            # registre "all" results
            consensus_seq_all += consensus_base_call
            consensus_q_scores_all.append(q_score)

        # 登録
        self.consensus_dict[self.my_aligner.refseq_list[refseq_idx].path.name] = [
            consensus_seq, 
            consensus_q_scores, 
            consensus_seq_all, 
            consensus_q_scores_all
        ]
    # register settings
    self.consensus_settings = {
        "sbq_pdf_version":sbq_pdf.file_version, 
        "P_N_dict_matrix":P_N_dict_dict_2_matrix(P_N_dict_dict), 
        "bases": bases
    }

def calculate_consensus(alignment_result, param_dict):
    # params
    P_N_dict_dict, P_N_dict_dict_2 = consensus_params(param_dict)

    # execute
    alignment_result.integrate_assigned_result_info()
    print()
    print("integration: DONE")

    print("Calculating consensus with prior information...")
    calc_consensus(alignment_result, sbq_pdf, P_N_dict_dict)
    print("\n\nCalculating consensus without prior information...")
    alignment_result_2 = copy.deepcopy(alignment_result)
    calc_consensus(alignment_result_2, sbq_pdf, P_N_dict_dict_2)

    return alignment_result_2

#@title # 5. Export results

def export_results(alignment_result, alignment_result_2, intermediate_results, save_dir, group_idx, compress_as_zip=False):

    all_file_paths = []
    # export settings
    print("Exporting logs...")
    save_path_log_1 = save_dir / "log_with_prior.txt"
    log = Log(alignment_result)
    log.save(save_path_log_1)
    all_file_paths.append(save_path_log_1)

    save_path_log_2 = save_dir / "log_without_prior.txt"
    log = Log(alignment_result_2)
    log.save(save_path_log_2)
    all_file_paths.append(save_path_log_2)

    # export text
    print("Exporting alignment results...")
    save_path_list_text = []
    text_list, save_path_list_text_1 = alignment_result.export_as_text(save_dir=save_dir)
    for file_path in save_path_list_text_1:
        path = file_path.parent / (file_path.stem + ".alignment_with_prior.txt")
        os.replace(src=file_path, dst=path.as_posix())
        save_path_list_text.append(path)
    text_list, save_path_list_text_2 = alignment_result_2.export_as_text(save_dir=save_dir)
    for file_path in save_path_list_text_2:
        path = file_path.parent / (file_path.stem + ".alignment_without_prior.txt")
        os.replace(src=file_path, dst=path.as_posix())
        save_path_list_text.append(path)
    all_file_paths += save_path_list_text

    # print("Aligned Sequences")
    # for text in text_list:
    #     print(text)

    # export alignment image
    print("Exporting alignment summary...")
    bar_graph_img_list, filename_for_saving_list = alignment_result.alignment_summary_bar_graphs()
    for bar_graph_img, filename_for_saving in zip(bar_graph_img_list, filename_for_saving_list):
        save_path = save_dir / filename_for_saving
        bar_graph_img.export_as_img(save_path=save_path)
        all_file_paths.append(save_path)

    # export score_summary
    print("Exporting summary...")
    save_path_summary_score = save_dir / "summary_scores.txt"
    score_summary = alignment_result.save_score_summary(save_path=save_path_summary_score)
    all_file_paths.append(save_path_summary_score)

    # print("Score Summary")
    # print(score_summary)

    # export summary image
    print("Exporting summary svg images...")
    save_path_summary_dictribution = save_dir / "summary_distribution.svg"
    save_path_summary_scatter = save_dir / "summary_scatter.svg"
    score_summary_df = pd.read_csv(save_path_summary_score, sep="\t")
    draw_distributions(score_summary_df, alignment_result.my_aligner.combined_fastq)
    plt.savefig(save_path_summary_dictribution)
    plt.close()
    draw_alignment_score_scatter(score_summary_df, alignment_result.score_threshold)
    plt.savefig(save_path_summary_scatter)
    plt.close()
    all_file_paths.extend([save_path_summary_dictribution, save_path_summary_scatter])

    # export consensus
    print("Exporting consensus fastq files...")
    consensus_path_list = []
    consensus_path_list_1 = alignment_result.save_consensus(save_dir=save_dir, id_suffix="with_prior")
    for file_path in consensus_path_list_1:
        path = file_path.parent / (file_path.stem + ".consensus_with_prior.fastq")
        os.replace(src=file_path, dst=(path).as_posix())
        consensus_path_list.append(path)
    consensus_path_list_2 = alignment_result_2.save_consensus(save_dir=save_dir, id_suffix="without_prior")
    for file_path in consensus_path_list_2:
        path = file_path.parent / (file_path.stem + ".consensus_without_prior.fastq")
        os.replace(src=file_path, dst=(path).as_posix())
        consensus_path_list.append(path)
    all_file_paths += consensus_path_list

    # export intermediate results (allow saving even if someone deleted during the run)
    if not intermediate_results.path.exists():  # 万一途中で消されたりしてた場合
        intermediate_results.save(intermediate_results.path)
    all_file_paths.append(intermediate_results.path)

    # make new folder
    idx = 0
    if len(alignment_result.my_aligner.combined_fastq.path) == 1:
        name_stem = alignment_result.my_aligner.combined_fastq.path[0].stem
    else:
        name_stem = f"results_{group_idx}"
    results_dir = save_dir / name_stem
    while os.path.exists(results_dir.as_posix()):
        idx += 1
        results_dir = save_dir / f"{name_stem} {idx}"
    os.makedirs(results_dir)

    # move files
    for file_path in all_file_paths:
        os.replace(src=file_path, dst=(results_dir / file_path.name).as_posix())
    # # intermediate fileはコピーして残す
    # shutil.copy(results_dir / intermediate_results_filename, save_dir / intermediate_results_filename)

    # compress as zip
    zip_path = results_dir.with_suffix(".zip")
    if compress_as_zip:
        os.chdir(results_dir)
        with zipfile.ZipFile(zip_path.as_posix(), 'w') as f:
            for file_path in all_file_paths:
                f.write(file_path.name)

    print("export: DONE")
    return results_dir, zip_path

if __name__ == "__main__":
    """
    Parameters
    """
    gap_open_penalty = 3   #@param {type:"integer"}
    gap_extend_penalty = 1 #@param {type:"integer"}
    match_score = 1        #@param {type:"integer"}
    mismatch_score = -2    #@param {type:"integer"}
    score_threshold = 0.3  #@param {type:"number"}
    error_rate = 0.0001   #@param {type:"number"}
    del_mut_rate = error_rate / 4     # e.g. "A -> T, C, G, del"
    ins_rate   = 0.0001 #@param {type:"number"}   # 挿入は独立に考える？

    param_dict = {i:globals()[i] for i in (
        'gap_open_penalty', 
        'gap_extend_penalty', 
        'match_score', 
        'mismatch_score', 
        'score_threshold', 
        'error_rate', 
        'del_mut_rate', 
        'ins_rate'
    )}

    """
    Example for calculating consensu sequence
    """
    true_refseq = "T"   # prior info
    event_list = [("T", 15), ("A", 10), ("A", 3), ("A", 20)]   # obtained data

    true_refseq = "G"   # prior info
    event_list = [("C", 23), ("C", 26), ("G", 40)]   # obtained data


    sbq_pdf = SequenceBasecallQscoreLibrary(io.StringIO(NanoporeStats_PDF_txt))
    P_N_dict_dict, P_N_dict_dict_2 = consensus_params(param_dict)

    P_N_dict = P_N_dict_dict_2[true_refseq.upper()] # P_N_dict_dict for with prior, P_N_dict_dict_2 for without prior
    p_list = [
        sbq_pdf.calc_consensus_error_rate(event_list, true_refseq=B, P_N_dict=P_N_dict, bases=bases)
        for B in bases
    ]
    p = min(p_list)
    q_score = np.round(-10 * np.log10(p)).astype(int)
    consensus_base_call = mixed_bases([b for b, tmp_p in zip(bases, p_list) if tmp_p == p])
    print(consensus_base_call, q_score)
    print(p_list)

    """
    Example of execution
    """
    # files
    plasmid_map_dir = Path("./resources/demo_data/my_plasmid_maps_dna")
    refseq_file_namd_list = [
        "M32_pmNeonGreen-N1.dna", 
        "M38_mCherry-Spo20.dna", 
        "M42_GFP-PASS_vecCMV.dna", 
        "M43_iRFP713-PASS_vecCMV.dna", 
        "M160_P18-CIBN-P2A-CRY2-mCherry-PLDs17_pcDNA3.dna", 
        "M161_CRY2-mCherry-PLDs27-P2A-CIBN-CAAX_pcDNA3.dna", 
    ]
    refseq_file_path_list = []
    for refseq_file_name in refseq_file_namd_list:
        plasmid_map_path = list(plasmid_map_dir.rglob(refseq_file_name))
        assert len(plasmid_map_path) == 1
        refseq_file_path_list.append(plasmid_map_path[0])
    fastq_file_path = Path("./resources/demo_data/my_fastq_files/Uematsu_n7x_1_MU-test1.fastq")
    assert fastq_file_path.exists()
    save_dir = Path("./resources/demo_data/results_analysis")   # 2:17:10.726045
    assert save_dir.exists()

    group_idx = 0

    t0 = datetime.now()

    refseq_list, combined_fastq = organize_files([fastq_file_path], refseq_file_path_list)
    combined_fastq = combined_fastq[:2]
    # 2. Execute alignment: load if any previous score_matrix if possible
    result_dict, my_aligner, intermediate_results = execute_alignment(refseq_list, combined_fastq, param_dict, save_dir)
    # 3. Set threshold for assignment
    alignment_result = set_threshold_for_assignment(result_dict, my_aligner, param_dict)
    # 4. Calculate consensus
    alignment_result_2 = calculate_consensus(alignment_result, param_dict)
    # 5. Export results
    results_dir, zip_path = export_results(alignment_result, alignment_result_2, intermediate_results, save_dir, group_idx)

    t1 = datetime.now()
    print(t1 - t0)


