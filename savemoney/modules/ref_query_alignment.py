# * coding: utf8 *

#@title # 1. Upload and select files

app_name = "SAVEMONEY"
version = "0.2.0"
description = "written by MU"

# total number of reads に対する insertion の割合が一定以上（2割とか3割とか？）になった場合を danger_zone と設定
# total number of reads に対する empty     の割合が一定以上（2割とか3割とか？）になった場合を empty_zone と設定
# danger_zone の周囲もdangerであるかを調べる

import re
import parasail
import numpy as np
import scipy.stats as stats

from . import my_classes as mc
from .cython_functions import alignment_functions as af

class MyAlignerBase(mc.MyCigarBase):
    alignment_algorithm_version = "aa_0.2.1"
    def __init__(self, param_dict):
        # params
        self.param_dict = param_dict
        self.gap_open_penalty = param_dict["gap_open_penalty"]
        self.gap_extend_penalty = param_dict["gap_extend_penalty"]
        self.match_score = param_dict["match_score"]
        self.mismatch_score = param_dict["mismatch_score"]
    @property
    def my_custom_matrix(self):
        return parasail.matrix_create("ACGT", self.match_score, self.mismatch_score)
    def my_special_sw(self, ref_seq, query_seq):
        result = parasail.sw_trace(query_seq, ref_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
        my_result = MyResult(result)
        my_result.my_cigar = (
            "H" * my_result.beg_ref + 
            "S" * my_result.beg_query + 
            my_result.my_cigar + 
            "S" * (len(query_seq) - my_result.end_query - 1) + 
            "H" * (len(ref_seq) - my_result.end_ref - 1)
        )
        my_result.beg_query = 0
        my_result.end_query = len(query_seq) - 1
        return my_result
    def nw_trace(self, ref_seq, query_seq):
        result = parasail.nw_trace(query_seq, ref_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
        return MyResult(result)
    def my_special_sg_qe_de(self, ref_seq, query_seq):
        """
        ATCGATCGATCGATCG
        ATCGATCG--------
        or
        ATCGATCG--------
        ATCGATCGATCGATCG
        """
        result = parasail.sg_qe_de_trace(query_seq, ref_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
        my_result = MyResult(result)
        assert my_result.beg_ref == 0
        assert my_result.beg_query == 0
        my_result.set_soft_clipping(self.gap_open_penalty, self.gap_extend_penalty, self.match_score, self.mismatch_score, side="beg")
        return my_result
    def my_special_sg_qb_db(self, ref_seq, query_seq):
        """
        ATCGATCGATCGATCG
        --------ATCGATCG
        or
        --------ATCGATCG
        ATCGATCGATCGATCG
        """
        result = parasail.sg_qb_db_trace(query_seq, ref_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
        my_result = MyResult(result)
        assert my_result.end_ref == len(ref_seq) - 1
        assert my_result.end_query == len(query_seq) - 1
        my_result.set_soft_clipping(self.gap_open_penalty, self.gap_extend_penalty, self.match_score, self.mismatch_score, side="end")
        return my_result
    # query の間はギャップペナルティを与えないアラインメント
    def my_special_DP(self, query_seq_1: str, query_seq_2: str, ref_seq: str):
        ################
        # アラインメント #
        ################
        if len(query_seq_1) > 0:
            my_result_1 = self.my_special_sg_qe_de(ref_seq, query_seq_1)
        else:
            my_result_1 = MyResult()
            my_result_1.my_cigar = "H" * len(ref_seq)
            my_result_1.beg_ref = -1
            my_result_1.end_ref = -1
        if len(query_seq_2) > 0:
            my_result_2 = self.my_special_sg_qb_db(ref_seq, query_seq_2)
        else:
            my_result_2 = MyResult()
            my_result_2.my_cigar = "H" * len(ref_seq)
            my_result_2.beg_ref = len(ref_seq)
            my_result_2.end_ref = len(ref_seq)
        ########
        # 結合 #
        ########
        if my_result_2.beg_ref - my_result_1.end_ref - 1 > 0:
            """
            ref         ?????????????????????     ??????         ?????????????????????
            my_cigar_1  =====ALIGNED-SEQ=====SSSSSHHHHHH         HHHHHHHHHHHHHHHHHHHHH
            my_cigar_2  HHHHHHHHHHHHHHHHHHHHH     HHHHHHSSSSSSSSS=====ALIGNED-SEQ=====
            combined    =====ALIGNED-SEQ=====SSSSSHHHHHHSSSSSSSSS=====ALIGNED-SEQ=====
            """
            len_H = my_result_2.beg_ref - my_result_1.end_ref - 1
            my_result = MyResult()
            my_result.my_cigar = (
                my_result_1.my_cigar[:len(my_result_1.my_cigar) - my_result_1.my_cigar.count("H")] + 
                "H" * len_H + 
                my_result_2.my_cigar[my_result_2.my_cigar.count("H"):]
            )
        else:
            """ ベストな区切りを探すよ
            ref         ??????????????????????????????????????????????????
            my_cigar_1  =====ALIGNED-SEQ===========SSSSSHHHHHHHHHHHHHHHHHH
            my_cigar_2  HHHHHHHHHHHHSSSSSSSSS=============ALIGNED-SEQ=====
            """
            from . import msa
            my_msa = msa.MyMSA.generate_msa(mc.MySeq(ref_seq), [query_seq_1, query_seq_2], q_scores_list=[[-1] * len(query_seq_1), [-1] * len(query_seq_2)], my_cigar_list=[my_result_1.my_cigar, my_result_2.my_cigar], param_dict=self.param_dict)
            # find best position to cut
            my_cigar_1_aligned, my_cigar_2_aligned = my_msa.my_cigar_list_aligned   # 逆端に O が入ることはあっても H が入ることはない
            cumsum_scores_1_from_left = self.cumsum_my_cigar_scores(my_cigar_1_aligned)
            cumsum_scores_2_from_right = self.cumsum_my_cigar_scores(my_cigar_2_aligned[::-1])[::-1]
            idx_to_cut = (np.array(cumsum_scores_1_from_left) + np.array(cumsum_scores_2_from_right)).argmax()
            len_S_1 = (
                len(cumsum_scores_1_from_left) - 1 - idx_to_cut - 
                my_cigar_1_aligned[idx_to_cut:].count("D") - 
                my_cigar_1_aligned[idx_to_cut:].count("N") - 
                my_cigar_1_aligned[idx_to_cut:].count("H") - 
                my_cigar_1_aligned[idx_to_cut:].count("O")
            )
            len_S_2 = (
                idx_to_cut - 
                my_cigar_2_aligned[:idx_to_cut].count("D") - 
                my_cigar_2_aligned[:idx_to_cut].count("N") - 
                my_cigar_2_aligned[:idx_to_cut].count("H") - 
                my_cigar_2_aligned[:idx_to_cut].count("O")
            )
            my_cigar = my_cigar_1_aligned[:idx_to_cut] + "S" * len_S_1 + "S" * len_S_2 + my_cigar_2_aligned[idx_to_cut:]
            my_cigar = my_cigar.replace("N", "").replace("O", "")
            """ # assert my_cigar.count("H") == 0 は不要: 例えば下記の場合に H が入る
            ref_seq     ACTTGTTTAGAAGA    AATAAT GGAA
            query_seq_1 ACTTGTTTAGAAGAGATG
            my_cigar_1  ==============SSSSHHHHHHOHHHH
            query_seq_2 ACTTGTTT    GA    GATAATGGGAG
            my_cigar_2  ========DDDD==NNNNX=====I===X
            combined    ==============SSSSH=====I===X
            """
            my_result = MyResult()
            my_result.my_cigar = my_cigar
        return my_result
    def cumsum_my_cigar_scores(self, my_cigar):
        cumsum_scores = [0]
        for LLL, L in self.generate_cigar_iter(my_cigar):
            N = len(LLL)
            if L == "=":
                cumsum_scores.extend(
                    np.arange(1, N + 1) * self.match_score + cumsum_scores[-1]
                )
            elif L == "X":
                cumsum_scores.extend(
                    np.arange(1, N + 1) * self.mismatch_score + cumsum_scores[-1]
                )
            elif L in "DI":
                s = cumsum_scores[-1] - self.gap_open_penalty
                cumsum_scores.extend(
                    [s] + list(s - np.arange(1, N) * self.gap_extend_penalty)
                )
            elif L in "SHON":
                cumsum_scores.extend([cumsum_scores[-1] for i in range(N)])
            else:
                raise Exception(f"error: {L}")
        return cumsum_scores

class MyOptimizedAligner(MyAlignerBase, mc.AlignmentBase):
    default_repeat_max = 5
    percentile_factor = 0.1 # 低いほど conserved region の quality は上がるが、計算時間がかかる
    omit_too_long = 1.75    # omit_too_long: 1.5-2.0 くらいの値: self.calc_circular_conserved_region の elif 文参照
    def __init__(self, ref_seq, param_dict) -> None:
        super().__init__(param_dict)
        self.ref_seq = ref_seq
        self.ref_seq_v = np.array([ord(char) for char in ref_seq])
        self.N_ref = len(self.ref_seq_v)
        self.ref_seq_v_repeated_list = [
            np.hstack([self.ref_seq_v for i in range(repeat)]).astype(np.int64)
            for repeat in range(2, self.default_repeat_max + 1)
        ]
        self.k_list = np.arange(0, self.N_ref)
        x_percentile = 1 / self.N_ref * self.percentile_factor # 偶然 threshold を超える position が 0.1個/ref_seq 以下になるようにする
        self.sigma = stats.norm.ppf(1 - x_percentile, loc=0, scale=1)  # 累積分布関数の逆関数
        self.is_all_ATCG = all([b.upper() in "ATCG" for b in ref_seq])
    def calc_circular_conserved_region(self, query_seq, omit_too_long=False):
        conserved_regions = self.calc_circular_conserved_region_core(query_seq)
        if len(conserved_regions) == 0:
            return None
        elif omit_too_long and (self.N_ref * self.omit_too_long <= len(query_seq)):
            return None
        else:
            longest_path_trace, longest_path_score = SearchLongestPath.exec(conserved_regions, N_ref=self.N_ref, N_query=len(query_seq))
            # 順序づけ
            conserved_regions.tidy_conserved_regions(longest_path_trace)
            return conserved_regions
    def execute_circular_alignment_using_conserved_regions(self, query_seq, conserved_regions):
        N_query = len(query_seq)
        # set offset
        query_end_idx_after_offset = conserved_regions.set_tidy_offset(len(self.ref_seq), N_query)   # ラストの non_conserved_region もしくは ラストの conserved_region の末端に query_end_idx_after_offset が含まれることを保証する
        self.ref_seq.set_offset(conserved_regions.ref_seq_offset)
        query_seq.set_offset(conserved_regions.query_seq_offset)
        # execute alignment for non_conserved_regiong
        result_master = MyResult()
        for length_of_previous_conserved_region, non_conserved_ref_start, non_conserved_ref_end, non_conserved_query_start, non_conserved_query_end in conserved_regions.iter_regions():
            # print(length_of_previous_conserved_region, non_conserved_ref_start, non_conserved_ref_end, non_conserved_query_start, non_conserved_query_end)
            # 最後のループはは常に -1 になっているはず (conserved_regions.iter_regions() の中で assert している)
            if non_conserved_ref_end != -1:
                # non_conserved_ref の長さが 0 の場合
                if non_conserved_ref_start > non_conserved_ref_end:
                    assert non_conserved_ref_start - 1 == non_conserved_ref_end
                    result_master.append_my_cigar("=" * length_of_previous_conserved_region)
                    result_master.append_my_cigar("I" * (non_conserved_query_end - non_conserved_query_start + 1))
                    continue
                # non_conserved_query の長さが 0 の場合（non_conserved_ref, non_conserved_query の長さが同時に 0 になることがあるが、多分大丈夫なはず）
                elif non_conserved_query_start > non_conserved_query_end:
                    assert non_conserved_query_start - 1 == non_conserved_query_end
                    result_master.append_my_cigar("=" * length_of_previous_conserved_region)
                    result_master.append_my_cigar("D" * (non_conserved_ref_end - non_conserved_ref_start + 1))
                    continue
                else:
                    ref_seq_extracted = self.ref_seq[non_conserved_ref_start:non_conserved_ref_end+1]
                    query_seq_extracted = query_seq[non_conserved_query_start:non_conserved_query_end+1]
                    # execute alignment # non_conserved_region の両端は完全一致している conserved_region のはずなので、ローカルアラインメントではない！
                    result = parasail.nw_trace(query_seq_extracted, ref_seq_extracted, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
                    my_result = MyResult(result)
            # ラストループ
            else:
                assert non_conserved_query_end == -1
                # non_conserved_ref の長さが 0 の場合
                if non_conserved_ref_start == self.N_ref:
                    result_master.append_my_cigar("=" * length_of_previous_conserved_region)
                    result_master.append_my_cigar("S" * (N_query - non_conserved_query_start))  # ソフトクリップ
                    continue
                # non_conserved_query の長さが 0 の場合（non_conserved_ref, non_conserved_query の長さが同時に 0 になることがあるが、多分大丈夫なはず）
                elif non_conserved_query_start == N_query:
                    result_master.append_my_cigar("=" * length_of_previous_conserved_region)
                    result_master.append_my_cigar("H" * (self.N_ref - non_conserved_ref_start)) # ハードクリップ
                    continue
                else:
                    ref_seq_extracted = self.ref_seq[non_conserved_ref_start:]
                    query_seq_extracted = query_seq[non_conserved_query_start:]
                    # execute alignment
                    query_seq_extracted_1 = query_seq[non_conserved_query_start:query_end_idx_after_offset + 1]
                    query_seq_extracted_2 = query_seq[query_end_idx_after_offset + 1:]
                    my_result = self.my_special_DP(
                        query_seq_extracted_1, 
                        query_seq_extracted_2, 
                        ref_seq_extracted, 
                    )

            # 結果を追加
            result_master.append_my_cigar("=" * length_of_previous_conserved_region)
            result_master.append_my_cigar(my_result.my_cigar)

        # self.print_alignment(self.ref_seq, query_seq, result_master.my_cigar)

        # ref_seq_offset が 0 になるような感じで、queyr_seq_offset を調整しよう (conserved_regions.query_seq_offset の場合は conserved_regions.ref_seq_offset != 0 であることがほとんど)
        new_query_seq_offset = result_master.set_ref_seq_offset_as_0(conserved_regions.ref_seq_offset, conserved_regions.query_seq_offset)
        result_master.set_cigar_score(self.gap_open_penalty, self.gap_extend_penalty, self.match_score, self.mismatch_score)

        # assertion
        self.ref_seq.set_offset(0)
        query_seq.set_offset(new_query_seq_offset)
        self.assert_alignment(self.ref_seq, query_seq, result_master.my_cigar)

        # prepare for nexe analysis
        query_seq.set_offset(0)

        # プラスミドダイマーの場合
        # if result_master.score < 0:
        #     # view alignment
        #     print()
        #     self.ref_seq.set_offset(0)
        #     query_seq.set_offset(new_query_seq_offset)
        #     self.print_alignment(self.ref_seq, query_seq, result_master.my_cigar)
        #     quit()
        return result_master
    def calc_circular_conserved_region_core(self, query_seq):
        # 文字列を数字に変換
        query_seq_v = np.array([ord(char) for char in query_seq], dtype=np.int64)
        N_query = len(query_seq_v)
        repeat = 1 + np.ceil((N_query - 1) / self.N_ref).astype(int)
        # アラインメントのたびに ref_seq_v_repeated を作らないようにする：本来は else 中の hstack のみで ok だけど、それだと時間がかかるかな？
        if repeat <= self.default_repeat_max:
            ref_seq_v_repeated = self.ref_seq_v_repeated_list[repeat - 2]
        else:
            ref_seq_v_repeated = np.hstack([self.ref_seq_v for i in range(repeat)]).astype(np.int64)

        #######################
        # k座位分だけ query をずらして並べたときの類似性（repeated ref_seq を固定して、query_seq のオフセットを右にずらしていくイメージ）
        #######################
        """
        # python version
        """
        # result_array = [(ref_seq_v_repeated[k:k + N_query] == query_seq_v).sum() for k in self.k_list]
        """
        # cython version 1
        """
        # result_array = np.empty_like(self.k_list, dtype=np.int64)
        # af.k_mer_offset_analysis(ref_seq_v_repeated, query_seq_v, result_array)
        """
        # cython version 2 (0.012740850448608398s)
        """
        result_array = af.k_mer_offset_analysis_2(ref_seq_v_repeated, query_seq_v, self.N_ref, len(query_seq_v))

        # 閾値を超える類似性が得られた場合の k座位を取得 (0.001313924789428711s)
        selected_k_list = np.where(result_array > np.median(result_array) + self.sigma * np.std(result_array))[0]

        # import matplotlib.pyplot as plt
        # plt.plot(self.k_list, result_array)
        # plt.axhline(np.median(result_array), color="result_array")
        # plt.axhline(np.median(result_array) + self.sigma * np.std(result_array), color="result_array")
        # plt.axhline(np.median(result_array) - self.sigma * np.std(result_array), color="result_array")
        # plt.show()

        # ウィンドウ解析
        conserved_region_start_idx_list_query = []
        conserved_region_end_idx_list_query = []
        conserved_region_start_idx_list_ref = []
        conserved_region_end_idx_list_ref = []

        for i, k in enumerate(selected_k_list):
            start_idx_sublist, end_idx_sublist = WindowAnalysis.exec(ref_seq_v_repeated[k:k + N_query], query_seq_v)
            conserved_region_start_idx_list_ref.extend(start_idx_sublist + k)
            conserved_region_end_idx_list_ref.extend(end_idx_sublist + k)
            conserved_region_start_idx_list_query.extend(start_idx_sublist)
            conserved_region_end_idx_list_query.extend(end_idx_sublist)
        conserved_regions_ref = np.vstack((conserved_region_start_idx_list_ref, conserved_region_end_idx_list_ref)).T
        conserved_regions_ref %= self.N_ref # リピートが 3 以上の場合は、複数回引く必要がある：でないと index エラーになる
        conserved_regions_query = np.vstack((conserved_region_start_idx_list_query, conserved_region_end_idx_list_query)).T
        return ConservedRegions(np.hstack((conserved_regions_ref, conserved_regions_query)))

class MyResult(mc.MyCigarBase):
    def __init__(self, parasail_result=None) -> None:
        if parasail_result is not None:
            self.my_cigar = self.cigar_to_my_cigar(parasail_result.cigar.decode.decode("ascii"))
            self.score = parasail_result.score
            self.beg_ref = parasail_result.cigar.beg_ref
            self.beg_query = parasail_result.cigar.beg_query
            self.end_ref = parasail_result.end_ref
            self.end_query = parasail_result.end_query
        else:
            self.my_cigar = ""
            self.score = 0
            self.beg_ref = 0
            self.beg_query = 0
            self.end_ref = 0
            self.end_query = 0
        # ref_seq_offset = 0 となるようなアラインメントが行われた時にのみ値が登録される (see self.set_ref_seq_offset_as_0)
        self.new_query_seq_offset = None
    def __str__(self) -> str:
        return f"cigar\t{self.my_cigar}\nscore\t{self.score}\nref\t{self.beg_ref}\t{self.end_ref}\nquery\t{self.beg_query}\t{self.end_query}\nnew_query_seq_offset\t{self.new_query_seq_offset}"
    def __len__(self):
        return len(self.my_cigar)
    def append_my_cigar(self, my_cigar):
        self.my_cigar += my_cigar
    @property
    def cigar(self):
        return "".join(f"{len(LLL)}{L}" for LLL, L in self.generate_cigar_iter(self.my_cigar))
    def set_cigar_score(self, gap_open_penalty, gap_extend_penalty, match_score, mismatch_score):
        self.score = 0
        for LLL, L in self.generate_cigar_iter(self.my_cigar):
            if L == "=":
                self.score += match_score * len(LLL)
            elif L == "X":
                self.score += mismatch_score * len(LLL)
            elif L in "DI":
                self.score -= gap_open_penalty + gap_extend_penalty * (len(LLL) - 1)
            elif L in "SHE":
                pass
            else:
                raise Exception(f"unknown letter code: {L}")
    def set_ref_seq_offset_as_0(self, ref_seq_offset, query_seq_offset):
        aligned_ref_seq_offset = self.get_aligned_ref_seq_offset(ref_seq_offset)
        aligned_query_seq_offset = self.get_aligned_query_seq_offset(query_seq_offset)
        diff = aligned_query_seq_offset - aligned_ref_seq_offset
        if aligned_query_seq_offset > aligned_ref_seq_offset:
            if aligned_ref_seq_offset != 0:
                new_query_seq_offset = diff - self.my_cigar[-aligned_query_seq_offset:-aligned_ref_seq_offset].count("D") - self.my_cigar[-aligned_query_seq_offset:-aligned_ref_seq_offset].count("H")
            else:
                new_query_seq_offset = diff - self.my_cigar[-aligned_query_seq_offset:].count("D") - self.my_cigar[-aligned_query_seq_offset:].count("H")
        else:
            # aligned_query_seq_offset != 0 の場合
            if aligned_query_seq_offset != 0:
                new_query_seq_offset = diff + self.my_cigar[-aligned_ref_seq_offset:-aligned_query_seq_offset].count("D") + self.my_cigar[-aligned_ref_seq_offset:-aligned_query_seq_offset].count("H")
            # aligned_ref_seq_offset != aligned_query_seq_offset == 0 の場合
            elif aligned_ref_seq_offset != 0:
                new_query_seq_offset = diff + self.my_cigar[-aligned_ref_seq_offset:].count("D") + self.my_cigar[-aligned_ref_seq_offset:].count("H")
            # aligned_ref_seq_offset == aligned_query_seq_offset == 0 の場合
            else:
                new_query_seq_offset = 0
        # 自身のアップデート
        self.my_cigar = self.my_cigar[-aligned_ref_seq_offset:] + self.my_cigar[:-aligned_ref_seq_offset]
        self.new_query_seq_offset = new_query_seq_offset
        return new_query_seq_offset
    def get_aligned_ref_seq_offset(self, ref_seq_offset):
        if ref_seq_offset == 0:
            return 0
        assert ref_seq_offset > 0
        aligned_ref_seq_offset = ref_seq_offset
        previous_N_ins = 0
        N_ins = self.my_cigar[-aligned_ref_seq_offset:].count("I") + self.my_cigar[-aligned_ref_seq_offset:].count("S")
        while ref_seq_offset != aligned_ref_seq_offset - N_ins:
            aligned_ref_seq_offset += N_ins - previous_N_ins
            previous_N_ins = N_ins
            N_ins = self.my_cigar[-aligned_ref_seq_offset:].count("I") + self.my_cigar[-aligned_ref_seq_offset:].count("S")
        return aligned_ref_seq_offset
    def get_aligned_query_seq_offset(self, query_seq_offset):
        if query_seq_offset == 0:
            return 0
        assert query_seq_offset > 0
        aligned_query_seq_offset = query_seq_offset
        previous_N_del = 0
        N_del = self.my_cigar[-aligned_query_seq_offset:].count("D") + self.my_cigar[-aligned_query_seq_offset:].count("H")
        while query_seq_offset != aligned_query_seq_offset - N_del:
            aligned_query_seq_offset += N_del - previous_N_del
            previous_N_del = N_del
            N_del = self.my_cigar[-aligned_query_seq_offset:].count("D") + self.my_cigar[-aligned_query_seq_offset:].count("H")
        return aligned_query_seq_offset
    def to_dict(self):
        d = {
            "cigar":self.cigar,     # 出力容量節約の為、my_cigarr でなく、cigar で出力する
            "score":self.score, 
            "new_query_seq_offset":self.new_query_seq_offset, 
        }
        return d
    def apply_dict_params(self, d):
        setattr(self, "score", int(d["score"]))
        setattr(self, "my_cigar", self.cigar_to_my_cigar(d["cigar"]))
        new_query_seq_offset = d["new_query_seq_offset"]
        if new_query_seq_offset == "None":
            setattr(self, "new_query_seq_offset", None)
        else:
            setattr(self, "new_query_seq_offset", int(new_query_seq_offset))
    # for pre-survey
    def calc_levenshtein_distance(self):
        return len(self.my_cigar) - self.my_cigar.count("=")
    # # for semi-global alignment
    # def organize_end(self):
    #     assert self.beg_ref == self.beg_query == 0
    #     m = re.search(r"^[=XDI]+?(I*)(D*)$", self.my_cigar)
    #     len_endI = len(m.group(1))
    #     len_endD = len(m.group(2))
    #     if len_endD == 0:
    #         len_endID = len_endI
    #         end_cigar = "S"
    #     else:
    #         len_endID = len_endD
    #         end_cigar = "H"
    #     alignment_len = len(self.my_cigar) - len_endID
    #     self.end_ref = alignment_len - self.my_cigar[:alignment_len].count("I") - 1
    #     self.end_query = alignment_len - self.my_cigar[:alignment_len].count("D") - 1
    #     assert self.end_ref >= 0
    #     assert self.end_query >= 0
    #     self.my_cigar = self.my_cigar[:alignment_len] + end_cigar * len_endID
    # def organize_beg(self):
    #     m = re.search(r"^(D*)(I*)[=XDI]+?$", self.my_cigar)
    #     len_begI = len(m.group(1))
    #     len_begD = len(m.group(2))
    #     if len_begD == 0:
    #         len_begID = len_begI
    #         end_cigar = "S"
    #     else:
    #         len_begID = len_begD
    #         end_cigar = "H"
    #     self.beg_ref = len_begID - self.my_cigar[:len_begID].count("I")
    #     self.beg_query = len_begID - self.my_cigar[:len_begID].count("D")
    #     self.my_cigar = "H" * self.beg_ref + self.my_cigar[self.beg_ref:]
    #     return self.beg_ref # = len_begD
    def set_soft_clipping(self, gap_open_penalty, gap_extend_penalty, match_score, mismatch_score, side):
        if side == "end":
            my_cigar = self.my_cigar[::-1]
        elif side == "beg":
            my_cigar = self.my_cigar
        else:
            raise Exception(f"error: {side}")
        my_cigar, len_aligned_ref = self.set_soft_clipping_core(my_cigar, gap_open_penalty, gap_extend_penalty, match_score, mismatch_score)
        if side == "end":
            self.my_cigar = my_cigar[::-1]
            self.beg_ref = my_cigar.count("H")
            self.beg_query = 0
        elif side == "beg":
            self.my_cigar = my_cigar
            self.end_ref = len_aligned_ref - 1
            self.end_query = len(my_cigar) - my_cigar.count("H") - my_cigar.count("D") - 1
    @staticmethod
    def set_soft_clipping_core(my_cigar, gap_open_penalty, gap_extend_penalty, match_score, mismatch_score):
        """ 下記のようなイメージにする
        ### PRE ###
        cigar   ==I=DX===XXXXXDDD
        ref     AA-AAAAAATTTTTTTT
        query   AAAA-TAAACCCCC

        ### POST ###
        cigar   ==I=DX===SSSSSHHHHHHHH
        ref     AA-AAAAAA     TTTTTTTT
        query   AAAA-TAAACCCCC
        """
        score_list = []
        cur_score = 0
        previous_L = ""
        for L in my_cigar:
            if L == "=":
                cur_score += match_score
            elif L == "X":
                cur_score += mismatch_score
            elif L in "ID":
                if previous_L == L:
                    cur_score -= gap_extend_penalty
                else:
                    cur_score -= gap_open_penalty
            else:
                raise Exception(f"error: {L}")
            score_list.append(cur_score)
            previous_L = L
        max_score_idx = np.argmax(score_list)
        len_aligned = max_score_idx + 1
        len_aligned_and_preS = len(score_list)
        len_aligned_and_preH = len(my_cigar)
        if score_list[max_score_idx] < 0:
            len_query = len_aligned_and_preS - my_cigar[:len_aligned_and_preS].count("D")
            len_ref = len_aligned_and_preH - my_cigar.count("I")
            new_my_cigar = "S" * len_query  + "H" * len_ref
            len_aligned_ref = 0
        else:
            len_query_S = len_aligned_and_preS - len_aligned - my_cigar[len_aligned:len_aligned_and_preS].count("D")
            len_ref_H = len_aligned_and_preH - len_aligned - my_cigar[len_aligned:].count("I")
            new_my_cigar = my_cigar[:len_aligned] + "S" * len_query_S + "H" * len_ref_H
            len_aligned_ref = len_aligned - my_cigar[:len_aligned].count("I")
        return new_my_cigar, len_aligned_ref
    ##############################
    # Used for polishing process #
    ##############################
    def get_mutated_regions(self):
        return [(m.group(0), m.start(), m.end() - 1, self.my_cigar[:m.start()].count("I"), self.my_cigar[:m.start()].count("D")) for m in re.finditer(r"([IDX]+)", self.my_cigar)]
    def get_corresponding_query_idx(self, ref_idx):
        cur_query_idx = -1
        cur_ref_idx = -1
        for c in self.my_cigar:
            if c in "=X":
                cur_ref_idx += 1
                cur_query_idx += 1
            elif c == "I":
                cur_query_idx += 1
            elif c == "D":
                cur_ref_idx += 1
            else:
                raise Exception(f"error: {c}")
            if cur_ref_idx == ref_idx:
                if c == "I":
                    raise Exception(f"error {c}")
                else:
                    return cur_query_idx
        print(cur_ref_idx, cur_query_idx)
        raise Exception("error")

class ConnectionMatrix():
    def __init__(self, connection_matrix, initial_node_idx=0, initial_node_score=0) -> None:
        self.connection_matrix = np.copy(connection_matrix)
        self.initial_node_idx = initial_node_idx
        self.initial_node_score = initial_node_score

        self.N_node = self.connection_matrix.shape[0]
        self.longest_path_trace_list = [None for i in range(self.N_node)]
        self.longest_path_trace_list[self.initial_node_idx] = [self.initial_node_idx]
        self.longest_path_score_list = [0 for i in range(self.N_node)]
        self.longest_path_score_list[self.initial_node_idx] += self.initial_node_score

        self.connection_matrix_remained_to_be_analyzed = None
    def remove_unreachable_branches(self):
        # 便宜上数値を1つだけ適当な値を代入（このクラスメソッドの最後で元に戻されるよ）
        self.connection_matrix[self.initial_node_idx, self.initial_node_idx] = 100  # 値は何でも良いが、必要

        N_idx_with_no_input_previous = []
        update = True
        while update:
            # 入力が無いノードへは行けない
            idx_with_no_input = np.where(self.connection_matrix.sum(axis=0) == 0)[0]
            N_idx_with_no_input = len(idx_with_no_input)
            if N_idx_with_no_input == N_idx_with_no_input_previous:
                update = False
            else:
                for idx in idx_with_no_input:
                    self.connection_matrix[idx, :] = 0
                # 後処理
                N_idx_with_no_input_previous = N_idx_with_no_input

        # 後処理
        self.connection_matrix[self.initial_node_idx, self.initial_node_idx] = 0
    def execute_dijkstras_algorithm(self):
        self.connection_matrix_remained_to_be_analyzed = self.connection_matrix > 0
        newly_determined_idx_list = [self.initial_node_idx]
        while self.connection_matrix_remained_to_be_analyzed.any(axis=None):
            determined_idx_list = []
            for idx in newly_determined_idx_list:
                for col in np.where(self.connection_matrix[idx, :] > 0)[0]:
                    score_so_far = self.longest_path_score_list[idx]
                    acquired_score = self.connection_matrix[idx, col]
                    total_score = score_so_far + acquired_score
                    # コネクション (i, j) が、既存よりより長いパスであるようなら、更新
                    if total_score > self.longest_path_score_list[col]:
                        self.longest_path_score_list[col] = total_score
                        self.longest_path_trace_list[col] = self.longest_path_trace_list[idx] + [col]
                    # 済をチェック -> all済なら、determined_idx_list に登録
                    self.connection_matrix_remained_to_be_analyzed[idx, col] = False
                    if not any(self.connection_matrix_remained_to_be_analyzed[:, col]):
                        determined_idx_list.append(col)
            newly_determined_idx_list = determined_idx_list
        # 一番長いパスを取得
        max_i = np.argmax(self.longest_path_score_list)
        return self.longest_path_trace_list[max_i], self.longest_path_score_list[max_i]
    def ignore_nodes(self, node_index_to_ignore):
        for node_idx in node_index_to_ignore:
            self.connection_matrix[node_idx, :] = 0
            self.connection_matrix[:, node_idx] = 0
    def __str__(self) -> str:
        return str(self.connection_matrix)

class ConservedRegions():
    """
    self.conserved_regions.shape = (N, 4)  # ref_start, ref_end, query_start, query_end
    """
    def __init__(self, conserved_regions) -> None:
        assert conserved_regions.shape[1] == 4
        self.conserved_regions = conserved_regions
        self.ref_seq_offset = 0
        self.query_seq_offset = 0
    def copy(self):
        return self.__class__(np.copy(self.conserved_regions))
    def tidy_conserved_regions(self, longest_path_trace):
        # re-order
        self.conserved_regions = self.conserved_regions[longest_path_trace, :]
    def set_tidy_offset(self, N_ref, N_query):
        assert len(self.conserved_regions) > 0
        assert all(self.conserved_regions[:, 2] >= 0)
        assert self.ref_seq_offset == self.query_seq_offset == 0
        assert self.conserved_regions[-1, 2] < self.conserved_regions[-1, 3]    # consered_regions の検出では ref を繰り返した配列を作成して query を動かしているので、queryの最初最後にまたがった conserved_region が作成されることはないはず
        # min(query_start) が、一番上の行にくるように並び替え -> これで、query の切れ目が最後の non_conserved_region に来ることが保証される
        self.conserved_regions = np.roll(self.conserved_regions, -self.conserved_regions[:, 2].argmin(), axis=0)
        # 最初の行を [0, x, 0, y] となるように offset を取る
        self.set_offset(self.conserved_regions[0, 0], self.conserved_regions[0, 2], N_ref, N_query)
        assert self.conserved_regions[0, 0] == self.conserved_regions[0, 2] == 0
        # query の切れ目がどこに属するかが返り値 (ref_end_idx_after_offset + 1 = ref_start_idx_after_offset であることに注意)
        return N_query - self.query_seq_offset - 1  # ref_end_idx_after_offset
        query_end_idx_after_offset = N_query - self.query_seq_offset - 1
        query_start_idx_after_offset = (N_query - self.query_seq_offset)%N_query
    def set_offset(self, ref_start, query_start, N_ref, N_query):
        self.ref_seq_offset = ref_start
        self.query_seq_offset = query_start
        self.conserved_regions[:, :2] -= self.ref_seq_offset
        self.conserved_regions[:, 2:] -= self.query_seq_offset
        # 負の index を補正
        below_zero_ref = self.conserved_regions < 0
        below_zero_query = np.copy(below_zero_ref)
        below_zero_ref[:, 2:] = False
        below_zero_query[:, :2] = False
        self.conserved_regions[below_zero_ref] += N_ref
        self.conserved_regions[below_zero_query] += N_query
    # @property
    # def non_conserved_regions(self):
    #     non_conserved_regions = np.empty_like(self.conserved_regions)
    #     for i, (ref_start, ref_end, query_start, query_end) in enumerate(self):
    #         non_conserved_regions[i, 0] = ref_end + 1
    #         non_conserved_regions[i, 2] = query_end + 1
    #         non_conserved_regions[i - 1, 1] = ref_start - 1
    #         non_conserved_regions[i - 1, 3] = query_start - 1
    #     return non_conserved_regions
    def iter_regions(self):
        """
        regions.shape = (N, 5)  # length_of_previous_conserved_region, non_conserved_ref_start, non_conserved_ref_end, non_conserved_query_start, non_conserved_query_end
        """
        regions = np.empty((self.conserved_regions.shape[0], 5), dtype=int)
        for i, (ref_start, ref_end, query_start, query_end) in enumerate(self):
            regions[i, 0] = ref_end - ref_start + 1
            regions[i, 1] = ref_end + 1
            regions[i, 3] = query_end + 1
            regions[i - 1, 2] = ref_start - 1
            regions[i - 1, 4] = query_start - 1
        assert regions[-1, 2] == regions[-1, 4] == -1
        return regions
    def __iter__(self):
        yield from self.conserved_regions
    def __str__(self) -> str:
        return self.conserved_regions.__str__()
    def __len__(self):
        return len(self.conserved_regions)

#####################
# メソッドのみのクラス #
#####################
class SearchLongestPath():
    @classmethod
    def exec(cls, conserved_regions:ConservedRegions, N_ref:int, N_query:int):   # conserved_regions.shape = (N, 4)  # ref_start, ref_end, query_start, query_end
        longest_path_trace_list = []
        longest_path_score_list = []
        # 各々の conserved_region を始点とした時の、最適スコアを計算する
        for ref_start, ref_end, query_start, query_end in conserved_regions.conserved_regions:
            # 各 conserved_region を始点とするように offset を調整して、簡易DPを実行
            conserved_regions_copied = conserved_regions.copy()
            conserved_regions_copied.set_offset(ref_start, query_start, N_ref, N_query)
            longest_path_trace, longest_path_score = cls.search_longest_path(conserved_regions_copied)
            longest_path_trace_list.append(longest_path_trace)
            longest_path_score_list.append(longest_path_score)

        # 三段階で initial_idx を決定 (maximum longest_path_score -> maximum query_start_end_gap -> maximum gap)
        # max のスコアを与える conserved_region のインデックスを取得
        longest_path_score = np.max(longest_path_score_list)
        initial_idx_list_with_max_score = np.where(longest_path_score_list == longest_path_score)[0]
        if len(initial_idx_list_with_max_score) == 1:   # この場合分けがないと、下記の np.diff でエラーが出る
            return longest_path_trace_list[initial_idx_list_with_max_score[0]], longest_path_score
        else:
            # スコア max のうち、query の start と end のギャップが最大のものを採用する (もっとも多くの場合は全部同じになるけど)
            query_start_end_gap_list = []   # __len__ = len(initial_idx_list_with_max_score)
            gap_max_list = []               # __len__ = len(initial_idx_list_with_max_score)
            for initial_idx in initial_idx_list_with_max_score:
                longest_path_trace = longest_path_trace_list[initial_idx]
                # longest_path_trace の数値は circular の切れ目である一箇所を除いて昇順で並んでいる ので、下記により gap の幅を求められる
                conserved_regions_ordered = conserved_regions.conserved_regions[longest_path_trace, :]
                query_start_idx = conserved_regions_ordered[:, 2].argmin()
                query_end_idx = conserved_regions_ordered[:, 3].argmax()
                query_start_end_gap_list.append((conserved_regions_ordered[query_start_idx, 0] - conserved_regions_ordered[query_end_idx, 1])%N_ref)
                # さらにその中で、query_start_gap 以外の gap が最大なものを選出する
                    # query_start_gap は np.diff の値がマイナスになるので自動的に除かれる
                    # 最初と最後の np.diff の値が計算結果に含まれないので別途追加する
                gap_max_list.append(max(
                    np.diff(conserved_regions_ordered[:, 2:].flatten())[1::2].max(), 
                    conserved_regions_ordered[0, 2] - conserved_regions_ordered[-1, 3]
                ))
            # query の start と end のギャップが最大ものも
            query_start_end_gap_max = max(query_start_end_gap_list)
            q_se_gap_max_idx_list = np.where(np.array(query_start_end_gap_list) == query_start_end_gap_max)[0]
            gap_max_idx = np.array(gap_max_list)[q_se_gap_max_idx_list].argmax()
            initial_idx = initial_idx_list_with_max_score[q_se_gap_max_idx_list][gap_max_idx]
            return longest_path_trace_list[initial_idx], longest_path_score

    @classmethod
    def search_longest_path(cls, conserved_regions_copied):  # conserved_regions.shape = (N, 4)  # ref_start, ref_end, query_start, query_end
        # conserved_regions = np.vstack((conserved_regions, [8000, 8100, 2000, 2100], [0, 80, 1000, 1100], [100, 400, 1300, 1500]))
        """
        2つの conserved_region (cr1, cr2) が cr1 -> cr2 の有向エッジをもつための条件
        - cr1_end_ref <= cr2_start_ref
        - cr1_end_query <= cr2_start_query
        - 上記 4 つの数値で囲われる長方形に完全に含まれるような conserved_region が無い
        """

        # # start >= end となっているものは除外する（circularな都合）
        # conserved_regions_copied.conserved_regions = conserved_regions_copied.conserved_regions[(np.diff(conserved_regions_copied.conserved_regions[:, :2], axis=1).flatten() > 0) * (np.diff(conserved_regions_copied.conserved_regions[:, 2:], axis=1).flatten() > 0), :]

        # 0, 0 から始まる conserved_region が必須！
        where_all_zero = np.where((conserved_regions_copied.conserved_regions[:, (0, 2)] == 0).all(axis=1))[0]
        assert len(where_all_zero) == 1
        initial_node_idx = where_all_zero[0]
        # コネクションマトリックスを生成
        connection_matrix_ref = cls.get_1d_connection(conserved_regions_copied.conserved_regions[:, :2])     # 以下で統合する
        connection_matrix_query = cls.get_1d_connection(conserved_regions_copied.conserved_regions[:, 2:])   # 以下で統合する
        connection_constraint = cls.get_connection_constraint(conserved_regions_copied.conserved_regions)    # 以下で統合する
        connection_matrix = ConnectionMatrix(
            (connection_matrix_ref + connection_matrix_query) * connection_constraint * np.diff(conserved_regions_copied.conserved_regions[:, :2], axis=1).flatten(),    # エッジの重みは、ノードの終点の conserved_region の長さとする（その後の解析で始点は共通で始めるので、こうすると最後に終点処理しいないでいい）。
            initial_node_idx=initial_node_idx, 
            initial_node_score=np.diff(conserved_regions_copied.conserved_regions[initial_node_idx, :2])[0]
        )
        # end < start  となっているものは除外する（circularな都合）
        connection_matrix.ignore_nodes(np.where(
            (np.diff(conserved_regions_copied.conserved_regions[:, :2], axis=1).flatten() < 0) + 
            (np.diff(conserved_regions_copied.conserved_regions[:, 2:], axis=1).flatten() < 0)
        )[0])
        connection_matrix.remove_unreachable_branches()
        longest_path_trace, longest_path_score = connection_matrix.execute_dijkstras_algorithm()

        # print(longest_path_trace)
        # cls.draw_connection(conserved_regions_copied.conserved_regions, connection_matrix.connection_matrix)
        # quit()

        return longest_path_trace, longest_path_score
    @classmethod
    def get_1d_connection(cls, conserved_1d_regions):    # conserved_1d_region.shape = (N, 2) # 2 means (start, end)
        conserved_1d_regions = np.copy(conserved_1d_regions).astype(float)

        # 以下が無いと、前の region の end と 次の region の start が同じだった場合、そのコネクトを許容するかどうかは region の登場順番に応じてランダムに決定されてしまう。
        # get_connection_constraint でどちらにしろフィルターされるからここでは無くても大丈夫か？と思いきや、どれとどれがコネクトされるかも変わってくるので必要
        conserved_1d_regions[:, 0] -= 0.1

        N_regions = conserved_1d_regions.shape[0]
        connection_matrix = np.zeros((N_regions, N_regions), dtype=bool)
        # cr1_end - cr2_start の間に他の cr があるかを検出
        start_end_argsort_idx = np.unravel_index(np.argsort(conserved_1d_regions, axis=None), (N_regions, 2))[0]
        """
        start_end_argsort_idx は同じ数字は 0-N_regions - 1 までの数が二回ずつ登場する一次元の配列 (start, end の idx)
        - コネクトの例
            1 1 85 85 (1->85)
            1 1 85 85 (1->85)
            16 16 28 25 28 25 (16->28, 16->25)
            26 29 26 27 29 27 23 23 (26->27, 29->23, 27->23)
        """
        N_of_appearance_list = np.zeros((N_regions), dtype=int) # 二回登場かつコネクションを持つ場合は、値を 3 とする：これ以上コネクションを持つ可能性が無い場合は -1 とする
        for idx in start_end_argsort_idx:
            # 初登場の場合：二回登場済みのもの（値が3も含まれる）とコネクションを形成
            if N_of_appearance_list[idx] == 0:
                # コネクション作成：すでに二回登場しているものとのみ、コネクションを作成
                edge_start_loc_list = N_of_appearance_list >= 2
                edge_start_idx_list = np.where(edge_start_loc_list)[0]
                connection_matrix[(edge_start_idx_list, [idx for i in edge_start_idx_list])] = 1

                # 後処理
                N_of_appearance_list[edge_start_loc_list] = 3
                N_of_appearance_list[idx] += 1
            # 二回目登場の場合：新たにコネクションが作られることはない
            else:
                # すでに二回登場して、コネクションを持ってるやつらを打ち切る
                N_of_appearance_list[N_of_appearance_list == 3] = -1
                # 後処理
                N_of_appearance_list[idx] += 1
        assert all(i in (-1, 2) for i in N_of_appearance_list)
        return connection_matrix
    @classmethod
    def get_connection_constraint(cls, conserved_regions):
        # ref, query, どちらにおいても, 次の region の start が前の region の end より大きくなっているもののみを選択
        constraint_ref = conserved_regions[:, 1][:, np.newaxis] < conserved_regions[:, 0]
        constraint_query = conserved_regions[:, 3][:, np.newaxis] < conserved_regions[:, 2]
        return constraint_ref * constraint_query
    @classmethod
    def draw_connection(cls, conserved_regions, connection_matrix):
        import matplotlib.pyplot as plt
        for i, (ref_start, ref_end, query_start, query_end) in enumerate(conserved_regions):
            plt.plot((ref_start, ref_end), (query_start, query_start), c="r", alpha=0.2, linestyle="--")
            plt.plot((ref_start, ref_end), (query_end, query_end), c="r", alpha=0.2, linestyle="--")
            plt.plot((ref_start, ref_start), (query_start, query_end), c="r", alpha=0.2, linestyle="--")
            plt.plot((ref_end, ref_end), (query_start, query_end), c="r", alpha=0.2, linestyle="--")
            plt.text(ref_start, query_end, f"{i}")
        for i, j in zip(*np.where(connection_matrix)):
            print(i, j)
            plt.plot(conserved_regions[([i, j], [1, 0])], conserved_regions[([i, j], [3, 2])])
        plt.xlabel("ref")
        plt.ylabel("query")
        plt.show()

class WindowAnalysis():
    consecutive_true_threshold = 20 # ref_seq (seq1) と query_seq (seq2) がこの値より大きい回数連続して同じ場合、保存領域の候補とする。
    @classmethod
    def exec(cls, seq1, seq2) -> None:  # they should be len(seq1) == len(seq2)
        a = (seq1 == seq2).astype(int)
        transition_list = a[1:] - a[:-1]
        true_start_idx_list = np.where(transition_list == 1)[0] + 1
        true_end_idx_list = np.where(transition_list == -1)[0]
        if a[0]:
            true_start_idx_list = np.hstack(([0], true_start_idx_list))
        if a[-1]:
            true_end_idx_list = np.hstack((true_end_idx_list, [len(a) - 1]))
        assert len(true_start_idx_list) == len(true_end_idx_list)
        # 最初と最後は繋げない (a[0] == a[-1] == True の場合でも、そのまま放置：まとめたりしないよ)
        
        # True が連続している長さを計算、threshold 以上の場所を取得する
        consecutive_true_length_list = true_end_idx_list - true_start_idx_list + 1
        over_threshold_where = consecutive_true_length_list > cls.consecutive_true_threshold
        return true_start_idx_list[over_threshold_where], true_end_idx_list[over_threshold_where]

        # import matplotlib.pyplot as plt
        # plt.plot(range(len(consecutive_true_length_list)), consecutive_true_length_list)
        # plt.show()
        # quit()



