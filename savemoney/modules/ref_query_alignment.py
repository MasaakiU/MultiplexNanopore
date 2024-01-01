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

class MyAlignerBase():
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
    def test(self, ref_seq, query_seq):
        return parasail.nw_trace(query_seq, ref_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
    # query の間はギャップペナルティを与えないアラインメント
    def __my_special_dp_python(self, query_seq_1, query_seq_2, ref_seq):
        gap_open_penalty = self.gap_open_penalty
        gap_extend_penalty = self.gap_extend_penalty
        match_score = self.match_score
        mismatch_score = self.mismatch_score

        N_ref_seq = len(ref_seq)
        N_query_seq_1 = len(query_seq_1)
        N_query_seq_2 = len(query_seq_2)
        assert N_ref_seq > N_query_seq_1 + N_query_seq_2

        ################
        # TRACE PARAMS #
        ################
        # origin:           42 (*)      # 解析に関係しない
        # match_trace:      61 (=)      # ナナメ、match
        # mismatch_trace:   88 (X)      # ナナメ、mismatch
        # insertion_trace:  73 (I)      # タテに進むと insertion
        # deletion_trace:   68 (D)      # ヨコに進むと deletion
        # special deletion: 72 (H)      # ヨコに進む & query_seq_q の最後の行限定

        #########
        # 初期化 #
        #########
        dp = np.zeros((N_query_seq_1 + N_query_seq_2 + 1, N_ref_seq + 1), dtype=int)   # 値を左から右に更新していき、trace は別に保存する
        dp[0, 1] = -gap_open_penalty
        for j in range(1, N_ref_seq):
            dp[0, j+1] = dp[0, j] - gap_extend_penalty
        dp[1, 0] = -gap_open_penalty
        for i in range(1, N_query_seq_1 + N_query_seq_2):
            dp[i+1, 0] = dp[i, 0] - gap_extend_penalty
        trace = np.zeros((N_query_seq_1 + N_query_seq_2 + 1, N_ref_seq + 1), dtype=int) # 0行目、0列目は dp とidx を合わせるための pseudo
        trace[:, 0] = 73     # タテに進むと insertion
        trace[0, :] = 68     # ヨコに進むと deletion
        trace[0, 0] = 42

        ##########
        # DP実行 #
        ##########
        if N_query_seq_1 > 0:
            # query_seq_1
            for i, q in enumerate(query_seq_1[:N_query_seq_1 - 1]):
                for j, r in enumerate(ref_seq):
                    # aligned_score
                    if q == r:
                        aligned_score = dp[i, j] + match_score
                        modifier = 0
                    else:
                        aligned_score = dp[i, j] + mismatch_score
                        modifier = 27
                    # insertion_score   # タテに進むと insertion
                    if trace[i, j+1] == 73:
                        insertion_score = dp[i, j+1] - gap_extend_penalty
                    else:
                        insertion_score = dp[i, j+1] - gap_open_penalty
                    # deletion_score   # ヨコに進むと deletion
                    if trace[i+1, j] == 68:
                        deletion_score = dp[i+1, j] - gap_extend_penalty
                    else:
                        deletion_score = dp[i+1, j] - gap_open_penalty
                    # set dp and trace
                    if aligned_score >= deletion_score:
                        if aligned_score >= insertion_score:
                            dp[i+1, j+1] = aligned_score
                            trace[i+1, j+1] = 61 + modifier
                        else:
                            dp[i+1, j+1] = insertion_score
                            trace[i+1, j+1] = 73
                    else:
                        if insertion_score >= deletion_score:
                            dp[i+1, j+1] = insertion_score
                            trace[i+1, j+1] = 73
                        else:
                            dp[i+1, j+1] = deletion_score
                            trace[i+1, j+1] = 68
            # query間
            i = N_query_seq_1 - 1
            q = query_seq_1[-1]
            for j, r in enumerate(ref_seq):
                # aligned_score
                if q == r:
                    aligned_score = dp[i, j] + match_score
                    modifier = 0
                else:
                    aligned_score = dp[i, j] + mismatch_score
                    modifier = 27
                # insertion_score   # タテに進むと insertion
                if trace[i, j+1] == 73:
                    insertion_score = dp[i, j+1] - gap_extend_penalty
                else:
                    insertion_score = dp[i, j+1] - gap_open_penalty
                # deletion_score   # ヨコに進むと deletion
                deletion_score = dp[i+1, j]     # special deletion: no penalty
                # set dp and trace
                if aligned_score >= deletion_score:
                    if aligned_score >= insertion_score:
                        dp[i+1, j+1] = aligned_score
                        trace[i+1, j+1] = 61 + modifier
                    else:
                        dp[i+1, j+1] = insertion_score
                        trace[i+1, j+1] = 73
                else:
                    if insertion_score >= deletion_score:
                        dp[i+1, j+1] = insertion_score
                        trace[i+1, j+1] = 73
                    else:
                        dp[i+1, j+1] = deletion_score
                        trace[i+1, j+1] = 72     # 特殊 deletion (no penalty)
        else:
            dp[0, :] = 0
            trace[0, 1:] = 72
        # query_seq_2
        for i, q in enumerate(query_seq_2):
            i += N_query_seq_1
            for j, r in enumerate(ref_seq):
                # aligned_score
                if q == r:
                    aligned_score = dp[i, j] + match_score
                    modifier = 0
                else:
                    aligned_score = dp[i, j] + mismatch_score
                    modifier = 27
                # insertion_score   # タテに進むと insertion
                if trace[i, j+1] == 73:
                    insertion_score = dp[i, j+1] - gap_extend_penalty
                else:
                    insertion_score = dp[i, j+1] - gap_open_penalty
                # deletion_score   # ヨコに進むと deletion
                if trace[i+1, j] == 68:
                    deletion_score = dp[i+1, j] - gap_extend_penalty
                else:
                    deletion_score = dp[i+1, j] - gap_open_penalty
                # set dp and trace
                if aligned_score >= deletion_score:
                    if aligned_score >= insertion_score:
                        dp[i+1, j+1] = aligned_score
                        trace[i+1, j+1] = 61 + modifier
                    else:
                        dp[i+1, j+1] = insertion_score
                        trace[i+1, j+1] = 73
                else:
                    if insertion_score >= deletion_score:
                        dp[i+1, j+1] = insertion_score
                        trace[i+1, j+1] = 73
                    else:
                        dp[i+1, j+1] = deletion_score
                        trace[i+1, j+1] = 68

        #############
        # traceback #
        #############
        i = N_query_seq_1 + N_query_seq_2
        j = N_ref_seq
        score_trace = []
        traceback = []
        while True:
            t = trace[i, j]
            traceback.append(t)
            score_trace.append(dp[i, j])
            if t == 61:
                i -= 1
                j -= 1
            elif t == 88:
                i -= 1
                j -= 1
            elif t == 73:
                i -= 1
            elif t == 68:
                j -= 1
            else:   # t == 72
                j -= 1
            # 終了判定
            if t == 42:
                break

        # 後処理 (hard/soft clipping)
        my_cigar = "".join(map(chr, traceback[::-1]))[1:]
        score_trace = score_trace[::-1][1:]
        trace_len = len(my_cigar)

        query_1_aligned_len = my_cigar.index("H")
        if query_1_aligned_len > 0:
            query_1_not_clipped_len = np.argmax(score_trace[:query_1_aligned_len]) + 1
            if (query_1_not_clipped_len == 1) and (score_trace[0] < 0):
                query_1_not_clipped_len = 0
        else:
            query_1_not_clipped_len = 0

        query_2_aligned_len = my_cigar[::-1].index("H")
        if query_2_aligned_len > 0:
            query_2_not_clipped_len = np.argmin(score_trace[::-1][:query_2_aligned_len]) + 1    # end 側なので、argmax ではなく argmin
            if (query_2_not_clipped_len == 1) and (score_trace[-2] > score_trace[-1]):
                query_2_not_clipped_len = 0
        else:
            query_2_not_clipped_len = 0

        my_cigar = (
            my_cigar[:query_1_not_clipped_len] + 
            "S" * (query_1_aligned_len - query_1_not_clipped_len - my_cigar[:query_1_aligned_len - query_1_not_clipped_len].count("D")) + 
            "H" * (
                trace_len - query_2_aligned_len - query_1_aligned_len
                + my_cigar[:query_1_aligned_len - query_1_not_clipped_len].count("D")
                + my_cigar[trace_len - (query_2_aligned_len - query_2_not_clipped_len):].count("D")
                - my_cigar[:query_1_aligned_len - query_1_not_clipped_len].count("I")
                - my_cigar[trace_len - (query_2_aligned_len - query_2_not_clipped_len):].count("I")
            ) + 
            "S" * (query_2_aligned_len - query_2_not_clipped_len - my_cigar[trace_len - (query_2_aligned_len - query_2_not_clipped_len):].count("D")) + 
            my_cigar[trace_len-query_2_not_clipped_len:]
        )
        return my_cigar
    def __my_special_dp_cython(self, query_seq_1: str, query_seq_2: str, ref_seq: str):
        # parasail　に合わせるために、文字列を逆順にしてアラインメントする
        traceback, score_trace = af.my_special_dp_cython(query_seq_2[::-1].encode("utf-8"), query_seq_1[::-1].encode("utf-8"), ref_seq[::-1].encode("utf-8"), self.gap_open_penalty, self.gap_extend_penalty, self.match_score, self.mismatch_score)
        # 後処理 (hard/soft clipping)
        my_cigar = "".join(map(chr, traceback))
        score_trace = score_trace
        trace_len = len(my_cigar)
        if "H" not in my_cigar:
            return my_cigar
        else:
            query_1_aligned_len = my_cigar.index("H")
            if query_1_aligned_len > 0:
                query_1_not_clipped_len = np.argmax(score_trace[:query_1_aligned_len]) + 1
                if (query_1_not_clipped_len == 1) and (score_trace[0] < 0):
                    query_1_not_clipped_len = 0
            else:
                query_1_not_clipped_len = 0

            query_2_aligned_len = my_cigar[::-1].index("H")
            if query_2_aligned_len > 0:
                query_2_not_clipped_len = np.argmin(score_trace[::-1][:query_2_aligned_len]) + 1    # end 側なので、argmax ではなく argmin
                if (query_2_not_clipped_len == 1) and (score_trace[-2] > score_trace[-1]):
                    query_2_not_clipped_len = 0
            else:
                query_2_not_clipped_len = 0

            my_cigar = (
                my_cigar[:query_1_not_clipped_len] + 
                "S" * (query_1_aligned_len - query_1_not_clipped_len - my_cigar[:query_1_aligned_len - query_1_not_clipped_len].count("D")) + 
                "H" * (
                    trace_len - query_2_aligned_len - query_1_aligned_len
                    + my_cigar[:query_1_aligned_len - query_1_not_clipped_len].count("D")
                    + my_cigar[trace_len - (query_2_aligned_len - query_2_not_clipped_len):].count("D")
                    - my_cigar[:query_1_aligned_len - query_1_not_clipped_len].count("I")
                    - my_cigar[trace_len - (query_2_aligned_len - query_2_not_clipped_len):].count("I")
                ) + 
                "S" * (query_2_aligned_len - query_2_not_clipped_len - my_cigar[trace_len - (query_2_aligned_len - query_2_not_clipped_len):].count("D")) + 
                my_cigar[trace_len-query_2_not_clipped_len:]
            )
            print("###")
            print(my_cigar)
            quit()
            return my_cigar
    def my_special_dp(self, query_seq_1: str, query_seq_2: str, ref_seq: str):
        ################
        # アラインメント #
        ################
        if len(query_seq_1) > 0:
            result_1 = parasail.sg_de_trace(query_seq_1, ref_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
            my_result_1 = MyResult(result_1)
            assert my_result_1.end_query == len(query_seq_1) - 1
            len_H_1_tmp = my_result_1.organize_end()
            assert my_result_1.end_ref + 1 + len_H_1_tmp == len(ref_seq)
            # ソフトクリッピング
            my_result_1.set_soft_clipping(self.gap_open_penalty, self.gap_extend_penalty, self.match_score, self.mismatch_score, side="beg")
            len_H_1 = my_result_1.my_cigar.count("H")
        else:
            len_H_1 = len(ref_seq)
            my_result_1 = MyResult()
            my_result_1.my_cigar = "H" * len_H_1
            my_result_1.beg_ref = -1
            my_result_1.end_ref = -1
        if len(query_seq_2) > 0:
            result_2 = parasail.sg_db_trace(query_seq_2, ref_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
            my_result_2 = MyResult(result_2)
            assert my_result_2.end_ref == len(ref_seq) - 1
            assert my_result_2.end_query == len(query_seq_2) - 1
            len_H_2_tmp = my_result_2.organize_beg()
            # ソフトクリッピング
            my_result_2.set_soft_clipping(self.gap_open_penalty, self.gap_extend_penalty, self.match_score, self.mismatch_score, side="end")
            len_H_2 = my_result_2.my_cigar.count("H")
        else:
            len_H_2 = len(ref_seq)
            my_result_2 = MyResult()
            my_result_2.my_cigar = "H" * len_H_2
            my_result_2.beg_ref = len(ref_seq)
            my_result_2.end_ref = len(ref_seq)
        ########
        # 結合 #
        ########
        # =====ALIGNED-SEQ=====SSSSSSHHHHHHSSSSSS=====ALIGNED-SEQ=====
        if my_result_2.beg_ref - my_result_1.end_ref > 0:
            my_result = MyResult()
            my_result.my_cigar = (
                my_result_1.my_cigar[:len(my_result_1.my_cigar) - len_H_1] + 
                "H" * (my_result_2.beg_ref - my_result_1.end_ref - 1) + 
                my_result_2.my_cigar[len_H_2:]
            )
        # =====ALIGNED-SEQ=====SSEEEEE(EEEEE)                   Replace "S" with "E (Excess sequence)"
        #                        SSSSSSSS=====ALIGNED-SEQ=====
        elif my_result_2.beg_ref - (my_result_1.end_ref - my_result_1.my_cigar.count("S")) - 1 > 0:
            E_len = my_result_1.end_ref - my_result_2.beg_ref + 1
            my_result = MyResult()
            my_result.my_cigar = (
                my_result_1.my_cigar[:len(my_result_1.my_cigar) - my_result_1.my_cigar.count("H") - my_result_1.my_cigar.count("S")] + 
                "S" * (my_result_1.my_cigar.count("S") - E_len) + 
                "E" * E_len + 
                "S" * my_result_2.my_cigar.count("S") +
                my_result_2.my_cigar[my_result_2.my_cigar.count("H") + my_result_2.my_cigar.count("S"):]
            )
        # =====ALIGNED-SEQ=====EEEEEEE(EEEEE)                    Replace "S" with "E (Excess sequence)"
        #                   FFFSSSSSSSSSSS=====ALIGNED-SEQ=====  Replace "S" with "E (Excess sequence)"     # Replace "S" with "F (next to E)"
        elif (my_result_2.beg_ref + my_result_2.my_cigar.count("S")) - (my_result_1.end_ref - my_result_1.my_cigar.count("S")) - 1 > 0:
            E_len_2 = (my_result_1.end_ref - my_result_1.my_cigar.count("S")) - my_result_2.beg_ref + 1
            my_result = MyResult()
            my_result.my_cigar = (
                my_result_1.my_cigar[:len(my_result_1.my_cigar) - my_result_1.my_cigar.count("H") - my_result_1.my_cigar.count("S")] + 
                "E" * my_result_1.my_cigar.count("S") + 
                "E" * E_len_2 + 
                "S" * (my_result_2.my_cigar.count("S") - E_len_2) + 
                my_result_2.my_cigar[my_result_2.my_cigar.count("H") + my_result_2.my_cigar.count("S"):]
            )
        # =====ALIGNED-SEQ=====SSSSSS
        #              SSSSSS=====ALIGNED-SEQ=====  Redo alignment
        else:
            # アラインメントやり直し
            result = parasail.nw_trace(query_seq_1 + query_seq_2, ref_seq, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
            my_result = MyResult(result)
        """ OLD FUNCTIONS
        # =====ALIGNED-SEQ=====SSSSSSS(SSSSS)
        #                   (SSSSS)SSSSSSSS=====ALIGNED-SEQ=====
        elif (my_result_2.beg_ref + my_result_2.my_cigar.count("S")) - (my_result_1.end_ref - my_result_1.my_cigar.count("S")) - 1 > 0:
            len_S_1 = my_result_1.my_cigar.count("S")
            len_S_2 = my_result_2.my_cigar.count("S")
            ref_seq_inter = ref_seq[my_result_1.end_ref - len_S_1 + 1:my_result_2.beg_ref + len_S_2]
            query_seq_inter = query_seq_1[len(query_seq_1) - len_S_1:] + query_seq_2[:len_S_2]
            # 中間配列のみアラインメント
            result_inter = parasail.nw_trace(query_seq_inter, ref_seq_inter, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
            my_result_inter = MyResult(result_inter)
            # 統合
            my_result = MyResult()
            my_result.my_cigar = (
                my_result_1.my_cigar[:len(my_result_1.my_cigar) - len_H_1 - len_S_1] + 
                my_result_inter.my_cigar +
                my_result_2.my_cigar[len_H_2 + len_S_2:]
            )
        # =====ALIGNED-SEQ=====SSSSSS
        #              SSSSSS=====ALIGNED-SEQ=====  # replace redundant sequence with "R"
        else:
            idx_2 = my_result_2.count("H") + my_result_2.count("S")
            len_1 = 0
            for idx_1, L in enumerate(my_result_1.my_cigar):
                if L in "=XD":
                    len_1 += 1
                elif L == "I":
                    continue
                else:
                    raise Exception("error")
                # 終了
                if len_1 == idx_2:
                    idx_1 += 1
                    break
            len_R = 0
            for L in my_result_1.my_cigar[idx_1:]:
                if L in "=XD":
                    len_R += 1
                    if my_result_2[idx_2] != "I":
                        idx_2 += 1
                    else:
                        while my_result_2[idx_2] == "I":
                            idx_2 += 1
                elif L == "I":
                    continue
                elif L in "HS":
                    break
                else:
                    raise Exception("error")
            my_result = MyResult()
            my_result.my_cigar = (
                my_result_1.my_cigar[:len(my_result_1.my_cigar) - my_result_1.my_cigar.count("H") - my_result_1.my_cigar.count("S")] + 
                "E" * my_result_1.my_cigar.count("S") + 
                "R" * len_R + 
                "E" * my_result_2.my_cigar.count("S") + 
                my_result_2.my_cigar[idx_2:]
            )
        """
        return my_result

class MyOptimizedAligner(MyAlignerBase, mc.AlignmentBase):
    default_repeat_max = 5
    percentile_factor = 0.1 # 低いほど conserved region の quality は上がるが、計算時間がかかる
    def __init__(self, ref_seq, param_dict) -> None:
        super().__init__(param_dict)
        self.ref_seq = ref_seq
        self.ref_seq_v = np.array([ord(char) for char in ref_seq])
        self.N_ref = len(self.ref_seq_v)
        self.ref_seq_v_repeated_list = [
            np.hstack([self.ref_seq_v for i in range(repeat)], dtype=np.int64)
            for repeat in range(2, self.default_repeat_max + 1)
        ]
        self.k_list = np.arange(0, self.N_ref)
        x_percentile = 1 / self.N_ref * self.percentile_factor # 偶然 threshold を超える position が 0.1個/ref_seq 以下になるようにする
        self.sigma = stats.norm.ppf(1 - x_percentile, loc=0, scale=1)  # 累積分布関数の逆関数
        self.is_all_ATCG = all([b.upper() in "ATCG" for b in ref_seq])
    def calc_circular_conserved_region(self, query_seq):
        conserved_regions = self.calc_circular_conserved_region_core(query_seq)
        if len(conserved_regions) == 0:
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
                    result_master.append_my_cigar("I" * (N_query - non_conserved_query_start))
                    continue
                # non_conserved_query の長さが 0 の場合（non_conserved_ref, non_conserved_query の長さが同時に 0 になることがあるが、多分大丈夫なはず）
                elif non_conserved_query_start == N_query:
                    result_master.append_my_cigar("=" * length_of_previous_conserved_region)
                    result_master.append_my_cigar("H" * (self.N_ref - non_conserved_ref_start))
                    continue
                else:
                    ref_seq_extracted = self.ref_seq[non_conserved_ref_start:]
                    query_seq_extracted = query_seq[non_conserved_query_start:]
                    # execute alignment: query の方が長い場合は、そのままグローバルアラインメントで ok
                    if (len(query_seq_extracted) >= len(ref_seq_extracted)):
                        result = parasail.nw_trace(query_seq_extracted, ref_seq_extracted, self.gap_open_penalty, self.gap_extend_penalty, self.my_custom_matrix)
                        my_result = MyResult(result)
                    else:
                        query_seq_extracted_1 = query_seq[non_conserved_query_start:query_end_idx_after_offset + 1]
                        query_seq_extracted_2 = query_seq[query_end_idx_after_offset + 1:]
                        my_result = self.my_special_dp(
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
            ref_seq_v_repeated = np.hstack([self.ref_seq_v for i in range(repeat)], np.int64)

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
        N_ins = self.my_cigar[-aligned_ref_seq_offset:].count("I") + self.my_cigar[-aligned_ref_seq_offset:].count("E")
        while ref_seq_offset != aligned_ref_seq_offset - N_ins:
            aligned_ref_seq_offset += N_ins - previous_N_ins
            previous_N_ins = N_ins
            N_ins = self.my_cigar[-aligned_ref_seq_offset:].count("I") + self.my_cigar[-aligned_ref_seq_offset:].count("E")
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
    # for semi-global alignment
    def organize_end(self):
        m = re.search(r"^[=XDI]+?(D*)$", self.my_cigar)
        assert self.beg_ref == self.beg_query == 0
        len_D = len(m.group(1))
        alignment_len = len(self.my_cigar) - len_D
        self.end_ref = alignment_len - self.my_cigar[:alignment_len].count("I") - 1
        self.end_query = alignment_len - self.my_cigar[:alignment_len].count("D") - 1
        assert self.end_ref >= 0
        assert self.end_query >= 0
        self.my_cigar = self.my_cigar[:alignment_len] + "H" * len_D
        return len_D    # = len_H
    def organize_beg(self):
        assert self.beg_ref == self.beg_query == 0
        m = re.search(r"^(D*)[=XDI]+?$", self.my_cigar)
        self.beg_ref = len(m.group(1))
        self.my_cigar = "H" * self.beg_ref + self.my_cigar[self.beg_ref:]
        return self.beg_ref # = len_H
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
            self.beg_ref = self.my_cigar.count("H")
        elif side == "beg":
            self.my_cigar = my_cigar
            self.end_ref = len_aligned_ref - 1
    @staticmethod
    def set_soft_clipping_core(my_cigar, gap_open_penalty, gap_extend_penalty, match_score, mismatch_score):
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
            elif L == "H":
                break
            else:
                raise Exception(f"error: {L}")
            score_list.append(cur_score)
            previous_L = L
        max_score_idx = np.argmax(score_list)
        len_non_SH = max_score_idx + 1
        len_non_H = len(score_list)
        len_my_cigar = len(my_cigar)
        if score_list[max_score_idx] < 0:
            len_query = len_non_H - my_cigar[:len_non_H].count("D")
            len_ref = len_my_cigar - my_cigar.count("I")
            assert len_ref >= len_query
            new_my_cigar = "S" * len_query  + "H" * (len_ref - len_query)
            len_aligned_ref = len_query
        else:
            len_query_S = len_non_H - len_non_SH - my_cigar[len_non_SH:len_non_H].count("D")
            len_ref_SH = len_my_cigar - len_non_SH - my_cigar[len_non_SH:len_non_H].count("I")
            assert len_ref_SH >= len_query_S    # len_query_S の方が長い場合でも、
            new_my_cigar = my_cigar[:len_non_SH] + "S" * len_query_S + "H" * (len_ref_SH - len_query_S)
            len_aligned_ref = len_non_SH - my_cigar[:len_non_SH].count("I") + len_query_S   # 159
        return new_my_cigar, len_aligned_ref
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
        self.connection_matrix[self.initial_node_idx, self.initial_node_idx] = 100

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



