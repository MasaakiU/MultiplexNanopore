# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
# cython: cdivision=True


# ビルド
# python alignment_functions_setup.py build_ext --inplace

# cimport cython
# # import numpy as np
# cimport numpy as cnp
# # DTYPEint64 = np.int64
# ctypedef cnp.int64_t DTYPEint64_t
# ctypedef cnp.int8_t DTYPEint8_t

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# @cython.nonecheck(False)

# cpdef k_mer_offset_analysis(
#         cnp.ndarray[DTYPEint64_t, ndim=1] ref_seq_v_repeated, 
#         cnp.ndarray[DTYPEint64_t, ndim=1] query_seq_v, 
#         cnp.ndarray[DTYPEint64_t, ndim=1] result_array, 
#     ):
#     cdef int k
#     cdef int i
#     cdef int s
#     s = 0
#     for k in range(len(result_array)):    # k はオフセット
#         for i in range(len(query_seq_v)):
#             if ref_seq_v_repeated[k + i] == query_seq_v[i]:
#                 s += 1
#         result_array[k] = s
#         s = 0


# ビルド: とりあえず cimport numpy しなければ、setup ファイルで numpy path を指定しなくても下記でいける
# python alignment_functions_setup.py build_ext --inplace
# or 
# cythonize -3 -a -i alignment_functions.pyx

from libcpp.vector cimport vector #ここでcppのvectorを呼び出す
ctypedef long long LL   # unsigned でも良いのか？
ctypedef vector[LL] vec
cpdef k_mer_offset_analysis_2(
        LL[:] ref_seq_v_repeated, 
        LL[:] query_seq_v, 
        LL N_ref, 
        LL len_query_seq_v
    ):
    cdef LL k
    cdef LL i
    cdef LL s
    cdef vec result_array
    result_array.reserve(N_ref)
    s = 0
    for k in range(N_ref):    # k はオフセット
        for i in range(len_query_seq_v):
            if ref_seq_v_repeated[k + i] == query_seq_v[i]:
                s += 1
        result_array.push_back(s)
        s = 0
    return result_array

from libcpp.map cimport map as c_map
from libcpp.string cimport string
from libcpp.algorithm cimport copy
ctypedef long double LD
ctypedef vector[char] char_vec
ctypedef vector[LD] LD_vec
ctypedef c_map[char, LD] chr_LD_map
ctypedef c_map[string, LD] str_LD_map
ctypedef c_map[string, LD_vec] str_LD_vec_map

cdef class SequenceBasecallQscorePDF:
    # default
    cdef str_LD_map P_base_calling_given_true_refseq_dict
    cdef str_LD_vec_map pdf_core
    cdef char_vec bases
    cdef LL N_bases
    # additional
    cdef string key
    def __cinit__(self, str_LD_map P_base_calling_given_true_refseq_dict, str_LD_vec_map pdf_core, char_vec bases):
        self.P_base_calling_given_true_refseq_dict = P_base_calling_given_true_refseq_dict
        self.pdf_core = pdf_core
        self.key.resize(3)
        self.key[1] = b"_"
        self.bases = bases
        self.N_bases = self.bases.size()

    def calc_consensus_error_rate(self, char_vec query_list, vec q_score_list, chr_LD_map P_N_dict):
        cdef LL N_query_list
        cdef LL base_idx
        cdef LL idx
        cdef LD_vec bunshi_list
        cdef LD bunshi_P_N
        cdef LD bunbo_bunshi_sum
        cdef char base
        cdef LD val
        cdef LD_vec p_list

        N_query_list = query_list.size()

        for base_idx in range(self.N_bases):
            # 初期化
            B = self.bases[base_idx]
            self.key[0] = B
            bunshi_list.clear()

            for idx in range(N_query_list):
                self.key[2] = query_list[idx]
                if q_score_list[idx] >= 0:
                    bunshi_list.push_back(self.P_base_calling_given_true_refseq_dict[self.key] * self.pdf_core[self.key][q_score_list[idx]])
                else:
                    bunshi_list.push_back(self.P_base_calling_given_true_refseq_dict[self.key])
            bunshi_P_N = P_N_dict[B]

            # inside sum
            bunbo_bunshi_sum = 0
            for base in self.bases:
                self.key[0] = base
                val = P_N_dict[base] / bunshi_P_N
                for idx in range(N_query_list):
                    self.key[2] = query_list[idx]
                    if q_score_list[idx] >= 0:
                        val *= (self.P_base_calling_given_true_refseq_dict[self.key] * self.pdf_core[self.key][q_score_list[idx]]) / bunshi_list[idx]
                    else:
                        val *= (self.P_base_calling_given_true_refseq_dict[self.key]) / bunshi_list[idx]
                bunbo_bunshi_sum += val
            p_list.push_back(1 - 1 / bunbo_bunshi_sum)
        return p_list

cpdef my_special_dp_cython(
        string query_seq_1, 
        string query_seq_2, 
        string ref_seq, 
        LL gap_open_penalty, 
        LL gap_extend_penalty, 
        LL match_score, 
        LL mismatch_score
    ):
    cdef size_t N_ref_seq
    cdef size_t N_query_seq_1
    cdef size_t N_query_seq_2
    N_ref_seq = ref_seq.size()
    N_query_seq_1 = query_seq_1.size()
    N_query_seq_2 = query_seq_2.size()
    assert N_ref_seq > N_query_seq_1 + N_query_seq_2

    ################
    # TRACE PARAMS #
    ################
    # origin:            42 (*)      # 解析に関係しない
    # match_trace:       61 (=)      # ナナメ、match
    # mismatch_trace:    88 (X)      # ナナメ、mismatch
    # insertion_trace:   73 (I)      # タテに進むと insertion
    # deletion_trace:    68 (D)      # ヨコに進むと deletion
    # ins OR del:        141(ID)     # スコアが同じ場合
    # ins OR mat:        134(I=)     # スコアが同じ場合
    # del OR mat:        129(D=)     # スコアが同じ場合
    # ins OR del OR mat: 202(ID=)    # スコアが同じ場合
    # ins OR mis:        161(IX)     # スコアが同じ場合
    # del OR mis:        156(DX)     # スコアが同じ場合
    # ins OR del OR mis: 229(IDX)    # スコアが同じ場合
    # special deletion:  72 (H)      # ヨコに進む & query_seq_q の最後の行限定
    #                    145(IH)
    #                    133(H=)
    #                    206(IH=)
    #                    160(HX)
    #                    233(IHX)

    #########
    # 初期化 #
    #########
    cdef vector[vec] dp     # shape = (N_query_seq_1 + N_query_seq_2 + 1, N_ref_seq + 1)
    cdef vector[vec] trace  # shape = (N_query_seq_1 + N_query_seq_2 + 1, N_ref_seq + 1)
    cdef vec dp_row
    cdef vec trace_row
    cdef size_t i
    cdef size_t j
    cdef char r
    cdef char q
    cdef LL aligned_score
    cdef LL insertion_score
    cdef LL deletion_score
    cdef LL modifier
    dp_row.reserve(N_ref_seq + 1)
    dp_row.push_back(0)
    dp.reserve(N_query_seq_1 + N_query_seq_2 + 1)
    trace_row.reserve(N_ref_seq + 1)
    trace_row.push_back(42)     # origin
    trace.reserve(N_query_seq_1 + N_query_seq_2 + 1)
    #####################
    # DP of query_seq_1 #
    #####################
    if N_query_seq_1 > 0:
        ###############
        # query_seq_1 #
        ###############
        # 初期化
        dp_row.push_back(-gap_open_penalty)
        trace_row.push_back(68)
        for j in range(1, N_ref_seq):
            dp_row.push_back(dp_row[j] - gap_extend_penalty)
            trace_row.push_back(68) # ヨコに進むと deletion
        dp.push_back(dp_row)
        trace.push_back(trace_row)
        # 実行
        for i in range(N_query_seq_1 - 1):
            q = query_seq_1[i]
            # 初期化
            dp_row.clear()
            if i == 0:
                dp_row.push_back(dp[i][0] - gap_open_penalty)
            else:
                dp_row.push_back(dp[i][0] - gap_extend_penalty)
            trace_row.clear()
            trace_row.push_back(73)     # タテに進むと insertion
            # 値格納していく
            for j in range(N_ref_seq):
                r = ref_seq[j]
                # aligned_score
                if q == r:
                    aligned_score = dp[i][j] + match_score
                    modifier = 0
                else:
                    aligned_score = dp[i][j] + mismatch_score
                    modifier = 27
                # insertion_score   # タテに進むと insertion
                if trace[i][j+1] in (73, 141, 134, 202, 161, 229):
                    insertion_score = dp[i][j+1] - gap_extend_penalty
                else:
                    insertion_score = dp[i][j+1] - gap_open_penalty
                # deletion_score   # ヨコに進むと deletion
                if trace_row[j] in (68, 141, 129, 202, 156, 229):
                    deletion_score = dp_row[j] - gap_extend_penalty
                else:
                    deletion_score = dp_row[j] - gap_open_penalty
                # set dp and trace
                if aligned_score > deletion_score:
                    if aligned_score > insertion_score:     # = or X
                        dp_row.push_back(aligned_score)
                        trace_row.push_back(61 + modifier)
                    elif aligned_score < insertion_score:   # I
                        dp_row.push_back(insertion_score)
                        trace_row.push_back(73)
                    else:                                   # I= or IX
                        dp_row.push_back(insertion_score)
                        trace_row.push_back(134 + modifier)
                elif aligned_score < deletion_score:
                    if insertion_score > deletion_score:    # I
                        dp_row.push_back(insertion_score)
                        trace_row.push_back(73)
                    elif insertion_score < deletion_score:  # D
                        dp_row.push_back(deletion_score)
                        trace_row.push_back(68)
                    else:                                   # ID
                        dp_row.push_back(deletion_score)
                        trace_row.push_back(141)
                else:
                    if aligned_score < insertion_score:     # I
                        dp_row.push_back(insertion_score)
                        trace_row.push_back(73)
                    elif aligned_score > insertion_score:   # D= or DX
                        dp_row.push_back(deletion_score)
                        trace_row.push_back(129 + modifier)
                    else:                                   # ID= or IDX
                        dp_row.push_back(deletion_score)
                        trace_row.push_back(202 + modifier)
            dp.push_back(dp_row)
            trace.push_back(trace_row)
        ###########
        # query間 #
        ###########
        i = N_query_seq_1 - 1
        q = query_seq_1[N_query_seq_1 - 1]
        # 初期化
        dp_row.clear()
        dp_row.push_back(dp[i][0] - gap_extend_penalty)
        trace_row.clear()
        trace_row.push_back(73)     # タテに進むと insertion
        # 値格納していく
        for j in range(N_ref_seq):
            r = ref_seq[j]
            # aligned_score
            if q == r:
                aligned_score = dp[i][j] + match_score
                modifier = 0
            else:
                aligned_score = dp[i][j] + mismatch_score
                modifier = 27
            # insertion_score   # タテに進むと insertion
            if trace[i][j+1] in (73, 141, 134, 202, 161, 229):
                insertion_score = dp[i][j+1] - gap_extend_penalty
            else:
                insertion_score = dp[i][j+1] - gap_open_penalty
            # deletion_score   # ヨコに進むと deletion
            deletion_score = dp_row[j]      # special deletion: no penalty
            # set dp and trace
            if aligned_score > deletion_score:
                if aligned_score > insertion_score:     # = or X
                    dp_row.push_back(aligned_score)
                    trace_row.push_back(61 + modifier)
                elif aligned_score < insertion_score:   # I
                    dp_row.push_back(insertion_score)
                    trace_row.push_back(73)
                else:                                   # I= or IX
                    dp_row.push_back(insertion_score)
                    trace_row.push_back(134 + modifier)
            elif aligned_score < deletion_score:
                if insertion_score > deletion_score:    # I
                    dp_row.push_back(insertion_score)
                    trace_row.push_back(73)
                elif insertion_score < deletion_score:  # H # 特殊 deletion
                    dp_row.push_back(deletion_score)
                    trace_row.push_back(72)
                else:                                   # IH
                    dp_row.push_back(deletion_score)
                    trace_row.push_back(145)
            else:
                if aligned_score < insertion_score:     # I
                    dp_row.push_back(insertion_score)
                    trace_row.push_back(73)
                elif aligned_score > insertion_score:   # H= or HX
                    dp_row.push_back(deletion_score)
                    trace_row.push_back(133 + modifier)
                else:                                   # IH= or IHX
                    dp_row.push_back(deletion_score)
                    trace_row.push_back(206 + modifier)
        dp.push_back(dp_row)
        trace.push_back(trace_row)
    else:
        # 初期化のみ
        for j in range(N_ref_seq):
            dp_row.push_back(0)
            trace_row.push_back(72)
        dp.push_back(dp_row)
        trace.push_back(trace_row)
    #####################
    # DP of query_seq_2 #
    #####################
    for i in range(N_query_seq_2):
        q = query_seq_2[i]
        i += N_query_seq_1
        # 初期化
        dp_row.clear()
        if i == 0:
            dp_row.push_back(dp[i][0] - gap_open_penalty)
        else:
            dp_row.push_back(dp[i][0] - gap_extend_penalty)
        trace_row.clear()
        trace_row.push_back(73)     # タテに進むと insertion
        # 値格納していく
        for j in range(N_ref_seq):
            r = ref_seq[j]
            # aligned_score
            if q == r:
                aligned_score = dp[i][j] + match_score
                modifier = 0
            else:
                aligned_score = dp[i][j] + mismatch_score
                modifier = 27
            # insertion_score   # タテに進むと insertion
            if trace[i][j+1] in (73, 141, 134, 202, 161, 229, 145, 206, 233):
                insertion_score = dp[i][j+1] - gap_extend_penalty
            else:
                insertion_score = dp[i][j+1] - gap_open_penalty
            # deletion_score   # ヨコに進むと deletion
            if trace_row[j] in (68, 141, 129, 202, 156, 229):
                deletion_score = dp_row[j] - gap_extend_penalty
            else:
                deletion_score = dp_row[j] - gap_open_penalty
            # set dp and trace
            if aligned_score > deletion_score:
                if aligned_score > insertion_score:     # = or X
                    dp_row.push_back(aligned_score)
                    trace_row.push_back(61 + modifier)
                elif aligned_score < insertion_score:   # I
                    dp_row.push_back(insertion_score)
                    trace_row.push_back(73)
                else:                                   # I= or IX
                    dp_row.push_back(insertion_score)
                    trace_row.push_back(134 + modifier)
            elif aligned_score < deletion_score:
                if insertion_score > deletion_score:    # I
                    dp_row.push_back(insertion_score)
                    trace_row.push_back(73)
                elif insertion_score < deletion_score:  # D
                    dp_row.push_back(deletion_score)
                    trace_row.push_back(68)
                else:                                   # ID
                    dp_row.push_back(deletion_score)
                    trace_row.push_back(141)
            else:
                if aligned_score < insertion_score:     # I
                    dp_row.push_back(insertion_score)
                    trace_row.push_back(73)
                elif aligned_score > insertion_score:   # D= or DX
                    dp_row.push_back(deletion_score)
                    trace_row.push_back(129 + modifier)
                else:                                   # ID= or IDX
                    dp_row.push_back(deletion_score)
                    trace_row.push_back(202 + modifier)
        dp.push_back(dp_row)
        trace.push_back(trace_row)
    #############
    # traceback #
    #############
    # 準備
    i = N_query_seq_1 + N_query_seq_2
    j = N_ref_seq
    cdef vec traceback
    cdef vec score_trace
    cdef LL t
    while True:
        t = trace[i][j]
        score_trace.push_back(dp[i][j])
        if t == 61:     # I
            i -= 1
            j -= 1
            traceback.push_back(61)     # =
        elif t == 88:   # X
            i -= 1
            j -= 1
            traceback.push_back(88)     # X
        elif t == 73:   # I
            i -= 1
            traceback.push_back(73)     # I
        elif t == 68:   # D
            j -= 1
            traceback.push_back(68)     # D
        elif t == 134:  # I=, =優先
            if traceback.empty() or (traceback.back() != 73):
                i -= 1
                j -= 1
                traceback.push_back(61) # =
            else:
                i -= 1
                traceback.push_back(73) # I
        elif t == 129:  # D=: =優先
            if traceback.empty() or (traceback.back() != 68):
                i -= 1
                j -= 1
                traceback.push_back(61) # =
            else:
                j -= 1
                traceback.push_back(68) # D
        elif t == 161:  # IX: X優先
            if traceback.empty() or (traceback.back() != 73):
                i -= 1
                j -= 1
                traceback.push_back(88) # X
            else:
                i -= 1
                traceback.push_back(73) # I
        elif t == 156:  # DX: X優先
            if traceback.empty() or (traceback.back() != 68):
                i -= 1
                j -= 1
                traceback.push_back(88) # X
            else:
                j -= 1
                traceback.push_back(68) # D

        elif t == 141:  # ID: D優先
            if traceback.empty() or (traceback.back() != 73):
                j -= 1
                traceback.push_back(68) # D どちらでも良い場合は deletion を優先？
            else:
                i -= 1
                traceback.push_back(73) # I
        elif t == 202:  # ID=: =優先
            if traceback.empty():
                i -= 1
                j -= 1
                traceback.push_back(61) # =
            elif traceback.back() == 68:
                j -= 1
                traceback.push_back(68) # D
            elif traceback.back() == 73:
                i -= 1
                traceback.push_back(73) # I
            else:
                i -= 1
                j -= 1
                traceback.push_back(61) # =
        elif t == 229:  # IDX: D優先
            if traceback.empty() or (traceback.back() != 73):
                j -= 1
                traceback.push_back(68) # D
            else:
                i -= 1
                traceback.push_back(73) # I
        elif t in (72, 145, 133, 206, 160, 233):    # H優先
            j -= 1
            traceback.push_back(72)
        # 終了判定
        else:
            assert t == 42
            break
    cdef vector[vec] result
    result.push_back(traceback)
    result.push_back(score_trace)
    return result
