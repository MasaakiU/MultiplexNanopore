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
ctypedef long long LL
ctypedef vector[LL] vec
cpdef k_mer_offset_analysis_(
        LL[:] ref_seq_v_repeated, 
        LL[:] query_seq_v, 
        LL N_ref, 
        LL len_query_seq_v
    ):
    cdef LL k
    cdef LL i
    cdef LL s
    cdef vec result_array
    s = 0
    for k in range(N_ref):    # k はオフセット
        for i in range(len_query_seq_v):
            if ref_seq_v_repeated[k + i] == query_seq_v[i]:
                s += 1
        result_array.push_back(s)
        s = 0
    return result_array
