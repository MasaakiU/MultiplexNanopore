# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
color_cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])

from ..modules import msa
from ..modules import my_classes as mc
from ..post_analysis import post_analysis_separate_paths_input
from ..post_analysis import post_analysis_core as pac

default_gen_stats_params_dict = {
    # for alignment
    "gap_open_penalty":3, 
    "gap_extend_penalty":1, 
    "match_score":1, 
    "mismatch_score":-2, 
    # number of k-mers that recognized on reference sequence
    # "base_length_to_observe":1,   # 奇数であること！
    "score_threshold": 0.8, 
}


def gen_stats(seq_info_list, save_dir_base:str = None, **param_dict):
    if save_dir_base is None:
        save_dir_base = Path(os.getcwd())

    df_list = []
    for ref_seq_path, fastq_path_list in seq_info_list:
        df_list.append(gen_stats_core(ref_seq_path, fastq_path_list, save_dir_base, **param_dict))
    # 結合
    master_df = df_list[0]
    for df in df_list[1:]:
        master_df = master_df + df

    # 上書き保存
    pdf_path = save_dir_base / "NanoporeStats_pdf_0.2.0.csv"
    if pdf_path.exists():
        pdf_path.unlink()
    with open(pdf_path, "a") as f:
        f.write("# file_version: 0.2.0\n")
        master_df.to_csv(f, index=True, sep="\t")

    # グラフ保存
    draw_summary_heatmap(master_df, base_order=msa.SequenceBasecallQscorePDF.bases)
    plt.savefig("NanoporeStats_pdf_0.2.0_summary.svg")
    plt.close()
    draw_density_plot_extra(master_df, base_order=msa.SequenceBasecallQscorePDF.bases)
    plt.savefig("NanoporeStats_pdf_0.2.0_density.svg")
    plt.close()


def gen_stats_core(ref_seq_path, fastq_path_list, save_dir_base, **param_dict):
    param_dict = {key: param_dict.get(key, val) for key, val in default_gen_stats_params_dict.items()}

    # execute alignment (and msa)
    save_dir = post_analysis_separate_paths_input([Path(ref_seq_path)], [Path(fastq_path) for fastq_path in fastq_path_list], save_dir_base, **param_dict)

    # get intermediate results
    ir_path = [path for path in save_dir.glob("*.*") if path.name.endswith(".intermediate_results.ir")]
    assert len(ir_path) == 1
    ir = pac.IntermediateResults()
    ir.load(ir_path[0])

    # assignment of reads
    ref_seq = mc.MyRefSeq(ref_seq_path)
    my_fastq = mc.MyFastQ.combine([mc.MyFastQ(fastq_path) for fastq_path in fastq_path_list])
    query_assignment = pac.normalize_scores_and_apply_threshold([ref_seq], my_fastq, ir.result_dict, param_dict)

    df_list = []
    for ref_seq, my_fastq_subset, result_list in query_assignment.iter_assignment_info(ir.result_dict):
        print(f"processing {ref_seq.path.name}...")
        my_msa_aligner = MyMSAligner4Stats(ref_seq, my_fastq_subset, result_list)
        my_msa = my_msa_aligner.execute_1st_MSA(param_dict)
        d = my_msa.generate_stats()
        # import textwrap
        # d = eval(textwrap.dedent("""
        #     {'A_A': [0, 0, 31, 294, 1053, 2118, 3065, 3905, 4331, 4254, 3900, 3709, 3689, 3604, 3595, 3504, 3535, 3598, 3693, 3788, 3893, 3900, 4154, 4334, 4371, 4745, 4925, 5171, 5479, 5969, 6345, 6822, 7509, 8175, 9066, 9684, 10958, 11328, 11728, 12062, 11967, 64002, 0], 'A_T': [0, 0, 1, 14, 26, 38, 33, 39, 38, 35, 15, 15, 8, 11, 7, 8, 7, 3, 1, 3, 4, 3, 2, 0, 1, 1, 1, 3, 3, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'A_C': [0, 0, 2, 21, 56, 72, 78, 75, 78, 52, 56, 42, 36, 24, 22, 22, 11, 13, 8, 10, 5, 4, 3, 2, 3, 1, 3, 0, 3, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], 'A_G': [0, 0, 0, 32, 88, 196, 192, 202, 193, 132, 116, 95, 75, 69, 37, 32, 36, 32, 24, 20, 10, 18, 7, 7, 9, 7, 7, 5, 6, 4, 4, 3, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0], 'A_-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2656], 'T_A': [0, 0, 1, 9, 29, 51, 53, 43, 46, 32, 32, 27, 22, 8, 14, 12, 8, 7, 1, 3, 8, 2, 4, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'T_T': [0, 0, 26, 282, 994, 1941, 2935, 3614, 4052, 4057, 3756, 3679, 3557, 3464, 3423, 3529, 3393, 3539, 3521, 3668, 3770, 3905, 4111, 4221, 4338, 4512, 4785, 5051, 5417, 5738, 6138, 6547, 7100, 7717, 8489, 9447, 10080, 11013, 11424, 12217, 12315, 68864, 0], 'T_C': [0, 0, 2, 33, 77, 148, 172, 196, 189, 140, 132, 95, 68, 50, 53, 35, 28, 25, 27, 14, 16, 13, 15, 17, 6, 9, 9, 6, 5, 2, 3, 5, 0, 3, 5, 1, 1, 2, 0, 0, 0, 3, 0], 'T_G': [0, 0, 1, 20, 55, 82, 101, 94, 62, 55, 43, 34, 36, 18, 24, 19, 16, 13, 6, 13, 5, 2, 3, 5, 1, 5, 3, 1, 2, 4, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], 'T_-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2298], 'C_A': [0, 0, 0, 16, 75, 109, 101, 102, 95, 80, 57, 50, 47, 31, 23, 32, 17, 13, 16, 8, 8, 4, 9, 9, 7, 2, 5, 4, 5, 0, 3, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0], 'C_T': [0, 0, 3, 44, 101, 176, 226, 244, 220, 222, 171, 125, 103, 112, 75, 62, 50, 41, 37, 36, 27, 19, 20, 12, 16, 14, 18, 8, 8, 4, 7, 0, 2, 4, 2, 1, 2, 0, 1, 0, 0, 3, 0], 'C_C': [0, 0, 39, 352, 1349, 2672, 4019, 4993, 5580, 5400, 5145, 4911, 4843, 4660, 4749, 4664, 4790, 4743, 4877, 5018, 5192, 5123, 5542, 5750, 6219, 6106, 6418, 6778, 7294, 7703, 8090, 8896, 9506, 10140, 11000, 11831, 12339, 12878, 13152, 13366, 13239, 70800, 0], 'C_G': [0, 0, 4, 24, 87, 158, 202, 147, 167, 145, 118, 71, 68, 52, 54, 35, 35, 41, 26, 21, 16, 18, 10, 12, 4, 8, 5, 6, 6, 3, 4, 4, 0, 3, 0, 1, 0, 1, 1, 0, 0, 2, 0], 'C_-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3417], 'G_A': [0, 0, 5, 36, 98, 166, 215, 235, 250, 191, 183, 140, 115, 98, 85, 74, 39, 47, 40, 27, 32, 20, 20, 23, 15, 13, 16, 11, 9, 11, 4, 2, 2, 3, 1, 4, 0, 4, 1, 0, 1, 0, 0], 'G_T': [0, 0, 1, 29, 70, 90, 115, 116, 107, 88, 63, 36, 35, 27, 23, 15, 17, 18, 9, 11, 12, 9, 6, 8, 3, 4, 3, 3, 0, 4, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0], 'G_C': [0, 0, 3, 32, 70, 145, 153, 135, 149, 118, 101, 81, 66, 47, 54, 36, 22, 31, 14, 15, 10, 13, 6, 5, 5, 10, 3, 2, 4, 2, 5, 4, 2, 1, 0, 1, 1, 2, 2, 1, 0, 0, 0], 'G_G': [0, 0, 28, 368, 1343, 2643, 3891, 5011, 5584, 5417, 5044, 4795, 4842, 4631, 4703, 4481, 4809, 4834, 5087, 5024, 5044, 5403, 5590, 5764, 6101, 6502, 6884, 7073, 7739, 8203, 8786, 9309, 10136, 10962, 12043, 12926, 13797, 14472, 14566, 14877, 14527, 74063, 0], 'G_-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3731], '-_A': [0, 0, 234, 583, 642, 618, 583, 511, 434, 398, 290, 263, 230, 212, 142, 137, 145, 117, 121, 99, 97, 92, 108, 97, 88, 88, 93, 76, 73, 96, 88, 111, 109, 101, 142, 121, 118, 148, 141, 144, 121, 609, 0], '-_T': [0, 1, 290, 655, 686, 692, 553, 551, 461, 334, 285, 256, 219, 200, 181, 140, 126, 118, 99, 93, 86, 95, 78, 97, 97, 102, 102, 104, 88, 99, 102, 116, 112, 102, 110, 139, 124, 127, 133, 132, 124, 716, 0], '-_C': [0, 0, 232, 464, 522, 522, 514, 493, 360, 315, 261, 228, 217, 184, 170, 148, 126, 133, 127, 101, 94, 88, 94, 110, 106, 102, 123, 94, 115, 101, 120, 102, 101, 100, 106, 131, 124, 126, 125, 128, 141, 577, 0], '-_G': [0, 2, 267, 504, 530, 525, 502, 474, 411, 360, 319, 253, 214, 205, 171, 130, 118, 146, 117, 106, 98, 95, 111, 117, 100, 99, 109, 91, 107, 97, 106, 124, 123, 118, 122, 121, 114, 151, 119, 149, 129, 594, 0], '-_-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4588497]}
        # """).strip())
        df_list.append(pd.DataFrame.from_dict(d).rename(index={42:-1}).sort_index())
    assert len(df_list) == 1
    return df_list[0]

class MyMSAligner4Stats(msa.MyMSAligner):
    def execute_1st_MSA(self, param_dict: dict):
        # set offset for queries
        query_seq_list = []
        q_scores_list = []
        my_cigar_list = []
        query_seq_offset_list = []
        for my_result, (query_seq, q_scores) in zip(self.result_list, self.my_fastq_subset.values()):
            query_seq_list.append(mc.MySeq.set_offset_core(query_seq, my_result.new_query_seq_offset))
            q_scores_list.append(mc.MySeq.set_offset_core(list(q_scores), my_result.new_query_seq_offset))
            my_cigar_list.append(my_result.my_cigar)
            query_seq_offset_list.append((my_result.new_query_seq_offset - 1)%len(query_seq) + 1)   # new_query_seq_offset == 0 の場合は、query_seq_offset = len(query_seq) とする
        # MSA 1st step
        return msa.MyMSA.generate_msa(self.ref_seq, query_seq_list, q_scores_list, my_cigar_list, query_seq_offset_list, param_dict)

def draw_summary_heatmap(df_stats: pd.DataFrame, base_order: str=None, **kwargs):
    summary_matrix = summary_df_2_matrix(df_stats, base_order=base_order, **kwargs)
    summary_matrix = summary_matrix / summary_matrix.sum(axis=1, keepdims=True)

    vmin, vmax = 0, 0.008#1#

    fig =plt.figure(figsize=(4, 4))
    ax = plt.subplot(1,1,1)
    im = plt.imshow(summary_matrix, cmap="YlGn", vmin=vmin, vmax=vmax)

    bar = plt.colorbar(im, fraction=0.046, pad=0.04)
    # Loop over data dimensions and create text annotations.
    for i in range(summary_matrix.shape[0]):
        for j in range(summary_matrix.shape[1]):
            if np.isnan(summary_matrix[i, j]):
                continue
            value = f"{np.round(summary_matrix[i, j], 3):0<5}"
            if np.absolute(summary_matrix[i, j]) < vmax/2:
                text = ax.text(j, i, value, ha="center", va="center", color="k", fontsize=10)
            else:
                text = ax.text(j, i, value, ha="center", va="center", color="w", fontsize=10)

    # Show all ticks and label them with the respective list entries
    # x
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(len(base_order)))
    ax.set_xticklabels(labels=base_order, fontsize=14)
    ax.set_xlabel("Base calling", fontsize=16, labelpad=10)
    
    # y
    ax.set_yticks(np.arange(len(base_order)))
    ax.set_yticklabels(labels=base_order, fontsize=14)
    ax.set_ylabel("True base", fontsize=16)
    plt.tight_layout()

def summary_df_2_matrix(summary_df:pd.DataFrame, base_order=None, **kwargs):
    crushed = summary_df.sum(axis=0)
    summary_matrix = np.zeros(shape=(len(base_order), len(base_order)), dtype=float)
    for k, v in crushed.items():
        m = re.match(r"(.+)_(.+)", k)
        ref_base = m.group(1)
        query_base = m.group(2)
        summary_matrix[base_order.index(ref_base), base_order.index(query_base)] = v
    return summary_matrix

def draw_density_plot_extra(combined_df, base_order=None, **kwargs):
    combined_df_normalized = combined_df.div(combined_df.sum(axis=0), axis=1)
    # 描画
    fig =plt.figure(figsize=(7.5, 7.5))
    # draw_core
    ax = None
    subplot_idx = 1
    N_rows = len(base_order)
    N_cols = len(base_order)
    for row, ref_base in enumerate(base_order):
        for col, query_base in enumerate(base_order[:-1]):  # "gap" does not have q-scores
            subplot_idx = row * N_cols + col + 1
            ax =fig.add_subplot(N_rows, N_cols, subplot_idx, sharex=ax, sharey=ax)
            counts = combined_df_normalized[f"{ref_base}_{query_base}"].values
            x = combined_df.index.values
            ax.hist(x=x, weights=counts, bins=np.arange(-1, 42), density=True, color=color_cycle[int(ref_base == query_base)])
            ax.axhline(y=0, c="gray", linewidth=0.5, zorder=0)
            # y-label
            if subplot_idx%N_cols != 1:
                ax.yaxis.set_ticks_position('none')
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(ref_base, rotation=0, fontsize=14, labelpad=25)
            # x-lable
            if (subplot_idx - 1) // N_cols < N_rows - 1:
                ax.xaxis.set_ticks_position('none')
                plt.setp(ax.get_xticklabels(), visible=False)
            if (subplot_idx - 1) // N_cols == 0:
                ax.xaxis.set_label_position('top')
                ax.set_xlabel(query_base, rotation=0, fontsize=14, labelpad=6)
                ax.set_xticks([0, 20, 40])
                ax.set_xticklabels(["0", "20", "40"])
            subplot_idx += 1
    # set aspect after setting the ylim
    plt.ylim(-0.01, 0.13)
    aspect = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    for ax in fig.axes:
        ax.set_aspect(aspect, adjustable='box')

    # add extra ax
    extra_ax = None
    for idx, b in enumerate(base_order[:-1]):    # ATCG except for "gap"
        subplot_idx = idx * 5 + 5
        extra_ax = fig.add_subplot(N_rows, N_cols, subplot_idx, sharex=ax, sharey=extra_ax)
        counts = combined_df_normalized[f"{b}_{b}"].values
        x = combined_df.index.values
        extra_ax.hist(x=x, weights=counts, bins=np.arange(-1, 42), density=True, color=color_cycle[1])
        extra_ax.axhline(y=0, c="gray", linewidth=0.5, zorder=0)
        # # y-label
        # ax.yaxis.set_ticks_position('none')
        # plt.setp(extra_ax.get_yticklabels(), visible=False)
        # x-lable
        if (subplot_idx - 1) // N_cols < N_rows - 2:
            extra_ax.xaxis.set_ticks_position('none')
            plt.setp(extra_ax.get_xticklabels(), visible=False)
        if (subplot_idx - 1) // N_cols == 0:
            extra_ax.xaxis.set_label_position('top')
            extra_ax.set_xlabel("full-size view of\ndiagonal panel", rotation=0, fontsize=12, labelpad=6)
    else:
        extra_ax.set_ylim(-0.01 * 4.2, 0.13 * 4.2)

    # label
    legend_elements = [
        Patch(facecolor=color_cycle[1], label='correct base calling'), 
        Patch(facecolor=color_cycle[0], label='incorrect base calling'), 
    ]
    fig.legend(handles=legend_elements, loc="lower right", borderaxespad=0.2)
    # fig.suptitle("alignment score scatter")

    plt.subplots_adjust(hspace=0, wspace=0, left=0.19, right=0.99, bottom=0.1, top=0.9)
    # suplabel
    fig.text(0.115, 0.5, 'density', va='center', rotation=90, fontsize=12)
    fig.text(0.05, 0.5, 'True base', va='center', rotation=90, fontsize=16)
    fig.text(0.5, 0.05, 'quality score', va='center', rotation=0, fontsize=12)
    fig.text(0.5, 0.96, 'Base calling', va='center', rotation=0, fontsize=16)

