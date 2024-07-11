# -*- coding: utf-8 -*-

import re
import io
import copy
import struct
import zipfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PilImage
from spoa import poa
from tqdm import tqdm
from typing import List
from pathlib import Path
from itertools import product
from collections import defaultdict, Counter
from multiprocessing import Pool
from numpy.core.memmap import uint8
from matplotlib.patches import Patch

from . import my_classes as mc
from . import ref_query_alignment as rqa
from .cython_functions import alignment_functions as af

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':[mc.sans_serif_font_master]})

class QueryAssignment():
    query_assignment_version = "qa_0.2.0"
    def __init__(self, ref_seq_list, my_fastq, result_dict) -> None:
        assert len(result_dict) == len(my_fastq)
        self.ref_seq_list = ref_seq_list
        self.my_fastq = my_fastq
        # assignment_info
        is_my_cigar_empty = np.array([result.my_cigar == "" for result_list in result_dict.values() for result in result_list]).reshape(self.N_query, self.N_ref_seq_list * 2)
        self.score_table = np.array([result.score for result_list in result_dict.values() for result in result_list], dtype=int).reshape(self.N_query, self.N_ref_seq_list * 2) # reverse compliment があるため、行は2倍
        normalized_score_table_pre = np.divide(self.score_table, np.array([len(ref_seq) for ref_seq in self.ref_seq_list for i in range(2)])[np.newaxis, :])                   # shape: (self.N_query, self.N_ref_seq_list * 2), dtype=float) # reverse compliment があるため、行は2倍
        normalized_score_table_pre[is_my_cigar_empty] = np.nan
        self.normalized_score_table = np.ma.masked_array(normalized_score_table_pre, mask=is_my_cigar_empty)
        self.normalized_score_table_summary = self.normalized_score_table.reshape(self.N_query, self.N_ref_seq_list, 2).max(axis=-1)
        self.normalized_score_table_summary[np.ma.getmask(self.normalized_score_table_summary)] = 0
        self.normalized_score_table_summary[self.normalized_score_table_summary < 0] = 0
        self.assignment_table = np.empty((self.N_query, 3), dtype=int)       # column_names: (classified_ref_seq_idx, is_reverse_compliment, assigned)
        self.score_threshold = None
    @property
    def N_query(self):
        return len(self.my_fastq)
    @property
    def N_ref_seq_list(self):
        return len(self.ref_seq_list)
    def set_assignment(self, score_threshold):
        self.score_threshold = score_threshold
        # classified_ref_seq_idx
        assigned_result_idx_list = (self.normalized_score_table).argmax(axis=1) # すべて np.nan の場合は 0 になる
        self.assignment_table[:, 0] = assigned_result_idx_list // 2
        # is_reverse_compliment
        self.assignment_table[:, 1] = assigned_result_idx_list % 2
        # assigned
        max_values = self.normalized_score_table[(range(self.N_query), assigned_result_idx_list)]
        max_values_count = (self.normalized_score_table == max_values[:, np.newaxis]).sum(axis=1)
        self.assignment_table[(max_values_count > 1).filled(True), :2] = -1                         # max値が複数ある場合、もしくは all nan の場合は classify/is_rc ができない
        self.assignment_table[:, 2] = (max_values > self.score_threshold) * (max_values_count == 1) # アサインされたかどうかを追加
        self.assignment_table[np.ma.getmask(max_values), 2] = -1                           # all nan の場合は アサイン を定義しない
    def iter_assignment_info(self, result_dict):
        assigned_groups = []
        for ref_seq_idx, ref_seq in enumerate(self.ref_seq_list):
            # ref_seq のうち、アサインされたもののサブセットを取得
            query_id_rc_list = [
                (query_id, is_rc) for ((classified_ref_seq_idx, is_rc, is_assigned), query_id) in zip(self.assignment_table, self.my_fastq.keys())
                if (classified_ref_seq_idx == ref_seq_idx) & (is_assigned == 1)
            ]
            my_fastq_subset = self.my_fastq.get_partially_rc_subset(query_id_rc_list)
            # results_dict から、目的のresult だけを取得
            result_list = [result_dict[query_id][ref_seq_idx * 2 + is_rc] for query_id, is_rc in query_id_rc_list]
            # 出力に追加
            assigned_groups.append([ref_seq, my_fastq_subset, result_list])
        return assigned_groups
    def save_scores(self, save_dir):
        ref_seq_column_names = [f"{ref_seq.path.name} (idx={ref_seq_idx}{t})" for ref_seq_idx, ref_seq in enumerate(self.ref_seq_list) for t in ["", ",rc"]]
        ref_seq_column_names_norm = [f"{ref_seq.path.name} (idx={ref_seq_idx}{t}, normalized)" for ref_seq_idx, ref_seq in enumerate(self.ref_seq_list) for t in ["", ",rc"]]
        df = pd.DataFrame(
            columns=["query_id"] + ref_seq_column_names + ref_seq_column_names_norm + ["classified_ref_seq_idx", "is_reverse_compliment", "assigned"]
        )
        df["query_id"] = self.my_fastq.keys()
        df.iloc[:, 1:self.N_ref_seq_list * 2 + 1] = self.score_table
        df.iloc[:, self.N_ref_seq_list * 2 + 1:self.N_ref_seq_list * 4 + 1] = np.ma.getdata(self.normalized_score_table)
        df.iloc[:, self.N_ref_seq_list * 4 + 1:] = self.assignment_table

        df.index.name = "query_idx"
        df = df. reset_index()
        df.to_csv(save_dir / f"{self.my_fastq.combined_name_stem}.summary_scores.csv", sep="\t", index=False, na_rep="NaN")

    # drawing functions
    def draw_distributions(self, save_dir=None, display_plot=None):
        assert self.N_query == self.score_table.shape[0] == self.normalized_score_table.shape[0] == self.assignment_table.shape[0]
        assert self.N_ref_seq_list * 2 == self.score_table.shape[1] == self.normalized_score_table.shape[1]

        # データ収集
        assignment_set_4_read_length = [[] for i in range(self.N_ref_seq_list + 1)]   # last one is for idx=-1 (not assigned)
        assignment_set_4_q_scores = [[] for i in range(self.N_ref_seq_list + 1)]   # last one is for idx=-1 (not assigned)
        for (classified_ref_seq_idx, is_reverse_compliment, assigned), (query_id, (query_seq, q_scores)) in zip(self.assignment_table, self.my_fastq.items()):
            if assigned == 0:
                classified_ref_seq_idx = -1
            assignment_set_4_read_length[classified_ref_seq_idx].append(len(query_seq))
            assignment_set_4_q_scores[classified_ref_seq_idx].extend(q_scores)

        # スタイル
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        focused_color1 = color_cycle[0]
        focused_color2 = color_cycle[1]
        not_assigned_color = "grey"
        # サイズ
        rows = self.N_ref_seq_list
        columns = 3
        wspace_unit = 1.0
        hspace_unit = 0.2
        left_margin_unit = 0.2
        top_margin_unit = 0.5
        right_margin_unit = 0.2
        bottom_margin_unit = 0.5
        left_header_unit = 1.3
        fig_w_unit = 4.0
        fig_h_unit = 2.0
        fig_width_unit = (fig_w_unit + wspace_unit) * (columns - 1) + left_margin_unit + right_margin_unit + left_header_unit
        fig_height_unit = (fig_h_unit + hspace_unit) * (rows - 1) + top_margin_unit + bottom_margin_unit + fig_h_unit
        left = left_margin_unit / fig_width_unit
        right = 1 - right_margin_unit / fig_width_unit
        top = 1 - top_margin_unit / fig_height_unit
        bottom = bottom_margin_unit / fig_height_unit
        wspace = wspace_unit * (columns - 1) / fig_width_unit
        hspace = hspace_unit * (rows - 1) / fig_height_unit
        # 描画
        fig = plt.figure(figsize=(fig_width_unit, fig_height_unit), clear=True)
        widths = [left_header_unit] + [fig_w_unit] * (columns - 1)
        heights = [fig_h_unit] * rows
        spec = fig.add_gridspec(ncols=columns, nrows=rows, width_ratios=widths, height_ratios=heights)

        # # 描画パラメータ
        # fig = plt.figure(figsize=(4 * columns, 2 * rows), clear=True)
        # fig.subplots_adjust(hspace=0.05, wspace=0.05)
        # widths = [1] + [2 for i in range(columns - 1)]
        # # heights = [1 for i in range(rows)]
        # spec = fig.add_gridspec(ncols=columns, nrows=rows, width_ratios=widths)#, height_ratios=heights)

        ###########
        # labeles #
        ###########
        column_idx = 0
        text_wrap = 15
        for ref_seq_idx, ref_seq in enumerate(self.ref_seq_list):
            ref_seq_name = ref_seq.path.name
            ax = fig.add_subplot(spec[ref_seq_idx, column_idx])
            refseq_name_wrapped = "\n".join([ref_seq_name[i:i+text_wrap] for i in range(0, len(ref_seq_name), text_wrap)])
            ax.text(0.1, 0.6, refseq_name_wrapped, ha='left', va='center', wrap=True, family="monospace")
            ax.set_axis_off()
        legend_elements = [
            Patch(facecolor=focused_color1, label='Focused plasmid'), 
            Patch(facecolor=focused_color2, label='Other plasmid(s)'), 
            Patch(facecolor=not_assigned_color, label='Not assigned')
        ]
        fig.legend(handles=legend_elements, loc="lower left", borderaxespad=0)

        ############################
        # read length distribution #
        ############################
        column_idx = 1
        # assignment ごとにヒートマップを描画
        bin_unit = 100
        bins = range(0, int(np.ceil(max(max(v) if len(v) > 0 else bin_unit for v in assignment_set_4_read_length) / bin_unit) * bin_unit), bin_unit)
        for ref_seq_idx in range(self.N_ref_seq_list):
            hist_params = dict(
                x=assignment_set_4_read_length[-2::-1] + assignment_set_4_read_length[-1:], 
                color=[focused_color1 if i == ref_seq_idx else focused_color2 for i in range(self.N_ref_seq_list)][::-1] + [not_assigned_color], 
                bins=bins, 
                histtype='bar', 
                stacked=True
            )
            # 描画
            ax0 = fig.add_subplot(spec[ref_seq_idx, column_idx])
            ax0.hist(**hist_params)
            ax0.set_ylabel("count")
            # # log scale
            # ax1 = fig.add_subplot(spec[ref_seq_idx, column_idx + 1])
            # ax1.hist(**hist_params)
            # ax1.set_yscale("log")
            # ax1.set_ylabel("count")
            if ref_seq_idx == 0:
                ax0.set_title("read length distribution")
                # ax1.set_title("read length distribution (log)")
            if ref_seq_idx == self.N_ref_seq_list - 1:
                ax0.set_xlabel("Read length (bp)")
                # ax1.set_xlabel("Read length (bp)")
            else:
                ax0.set_xticklabels([])
                # ax1.set_xticklabels([])

        ########################
        # q_score distribution #
        ########################
        column_idx = 2
        for ref_seq_idx in range(self.N_ref_seq_list):
            hist_params = dict(
                x=assignment_set_4_q_scores[-2::-1] + assignment_set_4_q_scores[-1:], 
                color=[focused_color1 if i == ref_seq_idx else focused_color2 for i in range(self.N_ref_seq_list)][::-1] + [not_assigned_color], 
                bins=np.arange(42), 
                histtype='bar', 
                stacked=True, 
                density=True
            )
            # 描画
            ax0 = fig.add_subplot(spec[ref_seq_idx, column_idx])
            ax0.hist(**hist_params)

            # labels
            ax0.set_ylabel("density")
            if ref_seq_idx == 0:
                ax0.set_title("Q-score distribution")
            if ref_seq_idx == self.N_ref_seq_list - 1:
                ax0.set_xlabel("Q-score")
            else:
                ax0.set_xticklabels([])

        fig.subplots_adjust(hspace=hspace, wspace=wspace, left=left, right=right, bottom=bottom, top=top)

        if save_dir is not None:
            plt.savefig(Path(save_dir) / f"{self.my_fastq.combined_name_stem}.summary_distribution.svg")
        if display_plot is None:
            return
        elif display_plot:
            plt.show()
        else:
            plt.clf()
            plt.close()
    def draw_alignment_score_scatters(self, save_dir=None, display_plot=False):
        assert self.N_query == self.score_table.shape[0] == self.normalized_score_table.shape[0] == self.assignment_table.shape[0]
        assert self.N_ref_seq_list * 2 == self.score_table.shape[1] == self.normalized_score_table.shape[1]

        # スタイル
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        focused_color1 = color_cycle[0]
        focused_color2 = color_cycle[1]
        not_assigned_color = "grey"
        # サイズ
        rows = columns = self.N_ref_seq_list + 1
        hspace_unit = wspace_unit = 0.1
        left_margin_unit = 0.0
        top_margin_unit = 0.2
        right_margin_unit = 0.2
        bottom_margin_unit = 0.5
        figsize_unit = 2.0
        left_header_unit = 2.0
        top_header_unit = 1.0
        fig_width_unit = (figsize_unit + wspace_unit) * (columns - 1) + left_header_unit + left_margin_unit + right_margin_unit
        fig_height_unit = (figsize_unit + hspace_unit) * (rows - 1) + top_header_unit + top_margin_unit + bottom_margin_unit
        left = left_margin_unit / fig_width_unit
        right = 1 - right_margin_unit / fig_width_unit
        top = 1 - top_margin_unit / fig_height_unit
        bottom = bottom_margin_unit / fig_height_unit
        wspace = wspace_unit * (columns - 1) / fig_width_unit
        hspace = hspace_unit * (rows - 1) / fig_height_unit
        # 描画
        fig = plt.figure(figsize=(fig_width_unit, fig_height_unit), clear=True)
        widths = [left_header_unit] + [figsize_unit] * (columns - 1)
        heights = [top_header_unit] + [figsize_unit] * (rows - 1)
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
        diagonal_hist_data = []
        other_axes = []
        for ref_seq_idx_1, ref_seq_idx_2 in product(range(self.N_ref_seq_list), range(self.N_ref_seq_list)):
            ax = fig.add_subplot(spec[ref_seq_idx_1 + 1, ref_seq_idx_2 + 1]) # 原点を左上にに取った！
            if ref_seq_idx_1 == ref_seq_idx_2:
                diagonal_axes.append(ax)
                hist_params = dict(
                    x=[
                        self.normalized_score_table_summary[(self.assignment_table[:, 0] == ref_seq_idx_1) * (self.assignment_table[:, 2] == 1), ref_seq_idx_1], 
                        self.normalized_score_table_summary[(self.assignment_table[:, 0] != ref_seq_idx_1) * (self.assignment_table[:, 2] == 1), ref_seq_idx_1], 
                        self.normalized_score_table_summary[self.assignment_table[:, 2] == 0, ref_seq_idx_1], 
                    ], 
                    color=[focused_color1, focused_color2, not_assigned_color], 
                    bins=np.linspace(0, 1, 100), 
                    histtype='bar', 
                    stacked=True, 
                    density=True
                )
                counts, bins, bars = ax.hist(**hist_params)
                diagonal_hist_data.append(counts[-1, 1:].max())
            else:
                other_axes.append(ax)
                scatter_params = dict(s=5, alpha=0.3)
                ax.scatter(
                    x=self.normalized_score_table_summary[(self.assignment_table[:, 0] == ref_seq_idx_2) * (self.assignment_table[:, 2] == 1), ref_seq_idx_2], 
                    y=self.normalized_score_table_summary[(self.assignment_table[:, 0] == ref_seq_idx_2) * (self.assignment_table[:, 2] == 1), ref_seq_idx_1], 
                    color=focused_color1, 
                    **scatter_params
                )
                ax.scatter(
                    x=self.normalized_score_table_summary[(self.assignment_table[:, 0] != ref_seq_idx_2) * (self.assignment_table[:, 2] == 1), ref_seq_idx_2], 
                    y=self.normalized_score_table_summary[(self.assignment_table[:, 0] != ref_seq_idx_2) * (self.assignment_table[:, 2] == 1), ref_seq_idx_1], 
                    color=focused_color2, 
                    **scatter_params
                )
                ax.scatter(
                    x=self.normalized_score_table_summary[self.assignment_table[:, 2] == 0, ref_seq_idx_2], 
                    y=self.normalized_score_table_summary[self.assignment_table[:, 2] == 0, ref_seq_idx_1], 
                    color=not_assigned_color, 
                    **scatter_params
                )
                plot_params = dict(c="k", linestyle="--", linewidth=1)
                ax.plot((self.score_threshold, self.score_threshold), (0, self.score_threshold), **plot_params)
                ax.plot((0, self.score_threshold), (self.score_threshold, self.score_threshold), **plot_params)
                ax.plot((self.score_threshold, 1), (self.score_threshold, 1), **plot_params)
                ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
            if ref_seq_idx_2 != 0:
                # ax.yaxis.set_ticks_position('none')
                ax.set(ylabel=None)
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                if ref_seq_idx_1 == ref_seq_idx_2:
                    ax.set_ylabel("density")
                else:
                    ax.set_ylabel("normalized alignment score")
                    ax.set_yticks(np.linspace(0, 1, 6))
                    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
            if ref_seq_idx_1 != rows - 2:
                # ax.xaxis.set_ticks_position('none')
                ax.set(xlabel=None)
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("normalized alignment score")

        text_wrap = 15
        for ref_seq_idx, ref_seq in enumerate(self.ref_seq_list):
            ref_seq_name = ref_seq.path.name
            ax = fig.add_subplot(spec[0, ref_seq_idx + 1])
            ref_seq_name_wrapped = "\n".join([ref_seq_name[i:i+text_wrap] for i in range(0, len(ref_seq_name), text_wrap)])
            ax.text(0.15, 0.1, ref_seq_name_wrapped, ha='left', va='bottom', wrap=True, family="monospace")
            ax.set_axis_off()

            ax = fig.add_subplot(spec[ref_seq_idx + 1, 0])
            ref_seq_name_wrapped = "\n".join([ref_seq_name[i:i+text_wrap] for i in range(0, len(ref_seq_name), text_wrap)])
            ax.text(0.1, 0.75, ref_seq_name_wrapped, ha='left', va='center', wrap=True, family="monospace")
            ax.set_axis_off()

        # set aspect after setting the ylim
        range_max = max(diagonal_hist_data) * 1.05
        for ax in diagonal_axes:
            ax.set_ylim(0, range_max)
        aspect_diagonal = np.diff(ax.get_xlim()) / np.diff(ax.get_ylim())
        for ax in diagonal_axes:
            ax.set_aspect(aspect_diagonal, adjustable='box')
        for ax in other_axes:
            ax.set_aspect(1, adjustable='box')

        fig.subplots_adjust(hspace=hspace, wspace=wspace, left=left, right=right, bottom=bottom, top=top)

        if save_dir is not None:
            plt.savefig(Path(save_dir) / f"{self.my_fastq.combined_name_stem}.summary_scatter.svg")
        if display_plot is None:
            return
        elif display_plot:
            plt.show()
        else:
            plt.clf()
            plt.close()
    def draw_alignment_score_scatters_rotated(self, save_dir=None, display_plot=False):
        assert self.N_query == self.score_table.shape[0] == self.normalized_score_table.shape[0] == self.assignment_table.shape[0]
        assert self.N_ref_seq_list * 2 == self.score_table.shape[1] == self.normalized_score_table.shape[1]

        # スタイル
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        focused_color1 = color_cycle[0]
        focused_color2 = color_cycle[1]
        not_assigned_color = "grey"
        # サイズ
        rows = columns = self.N_ref_seq_list + 1
        hspace_unit = 0.2
        wspace_unit = 0.6
        left_margin_unit = 0.0
        top_margin_unit = 0.0
        right_margin_unit = 0.4
        bottom_margin_unit = 0.8
        figsize_unit = 1.8
        left_header_unit = 1.0
        top_header_unit = 1.5
        fig_width_unit = (figsize_unit + wspace_unit) * (columns - 1) + left_header_unit + left_margin_unit + right_margin_unit
        fig_height_unit = (figsize_unit + hspace_unit) * (rows - 1) + top_header_unit + top_margin_unit + bottom_margin_unit
        left = left_margin_unit / fig_width_unit
        right = 1 - right_margin_unit / fig_width_unit
        top = 1 - top_margin_unit / fig_height_unit
        bottom = bottom_margin_unit / fig_height_unit
        wspace = wspace_unit * (columns - 1) / fig_width_unit
        hspace = hspace_unit * (rows - 1) / fig_height_unit
        # 範囲
        lim_min = -0.05
        lim_max = 1.05
        # 描画
        fig = plt.figure(figsize=(fig_width_unit, fig_height_unit), clear=True)
        widths = [left_header_unit] + [figsize_unit] * (columns - 1)
        heights = [top_header_unit] + [figsize_unit] * (rows - 1)
        spec = fig.add_gridspec(ncols=columns, nrows=rows, width_ratios=widths, height_ratios=heights)

        # label
        legend_elements = [
            Patch(facecolor=color_cycle[0], label='Focused plasmid'), 
            Patch(facecolor=color_cycle[1], label='Other plasmid(s)'), 
            Patch(facecolor="grey", label='Not assigned')
        ]
        fig.legend(handles=legend_elements, loc="lower right", borderaxespad=0.2)
        # fig.suptitle("alignment score scatter")

        ######################
        # score distribution #
        ######################
        rotation_matrix = np.array(
            [[2 ** (-1/2), -2 ** (-1/2)], 
             [2 ** (-1/2), 2 ** (-1/2)]]
        )
        # diagonal_axes = []
        other_axes = []
        for (ref_seq_idx_1, ref_seq_idx_2) in product(range(self.N_ref_seq_list), range(self.N_ref_seq_list)):
            ax = fig.add_subplot(spec[ref_seq_idx_1 + 1, ref_seq_idx_2 + 1]) # 原点を左上にに取った！
            if ref_seq_idx_1 == ref_seq_idx_2:
                fig.delaxes(ax)
            else:
                # 座標回転
                xy_coord_assigned = np.einsum("ij,kj->ik", rotation_matrix, self.normalized_score_table_summary[:, (ref_seq_idx_2, ref_seq_idx_1)]) # shape = (2, len(query))
                xy_coord_assigned[1, :] *= 2 ** (-1/2)                 # 回転した上で、max の値を 1 にする
                # 描画
                other_axes.append(ax)
                scatter_params = dict(s=5, alpha=0.3)
                ax.scatter(
                    x=xy_coord_assigned[0, (self.assignment_table[:, 0] == ref_seq_idx_2) * (self.assignment_table[:, 2] == 1)], 
                    y=xy_coord_assigned[1, (self.assignment_table[:, 0] == ref_seq_idx_2) * (self.assignment_table[:, 2] == 1)], 
                    color=focused_color1, 
                    **scatter_params
                )
                ax.scatter(
                    x=xy_coord_assigned[0, (self.assignment_table[:, 0] != ref_seq_idx_2) * (self.assignment_table[:, 2] == 1)], 
                    y=xy_coord_assigned[1, (self.assignment_table[:, 0] != ref_seq_idx_2) * (self.assignment_table[:, 2] == 1)], 
                    color=focused_color2, 
                    **scatter_params
                )
                ax.scatter(
                    x=xy_coord_assigned[0, self.assignment_table[:, 2] == 0], 
                    y=xy_coord_assigned[1, self.assignment_table[:, 2] == 0], 
                    color=not_assigned_color, 
                    **scatter_params
                )
                x_abs_max = np.absolute(ax.get_xlim()).max()
                ax.plot((0, 0), (self.score_threshold, 1), c="k", linestyle="--", linewidth=1)
                ax.plot((0, self.score_threshold * 2 ** (-1/2)), (self.score_threshold, self.score_threshold/2), c="k", linestyle="--", linewidth=1)
                ax.plot((0, -self.score_threshold * 2 ** (-1/2)), (self.score_threshold, self.score_threshold/2), c="k", linestyle="--", linewidth=1)

                # 軸描画
                ax.plot((0, 2 ** (-1/2) * (lim_max - lim_min)), (lim_min, 1/2), c="k", linewidth=1)
                ax.plot((0, -2 ** (-1/2) * (lim_max - lim_min)), (lim_min, 1/2), c="k", linewidth=1)
                ax.plot((0, 2 ** (-1/2) * (lim_max - lim_min)), (lim_max, 1/2), c="k", linewidth=1)
                ax.plot((0, -2 ** (-1/2) * (lim_max - lim_min)), (lim_max, 1/2), c="k", linewidth=1)
                ax.plot((x_abs_max, x_abs_max), (x_abs_max * 2 ** (-1/2) + lim_min, 1 - x_abs_max * 2 ** (-1/2) - lim_min), c="k", linewidth=2, linestyle=":")
                ax.plot((-x_abs_max, -x_abs_max), (x_abs_max * 2 ** (-1/2) + lim_min, 1 - x_abs_max * 2 ** (-1/2) - lim_min), c="k", linewidth=2, linestyle=":")

                # tick/ticklabels
                ax.set_ylim(lim_min, lim_max)
                ax.set_xlim(-x_abs_max, x_abs_max)
                ax.plot((-lim_min * 2 ** (-1/2), -lim_min * 2 ** (1/2) * 3/4), (lim_min/2, lim_min * 3/4), c="k", linewidth=1)
                ax.plot((lim_min * 2 ** (-1/2), lim_min * 2 ** (1/2) * 3/4), (lim_min/2, lim_min * 3/4), c="k", linewidth=1)
                ax.text(-lim_min * 2 ** (1/2) * 3/4, lim_min * 3/4, "0.0", ha="left", va="top")
                ax.text(lim_min * 2 ** (1/2) * 3/4, lim_min * 3/4, "0.0", ha="right", va="top")

                x_ticks = ax.get_xticks()
                x_tick_labels = ax.get_xticklabels()
                for x_tick, x_tick_label in zip(x_ticks, x_tick_labels):
                    if x_tick > 0:
                        if (x_tick - lim_min) * 2 ** (-1/2) >= x_abs_max:
                            continue
                        ax.plot(((x_tick - lim_min) * 2 ** (-1/2), (x_tick - lim_min * 3/2) * 2 ** (-1/2)), ((x_tick + lim_min)/2, (x_tick + lim_min * 3/2)/2), c="k", linewidth=1)
                        ax.text((x_tick - lim_min * 3/2) * 2 ** (-1/2), (x_tick + lim_min * 3/2)/2, x_tick_label.get_text(), ha="left", va="top")
                    elif x_tick < 0:
                        if (-x_tick - lim_min) * 2 ** (-1/2) >= x_abs_max:
                            continue
                        ax.plot(((x_tick + lim_min) * 2 ** (-1/2), (x_tick + lim_min * 3/2) * 2 ** (-1/2)), ((-x_tick + lim_min)/2, (-x_tick + lim_min * 3/2)/2), c="k", linewidth=1)
                        ax.text((x_tick + lim_min * 3/2) * 2 ** (-1/2), (-x_tick + lim_min * 3/2)/2, x_tick_label.get_text(), ha="right", va="top")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.set(ylabel=None)
                ax.set_yticks([])
                ax.set_yticks([], minor=True)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_xticks([])
                ax.set_xticks([], minor=True)
                plt.setp(ax.get_xticklabels(), visible=False)

                if (ref_seq_idx_1 == rows - 2) or ((ref_seq_idx_1 == rows - 3) and (ref_seq_idx_2 == columns - 2)):
                    ax.set_xlabel("\ndistance from the diagonal line\n[ P2  <––– –––>  P1 ]")

        text_wrap = 15
        for ref_seq_idx, ref_seq in enumerate(self.ref_seq_list):
            ref_seq_name = ref_seq.path.name
            ax = fig.add_subplot(spec[0, ref_seq_idx + 1])
            ref_seq_name_wrapped = "\n".join([ref_seq_name[i:i+text_wrap] for i in range(0, len(ref_seq_name), text_wrap)])
            ax.text(0.15, 0.1, ref_seq_name_wrapped, ha='left', va='bottom', wrap=True, family="monospace")
            ax.set_axis_off()

            ax = fig.add_subplot(spec[ref_seq_idx + 1, 0])
            ref_seq_name_wrapped = "\n".join([ref_seq_name[i:i+text_wrap] for i in range(0, len(ref_seq_name), text_wrap)])
            ax.text(0.1, 0.75, ref_seq_name_wrapped, ha='left', va='center', wrap=True, family="monospace")
            ax.set_axis_off()
        ax = fig.add_subplot(spec[0, 0])
        ax.text(0.75, 0.15, "P2", ha='right', va='top', wrap=True, fontsize=15)
        ax.text(0.85, 0.25, "P1", ha='left', va='bottom', wrap=True, fontsize=15)
        ax.plot((0.6, 1), (0.4, 0), c="k", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        fig.subplots_adjust(hspace=hspace, wspace=wspace, left=left, right=right, bottom=bottom, top=top)

        # set aspect after setting the ylim
        for ax in other_axes:
            aspect = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.set_aspect(aspect, adjustable='box')

        if save_dir is not None:
            plt.savefig(Path(save_dir) / f"{self.my_fastq.combined_name_stem}.summary_rotated.svg")
        if display_plot is None:
            return
        elif display_plot:
            plt.show()
        else:
            plt.clf()
            plt.close()

    # FOR LOG
    def get_assignment_summary(self):
        total = f"TOTAL({self.assignment_table.shape[0]})"
        classified = "  classified"
        assigned = "  assigned"
        not_assigned = "  not_assigned"
        garbage = "  garbage"
        multi_max = "  multi_max"
        read_number = "read#"       # column header
        read_number_rc = "read#_rc" # column header
        sum_number = "sum"          # column header
        index_width = max(len(total), len(classified), len(assigned), len(not_assigned), len(multi_max), len(garbage))
        column_width = max(len(str(self.assignment_table.shape[0])), len(read_number), len(read_number_rc), len(sum_number))

        # 作っていく
        summary_txt = f"{total:<{index_width}} {read_number:<{column_width}} {read_number_rc:<{column_width}} {sum_number:<{column_width}}"
        summary_txt += (
            f"\n{classified:<{index_width}}"    # (not garbage) and (not multi_max)
            f" {(self.assignment_table[:, 1] == 0).sum():<{column_width}}"
            f" {(self.assignment_table[:, 1] == 1).sum():<{column_width}}"
            f" {(self.assignment_table[:, 1] != -1).sum():<{column_width}}"
            f"\n{assigned:<{index_width}}"      # (not garbage) and (not multi max) and (over threshold)
            f" {((self.assignment_table[:, 1] == 0) * (self.assignment_table[:, 2] == 1)).sum():<{column_width}}"
            f" {((self.assignment_table[:, 1] == 1) * (self.assignment_table[:, 2] == 1)).sum():<{column_width}}"
            f" {((self.assignment_table[:, 1] != -1) * (self.assignment_table[:, 2] == 1)).sum():<{column_width}}"
            f"\n{not_assigned:<{index_width}}"  # (not garbage) and (not multi max) and (under threshold)
            f" {((self.assignment_table[:, 1] == 0) * (self.assignment_table[:, 2] == 0)).sum():<{column_width}}"
            f" {((self.assignment_table[:, 1] == 1) * (self.assignment_table[:, 2] == 0)).sum():<{column_width}}"
            f" {((self.assignment_table[:, 1] != -1) * (self.assignment_table[:, 2] == 0)).sum():<{column_width}}"
            f"\n{multi_max:<{index_width}}"     # multi max values
            f" {'-':<{column_width}}"
            f" {'-':<{column_width}}"
            f" {((self.assignment_table[:, 1] == -1) * (self.assignment_table[:, 2] == 0)).sum():<{column_width}}"
            f"\n{garbage:<{index_width}}"       # all NaN
            f" {'-':<{column_width}}"
            f" {'-':<{column_width}}"
            f" {(self.assignment_table[:, 2] == -1).sum():<{column_width}}"
        )
        # 各プラスミドの assignment 情報
        for ref_seq_idx, ref_seq in enumerate(self.ref_seq_list):
            summary_txt += (
                f"\n{ref_seq.path.name}"
                f"\n{classified:<{index_width}}"    # (not garbage) and (not multi_max)
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] == 0)).sum():<{column_width}}"
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] == 1)).sum():<{column_width}}"
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] != -1)).sum():<{column_width}}"
                f"\n{assigned:<{index_width}}"      # (not garbage) and (not multi max) and (over threshold)
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] == 0) * (self.assignment_table[:, 2] == 1)).sum():<{column_width}}"
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] == 1) * (self.assignment_table[:, 2] == 1)).sum():<{column_width}}"
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] != -1) * (self.assignment_table[:, 2] == 1)).sum():<{column_width}}"
                f"\n{not_assigned:<{index_width}}"  # (not garbage) and (not multi max) and (under threshold)
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] == 0) * (self.assignment_table[:, 2] == 0)).sum():<{column_width}}"
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] == 1) * (self.assignment_table[:, 2] == 0)).sum():<{column_width}}"
                f" {((self.assignment_table[:, 0] == ref_seq_idx) * (self.assignment_table[:, 1] != -1) * (self.assignment_table[:, 2] == 0)).sum():<{column_width}}"
            )
        return summary_txt

class MyMSAligner(mc.AlignmentBase):
    N_polish = 4
    def __init__(self, ref_seq, my_fastq_subset, result_list: List[rqa.MyResult]) -> None:
        self.ref_seq = ref_seq
        self.my_fastq_subset = my_fastq_subset
        self.result_list = result_list
        assert len(self.my_fastq_subset) == len(self.result_list)
    def execute(self, param_dict: dict):
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
        my_msa = MyMSA.generate_msa(self.ref_seq, query_seq_list, q_scores_list, my_cigar_list, param_dict)  ### time consuming ###
        my_msa.set_query_seq_offset_list(query_seq_offset_list)
        # MSA 2nd step: 4回 polish する (4回目の offset は最初の offet と同じ)
        actual_window = np.ceil(param_dict["window"] / (self.N_polish - 2)).astype(int) * (self.N_polish - 1)    # 1 nt ずつずらしながら polish するわけではないので、window を拡張する必要がある
        len_ref_seq_aligned = len(my_msa.ref_seq_aligned)
        my_msa.add_pseudo_ref_seq_as_query()    # polish の間、consensus 取るたびに ref_seq がズレていってしまうので、それを防ぐ
        with mc.MyTQDM(total=len_ref_seq_aligned * self.N_polish, ncols=100, leave=True, bar_format='{l_bar}{bar}{r_bar}', desc=f"polishing reads #{len(self.my_fastq_subset)}{'.' * max(0, 6 - len(str(len(self.my_fastq_subset))))}") as my_pbar:
            for i, offset in enumerate(np.array([0, 1, 2, 0]) * (actual_window // (self.N_polish - 1))):                        #"generating consensus..."
                my_pbar.offset_value = len_ref_seq_aligned * i
                my_msa = my_msa.polish(offset=offset, window=actual_window, topology_of_dna=param_dict["topology_of_dna"], my_pbar=my_pbar)
            my_pbar.offset_value = 0
            my_pbar.set_value(len_ref_seq_aligned * self.N_polish)
        # MSA 後処理
        my_msa.recover_query_seq_from_pseudo()
        my_msa.recover_q_scores(q_scores_list)       # polish には q_scores が必要がないので削ぎ落としている。再度追加する必要あり。
        my_msa.ref_seq_name = self.ref_seq.path.name
        my_msa.query_id_list = list(self.my_fastq_subset.keys())
        # MSA calc consensus
        my_msa = my_msa.execute_soft_clipping()     ### time consuming ###
        my_msa.calculate_consensus()
        # my_msa.print_alignment()
        return my_msa
    def get_racon_path(self):
        return Path(__file__).parents[1] / "racon/build/bin/racon"

class SequenceBasecallQscorePDF():
    bases = "ATCG-"
    assert bases[-1] == "-"
    def __init__(self, df_csv_path=None) -> None:
        with open(df_csv_path, "r") as f:
            self.NanoporeStats_PDF_version = re.match(r"^# file_version: ([0-9]+\.[0-9]+\.[0-9]+)$", f.readline().strip()).group(1)
        self.df_stats = pd.read_csv(df_csv_path, sep="\t", header=1, index_col=0)
        # initialize
        self.pdf_core = {}
        self.P_base_calling_given_true_refseq_dict = {}
        self.initialize_pdf()
        q_score_distribution = self.df_stats.sum(axis=1)
        assert list(q_score_distribution.index) == list(range(-1, mc.MyFastQ.maximum_q_score_allowed + 1))
        self.q_score_distribution = list(q_score_distribution)
    def calc_js_divergence(self, q_score_distribution):
        assert len(q_score_distribution) == len(self.q_score_distribution)
        # -1 (最初) は除いて計算する
        p_dist = np.array(self.q_score_distribution[1:]) + 1    # 0 の項をなくす
        q_dist = np.array(q_score_distribution[1:]) + 1         # 0 の項をなくす
        p_dist = p_dist / p_dist.sum()
        q_dist = q_dist / q_dist.sum()
        r_dist = (p_dist + q_dist) / 2
        return p_dist.dot(np.log(p_dist / r_dist)) + q_dist.dot(np.log(q_dist / r_dist))
    def initialize_pdf(self):
        assert all(self.df_stats.dtypes == np.int64)
        # 確率 0 となるのを避ける
        for c in self.df_stats.columns:
            if c.endswith("-"):
                continue
            for i in self.df_stats.index:
                if i < 1:
                    continue
                if self.df_stats.at[i, c] == 0:
                    self.df_stats.at[i, c] += 1
        # bunbo
        total_events_when_true_base = {}
        for column_names, values in self.df_stats.items():
            true_base = column_names.split("_")[0]
            if true_base not in total_events_when_true_base.keys():
                total_events_when_true_base[true_base] = values.sum()
            else:
                total_events_when_true_base[true_base] += values.sum()
        # calc probability
        for column_names, values in self.df_stats.items():
            true_base = column_names.split("_")[0]
            self.P_base_calling_given_true_refseq_dict[column_names] = values.sum() / total_events_when_true_base[true_base]
        # others
        for column_names, values in self.df_stats.items():
            assert all(values.index == np.arange(-1, 42))
            values /= values.sum()
            # マイナス1で最後のやつにアクセスできるようにする（さすがに50も間を開けてれば、q-scoreがかぶってくることは無いでしょう…）
            self.pdf_core[column_names] = list(values)[1:] + [0.0 for i in range(50)] + list(values)[:1]

        # cythonize
        self.cythonize()
    def cythonize(self):
        P_base_calling_given_true_refseq_dict_CYTHON = {
            key.encode("utf-8") : val
            for key, val in self.P_base_calling_given_true_refseq_dict.items()
        }
        pdf_core_CYTHON = {
            key.encode("utf-8") : val
            for key, val in self.pdf_core.items()
        }
        self.sbq_pdf_CYTHON = af.SequenceBasecallQscorePDF(P_base_calling_given_true_refseq_dict_CYTHON, pdf_core_CYTHON, self.bases.encode("utf-8"))
    def calc_P_event_given_true_refseq(self, event, true_refseq):
        readseq, q_score = event
        key = f"{true_refseq}_{readseq}"
        return self.P_base_calling_given_true_refseq_dict[key] * self.pdf_core[key][q_score]
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

class MyMSA(rqa.MyAlignerBase, mc.MyCigarBase):
    file_format_version = "ff_0.2.4"
    algorithm_version = "al_0.2.2"
    ref_seq_related_save_order = [
        ("add_sequence", "ref_seq_aligned"), 
        ("add_sequence", "with_prior_consensus_seq"), 
        ("add_sequence", "without_prior_consensus_seq"), 
        ("add_q_scores", "with_prior_consensus_q_scores"), 
        ("add_q_scores", "without_prior_consensus_q_scores"), 
    ]
    query_seq_related_save_order = [
        ("add_sequence", "query_seq_list_aligned"), 
        ("add_q_scores", "q_scores_list_aligned"), 
        # my_cigar_list_aligned は上記 2 つを用いて S, I 以外は後で再生性可能なので保存しない
        ("add_clipping_info", "clipping_info_list"),    # add "S" from my_cigar そして S は連続している
        # clipping info なしでも、S と I 以外は識別可能: なので S の情報のみを格納: 詳しくは self.gen_basic_cigar を参照
        ("add_clipping_info", "aligned_offset_info_list")
    ]
    bases = SequenceBasecallQscorePDF.bases
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
    sbq_pdf_major_version = "pdf_0.2"
    sbq_pdf_minor_version = None
    sbq_pdf = None
    default_print_options = {
        "center": 50, 
        "seq_range": 50, 
        "offset": 0, 
    }
    def __init__(self, ref_seq_aligned: str=None, query_seq_list_aligned: List[str]=None, q_scores_list_aligned: List[List[int]]=None, my_cigar_list_aligned: List[str]=None, param_dict: dict=None) -> None:
        self.ref_seq_name = None
        self.query_id_list = None
        self.ref_seq_aligned = ref_seq_aligned                  # ref_related
        self.query_seq_list_aligned = query_seq_list_aligned    # query_related
        self.q_scores_list_aligned = q_scores_list_aligned      # query_related
        self.my_cigar_list_aligned = my_cigar_list_aligned      # query_related
        self.query_seq_offset_list = None                       # set by self.set_query_seq_offset_list query_related, aligned_offset_info_list (= query_seq_offset_list_aligned) will be saved
        self.query_seq_offset_list_aligned = None               # set by self.calc_query_seq_offset_list_aligned()
        self.with_prior_consensus_seq = ""          # consensus_related
        self.with_prior_consensus_q_scores = []     # consensus_related
        self.with_prior_consensus_my_cigar = ""     # consensus_related
        self.without_prior_consensus_seq = ""       # consensus_related
        self.without_prior_consensus_q_scores = []  # consensus_related
        self.without_prior_consensus_my_cigar = ""  # consensus_related
        super().__init__(param_dict)
    @mc.classproperty
    def sbq_pdf_version(self):
        return f"{self.sbq_pdf_major_version}.{self.sbq_pdf_minor_version}"
    @property
    def ref_seq_NoDEL(self):
        return self.ref_seq_aligned.replace("-", "")
    @property
    def with_prior_consensus_seq_NoDEL(self):
        return self.with_prior_consensus_seq.replace("-", "")
    @property
    def with_prior_consensus_q_scores_NoDEL(self):
        previous_idx = 0
        q_scores = []
        for LLL, L in self.generate_cigar_iter(self.with_prior_consensus_my_cigar):
            N = len(LLL)
            if L in "=XI":
                q_scores += self.with_prior_consensus_q_scores[previous_idx:previous_idx + N]
                assert self.with_prior_consensus_seq[previous_idx:previous_idx + N].count("-") == 0
            elif L in "DN":
                assert self.with_prior_consensus_seq[previous_idx:previous_idx + N].count("-") == N
            else:
                raise Exception(f"unknown cigar: {L}")
            previous_idx += N
        return q_scores
    @property
    def without_prior_consensus_seq_NoDEL(self):
        return self.without_prior_consensus_seq.replace("-", "")
    @property
    def without_prior_consensus_q_scores_NoDEL(self):
        previous_idx = 0
        q_scores = []
        for LLL, L in self.generate_cigar_iter(self.without_prior_consensus_my_cigar):
            N = len(LLL)
            if L in "=XI":
                q_scores += self.without_prior_consensus_q_scores[previous_idx:previous_idx + N]
                assert self.without_prior_consensus_seq[previous_idx:previous_idx + N].count("-") == 0
            elif L in "DN":
                assert self.without_prior_consensus_seq[previous_idx:previous_idx + N].count("-") == N
            else:
                raise Exception(f"unknown cigar: {L}")
            previous_idx += N
        return q_scores
    @property
    def clipping_info_list(self):   # ['start_idx-end_idx', ...] (only contains "S" info)
        return [
            ",".join(
                f"{m.start()}-{m.end() - 1}" for m in re.finditer(r"[S]+", my_cigar)
            ) for my_cigar in self.my_cigar_list_aligned
        ]
    @clipping_info_list.setter
    def clipping_info_list(self, clipping_info_list: List[str]):
        self.regenerate_cigar()
        assert len(clipping_info_list) == len(self.query_seq_list_aligned)
        for query_idx, clipping_info in enumerate(clipping_info_list):
            my_cigar = self.my_cigar_list_aligned[query_idx]
            for ci in clipping_info.split(","):
                if ci == "":
                    continue
                start_S, end_S = ci.split("-")
                start_S = int(start_S)
                end_S = int(end_S)
                my_cigar = my_cigar[:start_S] + "S" * (end_S - start_S + 1) + my_cigar[end_S + 1:]
            self.my_cigar_list_aligned[query_idx] = my_cigar
    @property
    def aligned_offset_info_list(self):    # ['idx', ...]
        return list(map(str, self.query_seq_offset_list_aligned))
    @aligned_offset_info_list.setter
    def aligned_offset_info_list(self, aligned_offset_info_list: List[str]):
        self.query_seq_offset_list = [len(q[-o:].replace(" ", "").replace("-", "")) for o, q in zip(
            list(map(int, aligned_offset_info_list)), 
            self.query_seq_list_aligned
        )]
    @property
    def param_dict_str(self):
        # 下記のように単純にしてるので、トリッキーな文字 (コロンとか) 入れないでね
        return "\n".join(f"{k}:{v}" for k, v in self.param_dict.items())
    @param_dict_str.setter
    def param_dict_str(self, param_dict_str):
        from ..post_analysis.post_analysis import default_post_analysis_param_dict
        for k_v in param_dict_str.split("\n"):
            k_str, v_str = k_v.split(":")
            if k_str in default_post_analysis_param_dict.keys():    # 後方互換性のために必要：具体的には、default_post_analysis_param_dict から削除したパラメータ (del_mut_rate など) がある場合に必要
                self.param_dict[k_str] = type(default_post_analysis_param_dict[k_str])(v_str)
    ################
    # MSA 1st step #
    ################
    @staticmethod
    def generate_msa(ref_seq: mc.MySeq, query_seq_list: List[str], q_scores_list: List[int], my_cigar_list: List[str], param_dict:dict):
        """
        最後が必ず同時に終わるように、特殊シーケンスを追加 (こうしないと、query_seq_list のどれか一つのの最後が "I" である場合に正常終了しない)
        また、`elif my_cigar_list[i][my_cigar_idx_list[i] - 1] in "H":` の部分で idx=-1 が生じる可能性がある。
        それを考慮するために、my_cigar_list のみ、idx=-1 の要素を更に最後に追加しておく。
        """
        assert len(query_seq_list) == len(q_scores_list) == len(my_cigar_list)
        ref_seq += "Z"
        query_seq_list = [query_seq + "Z" for query_seq in query_seq_list]
        q_scores_list = [q_scores + [-2] for q_scores in q_scores_list]
        my_cigar_list = [my_cigar + "Z" + my_cigar[-1] for my_cigar in my_cigar_list]

        # 実行準備
        ref_seq_idx = 0
        my_cigar_idx_list = [0 for i in my_cigar_list]
        query_seq_idx_list = [0 for i in my_cigar_list]
        ref_seq_aligned = ""
        query_seq_list_aligned = ["" for i in my_cigar_list]
        q_scores_list_aligned = [[] for i in my_cigar_list]
        my_cigar_list_aligned =  ["" for i in my_cigar_list]

        # 実行
        while True:
            L_list = [my_cigar_list[i][my_cigar_idx] for i, my_cigar_idx in enumerate(my_cigar_idx_list)]
            I_not_in_L_list = "I" not in L_list
            S_not_in_L_list = "S" not in L_list
            # when ref_seq exists (not "I" nor "S")
            if I_not_in_L_list and S_not_in_L_list:
                R = ref_seq[ref_seq_idx]
                if R == "Z":    # 全て終了している (もしくはそもそも割り当てられたリードがない) ことを確認する
                    assert all(query_seq_list[i][query_seq_idx_list[i]] == "Z" for i in range(len(L_list))) or (len(query_seq_idx_list) == 0)
                    assert all(q_scores_list[i][query_seq_idx_list[i]] == -2 for i in range(len(L_list))) or (len(query_seq_idx_list) == 0)
                    assert all(L == "Z" for L in L_list) or (len(L_list) == 0)
                    break
                for i, L in enumerate(L_list):
                    my_cigar_list_aligned[i] += L
                    my_cigar_idx_list[i]     += 1
                    if L == "D":
                        query_seq_list_aligned[i] += "-"
                        q_scores_list_aligned[i]  += [-1]
                    elif L in "=X":
                        query_seq_list_aligned[i] += query_seq_list[i][query_seq_idx_list[i]]
                        q_scores_list_aligned[i]  += [q_scores_list[i][query_seq_idx_list[i]]]
                        query_seq_idx_list[i]     += 1
                    elif L == "H":
                        query_seq_list_aligned[i] += " "
                        q_scores_list_aligned[i]  += [-1]
                    else:
                        raise Exception(f"error!: {L}, {i}")
                ref_seq_aligned += R
                ref_seq_idx += 1
            # contains "I" but not "S"
            elif S_not_in_L_list:
                for i, L in enumerate(L_list):
                    if L == "I":
                        my_cigar_list_aligned[i]  += "I"
                        query_seq_list_aligned[i] += query_seq_list[i][query_seq_idx_list[i]]
                        q_scores_list_aligned[i]  += [q_scores_list[i][query_seq_idx_list[i]]]
                        my_cigar_idx_list[i]      += 1
                        query_seq_idx_list[i]     += 1
                    elif (
                        (my_cigar_list[i][my_cigar_idx_list[i]] == "H") or          # my_cigar_idx_list[i] は次回評価される idx: H の手前に O が追加される
                        (my_cigar_list[i][my_cigar_idx_list[i] - 1] == "H") or      # my_cigar_idx_list[i] が H でなくても、一つ前が H なら、現在の idx の手前 (一つ前の H の後) に O が追加される
                        ((my_cigar_list[i][my_cigar_idx_list[i]] == "Z") and (my_cigar_list[i][0] == "H"))  # 最初に I や S がくる場合は、while loop の前処理: my_cigar + "Z" + my_cigar[-1] for ... :で対処されてる
                    ):
                        my_cigar_list_aligned[i]  += "O"    # inside or adjascent to H region but skipped
                        query_seq_list_aligned[i] += " "
                        q_scores_list_aligned[i]  += [-1]
                    else:
                        my_cigar_list_aligned[i]  += "N"    # skipped
                        query_seq_list_aligned[i] += "-"
                        q_scores_list_aligned[i]  += [-1]
                ref_seq_aligned += "-"
            # contains "S": S と I が混在して存在するときも、S が常に連続するようになっている
            else:
                for i, L in enumerate(L_list):
                    if L == "S":
                        my_cigar_list_aligned[i]  += "S"
                        query_seq_list_aligned[i] += query_seq_list[i][query_seq_idx_list[i]]
                        q_scores_list_aligned[i]  += [q_scores_list[i][query_seq_idx_list[i]]]
                        my_cigar_idx_list[i]      += 1
                        query_seq_idx_list[i]     += 1
                    elif (
                        (my_cigar_list[i][my_cigar_idx_list[i]] == "H") or          # my_cigar_idx_list[i] は次回評価される idx: H の手前に O が追加される
                        (my_cigar_list[i][my_cigar_idx_list[i] - 1] == "H") or      # my_cigar_idx_list[i] が H でなくても、一つ前が H なら、現在の idx の手前 (一つ前の H の後) に O が追加される
                        ((my_cigar_list[i][my_cigar_idx_list[i]] == "Z") and (my_cigar_list[i][0] == "H"))  # 最初に I や S がくる場合は、while loop の前処理: my_cigar + "Z" + my_cigar[-1] for ... :で対処されてる
                    ):
                        my_cigar_list_aligned[i]  += "O"    # inside or adjascent to H region but skipped
                        query_seq_list_aligned[i] += " "
                        q_scores_list_aligned[i]  += [-1]
                    else:
                        my_cigar_list_aligned[i]  += "N"
                        query_seq_list_aligned[i] += "-"
                        q_scores_list_aligned[i]  += [-1]
                ref_seq_aligned += "-"
        # 最初に追加した余計なものを除く
        del ref_seq[-1]
            # query_seq_list = [query_seq[:-1] for query_seq in query_seq_list] # view ではないので、元に戻す必要はない
            # q_scores_list = [q_scores[:-1] for q_scores in q_scores_list]     # view ではないので、元に戻す必要はない
            # my_cigar_list = [my_cigar[:-1] for my_cigar in my_cigar_list]     # view ではないので、元に戻す必要はない
        return MyMSA(ref_seq_aligned, query_seq_list_aligned, q_scores_list_aligned, my_cigar_list_aligned, param_dict)
    ################
    # MSA 2nd step #
    ################
    def add_pseudo_ref_seq_as_query(self):    # polish の間、consensus 取るたびに ref_seq がズレていってしまうので、それを防ぐ
        if self.param_dict["topology_of_dna"] == 0:
            assert self.ref_seq_aligned[0] != "-"
        elif self.param_dict["topology_of_dna"] == 1:
            pass
        else:
            raise Exception(f"unknown topology_of_dna: {self.param_dict['topology_of_dna']}")
        self.query_seq_list_aligned.append(self.ref_seq_aligned)
        self.q_scores_list_aligned.append(list(self.custom_fill(self.ref_seq_aligned, some_list=[50 for i in self.ref_seq_NoDEL], empty_value=-1)))
        self.my_cigar_list_aligned.append("".join(self.custom_fill(self.ref_seq_aligned, some_list=["=" for i in self.ref_seq_NoDEL], empty_value="N")))
        self.query_seq_offset_list.append(len(self.ref_seq_aligned.replace(" ", "").replace("-", "")))
        self.query_seq_offset_list_aligned.append(0)
    def recover_query_seq_from_pseudo(self):
        """
        本処理の後は、"S+N*=", "=N*S+", "H+O*[SN]*=", "=[NS]*O*H+" が保証されなくなる (= の部分が変わりうる) が、それは許容しよう
        """
        # query_seq 末尾の pseudo_ref_seq で ref_seq を置換
        self.ref_seq_aligned = self.query_seq_list_aligned.pop(-1)
        del self.q_scores_list_aligned[-1]
        my_cigar_ref_seq_vs_pseudo = self.my_cigar_list_aligned.pop(-1)
        del self.query_seq_offset_list[-1]
        del self.query_seq_offset_list_aligned[-1]
        # my_cigar を修正していく
        for m in self.generate_cigar_match(my_cigar_ref_seq_vs_pseudo):
            L = m.group(2)
            if L in "=NO":   # =NO XDI HS
                continue
            else:
                s = m.start()
                e = m.end()
                ref_seq_chunk = list(self.ref_seq_aligned[s:e])
                if L in "XI":  # =XDH / ISON
                    for queruy_idx, query_seq_aligned in enumerate(self.query_seq_list_aligned):
                        new_my_cigar_chunk = ""
                        for r, q in zip(ref_seq_chunk, list(query_seq_aligned[s:e])):
                            if q == " ":
                                new_my_cigar_chunk += "H"
                            elif q == "-":
                                new_my_cigar_chunk += "D"
                            elif r == q:
                                new_my_cigar_chunk += "="
                            else:
                                new_my_cigar_chunk += "X"
                        my_cigar_aligned = self.my_cigar_list_aligned[queruy_idx]
                        self.my_cigar_list_aligned[queruy_idx] = my_cigar_aligned[:s] + new_my_cigar_chunk + my_cigar_aligned[e:]
                elif L in "DH":  # ION S / =XDH (S と同時に出現しうるのは SON のみなので、q サイドに S はありえない)
                    for queruy_idx, query_seq_aligned in enumerate(self.query_seq_list_aligned):
                        new_my_cigar_chunk = ""
                        for q in query_seq_aligned[s:e]:
                            if q == " ":
                                new_my_cigar_chunk += "O"
                            elif q == "-":
                                new_my_cigar_chunk += "N"
                            else:   # S は導入しない (L == "D" なので、consensus に寄与しなくてはだめだから)
                                new_my_cigar_chunk += "I"
                        my_cigar_aligned = self.my_cigar_list_aligned[queruy_idx]
                        self.my_cigar_list_aligned[queruy_idx] = my_cigar_aligned[:s] + new_my_cigar_chunk + my_cigar_aligned[e:]
                elif L == "S":  # =XDH / SON I (S と同時に出現しうるのは SON のみなので、q サイドに I はありえない)
                    for queruy_idx, query_seq_aligned in enumerate(self.query_seq_list_aligned):
                        new_my_cigar_chunk = ""
                        for q in query_seq_aligned[s:e]:
                            if q == " ":    # O
                                new_my_cigar_chunk += "H"
                            elif q == "-":  # N
                                new_my_cigar_chunk += "D"   # SSSNNNSSS みたいな場合は存在しないので大丈夫なはず
                            else:           # S
                                new_my_cigar_chunk += "S"
                        my_cigar_aligned = self.my_cigar_list_aligned[queruy_idx]
                        self.my_cigar_list_aligned[queruy_idx] = my_cigar_aligned[:s] + new_my_cigar_chunk + my_cigar_aligned[e:]
                else:
                    raise Exception(f"error: {L}")
    def recover_q_scores(self, q_scores_list: List[list]):
        assert len(q_scores_list) == len(self.query_seq_offset_list_aligned)
        aligned_len = len(self.ref_seq_aligned)
        for query_idx, (query_seq_aligned, query_seq_offset_aligned, q_scores) in enumerate(zip(self.query_seq_list_aligned, self.query_seq_offset_list_aligned, q_scores_list)):
            q_scores_after_offset = q_scores[query_seq_offset_aligned:] + q_scores[:query_seq_offset_aligned]
            q_scores_aligned = np.full(aligned_len, fill_value=-1, dtype=int)
            q_scores_aligned[[q not in "- " for q in query_seq_aligned]] = q_scores_after_offset
            self.q_scores_list_aligned[query_idx] = list(q_scores_aligned)
    def polish(self, offset, window, topology_of_dna, my_pbar: mc.MyTQDM=None):
        chunk_idx_start_aligned_list = []
        cur_chunk_len = 0
        target_chunk_len = offset   # 最初はオフセット
        pseudo_ref_seq = self.query_seq_list_aligned[-1]    # add_pseudo_ref_seq_as_query で追加された pseudo ref_seq
        for i, s in enumerate(pseudo_ref_seq):
            # 新規 chunk 開始
            if cur_chunk_len == target_chunk_len:
                chunk_idx_start_aligned_list.append(i)
                cur_chunk_len = 0
                target_chunk_len = window
            if s != "-":
                cur_chunk_len += 1
        # 最後の chunk の長さがが短ければ、一つ前のものと連結
        if len(chunk_idx_start_aligned_list) <= 1:  # 短すぎると chunk がなくなる
            raise Exception(f"Plasmid size is too small or window size is too large!")
        if len(pseudo_ref_seq[chunk_idx_start_aligned_list[-1]:].replace("-", "")) + offset < window:
            del chunk_idx_start_aligned_list[-1]

        # 最終 chunk は常に最初の offset 部分と連結される (ただし circular な場合に限る)。後でもとに戻すための準備をする。self.query_seq_offset_list も変更しなくてはならないので、その準備をする
        if topology_of_dna == 0:
            N_bases_before_1st_chunk = [len(query_seq_aligned[:chunk_idx_start_aligned_list[0]].replace(" ", "").replace("-", "")) for query_seq_aligned in self.query_seq_list_aligned]
        # linear な場合は、最初のchunkは常に0である必要がある（circularな場合と違って最初の chunk が最後の chunk と連結されないので）
        elif topology_of_dna == 1:
            chunk_idx_start_aligned_list[0] = 0 # 最初に 0 を insert する形だと、最初の chunk が短くなってしまう場合がある。
            chunk_idx_start_aligned_list.append(len(self.ref_seq_aligned))  # chunk_idx_start_aligned_list の最後に追加することで、 以下の with Pool 内でで回してる for loop の後処理を無くせる
        else:
            raise Exception(f"unknown topology_of_dna: {topology_of_dna}")

        # POA 実行
        with Pool(processes=self.param_dict["n_cpu"]) as p:
            mini_my_msa_list = []
            for s, e, mini_my_msa in p.imap(self.polish_core, zip(chunk_idx_start_aligned_list[:-1], chunk_idx_start_aligned_list[1:])):
                if my_pbar is not None:
                    my_pbar.set_value(s)
                mini_my_msa_list.append(mini_my_msa)
            # circular な場合、最終 chunk: 最初の offset 部分と連結する
            if topology_of_dna == 0:
                s = chunk_idx_start_aligned_list[-1]
                e = chunk_idx_start_aligned_list[0]
                if my_pbar is not None:
                    my_pbar.set_value(s)
                ref_seq_chunk_aligned = self.ref_seq_aligned[s:] + self.ref_seq_aligned[:e]
                query_seq_chunk_list_aligned = [query_seq_aligned[s:] + query_seq_aligned[:e] for query_seq_aligned in self.query_seq_list_aligned]
                my_cigar_chunk_list_aligned = [my_cigar_aligned[s:] + my_cigar_aligned[:e] for my_cigar_aligned in self.my_cigar_list_aligned]
                # 実行
                mini_my_msa = self.exec_chunk_poa(ref_seq_chunk_aligned, query_seq_chunk_list_aligned, my_cigar_chunk_list_aligned, s)
                # 格納
                mini_my_msa_list.append(mini_my_msa)
            # linear な場合は普通にPOA実行する
            elif topology_of_dna == 1:
                pass
            else:
                raise Exception(f"unknown topology_of_dna: {topology_of_dna}")

        # 結合
        ref_seq_aligned = "".join(mini_my_msa.ref_seq_aligned for mini_my_msa in mini_my_msa_list)
        query_seq_list_aligned = ["".join(chunks) for chunks in zip(*[mini_my_msa.query_seq_list_aligned for mini_my_msa in mini_my_msa_list])]
        my_cigar_list_aligned = ["".join(chunks) for chunks in zip(*[mini_my_msa.my_cigar_list_aligned for mini_my_msa in mini_my_msa_list])]
        # 結合部分の修正 # 各chunk内部では HHHNNN のようなことは起こってないが、稀に結合部分でそうなる可能性がある。
        for idx, (query_seq_aligned, my_cigar_aligned) in enumerate(zip(query_seq_list_aligned, my_cigar_list_aligned)):
            if " -" in query_seq_aligned:
                """
                Original
                    "ATCG      -------ATCG"
                    "====HHHHHHNNNNNNN===="
                Final
                    "ATCG             ATCG"
                    "====HHHHHHOOOOOOO===="
                """
                # 場所探索
                m_list = list(re.finditer(r" +-+", query_seq_aligned))
                assert len(m_list) == 1
                m = m_list[0]
                # クエリ修正
                query_seq_aligned = query_seq_aligned[:m.start()] + " " * (m.end() - m.start()) + query_seq_aligned[m.end():]
                # シガー修正
                N_start_idx = m.start() + m.group(0).find("-")
                assert my_cigar_aligned[N_start_idx:m.end()] == "N" * (m.end() - N_start_idx)
                my_cigar_aligned = my_cigar_aligned[:N_start_idx] + "O" * (m.end() - N_start_idx) + my_cigar_aligned[m.end():]
                # 格納
                query_seq_list_aligned[idx] = query_seq_aligned
                my_cigar_list_aligned[idx] = my_cigar_aligned
            if "- " in query_seq_aligned:
                """
                Original
                    "ATCG-------      ATCG"
                    "====NNNNNNNHHHHHH===="
                Final
                    "ATCG             ATCG"
                    "====OOOOOOOHHHHHH===="
                """
                m_list = list(re.finditer(r"-+ +", query_seq_aligned))
                assert len(m_list) == 1
                m = m_list[0]
                # クエリ修正
                query_seq_aligned = query_seq_aligned[:m.start()] + " " * (m.end() - m.start()) + query_seq_aligned[m.end():]
                # シガー修正
                N_end_idx = m.start() + m.group(0).find(" ")
                assert my_cigar_aligned[m.start():N_end_idx] == "N" * (N_end_idx - m.start())
                my_cigar_aligned = my_cigar_aligned[:m.start()] + "O" * (N_end_idx - m.start()) + my_cigar_aligned[N_end_idx:]
                # 格納
                query_seq_list_aligned[idx] = query_seq_aligned
                my_cigar_list_aligned[idx] = my_cigar_aligned
        # 後処理
        if topology_of_dna == 0:
            # circular な場合は、さらに最初と最後のつなぎ目部分で変な結合になってる場合を注意する
            for idx, (query_seq_aligned, my_cigar_aligned) in enumerate(zip(query_seq_list_aligned, my_cigar_list_aligned)):
                if (query_seq_aligned[0] == " ") and (query_seq_aligned[-1] == "-"):
                    """
                    Original
                        "   ATCG-------ATCG----"
                        "HHH====DDDDDDD====NNNN"
                    Final
                        "   ATCG-------ATCG    "
                        "HHH====DDDDDDD====OOOO"
                    """
                    m = re.search(r"-+$", query_seq_aligned)
                    hyphen_count = len(m.group(0))
                    query_seq_aligned = query_seq_aligned[:-hyphen_count] + " " * hyphen_count
                    my_cigar_aligned = my_cigar_aligned[:-hyphen_count] + "O" * hyphen_count
                    # 格納
                    query_seq_list_aligned[idx] = query_seq_aligned
                    my_cigar_list_aligned[idx] = my_cigar_aligned
                elif (query_seq_aligned[0] == "-") and (query_seq_aligned[-1] == " "):
                    """
                    Original
                        "---ATCG-------ATCG    "
                        "NNN====DDDDDDD====HHHH"
                    Final
                        "   ATCG-------ATCG    "
                        "OOO====DDDDDDD====HHHH"
                    """
                    m = re.search(r"^-+", query_seq_aligned)
                    hyphen_count = len(m.group(0))
                    query_seq_aligned = " " * hyphen_count + query_seq_aligned[hyphen_count:]
                    my_cigar_aligned = "O" * hyphen_count + my_cigar_aligned[hyphen_count:]
                    # 格納
                    query_seq_list_aligned[idx] = query_seq_aligned
                    my_cigar_list_aligned[idx] = my_cigar_aligned
            # 最後に入ってる ref_seq も始点と終点のあるリニアなリードとして扱われてるので、コンセンサスに insertion の部分が存在し、かつその位置が mapの切れ目の部分と一致してしまうと、リードが " " となってしまうが、実際は環状プラスミドなので、"-" としなくてはダメ
            query_seq_list_aligned[-1] = query_seq_list_aligned[-1].replace(" ", "-")
            cur_len = 0
            for i, s in enumerate(query_seq_list_aligned[-1][::-1]):  # pseudo_ref_seq_aligned
                if cur_len == offset:
                    offset_after_msa = i
                    break
                if s != "-":
                    cur_len += 1
            ref_seq_aligned = ref_seq_aligned[-offset_after_msa:] + ref_seq_aligned[:-offset_after_msa]
            query_seq_list_aligned = [query_seq_aligned[-offset_after_msa:] + query_seq_aligned[:-offset_after_msa] for query_seq_aligned in query_seq_list_aligned]
            my_cigar_list_aligned = [my_cigar_aligned[-offset_after_msa:] + my_cigar_aligned[:-offset_after_msa] for my_cigar_aligned in my_cigar_list_aligned]
            # MyMSA 再生成 (q_scores_list_aligned は適当)
            my_msa = MyMSA(ref_seq_aligned, query_seq_list_aligned, [[-1] * len(ref_seq_aligned) for i in range(len(query_seq_list_aligned))], my_cigar_list_aligned, self.param_dict)
            N_bases_before_1st_chunk_after_msa = [len(query_seq_aligned[:offset_after_msa].replace(" ", "").replace("-", "")) for query_seq_aligned in query_seq_list_aligned]
            query_seq_offset_list = np.mod(
                np.array(self.query_seq_offset_list) + np.array(N_bases_before_1st_chunk) - np.array(N_bases_before_1st_chunk_after_msa) - 1, 
                [len(query_seq_aligned.replace(" ", "").replace("-", "")) for query_seq_aligned in query_seq_list_aligned]
            ) + 1   # query_seq_offset に 0 は許されない: 詳しくは self.query_seq_offset_list_aligned を参照
            my_msa.set_query_seq_offset_list(list(query_seq_offset_list))
            return my_msa
        # linear の場合は query_seq_offset_list は本質的には必要ないが、exec_chunk_poa で必要なので登録して置く必要あり
        elif topology_of_dna == 1:
            my_msa = MyMSA(ref_seq_aligned, query_seq_list_aligned, [[-1] * len(ref_seq_aligned) for i in range(len(query_seq_list_aligned))], my_cigar_list_aligned, self.param_dict)
            for i, query_seq in enumerate(self.query_seq_list_aligned):
                assert self.query_seq_offset_list[i] == len(query_seq.replace(" ", "").replace("-", ""))
            my_msa.set_query_seq_offset_list(self.query_seq_offset_list)
            return my_msa
        else:
            raise Exception(f"unknown topology_of_dna: {topology_of_dna}")
    def polish_core(self, args):
        s, e = args
        ref_seq_chunk_aligned = self.ref_seq_aligned[s:e]
        query_seq_chunk_list_aligned = [query_seq_aligned[s:e] for query_seq_aligned in self.query_seq_list_aligned]
        my_cigar_chunk_list_aligned = [my_cigar_aligned[s:e] for my_cigar_aligned in self.my_cigar_list_aligned]
        # 実行
        mini_my_msa = self.exec_chunk_poa(ref_seq_chunk_aligned, query_seq_chunk_list_aligned, my_cigar_chunk_list_aligned, s)
        return s, e, mini_my_msa
    SN_re_beg = re.compile(r"^[N]*[S]+")
    SN_re_end = re.compile(r"[S]+[N]*$")
    all_HO_re = re.compile(r"^[HO]+$")
    all_ND_re = re.compile(r"^[ND]+$")
    with_SHO_re = re.compile(r"^.*[SHO]+.*$")
    def exec_chunk_poa(self, ref_seq_chunk_aligned: str, query_seq_chunk_list_aligned: List[str], my_cigar_chunk_list_aligned: List[str], s_idx: int) -> list:
        # S, H が含まれているかどうか
        chunk_type_list = []
        for my_cigar_chunk_aligned in my_cigar_chunk_list_aligned:
            # recover_query_seq_from_pseudo が実行されるまでは、HO と ND が、それら以外の要素を含まず混在することはない
            if self.all_HO_re.match(my_cigar_chunk_aligned) is not None:
                chunk_type_list.append("all_HO")
            elif self.all_ND_re.match(my_cigar_chunk_aligned) is not None:
                chunk_type_list.append("all_ND")
            elif self.with_SHO_re.match(my_cigar_chunk_aligned) is not None:
                chunk_type_list.append("with_SHO")
            else:
                chunk_type_list.append("valid4poa")
        # 両端処理 (適当な共通配列を付加することで、端のアラインメントが崩れることを防ぐ)
        appendix = "Z" 
        assert len(appendix) == 1
        query_seq_chunk_list = [query_seq_chunk_aligned.replace(" ", "").replace("-", "") for query_seq_chunk_aligned in query_seq_chunk_list_aligned]
        # 実行
        consensus, msa = poa(
            [appendix + ref_seq_chunk_aligned.replace("-", "") + appendix] + 
            [appendix + query_seq_chunk + appendix for query_seq_chunk, chunk_type in zip(query_seq_chunk_list, chunk_type_list) if chunk_type == "valid4poa"], 
            algorithm = 1,  # global alignment
            genmsa = False, #
            # m = 5     # m( 1) = self.param_dict["match_score"], 
            # n = -4    # n(-2) = self.param_dict["mismatch_score"], 
            # g = -8    # g(-3) = -self.param_dict["gap_open_penalty"], 
            # e = -6    # e(-1) = -min(self.param_dict["gap_extend_penalty"] * 2, self.param_dict["gap_open_penalty"])    # gap extension penalyt は少し強いほうが良いコンセンサスが得られる
        )
        # 各リードをリマップしていく
        my_cigar_list = []
        consensus_wo_appendix = consensus[1:-1] # appendix ありでアラインメントすると、"Z" のスコア (match, mismatch) が定義されていないのでバグる
        for idx, (chunk_type, query_seq_chunk) in enumerate(zip(chunk_type_list, query_seq_chunk_list)):
            if chunk_type == "valid4poa":
                beg_idx = len(self.ref_seq_aligned) - self.query_seq_offset_list_aligned[idx] - s_idx
                if beg_idx < 0:
                    beg_idx += len(self.ref_seq_aligned)
                query_seq_chunk_aligned = query_seq_chunk_list_aligned[idx]
                # 切れ目が領域内になければシンプルに
                if not (0 <= beg_idx < len(query_seq_chunk_aligned)):
                    my_cigar = self.nw_trace(ref_seq=consensus_wo_appendix, query_seq=query_seq_chunk).my_cigar
                # 切れ目が領域内にあれば、my_special_DP をする必要あり (S, H が無くても ピッタリアラインされてる可能性がある)
                else:
                    my_cigar = self.my_special_DP(
                        query_seq_1=query_seq_chunk_aligned[:beg_idx].replace(" ", "").replace("-", ""), 
                        query_seq_2=query_seq_chunk_aligned[beg_idx:].replace(" ", "").replace("-", ""), 
                        ref_seq=consensus_wo_appendix, 
                    ).my_cigar
            elif chunk_type == "all_HO":          # HHOOHH
                my_cigar = "H" * len(consensus_wo_appendix)
            elif chunk_type == "all_ND":          # DDNNDD
                my_cigar = "D" * len(consensus_wo_appendix)
            else:   # with_SHO: appendix なしでアラインメント
                my_cigar_chunk_aligned = my_cigar_chunk_list_aligned[idx]
                starts_with_OH = my_cigar_chunk_aligned[0] in "OH"
                ends_with_OH = my_cigar_chunk_aligned[-1] in "OH"
                starts_with_S = self.SN_re_beg.search(my_cigar_chunk_aligned) is not None
                ends_with_S = self.SN_re_end.search(my_cigar_chunk_aligned) is not None
                if starts_with_OH and ends_with_OH:         # HHHSSS===SSSHHH
                    my_cigar = self.my_special_sw(ref_seq=consensus_wo_appendix, query_seq=query_seq_chunk).my_cigar
                elif starts_with_OH and (not ends_with_OH): # HHHSSS=========
                    my_cigar = self.my_special_sg_qb_db(ref_seq=consensus_wo_appendix, query_seq=query_seq_chunk).my_cigar
                elif (not starts_with_OH) and ends_with_OH: # =========SSSHHH
                    my_cigar = self.my_special_sg_qe_de(ref_seq=consensus_wo_appendix, query_seq=query_seq_chunk).my_cigar
                # HEREAFTER: (not starts_with_OH) and (not ends_with_OH)
                elif starts_with_S and ends_with_S:         # SSSSSSSSS
                    raise Exception("error")
                elif starts_with_S and (not ends_with_S):   # SSSSSS=========
                    my_cigar = self.my_special_sg_qb_db(ref_seq=consensus_wo_appendix, query_seq=query_seq_chunk).my_cigar
                elif (not starts_with_S) and ends_with_S:   # =========SSSSSS
                    my_cigar = self.my_special_sg_qe_de(ref_seq=consensus_wo_appendix, query_seq=query_seq_chunk).my_cigar
                else:                                       # ===SSSHHHSSS===
                    query_seq_chunk_aligned = query_seq_chunk_list_aligned[idx]
                    S_beg_idx = len(self.ref_seq_aligned) - self.query_seq_offset_list_aligned[idx] - s_idx
                    if S_beg_idx < 0:
                        S_beg_idx += len(self.ref_seq_aligned)
                    assert 0 <= S_beg_idx < len(query_seq_chunk_aligned)
                    my_cigar = self.my_special_DP(
                        query_seq_1=query_seq_chunk_aligned[:S_beg_idx].replace(" ", "").replace("-", ""), 
                        query_seq_2=query_seq_chunk_aligned[S_beg_idx:].replace(" ", "").replace("-", ""), 
                        ref_seq=consensus_wo_appendix, 
                    ).my_cigar
            # 追加
            my_cigar_list.append(my_cigar)
        # 整頓する
        mini_my_msa = self.generate_msa(
            ref_seq=mc.MySeq(consensus_wo_appendix), 
            query_seq_list=query_seq_chunk_list, 
            q_scores_list=[[[1] for _q in query_seq_chunk] for query_seq_chunk in query_seq_chunk_list], 
            my_cigar_list=my_cigar_list, 
            param_dict=self.param_dict
        )
        mini_my_msa.correct_end_SNOH()  # generate_msa の都合、右端に O が来てしまいうる (circular な前提として msa してるため) また、=+S+の右隣はNではなくOであるべき
        return mini_my_msa
    beg_HO_re = re.compile(r"^O+[^HO]") # H ではない文字の前に O が来て、の形で始まるのはダメ
    end_HO_re = re.compile(r"[^HO]O+$") # H ではない文字の後に O が来て、そのまま終わるのはダメ
    beg_SN_re = re.compile(r"^(N+)S+[^SNOH]")
    end_SN_re = re.compile(r"[^SNOH]S+(N+)$")
    def correct_end_SNOH(self):
        for query_idx, my_cigar_aligned in enumerate(self.my_cigar_list_aligned):
            beg_m = self.beg_HO_re.search(my_cigar_aligned)
            end_m = self.end_HO_re.search(my_cigar_aligned)
            if beg_m is not None:
                assert end_m is None
                self.my_cigar_list_aligned[query_idx] = "N" * (beg_m.end() - 1) + my_cigar_aligned[beg_m.end() - 1:]
                self.query_seq_list_aligned[query_idx] = "-" * (beg_m.end() - 1) + self.query_seq_list_aligned[query_idx][beg_m.end() - 1:]
                continue
            if end_m is not None:
                assert beg_m is None
                self.my_cigar_list_aligned[query_idx] = my_cigar_aligned[:end_m.start() + 1] + "N" * (end_m.end() - end_m.start() - 1)
                self.query_seq_list_aligned[query_idx] = self.query_seq_list_aligned[query_idx][:end_m.start() + 1] + "-" * (end_m.end() - end_m.start() - 1)
                continue
            beg_m = self.beg_SN_re.search(my_cigar_aligned)
            end_m = self.end_SN_re.search(my_cigar_aligned)
            if beg_m is not None:
                assert end_m is None
                len_N = len(beg_m.group(1))
                self.my_cigar_list_aligned[query_idx] = "O" * len_N + my_cigar_aligned[len_N:]
                self.query_seq_list_aligned[query_idx] = " " * len_N + self.query_seq_list_aligned[query_idx][len_N:]
                continue
            if end_m is not None:
                assert beg_m is None
                len_N = len(end_m.group(1))
                self.my_cigar_list_aligned[query_idx] = my_cigar_aligned[:-len_N] + "O" * len_N
                self.query_seq_list_aligned[query_idx] = self.query_seq_list_aligned[query_idx][:-len_N] + " " * len_N
                continue

    @staticmethod
    def custom_fill(seq_aligned, some_list, empty_value):
        r = np.full(len(seq_aligned), fill_value=empty_value, dtype=type(some_list[0]))
        r[np.where(np.array(list(seq_aligned)) != "-")[0]] = some_list
        return r
    #############
    # MSA 後処理 #
    #############
    def set_query_seq_offset_list(self, query_seq_offset_list: List[int]):
        self.query_seq_offset_list = query_seq_offset_list
        """
        query_seq_offset に 0 は許されない: 0 のかわりに len(query_seq) になっている
        そうしないと、offset が 0 であるが aligned の状態が HO region から開始する場に本当の offset_aligned が得られない
        *** query_seq_offset_aligned については可能な場合は 0 を用いるよ (for loop の else 節) ***
        """
        self.query_seq_offset_list_aligned = []
        for query_seq_offset, my_cigar_aligned in zip(self.query_seq_offset_list, self.my_cigar_list_aligned):
            assert query_seq_offset > 0
            cur_len_at_the_end = 0
            for i, c in enumerate(my_cigar_aligned[::-1]):
                if cur_len_at_the_end == query_seq_offset:
                    break
                if c in "=XIS":
                    cur_len_at_the_end += 1
                elif c in "DNHO":
                    continue
                else:
                    raise Exception(f"error: {c}")
            else:
                i = 0   # break されなかったということは、最初が切れ目ということ
            self.query_seq_offset_list_aligned.append(i)
    def execute_soft_clipping(self):
        """
        S ごとに、それが存在することでスコアを上げることができるかを判定
        """
        # 行 (query) ごとに削除してく
        total_len = len(self.ref_seq_aligned)
        loc_to_remove = [[] for i in range(len(self.query_seq_list_aligned))]
        for query_idx, (query_seq_aligned, my_cigar_aligned, query_seq_offset_aligned) in enumerate(zip(self.query_seq_list_aligned, self.my_cigar_list_aligned, self.query_seq_offset_list_aligned)):
            """ S の左端処理 (右側に進みながら処理)
            Example 1:  SSS======HHH
            Example 2:  HHHSSS======
            Example 3:  ===HHHSSS===
            Example 4:  SSS===HHHSSS
            Example 5:  ===SSSHHH=== (運悪く polish の切れ目に S の切れ目がきてしてしまうと、左端でもこれが起こり得てしまう)
            """
            # soft-clipping の範囲を右側に向けて探索しながら処理
            assert query_seq_offset_aligned >= 0
            cur_idx = -query_seq_offset_aligned     # これが、soft-clipping の左端のインデックスである
            assert my_cigar_aligned[cur_idx] in "=XIS"
            left_side_S_idx_list = []
            score_diff_list = []
            while my_cigar_aligned[cur_idx] in "SNOH":   # 左 > 右に進みながら終わりを探す & スコア計算 # Example 5 の処理のため、OH も含める必要あり
                """
                # 情報エントロピーを計算
                entropy_full += self.calc_entropy(
                    [qs_aligned[cur_idx] for qs_aligned, mc_aligned in zip(self.query_seq_list_aligned, self.my_cigar_list_aligned) if mc_aligned[cur_idx] in "=XSDI"]    # omit "HON"
                )
                entropy_wo_cur_query += self.calc_entropy(
                    [qs_aligned[cur_idx] for i, (qs_aligned, mc_aligned) in enumerate(zip(self.query_seq_list_aligned, self.my_cigar_list_aligned)) if (mc_aligned[cur_idx] in "=XSDI") and (i != query_idx)]    # omit "HON"
                )
                """
                # スコア計算 (情報エントロピーだと、全て一致してるところに一箇所だけミスマッチを含む S の配列が与えられた時などに、omit されてしまう)
                score_diff_list.append(self.calc_score_diff(
                    query_seq_aligned[cur_idx], 
                    my_cigar_aligned[cur_idx], 
                    [qs_aligned[cur_idx] for i, qs_aligned in enumerate(self.query_seq_list_aligned) if (qs_aligned[cur_idx] not in " -") and (i != query_idx)]    # omit "ON" (S と同時に出現しうるのは SON のみ)
                ))
                left_side_S_idx_list.append(cur_idx%total_len)  # %計算は必要ないかも
                # 右に進んで次に備える
                cur_idx += 1
            # score_diff がプラスになるとこまでを保存、それ以外は消す
            cur_score = 0
            range_with_positive_score_len = 0
            for idx, score_diff in enumerate(score_diff_list[::-1]):     # 右 > 左 に戻りながら処理
                cur_score += score_diff
                if cur_score > 0:
                    range_with_positive_score_len = idx + 1
            loc_to_remove[query_idx].extend(left_side_S_idx_list[:len(left_side_S_idx_list)-range_with_positive_score_len])
            """ S の右端処理 (左側に進みながら処理)
            Example 1:  HHH======SSS
            Example 2:  ======SSSHHH
            Example 3:  ===SSSHHH===
            Example 4:  SSSHHH===SSS
            Example 5:  ===HHHSSS=== (運悪く polish の切れ目に S の切れ目がきてしてしまうと、右端でもこれが起こり得てしまう)
            """
            # まず soft-clipping の右端を探す
            cur_idx = total_len - query_seq_offset_aligned - 1 # これが、 soft-clipping 左端より一つ左のインデックス: これをもとに左に進みながら右端のインデックスを探していく
            while my_cigar_aligned[cur_idx] in "HO":
                cur_idx -= 1
            # soft-clipping の範囲を左側に向けて探索しながら処理
            assert my_cigar_aligned[cur_idx] in "=XISN" # HO D  # example 5 のために、N も許容する必要がある
            right_side_S_idx_list_inversed = []
            score_diff_list_inversed = []
            while my_cigar_aligned[cur_idx] in "SNOH":   # 右 > 左に進みながら終わりを探す & スコア計算 # Example 5 の処理のため、OH も含める必要あり
                # スコア計算
                score_diff_list_inversed.append(self.calc_score_diff(
                    query_seq_aligned[cur_idx], 
                    my_cigar_aligned[cur_idx], 
                    [qs_aligned[cur_idx] for i, qs_aligned in enumerate(self.query_seq_list_aligned) if (qs_aligned[cur_idx] not in " -") and (i != query_idx)]    # omit "HON" (S と同時に出現しうるのは SON のみ)
                ))
                right_side_S_idx_list_inversed.append(cur_idx%total_len) # %計算は必要ないかも
                # 左に進んで次に備える
                cur_idx -= 1
            # score_diff がプラスになるとこまでを保存、それ以外は消す
            cur_score = 0
            cur_max_score = 0
            range_until_max_score_len = 0
            for idx, score_diff in enumerate(score_diff_list_inversed[::-1]):     # 左 > 右 に進みながら処理
                cur_score += score_diff
                if cur_score > cur_max_score:
                    range_until_max_score_len = idx + 1
                    cur_max_score = cur_score
            loc_to_remove[query_idx].extend(right_side_S_idx_list_inversed[::-1][range_until_max_score_len:])
        # 新規 MyMSAを作り、削除を実行していく
        new_my_msa = copy.deepcopy(self)
        for query_idx, idx_list in enumerate(loc_to_remove):
            for idx in idx_list:
                new_my_msa.convert_to_H(query_idx, idx)
        new_my_msa.remove_empty_idx()
        return new_my_msa
    def calc_score_diff(self, subject_s, subject_my_c, other_s_list):
        if subject_my_c == "S":
            # 他のシーケンスがある場合
            len_others = len(other_s_list)
            if len_others > 0:
                score = 0
                for s in other_s_list:
                    if subject_s == s:
                        score += self.param_dict["match_score"]
                    else:
                        score += self.param_dict["mismatch_score"]
                return score / len_others
            # 他のシーケンスがない場合
            else:
                return -self.param_dict["gap_open_penalty"] * len(other_s_list)
        elif subject_my_c == "N":
            return 0
        elif subject_my_c == "O":
            return 0
        elif subject_my_c == "H":
            return -self.param_dict["gap_open_penalty"] * len(other_s_list)
        else:
            raise Exception(f"error: {subject_s}, {subject_my_c}")
    #########################
    # ireversible functions #
    #########################
    def convert_to_H(self, query_idx, idx):
        assert idx >= 0
        query_seq_aligned = self.query_seq_list_aligned[query_idx]
        my_cigar_aligned = self.my_cigar_list_aligned[query_idx]
        # H への変換を実行
        self.query_seq_list_aligned[query_idx] = query_seq_aligned[:idx] + " " + query_seq_aligned[idx+1:]
        self.q_scores_list_aligned[query_idx][idx] = -1
        self.my_cigar_list_aligned[query_idx] = my_cigar_aligned[:idx] + ("O" if self.ref_seq_aligned[idx] == "-" else "H") + my_cigar_aligned[idx+1:]
    def remove_empty_idx(self):
        idx_list_to_remove = []
        for idx in range(len(self.ref_seq_aligned)):
            my_cigar_list = [my_cigar_aligned[idx] for my_cigar_aligned in self.my_cigar_list_aligned]
            if all(c in "NO" for c in my_cigar_list) and (len(my_cigar_list) > 0):  # read が 0 だと 前半が True になってしまうので、len(my_cigar_list) のフィルターも必要
                idx_list_to_remove.append(idx)
        # 順序を保持するため、逆向きに削除していく
        for query_idx, (query_seq_aligned, q_scores_aligned, my_cigar_aligned) in enumerate(zip(self.query_seq_list_aligned, self.q_scores_list_aligned, self.my_cigar_list_aligned)):
            query_seq_aligned = list(query_seq_aligned)
            my_cigar_aligned = list(my_cigar_aligned)
            for idx in idx_list_to_remove[::-1]:
                del query_seq_aligned[idx]
                del q_scores_aligned[idx]
                del my_cigar_aligned[idx]
            self.query_seq_list_aligned[query_idx] = "".join(query_seq_aligned)
            self.my_cigar_list_aligned[query_idx] = "".join(my_cigar_aligned)
        # ref_seq からも削除
        ref_seq_aligned = list(self.ref_seq_aligned)
        for idx in idx_list_to_remove[::-1]:
            del ref_seq_aligned[idx]
        self.ref_seq_aligned = "".join(ref_seq_aligned)
        # 既に計算されていたら、consensus_seq からも削除
        if len(self.with_prior_consensus_seq) > 0:
            with_prior_consensus_seq = list(self.with_prior_consensus_seq)
            with_prior_consensus_my_cigar = list(self.with_prior_consensus_my_cigar)
            without_prior_consensus_seq = list(self.without_prior_consensus_seq)
            without_prior_consensus_my_cigar = list(self.without_prior_consensus_my_cigar)
            for idx in idx_list_to_remove[::-1]:
                del with_prior_consensus_seq[idx]
                del with_prior_consensus_my_cigar[idx]
                del without_prior_consensus_seq[idx]
                del without_prior_consensus_my_cigar[idx]
                del self.with_prior_consensus_q_scores[idx]
                del self.without_prior_consensus_q_scores[idx]
            self.with_prior_consensus_seq = "".join(with_prior_consensus_seq)
            self.with_prior_consensus_my_cigar = "".join(with_prior_consensus_my_cigar)
            self.without_prior_consensus_seq = "".join(without_prior_consensus_seq)
            self.without_prior_consensus_my_cigar = "".join(without_prior_consensus_my_cigar)
    ###################
    # print functions #
    ###################
    def print_alignment(self, **print_options):
        center = print_options.get("center", self.default_print_options["center"])
        seq_range = print_options.get("seq_range", self.default_print_options["seq_range"])
        offset = print_options.get("offset", self.default_print_options["offset"])
        N_ref_seq = len(self.ref_seq_NoDEL)
        assert 0 < center <= N_ref_seq
        # 1 スタートになるよう、インデックスを下げる
        center -= 1
        # ref_seq_aligned での index を求める (start)
        s_idx_in_ref_seq = max(0, center - seq_range)
        s_idx = self.ref_seq_idx_in_consensus(s_idx_in_ref_seq)
        c_idx = self.ref_seq_idx_in_consensus(center) + offset
        e_idx = self.ref_seq_idx_in_consensus(min(center + seq_range, N_ref_seq - 1))

        # prepare for color
        palette_max = 40
        q_score_palette = [("230;230;255", "≤", 10), ("200;220;255", "≤", 20), ("160;200;255", "≤", 30), ("110;170;255", "≤", palette_max), ("50;130;255", ">", palette_max)]
        q_score_color_dict = defaultdict(lambda: q_score_palette[-1][0])
        for q in range(palette_max + 1):
            for i, j, k in q_score_palette:
                if q <= k:
                    q_score_color_dict[q] = i
                    break

        # index header
        header_q_score = "".join(f"\033[48;2;{rgb}m {inequality}{score} " for rgb, inequality, score in q_score_palette) + "\033[0m"
        header_consensus_q_scores = "header_consensus_q_scores"
        header_pos = "position "    # 右揃えにするのでスペースは必須
        header_ref_seq_aligned = self.ref_seq_name
        header_consensus_with_prior = "consensus_with_prior"
        header_consensus_without_prior = "consensus_without_prior"
        header_width = max(*[len(i) for i in [header_consensus_q_scores, header_pos, header_ref_seq_aligned, header_consensus_with_prior, header_consensus_without_prior] + self.query_id_list]) + 1

        # positional string # 10毎にポジションを表示
        cur_pos = s_idx_in_ref_seq
        position_string = " " * ((10 - (cur_pos + 1))%10)
        assert self.ref_seq_aligned[s_idx] != "-"
        for ref in self.ref_seq_aligned[s_idx:e_idx+1]:
            # 10毎以外の場所は文字は加えない
            if cur_pos%10 != 9:
                if ref != "-":
                    cur_pos += 1
                else:
                    position_string += " "
            # 10毎以外の場所でも、"-" ならまだスルー
            elif ref == "-":
                position_string += " "
            else:
                position_string += f"{cur_pos + 1:<10}"
                cur_pos += 1
        if len(position_string) > e_idx - s_idx + 1:
            if (position_string[e_idx - s_idx] != position_string[e_idx - s_idx + 1] != " "):
                position_string = position_string[:e_idx - s_idx + 1]
                i = -1
                while position_string[i] != " ":
                    i -= 1
                position_string = position_string[:i+1] + " " * -(i+1)
            else:
                position_string = position_string[:e_idx - s_idx + 1]

        #########
        # PRINT #
        #########
        print(f"{header_pos:>{header_width}}{position_string}     Q-score {header_q_score}\033[0m")
        print(
            f"\033[1m{header_ref_seq_aligned:<{header_width}}"
            f"{self.ref_seq_aligned[s_idx:c_idx]}\033[48;2;180;180;180m{self.ref_seq_aligned[c_idx]}\033[0m{self.ref_seq_aligned[c_idx+1:e_idx+1]}"
            f"\033[0m     1{' '*8}10{' '*8}20{' '*8}30{' '*8}40{' '*8}50"
        )
        print(
            f"{header_consensus_with_prior:<{header_width}}"
            f"{self.q_score_color_string(self.with_prior_consensus_seq[s_idx:e_idx+1], self.with_prior_consensus_q_scores[s_idx:e_idx+1], q_score_color_dict, c_idx - s_idx, self.ref_seq_aligned[s_idx:e_idx+1])}\033[0m"
            f"{self.q_score_bar(None, None, self.with_prior_consensus_q_scores[c_idx], self.with_prior_consensus_my_cigar[c_idx], q_score_color_dict, show_asterisk=True)}\033[0m"
        )
        print(
            f"{header_consensus_without_prior:<{header_width}}"
            f"{self.q_score_color_string(self.without_prior_consensus_seq[s_idx:e_idx+1], self.without_prior_consensus_q_scores[s_idx:e_idx+1], q_score_color_dict, c_idx - s_idx, self.ref_seq_aligned[s_idx:e_idx+1])}\033[0m"
            f"{self.q_score_bar(None, None, self.without_prior_consensus_q_scores[c_idx], self.without_prior_consensus_my_cigar[c_idx], q_score_color_dict, show_asterisk=True)}\033[0m"
        )
        for query_id, query_seq, q_scores, my_cigar in zip(self.query_id_list, self.query_seq_list_aligned, self.q_scores_list_aligned, self.my_cigar_list_aligned):
            # print(f"{query_id:<{header_width}}{query_seq[s_idx:e_idx+1]}")
            print(
                f"{query_id:<{header_width}}{self.my_cigar_color_string(self.ref_seq_aligned[s_idx:e_idx+1], query_seq[s_idx:e_idx+1], my_cigar[s_idx:e_idx+1], c_idx - s_idx)}"
                f"{self.q_score_bar(self.ref_seq_aligned[c_idx], query_seq[c_idx], q_scores[c_idx], my_cigar[c_idx], q_score_color_dict, show_asterisk=False)}\033[0m"
            )
            # print(my_cigar)
    def ref_seq_idx_in_consensus(self, ref_seq_idx):
        idx = idx_in_ref_seq = ref_seq_idx
        N_diff = self.ref_seq_aligned[:idx + 1].count("-")
        while idx - N_diff != idx_in_ref_seq:
            previous_idx = idx
            idx += idx_in_ref_seq - (idx - N_diff)
            N_diff += self.ref_seq_aligned[previous_idx + 1:idx + 1].count("-")
        return idx
    @staticmethod
    def q_score_color_string(seq, q_scores, q_score_color_dict: defaultdict, make_it_bold: int, ref_seq):
        stdout_txt = ""
        for i, (s, q, r) in enumerate(zip(list(seq), q_scores, ref_seq)):
            if s == r:
                L = f"\033[48;2;{q_score_color_dict[q]}m{s}"
            else:
                L = f"\033[48;2;255;0;0m{s}\033[0m"
                # L = f"\033[48;2;{q_score_color_dict[q]}m\033[38;2;255;0;0m{s}"
            if i == make_it_bold:
                L = f"\033[1m{L}\033[0m"
            stdout_txt += L
        return stdout_txt
    @staticmethod
    def my_cigar_color_string(ref_seq, query_seq, my_cigar, make_it_bold: int):
        stdout_txt = ""
        for i, (r, q, c) in enumerate(zip(list(ref_seq), list(query_seq), list(my_cigar))):
            if c in "=NHO":
                L = q
            elif c in "XID":
                L = f"\033[48;2;255;176;176m{q}\033[0m"
            elif c == "S":
                L = q if r == q else f"\033[48;2;255;176;176m{q}\033[0m"
            else:
                raise Exception(f"error: {c}")
            if i == make_it_bold:
                L = f"\033[1m{L}\033[0m"
            stdout_txt += L
        return stdout_txt
    @staticmethod
    def q_score_bar(r, q, q_score, my_cigar, q_score_color_dict: defaultdict, show_asterisk):
        color = q_score_color_dict[q_score]
        if my_cigar in "=NHO":
            q_score_txt = f"{q_score:<3}"
        elif my_cigar in "S":
            q_score_txt = f"{q_score:<3}" if r == q else f"\033[48;2;255;176;176m{q_score:<2}\033[0m "
        else:
            q_score_txt = f"\033[48;2;255;176;176m{q_score:<2}\033[0m "
        if not show_asterisk:
            return f"  {q_score_txt}\033[48;2;{color}m\033[38;2;{color}m{'*' * q_score}"
        else:
            return f"  {q_score_txt}\033[48;2;{color}m{'*' * q_score}"
    ########################
    # consensus calculator #
    ########################
    @classmethod
    def get_sbq_pdf_info_list(cls):
        sbq_pdf_list = []
        sbq_pdf_minor_version = 0
        sbq_pdf_minor_version_list = []
        while True:
            df_csv_path = Path(__file__).parent / f"NanoporeStats_PDF/NanoporeStats_{cls.sbq_pdf_major_version}.{sbq_pdf_minor_version}.csv"
            if df_csv_path.exists():
                sbq_pdf_list.append(SequenceBasecallQscorePDF(df_csv_path=df_csv_path))
                sbq_pdf_minor_version_list.append(sbq_pdf_minor_version)
                sbq_pdf_minor_version += 1
            else:
                break
        return sbq_pdf_list, sbq_pdf_minor_version_list
    @classmethod
    def set_sbq_pdf(cls, my_fastq: mc.MyFastQ):
        q_score_distribution = my_fastq.get_q_score_distribution()
        sbq_pdf_list, sbq_pdf_minor_version_list = cls.get_sbq_pdf_info_list()
        js_scores = [sbq_pdf.calc_js_divergence(q_score_distribution) for sbq_pdf in sbq_pdf_list]
        min_idx = np.argmin(js_scores)
        cls.sbq_pdf = sbq_pdf_list[min_idx]
        cls.sbq_pdf_minor_version = sbq_pdf_minor_version_list[min_idx]
    def calculate_consensus(self):
        # params
        P_N_dict_dict_with_prior, P_N_dict_dict_without_prior = self.consensus_params(self.param_dict, make_it_for_CYTHON=True)
        # execute
        self.with_prior_consensus_seq, self.with_prior_consensus_q_scores, self.with_prior_consensus_my_cigar = self.calculate_consensus_core(P_N_dict_dict_with_prior)
        self.without_prior_consensus_seq, self.without_prior_consensus_q_scores, self.without_prior_consensus_my_cigar = self.calculate_consensus_core(P_N_dict_dict_without_prior)
    def calculate_consensus_core(self, P_N_dict_dict):
        consensus_seq = ""
        consensus_q_scores = []
        consensus_my_cigar = ""
        for consensus_idx, ref in tqdm(enumerate(self.ref_seq_aligned), ncols=100, mininterval=0.05, leave=True, bar_format='{l_bar}{bar}{r_bar}', desc="generating consensus...", total=len(self.ref_seq_aligned)):
            query_list = [i[consensus_idx] for i in self.query_seq_list_aligned]
            q_score_list = [i[consensus_idx] for i in self.q_scores_list_aligned]
            L_list = [i[consensus_idx] for i in self.my_cigar_list_aligned]
            event_list = [(i.upper(), j) for i, j, k in zip(query_list, q_score_list, L_list) if k not in "HO"]     # =XIDSN

            query_list = []
            q_score_list = []
            for query_idx, my_cigar_aligned in enumerate(self.my_cigar_list_aligned):
                if my_cigar_aligned[consensus_idx] not in "HO":      # =XIDSN
                    query_list.append(self.query_seq_list_aligned[query_idx][consensus_idx])
                    q_score_list.append(self.q_scores_list_aligned[query_idx][consensus_idx])

            if len(event_list) > 0:
                P_N_dict = P_N_dict_dict[ref.upper()]

                ### PYTHON ###
                # event_list = [(i.upper(), j) for i, j, k in zip(query_list, q_score_list, L_list)]
                # p_list = [
                #     self.sbq_pdf.calc_consensus_error_rate(event_list, true_refseq=B, P_N_dict=P_N_dict, bases=self.bases)
                #     for B in self.bases
                # ]
                ### CYTHON ###
                p_list = self.sbq_pdf.sbq_pdf_CYTHON.calc_consensus_error_rate("".join(query_list).encode("utf-8"), q_score_list, P_N_dict)
                ##############
                p = min(p_list)
                # p_idx_list = [i for i, v in enumerate(p_list) if v == p]
                consensus_base_call = self.mixed_bases([b for b, tmp_p in zip(self.bases, p_list) if tmp_p == p])

                # register
                if p >= 10 ** (-5):
                    q_score = np.round(-10 * np.log10(p)).astype(int)
                elif p < 0:
                    raise Exception("unknown error")
                else:
                    q_score = 50
            # アサインされたリード数が 0 の場合のために必要
            else:
                consensus_base_call = "-"
                q_score = -1

            # my_cigar に追加
            consensus_seq += consensus_base_call
            consensus_q_scores.append(q_score)
            if ref == consensus_base_call != "-":
                consensus_my_cigar += "="
            elif ref == "-" != consensus_base_call:
                consensus_my_cigar += "I"
            elif ref != consensus_base_call == "-":
                consensus_my_cigar += "D"
            elif "-" != ref != consensus_base_call != "-":
                consensus_my_cigar += "X"
            elif ref == consensus_base_call == "-":
                consensus_my_cigar += "N"
            else:
                raise Exception(f"error: {ref} {consensus_base_call}")
        return consensus_seq, consensus_q_scores, consensus_my_cigar
    @classmethod
    def consensus_params(cls, param_dict, make_it_for_CYTHON):
        ins_rate = param_dict["ins_rate"]
        error_rate = param_dict["error_rate"]
        # del_mut_rate = error_rate / 4, # e.g. "A -> T, C, G, del"
        len_bases = len(cls.bases)
        assert len_bases == 5 # "ATCG-"
        ##############
        # with prior #
        ##############
        P_N_dict_dict_with_prior = {   # 真のベースが b_key1 である場合に、b_key2 への mutation/deletion などが起こる確率
            b_key1:{b_key2:(1 - error_rate) / len(b_val1) if b_key2 in b_val1 else error_rate / (len_bases - len(b_val1)) for b_key2 in cls.bases} for b_val1, b_key1 in cls.letter_code_dict.items()
        }
        P_N_dict_dict_with_prior["-"] = {b_key2:ins_rate / 4 if b_key2 != "-" else 1 - ins_rate for b_key2 in cls.bases}
        #################
        # without prior #
        #################
        P_N_dict_dict_without_prior = {
            b_key1:{b_key2: 1 / len_bases for b_key2 in cls.bases} for b_key1 in "".join(cls.letter_code_dict.values()) + "-"
        }
        ######################
        # prepare for cython #
        ######################
        if make_it_for_CYTHON:
            P_N_dict_dict_with_prior = {
                key:{ord(k):v for k, v in val.items()}
                for key, val in P_N_dict_dict_with_prior.items()
            }
            P_N_dict_dict_without_prior = {
                key:{ord(k):v for k, v in val.items()}
                for key, val in P_N_dict_dict_without_prior.items()
            }
        return P_N_dict_dict_with_prior, P_N_dict_dict_without_prior
    def mixed_bases(self, base_list):
        if len(base_list) == 1:
            return base_list[0]
        elif "-" not in base_list:
            pass
        else:
            base_list.remove("-")
        letters = ""
        for b in self.bases[:-1]:
            if b in base_list:
                letters += b
        return self.letter_code_dict[letters]
    # FOR LOG
    @classmethod
    def P_N_dict_dict_2_matrix(cls, param_dict):
        P_N_dict_dict_with_prior, P_N_dict_dict_without_prior = cls.consensus_params(param_dict, make_it_for_CYTHON=False)
        def gen_matrix_from_dict_dict(P_N_dict_dict):
            r_matrix = np.empty((len(cls.bases), len(cls.bases)), dtype=float)
            for r, b_key1 in enumerate(cls.bases):
                for c, b_key2 in enumerate(cls.bases):
                    r_matrix[r, c] = P_N_dict_dict[b_key1][b_key2]
            return r_matrix
        return gen_matrix_from_dict_dict(P_N_dict_dict_with_prior), gen_matrix_from_dict_dict(P_N_dict_dict_without_prior)
    #######################
    # export/load results #
    #######################
    def export_consensus_fastq(self, save_dir):
        # with prior
        save_path = save_dir / f"{self.ref_seq_name}.consensus_with_prior.fastq"
        consensus_q_score_string = "".join([chr(q + 33) for q in self.with_prior_consensus_q_scores_NoDEL])
        consensus_fastq_txt = f"@{self.ref_seq_name}.with_prior:\n{self.with_prior_consensus_seq_NoDEL.upper()}\n+\n{consensus_q_score_string}"
        with open(save_path, "w") as f:
            f.write(consensus_fastq_txt)
        # without prior
        save_path = save_dir / f"{self.ref_seq_name}.consensus_without_prior.fastq"
        consensus_q_score_string = "".join([chr(q + 33) for q in self.without_prior_consensus_q_scores_NoDEL])
        consensus_fastq_txt = f"@{self.ref_seq_name}.without_prior:\n{self.without_prior_consensus_seq_NoDEL.upper()}\n+\n{consensus_q_score_string}"
        with open(save_path, "w") as f:
            f.write(consensus_fastq_txt)
    def export_gif(self, save_dir):
        save_path = save_dir / f"{self.ref_seq_name}.gif"
        # prepare
        N_array = np.empty((6, len(self.ref_seq_aligned)), int)
        tick_pos_list = []
        tick_label_list = []
        cur_ref_pos = 0
        for ref_idx, ref in enumerate(self.ref_seq_aligned):

            if ref != "-":
                cur_ref_pos += 1
                if cur_ref_pos%100 == 0:
                    tick_pos_list.append(ref_idx)
                    tick_label_list.append(cur_ref_pos)

            N_match = 0         # =
            N_mismatch = 0      # X
            N_insertion = 0     # I, S
            N_deletion = 0      # D
            N_skipped = 0       # N
            N_not_covered = 0   # H, O
            for my_cigar in self.my_cigar_list_aligned:
                L = my_cigar[ref_idx]
                if L == "=":    N_match += 1
                elif L == "X":  N_mismatch += 1
                elif L in "IS": N_insertion += 1
                elif L == "D":  N_deletion += 1
                elif L in "N":  N_skipped += 1
                elif L in "HO": N_not_covered += 1
                else:   raise Exception(f"unknown cigar string {L}")
            N_array[:, ref_idx] = [N_match, N_skipped, N_mismatch, N_insertion, N_deletion, N_not_covered]
        # 描画していく！
        bar_graph_img = MyGIF(N_array, tick_pos_list, tick_label_list)
        bar_graph_img.generate_bar_graph_ndarray()
        bar_graph_img.set_legend(legend_list=["match", "skipped", "mismatch", "insertion", "deletion", "not_covered"])
        bar_graph_img.set_legend_description(f"TOTAL READS: {len(self.query_id_list)}")
        bar_graph_img.export_as_img(save_path=save_path)
    def export_consensus_alignment(self, save_dir: Path):
        save_path = save_dir / f"{self.ref_seq_name}.ca"
        my_byte_str = self.to_binary()
        with zipfile.ZipFile(save_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.writestr(save_path.with_suffix(".b").name, my_byte_str.my_byte_str)
    def to_binary(self):
        """
        塩基は 6 ビット 単位で保存 (アルファベット大文字小文字および "=" をカバー可能): 3 byte (24 bit)で 4文字
        q_score も 6 ビット 単位で保存 (0-41 および -1 をカバー可能)
        """

        # header info
        my_byte_str_header = MyByteStr()
        my_byte_str_header.add_byte_str(self.file_format_version.encode("utf-8"))
        my_byte_str_header.add_byte_str(self.param_dict_str.encode("utf-8"))
        my_byte_str_header.add_byte_str(self.ref_seq_name.encode("utf-8"))
        for query_id in self.query_id_list:
            my_byte_str_header.add_byte_str(query_id.encode("utf-8"))

        # ref/consensus related
        my_byte_str_ref_related = MyByteStr()
        for func_name, attr_name in self.ref_seq_related_save_order:
            func = getattr(my_byte_str_ref_related, func_name)
            func(getattr(self, attr_name))

        # query/fastq related
        my_byte_str_query_related = MyByteStr()
        for func_name, attr_name in self.query_seq_related_save_order:
            func = getattr(my_byte_str_query_related, func_name)
            for q in getattr(self, attr_name):
                func(q)

        # データ追加
        my_byte_str = MyByteStr()
        my_byte_str.add_byte_str(my_byte_str_header.my_byte_str)
        my_byte_str.add_byte_str(my_byte_str_ref_related.my_byte_str)
        my_byte_str.add_byte_str(my_byte_str_query_related.my_byte_str)

        # cigar は再構築可能なので保存しない
        # self.my_cigar_list_aligned
        # self.with_prior_consensus_my_cigar
        # self.without_prior_consensus_my_cigar

        return my_byte_str
    def load_consensus_alignment(self, load_path: str):
        load_path = Path(load_path)
        with zipfile.ZipFile(load_path, "r") as z:
            with z.open(load_path.with_suffix(".b").name) as f:
                header = MyByteStr.read_buffer(f)
                ref_related = MyByteStr.read_buffer(f)
                query_related = MyByteStr.read_buffer(f)
        # header
        header_contents = MyByteStr.read_byte_string(header)
        assert self.file_format_version == header_contents[0]
        self.param_dict_str = header_contents[1]
        self.ref_seq_name = header_contents[2]
        self.query_id_list = header_contents[3:]
        N_query = len(self.query_id_list)

        # ref/consensus related
        format_txt = "".join(re.match(r"^add_(?=sequence|q_scores)(s|q).+$", func_name).group(1).upper() for func_name, attr_name in self.ref_seq_related_save_order)
        ref_related_contents = MyByteStr.read_byte_seq_q_scores(ref_related, format_txt)
        for (func_name, attr_name), content in zip(self.ref_seq_related_save_order, ref_related_contents):
            setattr(self, attr_name, content)

        # query/fastq related
        format_txt = "".join(re.match(r"^add_(?=sequence|q_scores|clipping_info)(s|q|c).+$", func_name).group(1).upper() * N_query for func_name, attr_name in self.query_seq_related_save_order)
        query_related_contents = MyByteStr.read_byte_seq_q_scores(query_related, format_txt)
        for i, (func_name, attr_name) in enumerate(self.query_seq_related_save_order):
            setattr(self, attr_name, query_related_contents[i * N_query: (i + 1) * N_query])

        # # cigar セット    # self.clipping_info_list の setattr で行われる
        # self.regenerate_cigar()
    def regenerate_cigar(self):
        assert all(len(query_seq) == len(self.ref_seq_aligned) for query_seq in self.query_seq_list_aligned) and (len(self.with_prior_consensus_seq) == len(self.without_prior_consensus_seq) == len(self.ref_seq_aligned))
        self.my_cigar_list_aligned = ["" for query_id in self.query_id_list]
        self.with_prior_consensus_my_cigar = ""
        self.without_prior_consensus_my_cigar = ""
        for query_idx, query_seq in enumerate(self.query_seq_list_aligned):
            for ref, query in zip(self.ref_seq_aligned, query_seq):
                self.my_cigar_list_aligned[query_idx] += self.gen_basic_cigar(ref, query)
        for ref, consensus in zip(self.ref_seq_aligned, self.with_prior_consensus_seq):
            self.with_prior_consensus_my_cigar += self.gen_basic_cigar(ref, consensus)
        for ref, consensus in zip(self.ref_seq_aligned, self.without_prior_consensus_seq):
            self.without_prior_consensus_my_cigar += self.gen_basic_cigar(ref, consensus)
    @staticmethod
    def gen_basic_cigar(ref, query):
        # (r, q) = ("A", "A") -> "="  # (ref, query) -> cigar
        # (r, q) = ("-", "-") -> "N"
        # (r, q) = ("-", "T") -> "I"
        # (r, q) = ("-", "T") -> "S"
        # (r, q) = ("-", " ") -> "O"

        # (r, q) = ("A", "T") -> "X"

        # (r, q) = ("A", "-") -> "D"
        # (r, q) = ("A", " ") -> "H"
        # self.generate_msa も参照
        if query == ref:
            if ref != "-":
                return "="
            else:
                assert ref != " "
                return "N"
        else:
            if ref == "-":
                if query == " ":
                    return "O"
                else:
                    return "I"  # or S -> fixed in @clipping_info_list.setter
            else:
                assert ref != " "
                if query == "-":
                    return "D"
                elif query == " ":
                    return "H"
                else:
                    return "X"
    def assert_data(self):
        for i, ref in enumerate([ref for ref in self.ref_seq_aligned]):
            for query_seq, q_scores, my_cigar in zip(self.query_seq_list_aligned, self.q_scores_list_aligned, self.my_cigar_list_aligned):
                query = query_seq[i]
                q_score = q_scores[i]
                L = my_cigar[i]
                if L == "=":
                    assert ref == query
                    assert q_score != -1
                elif L == "X":
                    assert ref != query
                    assert q_score != -1
                elif L == "I":
                    assert ref == "-"
                    assert q_score != -1
                elif L == "D":
                    assert query == "-"
                    assert q_score == -1
                elif L == "N":
                    assert query == "-"
                    assert q_score == -1
                else:
                    raise Exception(f"unknown cigar: {L}")
        for ref, con, q_score, L in zip(self.ref_seq_aligned, self.with_prior_consensus_seq, self.with_prior_consensus_q_scores, self.with_prior_consensus_my_cigar):
            if L == "=":
                assert ref == con != "-"
                assert q_score != -1
            elif L == "X":
                assert "-" != ref != con != "-"
                assert q_score != -1
            elif L == "D":
                assert con == "-" != ref
            elif L == "I":
                assert ref == "-" != con
                assert q_score != -1
            elif L == "N":
                assert con == ref == "-"
            else:
                raise Exception(f"unknown cigar: {L}")
        for ref, con, q_score, L in zip(self.ref_seq_aligned, self.without_prior_consensus_seq, self.without_prior_consensus_q_scores, self.without_prior_consensus_my_cigar):
            if L == "=":
                assert ref == con != "-"
                assert q_score != -1
            elif L == "X":
                assert "-" != ref != con != "-"
                assert q_score != -1
            elif L == "D":
                assert con == "-" != ref
            elif L == "I":
                assert ref == "-" != con
                assert q_score != -1
            elif L == "N":
                assert con == ref == "-"
            else:
                raise Exception(f"unknown cigar: {L}")
        return True
    def convert_to_bam(self, save_dir=None, replace_S_with_I=False, ext_not_exported=[".sorted.sam", ".bam.fastq"]):
        try:
            import pysam
        except ImportError:
            raise Exception("ImportError: Missing optional dependency 'pysam'.  Use pip to install pysam.")
        except ModuleNotFoundError:
            raise Exception("ModuleNotFoundError: Missing optional dependency 'pysam'.  Use pip to install pysam.")
        S_replacement = "I" if replace_S_with_I else "S"
        # Create a BAM file
        ref_len = len(self.ref_seq_NoDEL)
        header_text = f"@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:{self.ref_seq_name}\tLN:{ref_len}"
        header = pysam.AlignmentHeader.from_text(header_text)
        ref_id = header.get_tid(self.ref_seq_name)
        # =XDISHO
        m_re_A = re.compile(r"^([^HO].*?)[HO]+(.*?[^HO])$") # ====HHH====
        m_re_B = re.compile(r"^([HO]*)(.+?)[HO]*$")         # HHH=====HHH, HHH========, ========HHH, ===========
        # 一時的なファイルを作成
        with MyBamSet(ext_not_exported=ext_not_exported) as mbs:
            ### ファイルを作成
            with pysam.AlignmentFile(mbs.temp_bam_file_path.as_posix(), "wb", header=header) as bam_file:
                temp_my_fastq = mc.MyFastQ()
                for query_id, query_seq, q_scores_list_aligned, my_cigar in zip(self.query_id_list, self.query_seq_list_aligned, self.q_scores_list_aligned, self.my_cigar_list_aligned):
                    assert my_cigar[0] in "=XDSH"    # =XDISHO
                    query_name = query_id.split(" ")[0].strip("@")
                    # Create an alignment entry
                    m = m_re_A.match(my_cigar)
                    if m is not None:
                        # 準備
                        my_cigar_1 = m.group(1)
                        my_cigar_2 = m.group(2)
                        len_my_cigar_2 = len(my_cigar_2)
                        my_cigar_2_start = len(my_cigar) - len_my_cigar_2
                        query_2_start = my_cigar_2_start - my_cigar[:my_cigar_2_start].count("I") - my_cigar[:my_cigar_2_start].count("N") - my_cigar[:my_cigar_2_start].count("S") - my_cigar[:my_cigar_2_start].count("O")
                        # read 1 追加
                        read = pysam.AlignedSegment()
                        read.query_name = query_name + ".1"                                 # 1.QNAME
                        read.flag = 1 + 2 + 64                                              # 2.FLAG    paired, mapped properly, first in pair
                        read.reference_id = ref_id                                          # 3.RNAME
                        read.reference_start = 0                                            # 4.POS
                        read.mapping_quality = 40                                           # 5.MAPQ
                        read.cigarstring = "".join(f"{len(LLL)}{L}" for LLL, L in self.generate_cigar_iter(my_cigar_1.replace("N", "").replace("S", S_replacement)))         # 6.CIGAR
                        read.next_reference_id = ref_id                                     # 7.RNEXT
                        read.next_reference_start = query_2_start                           # 8.PNEXT
                        read.template_length = ref_len                                      # 9.TLEN
                        read.query_sequence = query_seq[:len(my_cigar_1)].replace("-", "")  # 10.SEQ
                        read.query_qualities = [q for q in q_scores_list_aligned[:len(my_cigar_1)] if q != -1]      # 11.QUAL
                        bam_file.write(read)
                        # fastq に追加
                        temp_my_fastq["@" + read.query_name] = [read.query_sequence, read.query_qualities]
                        # read 2 追加
                        read = pysam.AlignedSegment()
                        read.query_name = query_name + ".2"                                 # 1.QNAME
                        read.flag = 1 + 2 + 128                                             # 2.FLAG    paired, mapped properly, first in pair
                        read.reference_id = ref_id                                          # 3.RNAME
                        read.reference_start = query_2_start                                # 4.POS
                        read.mapping_quality = 40                                           # 5.MAPQ
                        read.cigarstring = "".join(f"{len(LLL)}{L}" for LLL, L in self.generate_cigar_iter(my_cigar_2.replace("N", "").replace("S", S_replacement)))         # 6.CIGAR
                        read.next_reference_id = ref_id                                     # 7.RNEXT
                        read.next_reference_start = 0                                       # 8.PNEXT
                        read.template_length = ref_len                                      # 9.TLEN
                        read.query_sequence = query_seq[-len_my_cigar_2:].replace("-", "")  # 10.SEQ
                        read.query_qualities = [q for q in q_scores_list_aligned[-len_my_cigar_2:] if q != -1]      # 11.QUAL
                        bam_file.write(read)
                        # fastq に追加
                        temp_my_fastq["@" + read.query_name] = [read.query_sequence, read.query_qualities]
                    else:
                        # 準備
                        m = m_re_B.match(my_cigar)
                        my_cigar_1 = m.group(1)
                        my_cigar_2 = m.group(2)
                        len_my_cigar_1 = len(my_cigar_1)
                        len_my_cigar_2 = len(my_cigar_2)
                        # read 1 追加
                        read = pysam.AlignedSegment()
                        read.query_name = query_name + ".0"                                 # 1.QNAME
                        read.flag = 1*0 + 2 + 64*0                                          # 2.FLAG    not paired, mapped properly, first in pair
                        read.reference_id = ref_id                                          # 3.RNAME
                        read.reference_start = len_my_cigar_1 - my_cigar_1.count("O")       # 4.POS
                        read.mapping_quality = 40                                           # 5.MAPQ
                        read.cigarstring = "".join(f"{len(LLL)}{L}" for LLL, L in self.generate_cigar_iter(my_cigar_2.replace("N", "").replace("S", S_replacement)))         # 6.CIGAR
                        read.next_reference_id = -1                                         # 7.RNEXT
                        read.next_reference_start = 0                                       # 8.PNEXT
                        read.template_length = len_my_cigar_2 - query_seq[len_my_cigar_1:len_my_cigar_1 + len_my_cigar_2].count("-")                        # 9.TLEN
                        read.query_sequence = query_seq[len_my_cigar_1:len_my_cigar_1 + len_my_cigar_2].replace("-", "")                                    # 10.SEQ
                        read.query_qualities = [q for q in q_scores_list_aligned[len_my_cigar_1:len_my_cigar_1 + len_my_cigar_2] if q != -1]                # 11.QUAL
                        bam_file.write(read)
                        # fastq に追加
                        temp_my_fastq["@" + read.query_name] = [read.query_sequence, read.query_qualities]

            ### ソート アンド インデックス
            temp_my_fastq.export(mbs.temp_bam_fastq_file_path.as_posix(), overwrite=True)
            pysam.sort("-o", mbs.temp_sorted_bam_file_path.as_posix(), mbs.temp_bam_file_path.as_posix())  # -o FILE    Write final output to FILE rather than standard output
            pysam.index(mbs.temp_sorted_bam_file_path.as_posix())

            ### sam 生成
            with pysam.AlignmentFile(mbs.temp_sorted_bam_file_path, "rb") as bam_file:
                with pysam.AlignmentFile(mbs.temp_sorted_sam_file_path, "wh", header=bam_file.header) as sam_file:
                    for read in bam_file:
                        sam_file.write(read)

            if save_dir is not None:
                mbs.save(Path(save_dir) / self.ref_seq_name)
            else:
                mbs.keep(True)
        return mbs
    def generate_stats(self):
        # q_scores 0~41 and -1 = 43
        d = {f"{i}_{j}":[0 for k in range(mc.MyFastQ.maximum_q_score_allowed + 2)] for i in SequenceBasecallQscorePDF.bases for j in SequenceBasecallQscorePDF.bases}
        for query_seq_aligned, q_scores_aligned in zip(self.query_seq_list_aligned, self.q_scores_list_aligned):
            query_seq_aligned = query_seq_aligned.upper()
            for idx, r in enumerate(self.ref_seq_aligned.upper()):
                q = query_seq_aligned[idx]
                if q == " ":
                    continue
                else:
                    d[f"{r}_{q}"][q_scores_aligned[idx]] += 1
        return d

class MyBamSet(mc.MyTempFiles):
    """
    一時的なファイルが作られるので、
    with MyBamSet() as mbs:
        mbs.save()
    みたいな感じで使う。
    作業が終了したら一時ファイルは削除される。
    """
    def __init__(self, ext_not_exported) -> None:
        default_ext_list = [".bam", ".sorted.bam", ".sorted.bam.bai", ".sorted.sam", ".bam.fastq"]
        self.ext_not_exported = ext_not_exported
        assert all(ext in default_ext_list for ext in self.ext_not_exported)
        # 初期化
        super().__init__(default_ext_list)
    def save(self, save_path_base):
        removed_temp_file_paths = []
        for ext in self.ext_not_exported:
            temp_file_path = self.temp_file_path_base.with_suffix(ext)
            self.temp_file_paths.remove(temp_file_path)
            removed_temp_file_paths.append(temp_file_path)
        # 保存
        super().save(save_path_base)
        # もとに戻す (self.__exit__ や self.__del__ に備える)
        self.temp_file_paths += removed_temp_file_paths

class MyByteStr():
    class MyBIT():
        my_bit = 6             # 一文字 6 ビットで保存する
        struct_bit = 24     
        seq_offset = 64
        edian = ">"
        @mc.classproperty
        def struct_N(cls):
            return cls.struct_bit // cls.my_bit
        @mc.classproperty
        def struct_byte(cls):
            return cls.struct_bit // 8
        @mc.classproperty
        def my_bit_struct_bitshift(cls):
            return np.arange(cls.struct_N)[::-1] * cls.my_bit
        @mc.classproperty
        def byte_struct_bitshift(cls):
            return np.arange(cls.struct_byte)[::-1] * 8
        @mc.classproperty
        def byte_bitmask(cls):
            return np.hstack([255 for i in range(cls.struct_byte)]) # 2 ** 8 -1
        @mc.classproperty
        def my_bit_bitmask(cls):
            return np.hstack([63 for i in range(cls.struct_N)])  # 2 ** 6 -1
        @classmethod
        def seq2byte(cls, seq: str):
            """
            str -> ord -> seq_64 (64引く)
            "A" -> 65  -> 1
            "Z" -> 90  -> 26
            "a" -> 97  -> 33
            "z" -> 122 -> 58

            str -> replace -> seq_64 (64引く)
            "-"(45) -> "~"(126) -> 62
            " "(32) -> "|"(124) -> 60
            """
            # 文字を数値に変換
            seq_64 = [ord(s) - cls.seq_offset for s in seq.replace("-", "~").replace(" ", "|")]
            return cls.value2byte_core(seq_64)
        @classmethod
        def score2byte(cls, seq_64: List[int]):
            """
            value -> seq_64
            -1 -> 62
            """
            return cls.value2byte_core([s if s != -1 else 62 for s in seq_64])
        @classmethod
        def value2byte_core(cls, seq_64: list):
            """ パディング用の文字
            "DEL"->127 -> 63 -> (127 "DEL")
            """
            # DEL 文字 (63) は使えないので注意
            assert (np.greater_equal(seq_64, 0) * np.less_equal(seq_64, 62)).all()
            # 全体のビット数が 8 の倍数になるように補完（6ビットの場合、4の倍数になる必要がある: 6x4 = 8x3）
            seq_64 += [63] * (cls.struct_N - len(seq_64)%cls.struct_N)  # 63 は "DEL"
            # 左シフトして足し算、右シフトにより違う幅で分割し直すことでバイトに直す
            seq_64_byte = np.bitwise_and(
                (
                    np.array(seq_64).reshape(-1, cls.struct_N) << cls.my_bit_struct_bitshift    # 左シフト
                ).sum(axis=1)[:, np.newaxis] >> cls.byte_struct_bitshift,                       # 足し算して右シフト
                cls.byte_bitmask                                                                # マスクして目的の桁より大きい桁を消す
            ).flatten()
            return struct.pack(f"{cls.edian}{len(seq_64_byte)}B", *list(seq_64_byte))
        @classmethod
        def byte2seq(cls, b: bytes):
            seq_64 = "".join(chr(v) for v in cls.byte2value_core(b) + cls.seq_offset)
            return seq_64.replace(chr(127), "").replace("~", "-").replace("|", " ")   # バッファー用の削除文字を削除, "~" を置換, "|" を空白文字を置換
        @classmethod
        def byte2q_score(cls, b: bytes):
            q_scores = cls.byte2value_core(b)
            while (len(q_scores) > 0) and (q_scores[-1] == 63):
                q_scores = np.delete(q_scores, -1)
            assert not (q_scores == 63).any()
            q_scores[q_scores == 62] = -1
            return list(q_scores)
        @classmethod
        def byte2value_core(cls, byte_values):
            values_in_byte = struct.unpack(f"{cls.edian}{len(byte_values)}B", byte_values)
            # 左シフトして足し算、右シフトにより違う幅で分割し直すことでバイトに直す
            seq_64 = np.bitwise_and(
                (
                    np.array(values_in_byte).reshape(-1, cls.struct_byte) << cls.byte_struct_bitshift   # 左シフト
                ).sum(axis=1)[:, np.newaxis] >> cls.my_bit_struct_bitshift,                             # 足し算して右シフト
                cls.my_bit_bitmask# マスクして目的の桁より大きい桁を消す
            ).flatten()
            return seq_64
    size_indicator_bytes = 4
    size_format = f"{MyBIT.edian}I"
    def __init__(self) -> None:
        self.my_byte_str = b""
    def __str__(self) -> str:
        return str(self.my_byte_str)
    def add_byte_str(self, byte_str):
        self.my_byte_str += struct.pack(self.size_format, len(byte_str)) + byte_str
    def add_sequence(self, seq: str):
        byte_seq = self.MyBIT.seq2byte(seq)
        self.add_byte_str(byte_seq)
    def add_q_scores(self, q_scores: List[int]):
        byte_q_scores = self.MyBIT.score2byte(q_scores)
        self.add_byte_str(byte_q_scores)
    def add_clipping_info(self, clipping_info: str):
        """ reserved characters
        ASCII code  symbol
        45          -
        48          0
        49          1
        50          2
        51          3
        52          4
        53          5
        54          6
        55          7
        56          8
        57          9
        Do NOT use 62 (reserved for -1)
        """
        clipping_info_ord = [ord(c) for c in clipping_info]
        byte_clipping_info_ord = self.MyBIT.score2byte(clipping_info_ord)
        self.add_byte_str(byte_clipping_info_ord)
    @classmethod
    def revert_byte_clipping_info_ord(cls, byte_clipping_info_ord: bytes):
        int_clipping_info_ord = cls.MyBIT.byte2q_score(byte_clipping_info_ord)
        return "".join(chr(c) for c in int_clipping_info_ord)
    @classmethod
    def read_buffer(cls, f: io.BufferedReader):
        size = struct.unpack(cls.size_format, f.read(cls.size_indicator_bytes))
        assert len(size) == 1
        return f.read(size[0])
    @classmethod
    def read_byte(cls, b: bytes):
        p = 0
        content_list = []
        N = len(b)
        while True:
            size = struct.unpack(cls.size_format, b[p:p + cls.size_indicator_bytes])
            assert len(size) == 1
            content_list.append(b[p + cls.size_indicator_bytes:p + cls.size_indicator_bytes + size[0]])
            p += cls.size_indicator_bytes + size[0]
            if p == N:
                break
        return content_list
    @classmethod
    def read_byte_string(cls, b: bytes):
        return [content.decode("utf-8") for content in cls.read_byte(b)]
    @classmethod
    def read_byte_seq_q_scores(cls, b: bytes, format: str):
        content_list = cls.read_byte(b)
        assert len(content_list) == len(format)
        r = []
        for fmt, content in zip(format, content_list):
            if fmt == "S":
                r.append(cls.MyBIT.byte2seq(content))
            elif fmt == "Q":
                r.append(cls.MyBIT.byte2q_score(content))
            elif fmt == "C":
                r.append(cls.revert_byte_clipping_info_ord(content))
            else:
                raise Exception(f"unknown format: {fmt}")
        return r
class MyGIF():
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # list of hex color "#ffffff" or tuple
    color_cycle = [(255, 252, 245), (255, 243, 220), (110, 110, 255), (110, 255, 110), (255, 110, 110), (255, 230, 230)]
    tick_color = (200, 200, 200)
    # numbers
    class MyCharacters(object):
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
        # words
        A = np.array([
            [0,1,1,0], 
            [1,0,0,1], 
            [1,1,1,1], 
            [1,0,0,1], 
            [1,0,0,1]
        ], dtype=dtype)
        B = np.array([
            [1,1,1,0], 
            [1,0,0,1], 
            [1,1,1,0], 
            [1,0,0,1], 
            [1,1,1,0]
        ], dtype=dtype)
        C = np.array([
            [0,1,1,0], 
            [1,0,0,1], 
            [1,0,0,0], 
            [1,0,0,1], 
            [0,1,1,0]
        ], dtype=dtype)
        D = np.array([
            [1,1,1,0], 
            [1,0,0,1], 
            [1,0,0,1], 
            [1,0,0,1], 
            [1,1,1,0]
        ], dtype=dtype)
        E = np.array([
            [1,1,1,1], 
            [1,0,0,0], 
            [1,1,1,0], 
            [1,0,0,0], 
            [1,1,1,1]
        ], dtype=dtype)
        F = np.array([
            [1,1,1,1], 
            [1,0,0,0], 
            [1,1,1,0], 
            [1,0,0,0], 
            [1,0,0,0]
        ], dtype=dtype)
        G = np.array([
            [0,1,1,1], 
            [1,0,0,0], 
            [1,0,1,1], 
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
        I = np.array([
            [1,1,1], 
            [0,1,0], 
            [0,1,0], 
            [0,1,0], 
            [1,1,1]
        ], dtype=dtype)
        J = np.array([
            [0,1,1,1], 
            [0,0,1,0], 
            [0,0,1,0], 
            [1,0,1,0], 
            [0,1,0,0]
        ], dtype=dtype)
        K = np.array([
            [1,0,0,1], 
            [1,0,1,0], 
            [1,1,0,0], 
            [1,0,1,0], 
            [1,0,0,1]
        ], dtype=dtype)
        L = np.array([
            [1,0,0,0], 
            [1,0,0,0], 
            [1,0,0,0], 
            [1,0,0,0], 
            [1,1,1,1]
        ], dtype=dtype)
        M = np.array([
            [1,0,0,0,1], 
            [1,1,0,1,1], 
            [1,0,1,0,1], 
            [1,0,0,0,1], 
            [1,0,0,0,1]
        ])
        N = np.array([
            [1,0,0,0,1], 
            [1,1,0,0,1], 
            [1,0,1,0,1], 
            [1,0,0,1,1], 
            [1,0,0,0,1]
        ], dtype=dtype)
        O = np.array([
            [0,1,1,1,0], 
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [0,1,1,1,0]
        ], dtype=dtype)
        P = np.array([
            [1,1,1,0], 
            [1,0,0,1], 
            [1,1,1,0], 
            [1,0,0,0], 
            [1,0,0,0]
        ], dtype=dtype)
        Q = np.array([
            [0,1,1,1,0], 
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [1,0,0,1,1], 
            [0,1,1,1,1]
        ], dtype=dtype)
        R = np.array([
            [1,1,1,0], 
            [1,0,0,1], 
            [1,1,1,0], 
            [1,0,1,0], 
            [1,0,0,1]
        ], dtype=dtype)
        S = np.array([
            [0,1,1,1], 
            [1,0,0,0], 
            [0,1,1,0], 
            [0,0,0,1], 
            [1,1,1,0]
        ], dtype=dtype)
        T = np.array([
            [1,1,1,1,1], 
            [0,0,1,0,0], 
            [0,0,1,0,0], 
            [0,0,1,0,0], 
            [0,0,1,0,0]
        ], dtype=dtype)
        U = np.array([
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [0,1,1,1,0]
        ], dtype=dtype)
        V = np.array([
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [0,1,0,1,0], 
            [0,0,1,0,0]
        ], dtype=dtype)
        W = np.array([
            [1,0,0,0,1], 
            [1,0,0,0,1], 
            [1,0,1,0,1], 
            [1,0,1,0,1], 
            [0,1,0,1,0]
        ], dtype=dtype)
        X = np.array([
            [1,0,0,0,1], 
            [0,1,0,1,0], 
            [0,0,1,0,0], 
            [0,1,0,1,0], 
            [1,0,0,0,1]
        ], dtype=dtype)
        Y = np.array([
            [1,0,0,0,1], 
            [0,1,0,1,0], 
            [0,0,1,0,0], 
            [0,0,1,0,0], 
            [0,0,1,0,0]
        ], dtype=dtype)
        Z = np.array([
            [1,1,1,1,1], 
            [0,0,0,1,0], 
            [0,0,1,0,0], 
            [0,1,0,0,0], 
            [1,1,1,1,1]
        ], dtype=dtype)
        vs = np.array([ # vertical space
            [0], 
            [0], 
            [0], 
            [0], 
            [0], 
        ])
        us = np.array([ # under score
            [0,0,0,0], 
            [0,0,0,0], 
            [0,0,0,0], 
            [0,0,0,0], 
            [1,1,1,1]
        ])
        colon = np.array([ # vertical space
            [0, 0, 0], 
            [0, 1, 0], 
            [0, 0, 0], 
            [0, 1, 0], 
            [0, 0, 0], 
        ])
        string2key_dict = {
            "0" : "zero", 
            "1" : "one", 
            "2" : "two", 
            "3" : "three", 
            "4" : "four", 
            "5" : "five", 
            "6" : "six", 
            "7" : "seven", 
            "8" : "eight", 
            "9" : "nine", 
            " " : "vs", 
            ":" : "colon", 
        }  # string to key
        insertion = np.hstack((I,vs,N,vs,S,vs,E,vs,R,vs,T,vs,I,vs,O,vs,N))
        deletion = np.hstack((D,vs,E,vs,L,vs,E,vs,T,vs,I,vs,O,vs,N))
        not_covered = np.hstack((N,vs,O,vs,T,us,vs,C,vs,O,vs,V,vs,E,vs,R,vs,E,vs,D))
        mismatch = np.hstack((M,vs,I,vs,S,vs,M,vs,A,vs,T,vs,C,vs,H))
        skipped = np.hstack((S,vs,K,vs,I,vs,P,vs,P,vs,E,vs,D))
        match = np.hstack((M,vs,A,vs,T,vs,C,vs,H))
        @classmethod
        def get_string_array(cls, keys):
            return np.hstack([getattr(cls, cls.string2key_dict.get(k, k)) for k in " ".join(keys)])
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
    def __init__(self, N_array, tick_pos_list: list, tick_label_list: list) -> None:
        self.N_array = N_array  # shape=(6, len(ref_seq_aligned))
        self.N_reads = self.N_array[:, 0].sum()
        assert (self.N_array.sum(axis=0) == self.N_reads).all()
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
        assert self.MyCharacters.dtype == np.uint8
        self.img_array_rgb = np.ones((self.img_pixel_h, self.img_pixel_w, 3), dtype=self.MyCharacters.dtype) * 255
        self.color_cycle_rgb = self._color_cycle_rgb()
    def _color_cycle_rgb(self):
        try:
            return [tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5)) for hex_color in self.color_cycle]
        except:
            return self.color_cycle # already rgb
    def generate_bar_graph_ndarray(self):
        # リードが割り当てられていない場合に warning を回避するために必要
        if (self.N_array.sum(axis=0) == 0).any():
            self.N_array[-2, self.N_array.sum(axis=0) == 0] += 1    # 全て deletion (idx=-2) とする

        # 最終的には 0 でないものは最低限 1 ピクセルは確保できるようにするが、とりま round (銀行丸めだが、まあいいや) する
        N_array_compositional = np.round(self.N_array / self.N_array.sum(axis=0) * self.bar_sum_h).astype(int)
        rounding_error = np.ones(N_array_compositional.shape[1], dtype=int) * self.bar_sum_h - N_array_compositional.sum(axis=0)
        # 誤差がもっとも少なくなるように、マジョリティなグループでバッファーする
        for c, e in enumerate(rounding_error):
            # 0 でないものは最低限 1 ピクセルは確保
            for r, v in enumerate(self.N_array[:, c]):
                if (v > 0) and (N_array_compositional[r, c] == 0):
                    N_array_compositional[r, c] = 1
                    e -= 1
            # マジョリティなグループでバッファーする
            if e == 0:
                continue
            else:
                e_len = np.absolute(e)
                e_unit = e // e_len
                composition = np.copy(N_array_compositional[:, c]).astype(float)
                denominator = np.ones_like(composition, dtype=float)
                while e_len > 0:
                    r = np.argmax(composition)
                    N_array_compositional[r, c] += e_unit
                    # アプデ
                    composition[r] *= denominator[r] / (denominator[r] + 1)
                    denominator[r] += 1
                    e_len -= 1
        assert (np.ones(N_array_compositional.shape[1], dtype=int) * self.bar_sum_h == N_array_compositional.sum(axis=0)).all()
        # 画像に追加していく
        ax_loc = [0, 0]
        bar_pos_x = 0
        ax_origin = self.get_ax_origin(ax_loc)
        for idx, array in enumerate(N_array_compositional.T):
            bar_pos_y = 0
            for c_cycle, bar_height in enumerate(array):
                self.draw_bar(bar_pos_x, bar_pos_y, bar_pos_y + bar_height, ax_origin, self.color_cycle_rgb[c_cycle])
                bar_pos_y += bar_height # （画像左上の座標が [0, 0]、ただし ax の内部では左下が原点）
            if idx in self.tick_pos_list: # 塩基は1スタートだが、tick_pos_list に登録されているものは 0 スタート
                tick_idx = self.tick_pos_list.index(idx)
                self.draw_tick(bar_pos_x=bar_pos_x, tick_label=self.tick_label_list[tick_idx], ax_origin=ax_origin) # 下から積み上げていく
            bar_pos_x += 1
            # N_array_compositional.T.shape[0] が、行の最後とぴったり一致して終了した場合は、改行しない（改行しても次の行に入力するものがないし、index out of range error になる）
            if idx == N_array_compositional.shape[1] - 1:
                break
            # 後の idx 処理
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
            loc_y += self.MyCharacters.letter_h + 3
            img_box_rgb = np.expand_dims(np.ones((self.MyCharacters.letter_h, self.MyCharacters.letter_h), dtype=self.MyCharacters.dtype), axis=-1) * np.array(color)
            self.fill_img(loc_x, loc_y, img_box_rgb)
            loc_x_new = loc_x + self.MyCharacters.letter_h + 5
            img_rgb = np.expand_dims(255 - getattr(self.MyCharacters, legend) * 255, axis=-1) * np.ones(3, dtype=self.MyCharacters.dtype)
            self.fill_img(loc_x_new, loc_y, img_rgb)
    def set_legend_description(self, description):
            loc_x = self.l_margin * 4
            loc_y = self.minimum_margin + self.MyCharacters.letter_h + 3
            img_rgb = np.expand_dims(255 - self.MyCharacters.get_string_array(description) * 255, axis=-1) * np.ones(3, dtype=self.MyCharacters.dtype)
            self.fill_img(loc_x, loc_y, img_rgb)
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
        loc_x = ax_origin[1] + bar_pos_x * self.bar_w - self.MyCharacters.number_w // 2
        for i, l in enumerate(str(tick_label)):
            loc_y = ax_origin[0] - self.bar_sum_h - self.tick_h - self.MyCharacters.letter_h - 1
            img_rgb = np.expand_dims(255 - getattr(self.MyCharacters, self.MyCharacters.string2key_dict[l]) * 255, axis=-1) * np.ones(3, dtype=self.MyCharacters.dtype)
            self.fill_img(loc_x, loc_y, img_rgb)
            loc_x += self.MyCharacters.number_w + 1
    def export_as_img(self, save_path):
        PilImage.fromarray(self.img_array_rgb).save(save_path)



