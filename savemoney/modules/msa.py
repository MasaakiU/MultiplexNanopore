# -*- coding: utf-8 -*-

import re
import io
import struct
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from PIL import Image as PilImage
from tqdm import tqdm
from pathlib import Path
from itertools import product
from collections import defaultdict
from numpy.core.memmap import uint8
from matplotlib import rc
from matplotlib.patches import Patch

from . import my_classes as mc
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
        fig_height_unit = (fig_h_unit + hspace_unit) * (rows - 1) + top_margin_unit + bottom_margin_unit
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
            plt.savefig(Path(save_dir) / f"{self.my_fastq.combined_name_stem}.summary_scatter.svg")
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
        top_margin_unit = 0.0
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
            plt.savefig(Path(save_dir) / f"{self.my_fastq.combined_name_stem}.summary_distribution.svg")
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
        top_header_unit = 1.0
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
                np.set_printoptions(linewidth=200, threshold=np.inf)
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
    def __init__(self, ref_seq, my_fastq_subset, result_list) -> None:
        self.ref_seq = ref_seq
        self.my_fastq_subset = my_fastq_subset
        self.result_list = result_list
        assert len(self.my_fastq_subset) == len(self.result_list)
    def execute(self, param_dict):
        # set offset for queries
        query_seq_list = []
        q_scores_list = []
        my_cigar_list = []
        for result, (query_seq, q_scores) in zip(self.result_list, self.my_fastq_subset.values()):
            query_seq_list.append(mc.MySeq.set_offset_core(query_seq, result.new_query_seq_offset))
            q_scores_list.append(mc.MySeq.set_offset_core(list(q_scores), result.new_query_seq_offset))
            my_cigar_list.append(result.my_cigar)

        # 1st MSA
        my_msa = MyMSA.generate_msa(self.ref_seq, query_seq_list, q_scores_list, my_cigar_list)
        my_msa.ref_seq_name = self.ref_seq.path.name
        my_msa.query_id_list = list(self.my_fastq_subset.keys())
        # my_msa.print_alignment()
        my_msa.calculate_consensus(param_dict)
        return my_msa

class SequenceBasecallQscorePDF():
    def __init__(self, df_csv_path=None) -> None:
        with open(df_csv_path, "r") as f:
            self.NanoporeStats_PDF_version = re.match(r"^# file_version: ([0-9]+\.[0-9]+\.[0-9]+)$", f.readline().strip()).group(1)
        self.df_stats = pd.read_csv(df_csv_path, sep="\t", header=1, index_col=0)

        # initialize
        self.pdf_core = {}
        self.P_base_calling_given_true_refseq_dict = {}
        self.initialize_pdf()
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

class MyMSA(mc.MyCigarBase):
    file_format_version = "ff_0.1.0"
    algorithm_version = "al_0.1.0"
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
    ]
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
    sbq_pdf_version="pdf_0.2.0"
    sbq_pdf = SequenceBasecallQscorePDF(df_csv_path=Path(__file__).parent / f"NanoporeStats_PDF/NanoporeStats_{sbq_pdf_version}.csv")
    default_print_options = {
        "center": 2000, 
        "seq_range": 50, 
        "offset": 0
    }
    def __init__(self, ref_seq_aligned=None, query_seq_list_aligned=None, q_scores_list_aligned=None, my_cigar_list_aligned=None) -> None:
        self.ref_seq_name = None
        self.query_id_list = None
        self.ref_seq_aligned = ref_seq_aligned                  # ref_related
        self.query_seq_list_aligned = query_seq_list_aligned    # query_related
        self.q_scores_list_aligned = q_scores_list_aligned      # query_related
        self.my_cigar_list_aligned = my_cigar_list_aligned      # query_related
        self.with_prior_consensus_seq = ""          # consensus_related
        self.with_prior_consensus_q_scores = []     # consensus_related
        self.with_prior_consensus_my_cigar = ""       # consensus_related
        self.without_prior_consensus_seq = ""       # consensus_related
        self.without_prior_consensus_q_scores = []  # consensus_related
        self.without_prior_consensus_my_cigar = ""    # consensus_related
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
    @staticmethod
    def generate_msa(ref_seq: mc.MyRefSeq, query_seq_list: List[str], q_scores_list: List[int], my_cigar_list: List[str]):
        # 最後が必ず同時に終わるように、特殊シーケンスを追加 (こうしないと、query_seq_list のどれか一つのの最後が "I" である場合に正常終了しない)
        assert len(query_seq_list) == len(q_scores_list) == len(my_cigar_list)
        ref_seq += "Z"
        query_seq_list = [query_seq + "Z" for query_seq in query_seq_list]
        q_scores_list = [q_scores + [-2] for q_scores in q_scores_list]
        my_cigar_list = [my_cigar + "Z" for my_cigar in my_cigar_list]

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
            if "I" not in L_list:
                R = ref_seq[ref_seq_idx]
                if R == "Z":    # 全て終了していることを確認する
                    assert [query_seq_list[i][query_seq_idx_list[i]] == "Z" for i in range(len(L_list))]
                    assert [q_scores_list[i][query_seq_idx_list[i]] == -2 for i in range(len(L_list))]
                    assert all(L == "Z" for L in L_list)
                    break
                for i, L in enumerate(L_list):
                    """
                    ハード/ソフトクリッピング はない (環状 DNA で読めてない部分は deletion として扱う)
                    行うとしたら ref_query_alignmeent.MyOptimizedAligner.execute_circular_alignment_using_conserved_regions だけど、ちょっとだけ面倒だし、そもそもハードクリッピングを行うのが正しいのかもわからない
                    シーケンスがうまく行ってれば、ハード/ソフトクリッピングを行わなくても出てくる影響はごくわずかだし、うまく行ってないなら行ってないで、それがきちんと示されることが大切かなと。
                    """
                    my_cigar_list_aligned[i] += L
                    my_cigar_idx_list[i]     += 1
                    if L in "D":
                        query_seq_list_aligned[i] += "-"
                        q_scores_list_aligned[i]  += [-1]
                    elif L == "=":
                        query_seq_list_aligned[i] += query_seq_list[i][query_seq_idx_list[i]]
                        q_scores_list_aligned[i]  += [q_scores_list[i][query_seq_idx_list[i]]]
                        query_seq_idx_list[i]     += 1
                    elif L == "X":
                        query_seq_list_aligned[i] += query_seq_list[i][query_seq_idx_list[i]]
                        q_scores_list_aligned[i]  += [q_scores_list[i][query_seq_idx_list[i]]]
                        query_seq_idx_list[i]     += 1
                    else:
                        print(L)
                        raise Exception("error!")
                ref_seq_aligned += R
                ref_seq_idx += 1
            else:
                for i, L in enumerate(L_list):
                    if L == "I":
                        my_cigar_list_aligned[i]  += "I"
                        query_seq_list_aligned[i] += query_seq_list[i][query_seq_idx_list[i]]
                        q_scores_list_aligned[i]  += [q_scores_list[i][query_seq_idx_list[i]]]
                        my_cigar_idx_list[i]      += 1
                        query_seq_idx_list[i]     += 1
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
        return MyMSA(ref_seq_aligned, query_seq_list_aligned, q_scores_list_aligned, my_cigar_list_aligned)
    def print_alignment(self, print_options={}):
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
        position_string = " " * (10 - (cur_pos + 1)%10)
        assert self.ref_seq_aligned[s_idx] != "-"
        for ref in self.ref_seq_aligned[s_idx:e_idx+1]:
            if cur_pos%10 != 9:
                pass
            else:
                position_string += f"{cur_pos + 1:<10}"
            if ref != "-":
                cur_pos += 1
            else:
                position_string += " "
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
            f"{self.ref_seq_aligned[s_idx:c_idx]}\033[38;2;255;0;0m{self.ref_seq_aligned[c_idx]}\033[39m{self.ref_seq_aligned[c_idx+1:e_idx+1]}"
            f"\033[0m     1{' '*8}10{' '*8}20{' '*8}30{' '*8}40{' '*8}50"
        )
        print(
            f"{header_consensus_with_prior:<{header_width}}"
            f"{self.q_score_color_string(self.with_prior_consensus_seq[s_idx:e_idx+1], self.with_prior_consensus_q_scores[s_idx:e_idx+1], q_score_color_dict, c_idx - s_idx)}\033[0m"
            f"{self.q_score_bar(self.with_prior_consensus_q_scores[c_idx], self.with_prior_consensus_my_cigar[c_idx], q_score_color_dict, show_asterisk=True)}\033[0m"
        )
        print(
            f"{header_consensus_without_prior:<{header_width}}"
            f"{self.q_score_color_string(self.without_prior_consensus_seq[s_idx:e_idx+1], self.without_prior_consensus_q_scores[s_idx:e_idx+1], q_score_color_dict, c_idx - s_idx)}\033[0m"
            f"{self.q_score_bar(self.without_prior_consensus_q_scores[c_idx], self.without_prior_consensus_my_cigar[c_idx], q_score_color_dict, show_asterisk=True)}\033[0m"
        )
        for query_id, query_seq, q_scores, my_cigar in zip(self.query_id_list, self.query_seq_list_aligned, self.q_scores_list_aligned, self.my_cigar_list_aligned):
            # print(f"{query_id:<{header_width}}{query_seq[s_idx:e_idx+1]}")
            print(
                f"{query_id:<{header_width}}{self.my_cigar_color_string(query_seq[s_idx:e_idx+1], my_cigar[s_idx:e_idx+1], c_idx - s_idx)}"
                f"{self.q_score_bar(q_scores[c_idx], my_cigar[c_idx], q_score_color_dict, show_asterisk=False)}\033[0m"
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
    def q_score_color_string(seq, q_scores, q_score_color_dict: defaultdict, make_it_bold: int):
        stdout_txt = ""
        for i, (s, q) in enumerate(zip([s for s in seq], q_scores)):
            L = f"\033[48;2;{q_score_color_dict[q]}m{s}"
            if i == make_it_bold:
                L = f"\033[1m{L}\033[0m"
            stdout_txt += L
        return stdout_txt
    @staticmethod
    def my_cigar_color_string(seq, my_cigar, make_it_bold: int):
        stdout_txt = ""
        for i, (s, c) in enumerate(zip([s for s in seq], [c for c in my_cigar])):
            if c in "=N":
                L = f"{s}"
            else:
                L = f"\033[48;2;255;176;176m{s}\033[0m"
            if i == make_it_bold:
                L = f"\033[1m{L}\033[0m"
            stdout_txt += L
        return stdout_txt
    @staticmethod
    def q_score_bar(q_score, my_cigar, q_score_color_dict: defaultdict, show_asterisk):
        color = q_score_color_dict[q_score]
        if (my_cigar == "=") or (my_cigar == "N"):
            q_score_txt = f"{q_score:<3}"
        else:
            q_score_txt = f"\033[48;2;255;176;176m{q_score:<2}\033[0m "
        if not show_asterisk:
            return f"  {q_score_txt}\033[48;2;{color}m\033[38;2;{color}m{'*' * q_score}"
        else:
            return f"  {q_score_txt}\033[48;2;{color}m{'*' * q_score}"
    def calculate_consensus(self, param_dict):
        # params
        P_N_dict_dict_with_prior, P_N_dict_dict_without_prior = self.consensus_params(param_dict)
        # execute
        self.with_prior_consensus_seq, self.with_prior_consensus_q_scores, self.with_prior_consensus_my_cigar = self.calculate_consensus_core(P_N_dict_dict_with_prior)
        self.without_prior_consensus_seq, self.without_prior_consensus_q_scores, self.without_prior_consensus_my_cigar = self.calculate_consensus_core(P_N_dict_dict_without_prior)
    def calculate_consensus_core(self, P_N_dict_dict):
        consensus_seq = ""
        consensus_q_scores = []
        consensus_my_cigar = ""
        for consensus_idx, ref in tqdm(enumerate(self.ref_seq_aligned), ncols=100, mininterval=0.05, leave=True, bar_format='{l_bar}{bar}{r_bar}', total=len(self.ref_seq_aligned)):
            query_list = [i[consensus_idx] for i in self.query_seq_list_aligned]
            q_score_list = [i[consensus_idx] for i in self.q_scores_list_aligned]
            L_list = [i[consensus_idx] for i in self.my_cigar_list_aligned]
            event_list = [(i.upper(), j) for i, j, k in zip(query_list, q_score_list, L_list)]

            if len(event_list) > 0:
                P_N_dict = P_N_dict_dict[ref.upper()]
                p_list = [
                    self.sbq_pdf.calc_consensus_error_rate(event_list, true_refseq=B, P_N_dict=P_N_dict, bases=self.bases)
                    for B in self.bases
                ]
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
    def consensus_params(cls, param_dict):
        ins_rate = param_dict["ins_rate"]
        error_rate = param_dict["error_rate"]
        del_mut_rate = param_dict["del_mut_rate"]
        default_value_with_prior = {b_key2:ins_rate / 4 if b_key2 != "-" else 1 - ins_rate for b_key2 in cls.bases}

        P_N_dict_dict_with_prior = defaultdict(
            lambda: default_value_with_prior, 
            {   # 真のベースが b_key1 である場合に、b_key2 への mutation/deletion などが起こる確率
                b_key1:{b_key2:1 - error_rate if b_key2 == b_key1 else del_mut_rate for b_key2 in cls.bases} for b_key1 in cls.bases[:-1]  # remove "-" from b_key1
            }
        )
        P_N_dict_dict_with_prior["-"] = default_value_with_prior

        default_value_without_prior = {b_key2:0.2 / 4 if b_key2 != "-" else 0.8 for b_key2 in cls.bases}
        P_N_dict_dict_without_prior = defaultdict(
            lambda: default_value_without_prior, 
            {
                b_key1:{b_key2: 0.2 for b_key2 in cls.bases} for b_key1 in cls.bases[::-1]
            }
        )
        P_N_dict_dict_without_prior["-"] = default_value_without_prior

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
        P_N_dict_dict_with_prior, P_N_dict_dict_without_prior = cls.consensus_params(param_dict)
        def gen_matrix_from_dict_dict(P_N_dict_dict):
            r_matrix = np.empty((len(cls.bases), len(cls.bases)), dtype=float)
            for r, b_key1 in enumerate(cls.bases):
                for c, b_key2 in enumerate(cls.bases):
                    r_matrix[r, c] = P_N_dict_dict[b_key1][b_key2]
            return r_matrix
        return gen_matrix_from_dict_dict(P_N_dict_dict_with_prior), gen_matrix_from_dict_dict(P_N_dict_dict_without_prior)
    #######################
    # EXPORT/LOAD RESULTS #
    #######################
    def export_consensus_fastq(self, save_dir, ref_seq_name: str):
        # with prior
        save_path = save_dir / f"{ref_seq_name}.consensus_with_prior.fastq"
        consensus_q_score_string = "".join([chr(q + 33) for q in self.with_prior_consensus_q_scores_NoDEL])
        consensus_fastq_txt = f"@{ref_seq_name}.with_prior:\n{self.with_prior_consensus_seq_NoDEL.upper()}\n+\n{consensus_q_score_string}"
        with open(save_path, "w") as f:
            f.write(consensus_fastq_txt)
        # without prior
        save_path = save_dir / f"{ref_seq_name}.consensus_without_prior.fastq"
        consensus_q_score_string = "".join([chr(q + 33) for q in self.without_prior_consensus_q_scores_NoDEL])
        consensus_fastq_txt = f"@{ref_seq_name}.without_prior:\n{self.without_prior_consensus_seq_NoDEL.upper()}\n+\n{consensus_q_score_string}"
        with open(save_path, "w") as f:
            f.write(consensus_fastq_txt)
    def export_gif(self, save_dir, ref_seq_name: str):
        save_path = save_dir / f"{ref_seq_name}.gif"
        # prepare
        N_array = np.empty((5, len(self.ref_seq_aligned)), int)
        tick_pos_list = []
        tick_label_list = []
        cur_ref_pos = 0
        for ref_idx, ref in enumerate(self.ref_seq_aligned):

            if ref != "-":
                cur_ref_pos += 1
                if cur_ref_pos%100 == 0:
                    tick_pos_list.append(ref_idx)
                    tick_label_list.append(cur_ref_pos)

            N_match = 0     # =
            N_mismatch = 0  # X
            N_insertion = 0 # I
            N_deletion = 0  # D
            N_omitted = 0   # N, H, S
            for my_cigar in self.my_cigar_list_aligned:
                L = my_cigar[ref_idx]
                if L == "=":    N_match += 1
                elif L == "X":  N_mismatch += 1
                elif L == "I":  N_insertion += 1
                elif L == "D":  N_deletion += 1
                elif L in "NHS":    N_omitted += 1
                else:   raise Exception(f"unknown cigar string {L}")
            N_array[:, ref_idx] = [N_match, N_omitted, N_mismatch, N_insertion, N_deletion]
        # 描画していく！
        bar_graph_img = MyGIF(N_array, tick_pos_list, tick_label_list)
        bar_graph_img.generate_bar_graph_ndarray()
        bar_graph_img.set_legend(legend_list=["match", "omitted", "mismatch", "insertion", "deletion"])
        bar_graph_img.set_legend_description(f"TOTAL READS: {len(self.query_id_list)}")
        bar_graph_img.export_as_img(save_path=save_path)
    def export_consensus_alignment(self, save_dir: Path, ref_seq_name: str):
        save_path = save_dir / f"{ref_seq_name}.ca"
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
        self.ref_seq_name = header_contents[1]
        self.query_id_list = header_contents[2:]
        N_query = len(self.query_id_list)

        # ref/consensus related
        format_txt = "".join(re.match(r"^add_(?=sequence|q_scores)(s|q).+$", func_name).group(1).upper() for func_name, attr_name in self.ref_seq_related_save_order)
        ref_related_contents = MyByteStr.read_byte_seq_q_scores(ref_related, format_txt)
        for (func_name, attr_name), content in zip(self.ref_seq_related_save_order, ref_related_contents):
            setattr(self, attr_name, content)

        # query/fastq related
        format_txt = "".join(re.match(r"^add_(?=sequence|q_scores)(s|q).+$", func_name).group(1).upper() * N_query for func_name, attr_name in self.query_seq_related_save_order)
        query_related_contents = MyByteStr.read_byte_seq_q_scores(query_related, format_txt)
        for i, (func_name, attr_name) in enumerate(self.query_seq_related_save_order):
            setattr(self, attr_name, query_related_contents[i * N_query: (i + 1) * N_query])

        # cigar セット
        self.regenerate_cigar()
    def regenerate_cigar(self):
        assert all(len(query_seq) == len(self.ref_seq_aligned) for query_seq in self.query_seq_list_aligned) and (len(self.with_prior_consensus_seq) == len(self.without_prior_consensus_seq) == len(self.ref_seq_aligned))
        def gen_cigar(ref, query):
            if query == ref != "-":
                return "="
            elif query != ref == "-":
                return "I"
            elif "-" == query != ref:
                return "D"
            elif "-" != query != ref != "-":
                return "X"
            elif query == ref == "-":
                return "N"
            else:
                raise Exception(f"error: {ref} {query}")
        self.my_cigar_list_aligned = ["" for query_id in self.query_id_list]
        self.with_prior_consensus_my_cigar = ""
        self.without_prior_consensus_my_cigar = ""
        for query_idx, query_seq in enumerate(self.query_seq_list_aligned):
            for ref, query in zip(self.ref_seq_aligned, query_seq):
                self.my_cigar_list_aligned[query_idx] += gen_cigar(ref, query)
        for ref, consensus in zip(self.ref_seq_aligned, self.with_prior_consensus_seq):
            self.with_prior_consensus_my_cigar += gen_cigar(ref, consensus)
        for ref, consensus in zip(self.ref_seq_aligned, self.without_prior_consensus_seq):
            self.without_prior_consensus_my_cigar += gen_cigar(ref, consensus)
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

class MyByteStr():
    class MyBIT():
        my_bit = 6             # 一文字 6 ビットで保存する
        struct_bit = 24     
        seq_offset = 64
        edian = ">"
        @classmethod
        @property
        def struct_N(cls):
            return cls.struct_bit // cls.my_bit
        @classmethod
        @property
        def struct_byte(cls):
            return cls.struct_bit // 8
        @classmethod
        @property
        def my_bit_struct_bitshift(cls):
            return np.arange(cls.struct_N)[::-1] * cls.my_bit
        @classmethod
        @property
        def byte_struct_bitshift(cls):
            return np.arange(cls.struct_byte)[::-1] * 8
        @classmethod
        @property
        def byte_bitmask(cls):
            return np.hstack([255 for i in range(cls.struct_byte)]) # 2 ** 8 -1
        @classmethod
        @property
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
            """
            # 文字を数値に変換
            seq_64 = [ord(s) - cls.seq_offset for s in seq.replace("-", "~")]
            return cls.value2byte_core(seq_64)
        @classmethod
        def q_score2byte(cls, seq_64: list):
            """
            value -> seq_64
            -1 -> 62
            """
            return cls.value2byte_core([s if s != -1 else 62 for s in seq_64])
        @classmethod
        def value2byte_core(cls, seq_64: list):
            """
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
            return seq_64.replace(chr(127), "").replace("~", "-")   # バッファー用の削除文字を削除, "~" を置換
        @classmethod
        def byte2q_score(cls, b: bytes):
            q_scores = cls.byte2value_core(b)
            while q_scores[-1] == 63:
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
    def add_byte_str(self, byte_str):
        self.my_byte_str += struct.pack(self.size_format, len(byte_str)) + byte_str
    def add_sequence(self, seq):
        byte_seq = self.MyBIT.seq2byte(seq)
        self.add_byte_str(byte_seq)
    def add_q_scores(self, q_scores):
        byte_q_scores = self.MyBIT.q_score2byte(q_scores)
        self.add_byte_str(byte_q_scores)
    def __str__(self) -> str:
        return str(self.my_byte_str)
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
        return r
class MyGIF():
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # list of hex color "#ffffff" or tuple
    color_cycle = [(255, 252, 245), (255, 243, 220), (110, 110, 255), (110, 255, 110), (255, 110, 110)]
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
        mismatch = np.hstack((M,vs,I,vs,S,vs,M,vs,A,vs,T,vs,C,vs,H))
        omitted = np.hstack((O,vs,M,vs,I,vs,T,vs,T,vs,E,vs,D))
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
        assert self.MyCharacters.dtype == np.uint8
        self.img_array_rgb = np.ones((self.img_pixel_h, self.img_pixel_w, 3), dtype=self.MyCharacters.dtype) * 255
        self.color_cycle_rgb = self._color_cycle_rgb()
    def _color_cycle_rgb(self):
        try:
            return [tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5)) for hex_color in self.color_cycle]
        except:
            return self.color_cycle # already rgb
    def generate_bar_graph_ndarray(self):

        # if (self.N_array.sum(axis=0) == 0).any():
        #     self.N_array[-1, self.N_array.sum(axis=0) == 0] += 1

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



