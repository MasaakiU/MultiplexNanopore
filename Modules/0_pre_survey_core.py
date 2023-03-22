# -*- coding: utf-8 -*-

"""
0. Pre-survey
"""

"""
dependencies
"""
import sys, os
import re
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import numpy as np
import itertools
import gc
import parasail
import xml.etree.ElementTree as ET
import cairo
from . import my_classes as mc
from pathlib import Path
from snapgene_reader import snapgene_file_to_dict, snapgene_file_to_seqrecord
from scipy.cluster.hierarchy import linkage
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

"""
functions
"""
def pre_survery(refseq_list, param_dict):
    N = len(refseq_list)
    total_N = N ** 2
    i = 0
    score_matrix = np.empty((N, N), dtype=float)
    for r, my_refseq in enumerate(refseq_list):
        for c, query in enumerate(refseq_list):
            print(f"\rProcessing... {i+1}/{total_N}", end="")
            i += 1
            if r != c:
                duplicated_refseq_seq = my_refseq.seq + my_refseq.seq
                score_matrix[r, c] = calc_corrected_alignment_score(duplicated_refseq_seq, query.seq, param_dict)
            else:
                score_matrix[r, c] = 1
    print()
    return score_matrix

def calc_corrected_alignment_score(duplicated_refseq_seq, query_seq, param_dict):
    gap_open_penalty = param_dict["gap_open_penalty"]
    gap_extend_penalty = param_dict["gap_extend_penalty"]
    match_score = param_dict["match_score"]
    mismatch_score = param_dict["mismatch_score"]
    my_custom_matrix =parasail.matrix_create("ACGT", match_score, mismatch_score)
    result = parasail.sw_trace(query_seq, duplicated_refseq_seq, gap_open_penalty, gap_extend_penalty, my_custom_matrix)
    result = MyResult_Minimum(result)
    gc.collect()
    return result.score / (len(duplicated_refseq_seq) / 2)

def recommended_combination(score_matrix, score_threshold):

    distance_matrix = 1 - score_matrix
    N = len(distance_matrix)
    for i in range(N - 1):
        for j in range(i + 1, N):
            v = min(distance_matrix[i, j], distance_matrix[j, i])
            distance_matrix[i, j] = v
            distance_matrix[j, i] = v

    # draw_heatmap_core(a, x_labels=np.arange(score_matrix.shape[1]), y_labels=np.arange(score_matrix.shape[1]))
    # plt.show()

    # clustering by sequence similarity (similar sequences will be grouped)
    dArray = distance.squareform(distance_matrix)
    result = linkage(dArray, method='complete')
    # np.set_printoptions(suppress=True)
    # print(result)
    # distance_matrix_sorted, dendro_levels = draw_dendrogram_core(distance_matrix, result)
    # plt.show()
    # quit()

    # organize result
    threshold = 1 - score_threshold
    grouping = [[i] for i in range(N)]
    for idx1, idx2, d, number_of_sub_cluster in result:
        if d > threshold:
            break
        grouping.append(grouping[int(idx1)] + grouping[int(idx2)])
        grouping[int(idx1)] = None
        grouping[int(idx2)] = None
    grouping = [g for g in grouping if g is not None]

    print(grouping)

    # distance_matrix_sorted, dendro_levels = draw_dendrogram_core(distance_matrix, result)
    # plt.show()

    score_matrix_tmp = np.copy(score_matrix)
    def calc_group_score(index_list):
        combination_of_index = list(itertools.combinations(index_list, 2))
        scores = score_matrix_tmp[tuple(zip(*combination_of_index))]
        return scores.max() * len(scores)

    # 似た配列がコンビにならないように、組み合わせを選出（グループ：似た者同士の集合、コンビ：違う者同士の集合）
    N_combination = max(len(g) for g in grouping)
    combination_list = [[] for i in range(N_combination)]
    # 大きいグループから処理していく
    for c in range(N_combination, 0, -1):
        # 指定の長さのグループを抽出、None を追加して長さを len(combination_list) に合わせる
        selected_groups = [g + [None for i in range(N_combination - c)] for g in grouping if len(g) == c]
        # どのような組み合わせでコンビに追加するかを全通り書き出す
        selected_group_permuation = [list(set(itertools.permutations(g, N_combination))) for g in selected_groups]
        product_of_selected_group_permutation = list(itertools.product(*selected_group_permuation))
        # スコアの平均を計算
        average_score_list = []
        for prod in product_of_selected_group_permutation:
            scores = [calc_group_score(combination_list[i] + [p_sub for p_sub in p if p_sub is not None]) for i, p in enumerate(zip(*prod))]
            if len(scores):
                average_score_list.append(np.average(scores))
            else:
                average_score_list.append(np.nan)
        selected_prod = product_of_selected_group_permutation[np.argmin(average_score_list)]
        for i, p in enumerate(zip(*selected_prod)):
            combination_list[i].extend([p_sub for p_sub in p if p_sub is not None])
    print(combination_list)
    return combination_list

class MyRefSeq_Minimum():
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

class MyResult_Minimum():
    def __init__(self, parasail_result) -> None:
        self.cigar = parasail_result.cigar.decode.decode("ascii")
        self.score = parasail_result.score
        self.beg_ref = parasail_result.cigar.beg_ref
        self.beg_query = parasail_result.cigar.beg_query
        self.end_ref = parasail_result.end_ref
        self.end_query = parasail_result.end_query

class StringSizeWithSuffix():
    def __init__(self, size_with_suffix):
        if isinstance(size_with_suffix, str):
            m = re.match(r"([0-9.]+)([a-z]+)", size_with_suffix)
            self.size = float(m.group(1))
            self.suffix = m.group(2)
        elif isinstance(size_with_suffix, list):
            self.size = size_with_suffix[0]
            self.suffix = size_with_suffix[1]
        else:
            raise Exception("error!")
    def __add__(self, v):
        return StringSizeWithSuffix([self.size + v, self.suffix])
    def __sub__(self, v):
        return StringSizeWithSuffix([self.size - v, self.suffix])
    def __mul__(self, v):
        return StringSizeWithSuffix([self.size * v, self.suffix])
    def __truediv__(self, v):
        return StringSizeWithSuffix([self.size / v, self.suffix])
    def __floordiv__(self, v):
        return StringSizeWithSuffix([self.size // v, self.suffix])
    def __iadd__(self, v):
        self.size += v
        return self
    def __isub__(self, v):
        self.size -= v
        return self
    def __imul__(self, v):
        self.size *= v
        return self
    def __itruediv__(self, v):
        self.size /= v
        return self
    def __ifloordiv__(self, v):
        self.size //= v
        return self
    def __str__(self):
        return f"{self.size}{self.suffix}"

class Svg():
    def __init__(self, path):
        ET.register_namespace("","http://www.w3.org/2000/svg")
        tree = ET.parse(path)
        self.svg = tree.getroot()   # svg
    def adjust_margin(self, path, l, r, t, b):  # l, r, t, b represents margins to add to the viewbox of svg file.
        self.svg.attrib["width"] = str(StringSizeWithSuffix(self.svg.attrib["width"]) + l + r)
        self.svg.attrib["height"] = str(StringSizeWithSuffix(self.svg.attrib["height"]) + t + b)
        view_box = list(map(float, self.svg.attrib["viewBox"].split(" ")))
        view_box[0] -= l    # x0
        view_box[1] -= t    # y0
        view_box[2] += l + r    # x1
        view_box[3] += t + b    # y1
        self.svg.attrib["viewBox"] = " ".join(map(str, view_box))
    def draw_path(self, d, stroke="#000000", stroke_width=2, stroke_linecap="butt", stroke_linejoin="miter", stroke_opacity=1, stroke_miterlimit=4, stroke_dasharray="none"):
        self.svg.append(ET.Element(
            'path', 
            attrib={
                "style" :f"fill:none;stroke:{stroke};stroke-width:{stroke_width};stroke-linecap:{stroke_linecap};stroke-linejoin:{stroke_linejoin};stroke-opacity:{stroke_opacity};stroke-miterlimit:{stroke_miterlimit};stroke-dasharray:{stroke_dasharray}", 
                "d"     :d
            }
        ))
    def draw_text(self, string, x, y, font_style="normal", font_weight="normal", font_size="30px", line_height=1.25, font_family="sans-serif", fill="#000000", fill_opacity=1, stroke="none", stroke_width=0.75, text_anchor="middle", text_align="center"):
        text = ET.Element(
            "text", 
            attrib={
                "xml:space" :"preserve", 
                "style"     :f"font-style:{font_style};font-weight:{font_weight};font-size:{font_size};line-height:{line_height};font-family:{font_family};fill:{fill};fill-opacity:{fill_opacity};stroke:{stroke};stroke-width:{stroke_width};text-anchor:{text_anchor};text-align:{text_align}", 
                "x"         :str(x), 
                "y"         :str(y), 
            }
        )
        text.text = string
        self.svg.append(text)
    @staticmethod
    def textsize(text, fontsize, font_style):
        tmp_svg_path = "undefined.svg"
        surface = cairo.SVGSurface(tmp_svg_path, 1280, 200)
        cr = cairo.Context(surface)
        cr.select_font_face(font_style, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(fontsize)
        xbearing, ybearing, width, height, xadvance, yadvance = cr.text_extents(text)
        os.remove(tmp_svg_path)
        return {
            "xbearing":xbearing, 
            "ybearing":ybearing, 
            "width":width, 
            "height":height, 
            "xadvance":xadvance, 
            "yadvance":yadvance
        }
    def save(self, path):   # 上書き保存されます
        tree = ET.ElementTree(element=self.svg)
        tree.write(path, encoding='utf-8', xml_declaration=True)

class D(str):
    def __new__(cls, path_command_list=[]):
        initial_string = " ".join([f"{path_command} {','.join(map(str, values))}" for path_command, values in path_command_list])
        self = super().__new__(cls, initial_string)
        return self
    def append(self, path_command, values):
        # example of horizontal line: "M 6.7760638,-8.370432 H 169.80019"
        self += f"{path_command} {','.join(map(str, values))}"

def draw_heatmap(score_matrix, refseq_names, comb_idx_list, threshold_used, save_path, tmp_names=None):
    # label
    font_style = "Helvetica"
    if tmp_names is None:
        tmp_names = [f"P{i+1}" for i in range(len(refseq_names))]
    comb_txt_list = [f"threshold={threshold_used}"] + [', '.join([tmp_names[i] for i in comb_idx_list])]
    details_list = [f"{i: <4}: {j}" for i, j in zip(tmp_names, refseq_names)] + [""] + comb_txt_list

    # size
    dpi = 72
    value_font_size=10
    tick_font_size=12
    label_font_size = 14
    details_font_size = 8
    left_top_margin = (tick_font_size + label_font_size) * 2
    bottom_margin = details_font_size * (len(details_list) + 1)
    details_width = max([Svg.textsize(details, details_font_size, font_style)["xadvance"] for details in details_list])

    # make matplotlib fig
    fig, ax = draw_heatmap_core(score_matrix, x_labels=tmp_names, y_labels=tmp_names, value_font_size=value_font_size, tick_font_size=tick_font_size)

    # highlight combination
    for i in comb_idx_list:
        for j in comb_idx_list:
            if i == j:
                continue
            else:
                highlight_cell(i, j, ax=ax, color="r")

    # add titles etc.
    ax.set_xlabel("query", fontsize=label_font_size, labelpad=10)
    ax.set_ylabel("reference", fontsize=label_font_size)
    plt.savefig(save_path, dpi=dpi)

    # adjust saved svg
    svg = Svg(save_path)
    x0, y0, x1, y1 = svg.svg.attrib["viewBox"].split(" ")
    assert float(x0) == float(y0) == 0
    for i, details in enumerate(details_list):
        svg.draw_text(details, x=label_font_size - left_top_margin, y=float(y1) + float(y0) + details_font_size * (i + 1), font_size=details_font_size, text_anchor="left", font_style=font_style)
    right_margin = max(details_width + label_font_size - left_top_margin - float(x1), (tick_font_size + label_font_size) * 2)
    svg.adjust_margin(save_path, l=left_top_margin, r=right_margin, t=left_top_margin, b=bottom_margin)
    svg.save(save_path)

def draw_heatmap_core(score_matrix, x_labels, y_labels, value_font_size=10, tick_font_size=14, subplot=[1,1,1]):
    assert score_matrix.shape == (len(y_labels), len(x_labels))
    figsize_unit = 0.5

    fig =plt.figure(figsize=(len(x_labels) * figsize_unit, len(y_labels) * figsize_unit))
    ax = plt.subplot(*subplot)
    im = plt.imshow(score_matrix, cmap="YlGn", vmin=0, vmax=1)
    bar = plt.colorbar(im, fraction=0.046, pad=0.04)
    # Loop over data dimensions and create text annotations.
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            if np.isnan(score_matrix[i, j]):
                continue
            value = f"{np.round(score_matrix[i, j], 3):0<5}"
            if np.absolute(score_matrix[i, j]) < score_matrix.max()/2:
                text = ax.text(j, i, value, ha="center", va="center", color="k", fontsize=value_font_size)
            else:
                text = ax.text(j, i, value, ha="center", va="center", color="w", fontsize=value_font_size)

    # Show all ticks and label them with the respective list entries
    # x
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(labels=x_labels, fontsize=tick_font_size)
    
    # y
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(labels=y_labels, fontsize=tick_font_size)
    plt.subplots_adjust(bottom=0.0, left=0.0, right=1, top=1)
    return fig, ax

def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x-0.45, y-0.45), 0.9, 0.9, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def main(uploaded_refseq_file_paths, param_dict, save_dir, score_matrix=None, tmp_names=None):
    if not len(uploaded_refseq_file_paths) > 1:
        raise Exception("Please upload at least 2 reference files under the 'sample_data' directory!")

    # open files
    my_refseq_list = [
        MyRefSeq_Minimum(refseq_file_path) for refseq_file_path in uploaded_refseq_file_paths
    ]

    # calc distance & propose optimized combination
    if score_matrix is None:
        score_matrix = pre_survery(my_refseq_list, param_dict)
    comb = recommended_combination(score_matrix, param_dict["score_threshold"])
    refseq_names = [refseq.path.name for refseq in my_refseq_list]

    # remove before make
    for i in save_dir.glob(f"recommended_group_*.svg"):
        i.unlink()

    # draw histogram(s)
    for group_idx, comb_idx_list in enumerate(comb):
        save_path = save_dir / (f"recommended_group_{group_idx}.svg")
        draw_heatmap(score_matrix, refseq_names, comb_idx_list, param_dict["score_threshold"], save_path, tmp_names=tmp_names)

    print()
    print("#########################")
    print("# Recommended groupings #")
    print("#########################")
    for group_idx, comb_idx_list in enumerate(comb):
        print(f"Group{group_idx}")
        for comb_idx in comb_idx_list:
            print(f"{tmp_names[comb_idx]: <4}: " + refseq_names[comb_idx])
        print()
    return comb, score_matrix

if __name__ == "__main__":
    # params
    gap_open_penalty = 3   #@param {type:"integer"}
    gap_extend_penalty = 1 #@param {type:"integer"}
    match_score = 1        #@param {type:"integer"}
    mismatch_score = -2    #@param {type:"integer"}
    score_threshold = 0.97  #@param {type:"number"}
    param_dict = {i:globals()[i] for i in ('gap_open_penalty', 'gap_extend_penalty', 'match_score', 'mismatch_score', 'score_threshold')}

    # files
    uploaded_refseq_file_names = [
        "M32_pmNeonGreen-N1.dna", 
        "M38_mCherry-Spo20.dna", 
        "M42_GFP-PASS_vecCMV.dna", 
        "M43_iRFP713-PASS_vecCMV.dna", 
        "M160_P18-CIBN-P2A-CRY2-mCherry-PLDs17_pcDNA3.dna", 
        "M161_CRY2-mCherry-PLDs27-P2A-CIBN-CAAX_pcDNA3.dna", 
    ]
    tmp_names = [f"P{i}" for i in range(len(uploaded_refseq_file_names))]

    refseq_dir = Path("./demo_data/my_plasmid_maps_dna")
    uploaded_refseq_file_paths = []
    for refseq_file_name in uploaded_refseq_file_names:
        plasmid_map_path = list(refseq_dir.rglob(refseq_file_name))
        assert len(plasmid_map_path) == 1
        uploaded_refseq_file_paths.append(plasmid_map_path[0])
    save_dir = Path("./demo_data/results_pre_survey")
    assert save_dir.exists()
    recommended_groupings_path = save_dir / "recommended_groupings.txt"

    # load if any previous score_matrix
    skip = False
    if recommended_groupings_path.exists():
        recommended_groupings = mc.RecommendedGroupings()
        recommended_groupings.load(recommended_groupings_path)
        if (recommended_groupings.uploaded_refseq_file_names == uploaded_refseq_file_names) and (recommended_groupings.tmp_names == tmp_names):
            score_matrix = recommended_groupings.score_matrix
            comb, score_matrix = main(uploaded_refseq_file_paths, param_dict, save_dir, score_matrix=score_matrix, tmp_names=tmp_names)
            skip = True
    if not skip:
        comb, score_matrix = main(uploaded_refseq_file_paths, param_dict, save_dir, score_matrix=None, tmp_names=tmp_names)
    recommended_groupings = mc.RecommendedGroupings(score_matrix, comb, uploaded_refseq_file_paths, tmp_names, param_dict)
    # save results
    recommended_groupings.save(save_path = recommended_groupings_path)





