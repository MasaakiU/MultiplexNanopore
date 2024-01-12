# -*- coding: utf-8 -*-

import re
import textwrap
import numpy as np
import scipy.spatial.distance as distance
from tqdm import tqdm
from typing import List
from pathlib import Path
from itertools import combinations
from scipy.cluster.hierarchy import linkage, leaves_list
from pulp import LpProblem, LpVariable, PULP_CBC_CMD, LpStatus, LpBinary, LpMinimize, lpSum
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..modules import my_classes as mc
from ..modules import ref_query_alignment as rqa
rc('font', **{'family':'sans-serif','sans-serif':[mc.sans_serif_font_master]})

__all__ = ["execute_grouping"]

def execute_grouping(ref_seq_list: List[mc.MyRefSeq], param_dict: dict, save_dir: Path, ref_seq_aliases: list=None):
    assert (ref_seq_aliases is None) or (len(ref_seq_list) == len(ref_seq_aliases))
    recommended_grouping_path = save_dir.parent / RecommendedGrouping.file_name
    # load if any previous score_matrix
    skip = False
    if recommended_grouping_path.exists():
        recommended_grouping = RecommendedGrouping()
        recommended_grouping.load(recommended_grouping_path)
        if recommended_grouping.assert_data(ref_seq_list, param_dict):
            recommended_grouping.param_dict = param_dict
            print("calculation of distance matrix: SKIPPED")
            skip = True
    recommended_grouping_path = save_dir / RecommendedGrouping.file_name
    if not skip:
        print("calculating distance matrix...")
        recommended_grouping = RecommendedGrouping(ref_seq_list, param_dict, ref_seq_aliases)
        # 距離行列/linkage 計算
        recommended_grouping.calc_distance_linkage()
        recommended_grouping.save(recommended_grouping_path)
        print("calculation: DONE")
    #　コンビネーション計算実行
    # print("\n### distance matrix ###")
    # print(recommended_grouping.distance_matrix)
    # print("\n### linkage result ###")
    # print(recommended_grouping.linkage_result.astype(int))
    print("\ndetermining number of groups...")
    recommended_grouping.determine_number_of_groups()
    print(f"determination: DONE\n\n{recommended_grouping.message}")
    # print("\n### cluster list ###")
    # print(recommended_grouping.cluster_list)
    print("\nexecuting plasmid assignment...")
    recommended_grouping.execute_knapsac()
    print("\nassignment: DONE")
    return recommended_grouping

class RecommendedGrouping(mc.MyTextFormat, mc.MyHeader):
    grouping_algorithm_version = "ga_0.2.1"
    linkage_method = "single"   # 最短距離法: クラスタ間で最も近いデータの距離
    file_name = "recommended_grouping.txt"
    wrap_width = 100
    def __init__(self, ref_seq_list: List[mc.MyRefSeq]=None, param_dict: dict=None, ref_seq_aliases: list=None):
        super().__init__()
        self.header += (
            f"\ngrouping_algorithm_version: {self.grouping_algorithm_version}"
            f"\nassignment_solver_version: {PlasmidAssignmentSolver.assignment_solver_version}"
        )
        self.ref_seq_list = ref_seq_list
        self.param_dict = param_dict
        self.distance_matrix = None
        self.linkage_result = None
        self.adopted_number_of_groups = None
        # attributes below can be empty when saving
        self.message = ""
        self.cluster_list = []
        self.assignment_matrix = np.array([[]], dtype=int)
        # hidden attributes
        self._ref_seq_aliases = ref_seq_aliases
        self.recommended_grouping_from_txt = None
        self.keys = [
            ("header", "str"), 
            ("datetime", "str"), 
            ("ref_seq_paths", "listPath"), 
            ("ref_seq_aliases", "list"), 
            ("ref_seq_hash_list", "list"), 
            ("param_dict", "dict"), 
            ("distance_matrix", "ndarray"), 
            ("linkage_result", "ndarray"), 
            ("linkage_method", "str"), 
            ("message", "str"), 
            # ("cluster_list", "listlist"),         # assignment_matrix(_str) からに完全に含まれる情報
            ("assignment_matrix_txt", "str"),       # アサインされてないところを "*" で表現するために、string で保存
            ("recommended_grouping_txt", "str"),    # assignment_matrix_txt より後に読み込まれる必要がある（可能ならすべてすべて全ての属性の中で最後に、かな？）。
        ]
    @property
    def ref_seq_paths(self) -> List[Path]:
        return [ref_seq.path for ref_seq in self.ref_seq_list]
    @ref_seq_paths.setter
    def ref_seq_paths(self, ref_seq_paths: list):
        if self.ref_seq_list is not None:
            raise Exception("error: self.ref_seq_list is already assigned")
        else:
            try:
                self.ref_seq_list = [mc.MyRefSeq(ref_seq_path) for ref_seq_path in ref_seq_paths]
            except: # 存在しないファイルが一つでもある場合: (ほぼ) path 情報だけ保存するシュードインスタンスを作成
                self.ref_seq_list = [mc.MyRefSeq.Pseudo(ref_seq_path) for ref_seq_path in ref_seq_paths]
    @property
    def ref_seq_aliases(self):
        if (self._ref_seq_aliases is None) or (len(self._ref_seq_aliases) != self.N_plasmids):
            return [f"P{i + 1}" for i in range(self.N_plasmids)]
        else:
            return self._ref_seq_aliases
    @ref_seq_aliases.setter
    def ref_seq_aliases(self, ref_seq_aliases):
        self._ref_seq_aliases = ref_seq_aliases
    @property
    def ref_seq_hash_list(self) -> List[str]:
        return [ref_seq.my_hash for ref_seq in self.ref_seq_list]
    @property
    def N_plasmids(self) -> int:
        return len(self.ref_seq_list)
    @property
    def notice_beg(self):
        return f"{'=' * (self.wrap_width // 2 - 4)} NOTICE {'=' * (self.wrap_width // 2 - 4)}"
    @property
    def notice_end(self):
        return f"{'=' * self.wrap_width}"
    """
    # 上下関係: 
        - self.cluster_list
        - self.assignment_matrix
        - self.assignment_matrix_txt (self.assignment_matrix から動的生成)
        - self.recommended_grouping (self.assignemnt_matrix から動的生成)
        - self.recommended_grouping_txt (self.recommended_grouping から動的生成)
    # 保存されるのは
        - self.assignment_matrix_txt
        - self.recommended_grouping_txt
    # テキストファイルから読み込まれる場合
        - self.assignment_matrix_txt
        から
        - self.cluster_list
        を生成
        self.recommended_grouping_txt から一時的に recommended_grouping を生成する
        それが self.assignment_matrix に矛盾が発生する場合 (ユーザーが勝手に書き換えた場合位など) は self.recommended_grouping_txt の内容を優先する
    """
    @property
    def recommended_grouping(self):
        if self.recommended_grouping_from_txt is None:
            return [[idx for idx in assignment_row if idx != -1] for assignment_row in self.assignment_matrix.T]
        else:
            return self.recommended_grouping_from_txt
    @property
    def recommended_grouping_txt(self):
        grouping_str = ""
        for group_idx, plasmid_idx_list in enumerate(self.recommended_grouping):
            grouping_str += f"=== Group {group_idx + 1} ===\n"
            for idx in plasmid_idx_list:
                grouping_str += f"{self.ref_seq_aliases[idx]}\t{self.ref_seq_paths[idx].name}\n"
            grouping_str += "\n"
        return grouping_str
    @recommended_grouping_txt.setter
    def recommended_grouping_txt(self, recommended_grouping_txt):
        """
        "=== Group ([0-9]+) ===" と 改行 をもとにグループを detect する
        それをもとに recommended_grouping を作り、self.assignment_matrix と比較して矛盾がないかを検証する
        """
        group_match_list = re.findall(r"=== Group ([0-9]+) ===\n((?:.+\t.+\n)*)\n", recommended_grouping_txt.strip() + "\n\n")
        plasmid_match_re = re.compile(r"(.+)\t(.+)\n")
        recommended_grouping = []
        for gropu_idx, (group_number, match) in enumerate(group_match_list):
            assert gropu_idx == int(group_number) - 1
            group = []
            for ref_seq_alias, ref_seq_name in plasmid_match_re.findall(match):
                # パス違いでファイル名が同じ可能性も考慮して、、プラスミド名が一致するかを確認
                idx = self.ref_seq_aliases.index(ref_seq_alias)
                assert self.ref_seq_paths[idx].name == ref_seq_name
                group.append(idx)
            recommended_grouping.append(group)
        """
        # 検証
            recommended_grouping_txt をもとに作った recommended_grouping
        と
            assignment_matrix (テキストを読み込む時の由来は assignment_matrix_txt) から作った self.recommended_grouping
        の間にに矛盾がないかを確認
        """
        if len(recommended_grouping) == len(self.recommended_grouping) == 0:
            print(f"\033[91m\n{self.notice_beg}\n{textwrap.fill('`recommended_grouping` was not set either from `assignment_marix_txt` or `recommended_groupoing_txt`.', self.wrap_width)}\n{self.notice_end}\033[0m\n")
            self.adopted_number_of_groups = None
        elif not all(sorted(rg1) == sorted(rg2) for rg1, rg2 in zip(recommended_grouping, self.recommended_grouping)):
            self.recommended_grouping_from_txt = recommended_grouping
            print(f"\033[91m\n{self.notice_beg}\n{textwrap.fill('Conflicts were found between provided `assignment_marix_txt` and `recommended_groupoing_txt`. Constraints of `assignment_marix_txt` was ignored.', self.wrap_width)}\n{self.notice_end}\033[0m\n")
            self.adopted_number_of_groups = len(group_match_list)
        else:
            self.adopted_number_of_groups = self.assignment_matrix.shape[1]
    @property
    def assignment_matrix_txt(self):
        assignment_matrix_txt = "\n".join(["\t".join([str(idx) if idx != -1 else "." for idx in assignment]) for assignment in self.assignment_matrix])
        return assignment_matrix_txt
    @assignment_matrix_txt.setter
    def assignment_matrix_txt(self, assignment_matrix_txt):
        if len(assignment_matrix_txt) != 0:
            self.assignment_matrix = np.array([[int(idx) if idx != "." else -1 for idx in assignment.split("\t")] for assignment in assignment_matrix_txt.split("\n")])
            self.cluster_list = [[idx for idx in assignment if idx != -1] for assignment in self.assignment_matrix.T]
        else:
            self.assignment_matrix = np.array([[]], dtype=int)
    @property
    def adopted_distance_threshold(self):
        """
        self.adopted_number_of_groups は、self.assignment_matrix(_txt) および self.recommended_grouping(_txt) と関連してくるので、
        def recommended_grouping_txt 内で代入が行われる。
        self.adopted_distance_threshold については、あまり気にしなくても良さそうなので、単純に self.message から正規表現で取得する。
        """
        m = re.search(r"\nadopted `distance_threshold`\t([0-9]+)\n", self.message)
        if m is None:
            return None
        else:
            return int(m.group(1))
    def assert_data(self, ref_seq_list: List[mc.MyRefSeq], param_dict: dict):
        if len(ref_seq_list) != len(self.ref_seq_list):
            return False
        elif len(ref_seq_list) != len(self._ref_seq_aliases):
            return False
        elif all(self.param_dict[key] == param_dict[key] for key in ['gap_open_penalty', 'gap_extend_penalty', 'match_score', 'mismatch_score']):
            return False
        else:
            hash_list_new = [ref_seq.my_hash for ref_seq in ref_seq_list]
            if hash_list_new == self.ref_seq_hash_list:
                return True
            else:
                return False
    def calc_distance_linkage(self):
        self.set_distance_matrix()
        self.set_linkage()
    def set_distance_matrix(self):
        self.distance_matrix = np.zeros((self.N_plasmids, self.N_plasmids), dtype=int)
        ref_seq_idx_list = np.arange(self.N_plasmids)
        ref_seq_idx_pairs = combinations(ref_seq_idx_list, 2)
        for idx_1, idx_2 in tqdm(ref_seq_idx_pairs, ncols=100, mininterval=0.05, leave=True, bar_format='{l_bar}{bar}{r_bar}', total=self.N_plasmids * (self.N_plasmids - 1) // 2):
            ref_seq_1 = self.ref_seq_list[idx_1]
            ref_seq_2 = self.ref_seq_list[idx_2]
            self.distance_matrix[idx_1, idx_2] = self.distance_matrix[idx_2, idx_1] = self.calc_distance(ref_seq_1, ref_seq_2)
        assert (self.distance_matrix >= 0).all()
    def calc_distance(self, ref_seq_1: mc.MyRefSeq, ref_seq_2: mc.MyRefSeq):
        my_optimized_aligner = rqa.MyOptimizedAligner(ref_seq_1, self.param_dict)
        ref_seq_2 = mc.MySeq(ref_seq_2._seq)
        ref_seq_2_rc = ref_seq_2.reverse_complement()
        conserved_regions = my_optimized_aligner.calc_circular_conserved_region(ref_seq_2)
        conserved_regions_rc = my_optimized_aligner.calc_circular_conserved_region(ref_seq_2_rc)
        if conserved_regions is not None:
            my_result = my_optimized_aligner.execute_circular_alignment_using_conserved_regions(ref_seq_2, conserved_regions)
        else:
            my_result = rqa.MyResult()
        if conserved_regions_rc is not None:
            my_result_rc = my_optimized_aligner.execute_circular_alignment_using_conserved_regions(ref_seq_2_rc, conserved_regions_rc)
        else:
            my_result_rc = rqa.MyResult()
        # return distance
        if (my_result.score <= 0) and (my_result_rc.score <= 0):
            return len(ref_seq_1) + len(ref_seq_2)
        elif my_result.score >= my_result_rc.score:
            return my_result.calc_levenshtein_distance()
        else:
            return my_result_rc.calc_levenshtein_distance()
    def set_linkage(self):
        dArray = distance.squareform(self.distance_matrix)
        self.linkage_result = linkage(dArray, method=self.linkage_method).astype(int)   # distance が全て整数で、かつ linkage_method が average などでないため整数型にして ok
        # print(leaves_list(self.linkage_result))
        # print(self.linkage_result)
    def determine_number_of_groups(self):
        # 指定されたグループと distance_threshold の調整
        end_row_to_iter_plus, adopted_distance_threshold, self.adopted_number_of_groups = self.investigate_linkage_result()
        assert self.param_dict["distance_threshold"] <= adopted_distance_threshold
        assert self.param_dict["number_of_groups"] <= self.adopted_number_of_groups

        # メッセージ
        self.message = (
            f"provided `distance_threshold`\t{self.param_dict['distance_threshold']}\n"
            f"provided `number_of_groups`\t{self.param_dict['number_of_groups']}\n"
            f"adopted `distance_threshold`\t{adopted_distance_threshold}\n"
            f"adopted `number_of_groups`\t{self.adopted_number_of_groups}\n"
            f"`end_row_to_iter_plus`\t{end_row_to_iter_plus}"
        )
        # [CASE 5]
        if (self.param_dict["distance_threshold"] == adopted_distance_threshold) and (self.param_dict["number_of_groups"] == self.adopted_number_of_groups):
            core_message = ""
        # [CASE 2, 3, 6]
        elif (self.param_dict["distance_threshold"] < adopted_distance_threshold) and (self.param_dict["number_of_groups"] == self.adopted_number_of_groups):
            core_message = "\n" + textwrap.fill(
                f"Higher `distance_threshold` {adopted_distance_threshold} was used and the provided value {self.param_dict['distance_threshold']} was ignored."
                f" This is because the use of higer `distance_threshold` returns safer pre-survey results, and `distance_threshold` was able to be increased without changing the provided `number_of_groups` {self.param_dict['number_of_groups']}."
                , self.wrap_width
            )
        # [CASE 4]
        elif self.param_dict["number_of_groups"] < self.adopted_number_of_groups:
            core_message = (
                f"Provided `number_of_groups` {self.param_dict['number_of_groups']} is to small to meet the `distance_threshold` {self.param_dict['distance_threshold']}."
                f" The `number_of_groups` {self.adopted_number_of_groups} will be used instead."
            )
            if self.param_dict["distance_threshold"] > 1:
                core_message += f" To achieve number of groups {self.param_dict['number_of_groups']}, lower the `distance_threshold`."
            else:
                core_message += f" Number of groups lower than {self.param_dict['number_of_groups']} cannot be achieved as some plasmid maps share exactly the same sequences."
            if self.param_dict["distance_threshold"] < adopted_distance_threshold:
                core_message += (
                    f" Apart from that, higher `distance_threshold` {adopted_distance_threshold} was used and the provided value {self.param_dict['distance_threshold']} was ignored."
                    f" This is because the use of higer `distance_threshold` returns safer pre-survey results, "
                    f"and `distance_threshold` was able to be increased without changing the new `number_of_groups` {self.adopted_number_of_groups}."
                )
            core_message = "\n" + textwrap.fill(core_message, self.wrap_width)
        else:
            raise Exception(f"error:\n{self.message}\end_row_to_iter_plus:\t{end_row_to_iter_plus}")
        self.message = f"{self.notice_beg}\n{self.message}{core_message}\n{self.notice_end}"

        # クラスタ形成
        self.cluster_list = [[i] for i in range(self.N_plasmids)]
        for idx1, idx2, d, number_of_sub_cluster in self.linkage_result[:end_row_to_iter_plus, :]:
            self.cluster_list.append(self.cluster_list[int(idx1)] + self.cluster_list[int(idx2)])
            self.cluster_list[int(idx1)] = None
            self.cluster_list[int(idx2)] = None
        self.cluster_list = [c for c in self.cluster_list if c is not None]
    def investigate_linkage_result(self):
        """
        # about conditions
            distance_threshold は大きい方が良い (安全)
            number_of_groups は小さい方が良い (安い)
        # distance_threshold "未満" の距離のものは同じクラスターに含めなくてはいけないと定めている (前のバージョンのSAVEMONEYも同じ)。
            distance_threshold が 7 の時は、距離が 7 のプラスミドペアは、同じクラスタに含めなくともよい
            言い換えると、グループ内での最低距離が 7 "以上" になるようにするということ。
        # linkage results:
            [[idx1, idx2, dist, N_gr], 
             [idx1, idx2, dist, N_gr], 
             ...
             [idx1, idx2, dist, N_gr]]
        # Example of self.linkage_result (when distance_threshold=7, number_of_groups=3)
            実際に使われる数字は、 distance_threshold - 1 = 6 となる。
            この場合、distance <= 6 のペアは同じクラスタに入る -> グループ内の distance >=7 が保証される。
            row  dist  N_gr/max_number_of_sub_cluster [CASE 1]
            0      [1]   [2]
            1      [2]   [3]
            2      [5]   [3]
            3      [6]   [3]
            4      [6]    4 **** distance_threshold - 1 = 10 - 1, number_of_groups = max(4, 3)
            5      10     4
            row  dist  N_gr/max_number_of_sub_cluster [CASE 2]
            0      [1]   [2]
            1      [6]   [3]
            2      [6]   [3]
            3       7    [3] **** distance_threshold - 1 = 8 - 1, number_of_groups = max(3, 3)
            4       8     4
            5      10     4
            row  dist  N_gr/max_number_of_sub_cluster [CASE 3]
            0      [6]    2
            1      [6]    3
            2       7    [3]
            3       7    [3]
            4      10    [3] **** distance_threshold - 1 = 11 - 1, number_of_groups = max(3, 3)
            5      11     4
            row  dist  N_gr/max_number_of_sub_cluster [CASE 4]
            0      [1]   [2]
            1      [1]   [3]
            2      [1]   [3]
            3      [1]   [3]
            4      [6]    4
            5      [6]    4  この行で止めてしまうと、違うクラスタ内のプラスミドペアで distance_threshold を満たさないペアができる可能性がある
            6      [6]    5  **** distance_threshold = max(6+1, 7), number_of_groups = 5
            row  dist  N_gr/max_number_of_sub_cluster [CASE 5]
            0      [1]   [2]
            1      [1]   [3]
            2      [1]   [3]
            3      [1]   [3] **** distance_threshold - 1 = 7 - 1, number_of_groups = max(3, 3)
            4       7     4
            5       7     4
            6       7     5
            row  dist  N_gr/max_number_of_sub_cluster [CASE 6]
            0       9    [2] **** distance_threshold - 1 = 9 - 1, number_of_groups = max(2, 3)
            1       9     4
            2       9     4
        """
        # self.linkage_result を上の行から見て行って、その行までに登場した最大の number_of_sub_cluster のリストを作る
        assert self.linkage_result[0, 3] == 2
        assert self.linkage_result[-1, 3] == self.linkage_result[:, 3].max() == self.N_plasmids
        assert 0 < self.param_dict["distance_threshold"] #<= self.linkage_result[:, 2].max() + 1
        assert 0 < self.param_dict["number_of_groups"] <= self.N_plasmids
        # 事前準備
        max_n = 0
        max_number_of_sub_cluster = []
        for number_of_sub_cluster in self.linkage_result[:, 3]:
            if number_of_sub_cluster > max_n:
                max_n = number_of_sub_cluster
            max_number_of_sub_cluster.append(max_n)
        # 実行
        for end_of_iter_plus, (dist, number_of_sub_cluster) in enumerate(zip(self.linkage_result[:, 2], max_number_of_sub_cluster)):
            if (dist > self.param_dict["distance_threshold"] - 1) and (number_of_sub_cluster > self.param_dict["number_of_groups"]):
                if end_of_iter_plus == 0:
                    return end_of_iter_plus, dist, 1
                else:
                    return end_of_iter_plus, dist, max(max_number_of_sub_cluster[end_of_iter_plus-1], self.param_dict["number_of_groups"])
        return self.linkage_result.shape[0], max(dist+1, self.param_dict["distance_threshold"]), number_of_sub_cluster
    def execute_knapsac(self):
        cluster_size_list = [len(cluster) for cluster in self.cluster_list]
        assert max(cluster_size_list) <= self.adopted_number_of_groups
        assert sum(cluster_size_list) == self.distance_matrix.shape[0]

        """
        # 目的関数までまとめて一気にゴリ押ししてsolveできるのでは?
        # 目的関数: idx_pairs_to_be_separated の組み合わせで、採用されなかったものの idx が最大になるようにすること
        # 時間がかかりすぎるようなら、やはり二段階に分離して行うべきか（まずプラスミドのクラスタ - グループの位置を決めて、そのあとプラスミドを配置していく）
        """
        # 許される組み合わせのペアの中で、最も距離が近いものを昇順に取得する
        inf_value = self.distance_matrix.max() + 1
        distance_matrix = self.distance_matrix * np.tri(*self.distance_matrix.shape, k=0, dtype=int) + np.tri(*self.distance_matrix.shape, k=0, dtype=int).T * inf_value    # 対角線より上の部分を -1 にする
        for cluster in self.cluster_list:   # 同じクラスタに含まれるプラスミドのペアは、ここでは考慮しない（LP の制約条件で記述する）
            for idx1, idx2 in combinations(cluster, 2):
                distance_matrix[idx1, idx2] = distance_matrix[idx2, idx1] = inf_value
        unique, inverse = np.unique(distance_matrix, return_inverse=True)   # 距離が小さいものから順に取得
        inverse = inverse.reshape(distance_matrix.shape)
        idx_pairs_in_order_of_distance = []                                          # ペアの idx を取得
        for i, distance in enumerate(unique[:-1]):
            idx_pairs_in_order_of_distance.extend([PlasmidAssignmentSolver.sort_idx(idx1, idx2) for idx1, idx2 in zip(*np.where(inverse == i))])

        # """
        # # まず、クラスタ内のプラスミドの区別なく、組分けを行う。これにより、グループ間のプラスミドの個数がバラつくことを防ぐ
        # # その後、距離が短いものを最優先に、異なるクラスタに分類していく
        # """

        # 実行
        pas = PlasmidAssignmentSolver(self.adopted_number_of_groups, self.cluster_list, idx_pairs_in_order_of_distance) # 一気に解くのは線形じゃないからダメそう
        self.assignment_matrix = pas.execute()
    def load(self, load_path):
        ver_tmp = self.grouping_algorithm_version
        linkage_method_tmp = self.linkage_method
        super().load(load_path)
        if ver_tmp != self.get_version("grouping_algorithm_version"):
            print(
                f"\033[91m\n{self.notice_beg}\nmismatch of file version was found in `{load_path}`"
                f"\nexpected file version:\t{ver_tmp}\ndetected file version:\t{self.get_version('grouping_algorithm_version')}\n{self.notice_end}\033[0m"
            )
        if linkage_method_tmp != self.linkage_method:
            print(
                f"\033[91m\n{self.notice_beg}\nmismatch of `linkage_method` was found in `{load_path}`"
                f"\nexpected `linkage_method`:\t{linkage_method_tmp}\ndetected `linkage_method`:\t{self.linkage_method}\n{self.notice_end}\033[0m"
            )
        if PlasmidAssignmentSolver.assignment_solver_version != self.get_version("assignment_solver_version"):
            print(
                f"\033[91m\n{self.notice_beg}\nmismatch of solver version was found in `{load_path}`"
                f"\nexpected solver version:\t{ver_tmp}\ndetected solver version:\t{self.get_version('assignment_solver_version')}\n{self.notice_end}\033[0m"
            )
        self.path = load_path
    def draw_heatmaps(self, save_dir=None, display_plot=None):
        # 複数 PDF 作成準備
        if save_dir is not None:
            pdf = PdfPages((Path(save_dir) / self.file_name).with_suffix(".pdf"))

        # 描画
        for group_idx, group in enumerate(self.recommended_grouping):
            self.draw_heatmap(group_idx, group)
            # 保存? 表示? そのまま閉じる?
            if save_dir is not None:
                pdf.savefig()
            if display_plot is None:
                return
            elif display_plot:
                plt.show()
            else:
                plt.clf()
                plt.close()

        # 複数 PDF 作成事後処理
        if save_dir is not None:
            pdf.close()
    def draw_heatmap(self, group_idx, plasmid_idx_list_to_highlight):
        # font style params
        max_alias_width = max(len(ref_seq_alias) for ref_seq_alias in self.ref_seq_aliases)
        y_tick_laabels = [f"{ref_seq_alias:<{max_alias_width}} {ref_seq_path.name}" for ref_seq_alias, ref_seq_path in zip(self.ref_seq_aliases, self.ref_seq_paths)]
        tick_font_size = 10
        # fig size parmas (fixed margin with absoute values, not percentile)
        heatmap_cell_unit = 0.5
        top_margin_unit = 0.6
        left_margin_unit = 1.0
        bottom_margin_unit = 0.8
        right_margin_unit = 0.5 + tick_font_size/100 * 0.9 * max(len(y_tick_laabel) for y_tick_laabel in y_tick_laabels)
        bar_width_percentile = 3            # color bar のサイズは、heatmap サイズに応じて決める
        bbox_to_anchor_x0_percentile = -5   # color bar のサイズは、heatmap サイズに応じて決める
        bottom_title_y0_percentile = -0.4 / (heatmap_cell_unit * self.N_plasmids) * 100     # タイトルは、絶対値で決める
        fig_width = heatmap_cell_unit * self.N_plasmids + left_margin_unit + right_margin_unit
        fig_height = heatmap_cell_unit * self.N_plasmids + top_margin_unit + bottom_margin_unit
        # color styel params
        cbar_max = 30   #score_matrix.max()#
        
        # make matplotlib fig
        fig, ax= plt.subplots(1, 1, figsize=(fig_width, fig_height))
        im = ax.imshow(self.distance_matrix, cmap="YlGn", vmin=0, vmax=cbar_max)
        axins = inset_axes(ax, width=f"{bar_width_percentile}%", height="100%", loc='lower left', bbox_to_anchor=(bbox_to_anchor_x0_percentile/100, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        bar = plt.colorbar(im, cax=axins)
        axins.yaxis.set_ticks_position('left')

        # Loop over data dimensions and create text annotations.
        for i in range(self.N_plasmids):
            for j in range(self.N_plasmids):
                value = self.distance_matrix[i, j]
                if value > 9999:
                    value_font_size = 8
                else:
                    value_font_size = 10
                if value < cbar_max / 2:
                    text = ax.text(j, i, str(value), ha="center", va="center", color="k", fontsize=value_font_size)
                else:
                    text = ax.text(j, i, str(value), ha="center", va="center", color="w", fontsize=value_font_size)

        # Show all ticks and label them with the respective list entries
        # x
        if max_alias_width > 5:
            x_tick_font_size = tick_font_size * 5 / max_alias_width
        else:
            x_tick_font_size = tick_font_size
        ax.xaxis.set_ticks_position("top")
        ax.set_xticks(np.arange(self.N_plasmids))
        ax.set_xticklabels(labels=[i for i in self.ref_seq_aliases], fontsize=x_tick_font_size, fontname="monospace")

        # y
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks(np.arange(self.N_plasmids))
        ax.set_yticklabels(labels=y_tick_laabels, fontsize=tick_font_size, fontname="monospace")

        # highlight heatmap cells
        for i in plasmid_idx_list_to_highlight:
            for j in plasmid_idx_list_to_highlight:
                if i == j:
                    continue
                else:
                    rect = plt.Rectangle((i-0.45, j-0.45), 0.9, 0.9, fill=False, color="r")
                    ax.add_patch(rect)
        # highlight tick labels
        for idx, xtick in enumerate(ax.get_xticklabels()):
            if idx in plasmid_idx_list_to_highlight:
                xtick.set_color("r")
        for idx, ytick in enumerate(ax.get_yticklabels()):
            if idx in plasmid_idx_list_to_highlight:
                ytick.set_color("r")

        ax.set_title(
            f'Group {group_idx+1}: {{{", ".join([alias for i, alias in enumerate(self.ref_seq_aliases) if i in plasmid_idx_list_to_highlight])}}}'
            f'{" "*4}(distance_threshold={self.adopted_distance_threshold or "?"}, number_of_groups={self.adopted_number_of_groups or "?"})', 
            y=bottom_title_y0_percentile/100, loc='left', weight='bold'
        )
        plt.subplots_adjust(left=left_margin_unit/fig_width, bottom=bottom_margin_unit/fig_height, right=1-right_margin_unit/fig_width, top=1-top_margin_unit/fig_height)
    def get_ref_seq_path_list_in_group(self, group_idx):
        return [self.ref_seq_paths[idx] for idx in self.recommended_grouping[group_idx]]

class PlasmidAssignmentSolver():
    assignment_solver_version = "as_0.1.0"
    def __init__(self, adopted_number_of_groups, cluster_list, idx_pairs_in_order_of_distance) -> None:
        self.N_groups = adopted_number_of_groups
        self.cluster_list = cluster_list
        self.idx_pairs_in_order_of_distance = idx_pairs_in_order_of_distance    #　近い順
        self.idx_pairs_removed = []
        self.N_plasmids = sum(len(cluster) for cluster in self.cluster_list)
        self.N_clusters = len(self.cluster_list)
        self.N_plasmids_in_group_list = [self.N_plasmids // self.N_groups + 1 if i < self.N_plasmids%self.N_groups else self.N_plasmids // self.N_groups for i in range(self.N_groups)]
        assert (self.N_plasmids == sum(self.N_plasmids_in_group_list)) and (max(self.N_plasmids_in_group_list) - min(self.N_plasmids_in_group_list) <= 1)
        # 辞書的に使用する
        self.cluster_idx_from_plasmid_idx = np.empty(self.N_plasmids, dtype=int)
        for cluster_idx, cluster in enumerate(self.cluster_list):
            self.cluster_idx_from_plasmid_idx[cluster] = cluster_idx
    @staticmethod
    def sort_idx(idx1, idx2):
        if idx1 < idx2:
            return (idx1, idx2)
        elif idx1 > idx2:
            return (idx2, idx1)
        else:
            raise Exception(f"error: {idx1, idx2}")
    # 実行関数
    def execute(self):
        assignment_example_not_None = self.generate_example()
        pbar = tqdm(self.idx_pairs_in_order_of_distance, ncols=100, mininterval=0.02, leave=True, bar_format='{l_bar}{bar}{r_bar}')#, total=self.idx_pairs_in_order_of_distance)
        str_width = 4 + len(str(self.N_plasmids - 1)) * 2
        for idx_pair in pbar:
            pbar.set_postfix_str(f"examining edge {str(idx_pair):{str_width}}")
            self.idx_pairs_removed.append(idx_pair)
            assignment_example = self.generate_example()
            if assignment_example is None:
                del self.idx_pairs_removed[-1]
            else:
                assignment_example_not_None = assignment_example
        assert assignment_example_not_None is not None
        return assignment_example_not_None
    def generate_example(self):
        ########
        # 問題 #
        ########
        problem = LpProblem("GraphAssertion")
        ########
        # 変数 #
        ########
        # variable_box = (N_clusters, N_groups, N_plasmids)
        variable_box = [
            [
                [
                    LpVariable(f"c{cluster_idx}g{group_idx}p{plasmid_idx}", cat=LpBinary) for plasmid_idx in range(self.N_plasmids)
                ] for group_idx in range(self.N_groups)
            ] for cluster_idx in range(self.N_clusters)
        ]
        ###########
        # 制約条件 #
        ###########
        # クラスタ -- プラスミド 面からみた条件
        for cluster_idx, cluster in enumerate(self.cluster_list):
            for plasmid_idx in range(self.N_plasmids):
                if plasmid_idx in cluster:
                    problem += lpSum(variable_box[cluster_idx][group_idx][plasmid_idx] for group_idx in range(self.N_groups)) == 1
                else:
                    problem += lpSum(variable_box[cluster_idx][group_idx][plasmid_idx] for group_idx in range(self.N_groups)) == 0
        # クラスタ -- グループ 面からみた条件
        for cluster_idx in range(self.N_clusters):
            for group_idx in range(self.N_groups):
                problem += lpSum(variable_box[cluster_idx][group_idx][plasmid_idx] for plasmid_idx in range(self.N_plasmids)) <= 1
        # グループ の条件
        for group_idx, N_plasmids_in_group in enumerate(self.N_plasmids_in_group_list):
            problem += lpSum(variable_box[cluster_idx][group_idx][plasmid_idx] for cluster_idx in range(self.N_clusters) for plasmid_idx in range(self.N_plasmids)) == N_plasmids_in_group
        # 使ってはいけないペアの条件
        for plasmid_idx1, plasmid_idx2 in self.idx_pairs_removed:
            cluster_idx1 = self.cluster_idx_from_plasmid_idx[plasmid_idx1]
            cluster_idx2 = self.cluster_idx_from_plasmid_idx[plasmid_idx2]
            for group_idx in range(self.N_groups):
                problem += lpSum([variable_box[cluster_idx1][group_idx][plasmid_idx1], variable_box[cluster_idx2][group_idx][plasmid_idx2]]) <= 1
        ########
        # 求解 #
        ########
        solver = PULP_CBC_CMD(msg = False)
        result = problem.solve(solver=solver)
        status = LpStatus[result]
        if status == "Optimal":
            assignment_example = np.empty((self.N_clusters, self.N_groups), dtype=int)
            for cluster_idx in range(self.N_clusters):
                for group_idx in range(self.N_groups):
                    for plasmid_idx in range(self.N_plasmids):
                        if variable_box[cluster_idx][group_idx][plasmid_idx].value():
                            assignment_example[cluster_idx, group_idx] = plasmid_idx
                            break
                    else:
                        assignment_example[cluster_idx, group_idx] = -1
            return assignment_example
        else:
            return None

### これ単体で使用するには少し不完全だけど、めちゃめちゃプラスミド数が大きくなった時に何かの役に立つかもしれない ###
"""
一応、実行するには下記。
plasmid_loc_in_cluster_group_candidates = PlasmidAssignmentSolver.solve_simple_picross(distance_matrix.shape[0], adopted_number_of_groups, cluster_size_list)
for plasmid_loc_in_cluster_group in plasmid_loc_in_cluster_group_candidates:
    pa = PlasmidAssignment(plasmid_loc_in_cluster_group, self.cluster_list, unfavorable_idx_pairs)
    pa.execute()
"""
class PlasmidAssignment():
    def __init__(self, plasmid_loc_in_cluster_group, cluster_list, unfavorable_idx_pairs) -> None:
        assert plasmid_loc_in_cluster_group.dtype == int
        self.plasmid_loc_in_cluster_group = plasmid_loc_in_cluster_group
        self.cluster_list = cluster_list
        self.unfavorable_idx_pairs = unfavorable_idx_pairs
        self.unfavorable_idx_pairs_kept = []
        self.unfavorable_idx_pairs_removed = []
        self.cluster_size_list = np.array([len(cluster) for cluster in self.cluster_list])
        self.N_plasmids = sum(self.cluster_size_list)
        self.N_group = self.plasmid_loc_in_cluster_group.shape[1]
        # 辞書的に使用する
        self.cluster_idx_from_plasmid_idx = np.empty(self.N_plasmids, dtype=int)
        for cluster_idx, cluster in enumerate(self.cluster_list):
            self.cluster_idx_from_plasmid_idx[cluster] = cluster_idx
        # ややこしい
        """
        # pattern_list_for_each_plasmid example: 同じグループに所属しうるやつらが他のグループにあと何個残っているかを示すマトリックス。本関数内でアップデートされてく。
            cluster_list = [[0], [2], [5], [6], [11], [4, 7], [8, 13], [3, 12], [10, 1, 9]]
            where_plasmid = (shape = (len(cluster_list), adopted_number_of_groups))
                [[0 0 1]
                 [0 1 0]
                 [1 0 0]
                 [1 0 0]
                 [1 0 0]
                 [1 1 0]
                 [0 1 1]
                 [0 1 1]
                 [1 1 1]]

            [[-1  1  1  1  1  1  1  1  1  1  1  1  1  1]
             [ 1  1 -1  1  1  1  1  1  1  1  1  1  1  1]
             [ 1  1  1  1  1 -1  1  1  1  1  1  1  1  1]
             [ 1  1  1  1  1  1 -1  1  1  1  1  1  1  1]
             [ 1  1  1  1  1  1  1  1  1  1  1 -1  1  1]
             [ 2  2  2  2 -1  2  2 -1  2  2  2  2  2  2]
             [ 2  2  2  2  2  2  2  2 -1  2  2  2  2 -1]
             [ 2  2  2 -1  2  2  2  2  2  2  2  2 -1  2]
             [ 3 -1  3  3  3  3  3  3  3 -1 -1  3  3  3]]

            row: cluster_idx \ col: plasmid_idx
            [[ -1   1   1   1   1   1   1   1   1   1   1   1   1   1 ]  <- 基本的には、len(cluster_list[0]) の値が初期値として入る
             [  1   1  -1   1   1   1   1   1   1   1   1   1   1   1 ]  <- 基本的には、len(cluster_list[1]) の値が初期値として入る
             [  1   1   1   1   1  -1   1   1   1   1   1   1   1   1 ]  <- 基本的には、len(cluster_list[2]) の値が初期値として入る
             [  1   1   1   1   1   1  -1   1   1   1   1   1   1   1 ]  <- 基本的には、len(cluster_list[3]) の値が初期値として入る
             [  1   1   1   1   1   1   1   1   1   1   1   1  -1   1 ]  <- 基本的には、len(cluster_list[4]) の値が初期値として入る
             [  2   2   2   2 *-1*  2   2 *-1*  2   2   2   2   2   2 ]  <- 基本的には、len(cluster_list[5]) の値が初期値として入る ただし、[5(=cluster_idx), 4(=plasmid_idx)] と [5, 7] の場所の値は 1 となり、それは更新されない。
             [  2   2   2   2   2   2   2   2  -1   2   2   2   2  -1 ]
             [  2   2   2  -1   2   2   2   2   2   2   2   2  -1   2 ]
             [  3  -1   3   3   3   3   3   3   3  -1  -1   3   3   3 ]]
        # unfavorable_idx_pairs を iter して、pattern_list_for_each_plasmid をアップデートしていく。
            un  f avo rab le_i dx_ pai rs[ 0]  = ( 8,  3)  の処理 後を下 記に 示す。
            [[ -1   1   1   1   1   1   1   1   1   1   1   1   1   1 ]
             [  1   1  -1   1   1   1   1   1   1   1   1   1   1   1 ]
             [  1   1   1   1   1  -1   1   1   1   1   1   1   1   1 ]
             [  1   1   1   1   1   1  -1   1   1   1   1   1   1   1 ]
             [  1   1   1   1   1   1   1   1   1   1   1   1  -1   1 ]
             [  2   2   2   2  -1   2   2  -1   2   2   2   2   2   2 ]
             [  2   2   2  *1*  2   2   2   2  -1   2   2   2   2  -1 ] <- plasmid_idx = 3-8 はコネクトできなくなるので、[6(=cluster_idx where 8 is in), 3 (=plasmid_idx)] の位置と
             [  2   2   2  -1   2   2   2   2  *1*  2   2   2  -1   2 ] <- [7(=cluster_idx where 3 is in), 8 (=plasmid_idx)] の位置との値を 1 減らす。
             [  3  -1   3   3   3   3   3   3   3  -1  -1   3   3   3 ]]
        # さらに、unfavorable_idx_pairs[1] = (13, 3) の処理後を下記に示す。
            [[ -1   1   1   1   1   1   1   1   1   1   1   1   1   1  ]
             [  1   1  -1   1   1   1   1   1   1   1   1   1   1   1  ]
             [  1   1   1   1   1  -1   1   1   1   1   1   1   1   1  ]
             [  1   1   1   1   1   1  -1   1   1   1   1   1   1   1  ]
             [  1   1   1   1   1   1   1   1   1   1   1   1  -1   1  ]
             [  2   2   2   2  -1   2   2  -1   2   2   2   2   2   2  ]
             [  2   2   2  *0*  2   2   2   2  -1   2   2   2   2  -1  ] <- [6(=cluster_idx where 13 is in), 3 (=plasmid_idx)] の位置と
             [  2   2   2  -1   2   2   2   2   1   2   2   2  -1  *1* ] <- [7(=cluster_idx where 3 is in), 13 (=plasmid_idx)] の位置との値を 1 減らす。
             [  3  -1   3   3   3   3   3   3   3  -1  -1   3   3   3  ]]
        # この時点で、上のマトリックスの 3 列目 (Aとする) を満たせるような where_plasmid の列 (Bとする) が存在しない！計算式はちょっとややこしいので直下の実際の関数を参照してくれ
        # なので unfavorable_idx_pairs[1] = (13, 3) はスキップして、次へ進む
        """
        self.pattern_list_for_each_plasmid = np.ones((len(self.cluster_list), self.N_plasmids), dtype=int) * self.cluster_size_list[:, np.newaxis]
        for cluster_idx, cluster in enumerate(self.cluster_list):
            for plasmid_idx in cluster:
                self.pattern_list_for_each_plasmid[cluster_idx, plasmid_idx] = -1
        # ややこしい
        """ EXAMPLE: shape = (self.N_group, self.N_plasmids)
        # 列に True が 1 つしか出現しなくなったら、その plasmid の所属グループが決定したということ。
        [[False  True False False  True  True  True  True False  True  True  True False False]
         [False  True  True  True  True False False  True  True  True  True False  True  True]
         [ True  True False  True False False False False  True  True  True False  True  True]]
        """
        self.available_group_for_plasmid = np.zeros((self.N_group, self.N_plasmids), dtype=bool)  # bool
        for cluster_idx, cluster in enumerate(self.cluster_list):
            for plasmid_idx in cluster:
                for group_idx, plasmid_exists in enumerate(self.plasmid_loc_in_cluster_group[cluster_idx, :]):
                    if plasmid_exists:
                        self.available_group_for_plasmid[group_idx, plasmid_idx] = 1
        assert self.available_group_for_plasmid.any(axis=0).all()

        print("###")
        np.set_printoptions(linewidth=300)
        print(self.cluster_list)
        print(self.cluster_idx_from_plasmid_idx)
        print(self.plasmid_loc_in_cluster_group)
        print(self.unfavorable_idx_pairs)
        print(self.available_group_for_plasmid.astype(int))

        # 
        self.available_group_for_plasmid_initiated()


        # 使わないかも
        # adjacency_matrix = np.zeros((N_plasmids, N_plasmids), dtype=int)  # bool
        # for idx1, idx2 in unfavorable_idx_pairs:
        #     adjacency_matrix[idx1, idx2] = 1
        #     adjacency_matrix[idx2, idx1] = 1
    @staticmethod
    def solve_simple_picross(N_plasmids, adopted_number_of_groups, cluster_size_list):
        """
        # 配置
            縦 (axis=0) にクラスタ
            横 (axis=1) にグループ
        # 制約
            row の合計が cluster_size_list
            col の合計が N_plasmids_in_group_list
        # EXAMPLE
            [[0,  0,  1]    cluster_1 (size=1)
             [1,  0,  0]    cluster_2 (size=1)
             [0,  1,  0]    cluster_3 (size=1)
             [1,  0,  1]    cluster_4 (size=2)
             [1,  1,  1]]   cluster_5 (size=3)
              g1, g2, g3
              3   2   3 <- グループ間のプラスミドの個数の差は 1 以下になるようにする
        """
        N_plasmids_in_group_list = [N_plasmids // adopted_number_of_groups + 1 if i < N_plasmids%adopted_number_of_groups else N_plasmids // adopted_number_of_groups for i in range(adopted_number_of_groups)]
        assert (N_plasmids == sum(N_plasmids_in_group_list)) and (max(N_plasmids_in_group_list) - min(N_plasmids_in_group_list) <= 1)

        row_constraint = np.array(cluster_size_list)
        col_constraint = np.array(N_plasmids_in_group_list)
        N_row = len(row_constraint)
        N_col = len(col_constraint)

        # 問題
        problem = LpProblem("SimplePicross")
        # 変数（どこを 1 にするか）
        plasmid_loc_in_cluster_group = [
            [LpVariable(f"r{r}c{c}", cat=LpBinary) for c in range(N_col)]
            for r in range(N_row)
        ]
        # 制約条件の追加（目的関数はない：すべて数え上げる）
        for r in range(N_row):
            problem += lpSum(plasmid_loc_in_cluster_group[r][c] for c in range(N_col)) == row_constraint[r]
        for c in range(N_col):
            problem += lpSum(plasmid_loc_in_cluster_group[r][c] for r in range(N_row)) == col_constraint[c]
        # 求解
        solver = PULP_CBC_CMD()
        result = problem.solve(solver=solver)
        status = LpStatus[result]
        if status == "Optimal":
            result_array = np.empty((N_row, N_col), dtype=int)
            for i in range(N_row):
                for j in range(N_col):
                    result_array[i, j] = plasmid_loc_in_cluster_group[i][j].value()
        else:
            print("解なし")

        plasmid_loc_in_cluster_group_candidates = []  # 解を書き出す予定: TODO
        plasmid_loc_in_cluster_group_candidates.append(result_array)
        return plasmid_loc_in_cluster_group_candidates
    @staticmethod
    def __knapsac_core(adopted_number_of_groups, cluster_list, unfavorable_idx_pairs):
        N_plasmids = sum(len(cluster) for cluster in cluster_list)
        N_cluster = len(cluster_list)
        cluster_size_list = [len(cluster) for cluster in cluster_list]
        max_clusster_size = max(cluster_size_list)
        N_plasmids_in_group_list = [N_plasmids // adopted_number_of_groups + 1 if i < N_plasmids%adopted_number_of_groups else N_plasmids // adopted_number_of_groups for i in range(adopted_number_of_groups)]
        assert (N_plasmids == sum(N_plasmids_in_group_list)) and (max(N_plasmids_in_group_list) - min(N_plasmids_in_group_list) <= 1)

        ########
        # 問題 #
        ########
        problem = LpProblem("SimplePicross", LpMinimize)

        ########
        # 変数 #
        ########
        # どこにどのプラスミドを入れるかのインデックスを格納した3次元配列
        # where_plasmid[cluster_idx].shape = (adopted_number_of_groups, cluster_size_list[i])
        where_plasmid = []
        for cluster_idx, cluster_size in enumerate(cluster_size_list):
            group_and_in_cluster = []
            for group_idx in range(adopted_number_of_groups):
                in_cluster = []
                for idx_in_cluster in range(max_clusster_size):  # テンソルにするために、敢えて余計に入れる
                    var = LpVariable(f"c{cluster_idx}g{group_idx}i{idx_in_cluster}", cat=LpBinary)
                    in_cluster.append(var)
                    if idx_in_cluster > cluster_size - 1:
                        problem += var == 0
                group_and_in_cluster.append(in_cluster)
            where_plasmid.append(group_and_in_cluster)

        ################
        # 制約条件の追加 #
        ################
        # group に含まれるプラスミドの個数
        for group_idx in range(adopted_number_of_groups):
            problem += lpSum(
                where_plasmid[cluster_idx][group_idx][idx_in_cluster]
                for cluster_idx, cluster_size in enumerate(cluster_size_list)
                for idx_in_cluster in range(cluster_size)
            ) == N_plasmids_in_group_list[group_idx]
        # idx_in_cluster に含まれるプラスミドの個数
        for cluster_idx, cluster_size in enumerate(cluster_size_list):
            for idx_in_cluster in range(cluster_size):
                problem += lpSum(where_plasmid[cluster_idx][group_idx][idx_in_cluster] for group_idx in range(adopted_number_of_groups)) == 1
        # 同じグループに同一のクラスタ由来のプラスミドがが含まれないようにする
        for cluster_idx, cluster_size in enumerate(cluster_size_list):
            for group_idx in range(adopted_number_of_groups):
                problem += lpSum(where_plasmid[cluster_idx][group_idx][idx_in_cluster] for idx_in_cluster in range(cluster_size)) <= 1

        # 目的（最小化）関数：unfavorable_idx_pairs のスコアを、2**(max_idx - idx) などとしたら、最小化問題に帰着できる気がする
        # where_plasmid を、テンソルにする必要がある（0 を加えてクラスターサイズを揃えてある）
        score_tensor = np.arange()
        print()
        print(cluster_size_list)
        print(where_plasmid)
        problem += lpSum(

        )

        print(problem)
        quit()


        # 求解
        solver = PULP_CBC_CMD()
        result = problem.solve(solver=solver)
        status = LpStatus[result]
        if status == "Optimal":
            result_array = np.empty((N_cluster, adopted_number_of_groups), dtype=int)
            for cluster_idx, cluster_size in enumerate(cluster_size_list):
                for group_idx in range(adopted_number_of_groups):
                    idx_in_cluster_with_plasmid = -1
                    for idx_in_cluster in range(cluster_size):
                        i = where_plasmid[cluster_idx][group_idx][idx_in_cluster].value()
                        print(f"c{cluster_idx}g{group_idx}i{idx_in_cluster}: {int(i)}")
                        if i != 0:
                            assert idx_in_cluster_with_plasmid == -1
                            idx_in_cluster_with_plasmid = idx_in_cluster
                    result_array[cluster_idx, group_idx] = idx_in_cluster_with_plasmid
            print(result_array)
        else:
            print("解なし")


        quit()
    ##################
    # MAIN FUNCTIONS #
    ##################
    def available_group_for_plasmid_initiated(self):
        # pattern_list_for_each_plasmid/unfavorable_idx_pairs の処理
        for plasmid_idx1, available_group in enumerate(self.available_group_for_plasmid.T):
            for plasmid_idx2, allowed in enumerate(self.available_group_for_plasmid[available_group, :].sum(axis=0)):
                if (not allowed) and (plasmid_idx1 < plasmid_idx2):
                    self.remove_plasmid_connection_core(plasmid_idx1, plasmid_idx2)
        self.process_lonly_connection()
    # 実行関数
    def execute(self):
        """
        1. エッジを距離の短い順に除去する (self.remove_plasmid_connection)
            1-1. 除去すると矛盾が生じてしまう場合、エッジの除去を取りやめて次のループへ
            1-2. 除去 "できそう" な場合 (できるとはまだ言っていいない)
                1-2-0. 以下は、必要十分なエッジの除去を行う操作であり、恣意性は入らない（はず）
                1-2-1. 
        """
        print("EXECUTE")
        for idx1, idx2 in self.unfavorable_idx_pairs:
            print("IDX", idx1, idx2)
            if (idx1, idx2) in self.unfavorable_idx_pairs_kept:
                continue
            elif (idx1, idx2) in self.unfavorable_idx_pairs_removed:
                continue
            processed = self.remove_plasmid_connection(idx1, idx2)
            if processed:
                continue
            else:
                continue
        # チェック
        self.assert_available_group_for_plasmid()
        print(self.pattern_list_for_each_plasmid)
        print(self.available_group_for_plasmid)
        print(len(self.unfavorable_idx_pairs))
        print(len(self.unfavorable_idx_pairs_kept))
        print(len(self.unfavorable_idx_pairs_removed))
        print(self.unfavorable_idx_pairs)
        print(self.unfavorable_idx_pairs_kept)
        print(self.unfavorable_idx_pairs_removed)
        quit()
    def generate_available_group(self, idx):
        # self.available_group_for_plasmid はまだアプデされない
        # self.pattern_list_for_each_plasmid[:, idx] について、グループに所属できるかかどうかを　False, True のリスト (len=self.N_group) で返す。
        available_group = ((1 - self.pattern_list_for_each_plasmid[:, idx:idx+1] + self.plasmid_loc_in_cluster_group) != 2).all(axis=0)
        """
        available_group が適応された場合、今後矛盾なくプロセスが進められる可能性が確実に残っているかどうかを調べる。
        必要十分な解析をする必要がある。
        """
        # 全く True が無ければそもそもダメだね
        if not available_group.any():
            contradict = True
        else:
            contradict = False
            # available_group_for_plasmid = np.copy(self.available_group_for_plasmid)
            # print(available_group_for_plasmid.astype(int))
            # quit()
        if contradict:
            return None
        else:
            return available_group
    def remove_plasmid_connection(self, idx1, idx2):
        new_value1, new_value2 = self.remove_plasmid_connection_core(idx1, idx2)
        if (new_value1 != 0) and (new_value2 != 0):
            if (new_value1 == 1) or (new_value2 == 1):
                self.process_lonly_connection() # 新たに 1 が誕生しても、それが属するグループが決定していいなければ self.unfavorable_idx_pairs_kept には登録されないようになっている
            return True     # not processed
        else:
            if new_value1 == 0:
                available_group_1 = self.generate_available_group(idx1) # self.available_group_for_plasmid はまだアプデされない
                # グループを割り当てることが不可能な場合
                if available_group_1 is None:
                    self.undo_remove_plasmid_connection(idx1, idx2)
                    return False     # not processed
                else:
                    self.undo_remove_plasmid_connection(idx1, idx2)     # 直下の行の update_available_group_for_plasmid で統一的に書くため、一旦戻す
                    self.update_available_group_for_plasmid(idx1, available_group_1)
                    assert (self.available_group_for_plasmid.sum(axis=1) >= self.plasmid_loc_in_cluster_group.sum(axis=0)).all()
                    return True     # processed
            if new_value2 == 0:
                available_group_2 = self.generate_available_group(idx2) # self.available_group_for_plasmid はまだアプデされない
                # グループを割り当てることが不可能な場合
                if available_group_2 is None:
                    self.undo_remove_plasmid_connection(idx1, idx2)
                    return False     # not processed
                # グループを割り当てることが可能な場合、アプデを検討
                else:
                    self.undo_remove_plasmid_connection(idx1, idx2)     # 直下の行の update_available_group_for_plasmid で統一的に書くため、一旦戻す
                    self.update_available_group_for_plasmid(idx2, available_group_2)
                    assert (self.available_group_for_plasmid.sum(axis=1) >= self.plasmid_loc_in_cluster_group.sum(axis=0)).all()
                    return True     # processed
    def remove_plasmid_connection_core(self, idx1, idx2):
        # must be idx1 < idx2   (あえて assersion しないよ!)
        assert (idx1, idx2) not in self.unfavorable_idx_pairs_removed
        self.unfavorable_idx_pairs_removed.append((idx1, idx2))
        cluster_idx1 = self.cluster_idx_from_plasmid_idx[idx1]
        cluster_idx2 = self.cluster_idx_from_plasmid_idx[idx2]
        self.pattern_list_for_each_plasmid[cluster_idx1, idx2] -= 1
        self.pattern_list_for_each_plasmid[cluster_idx2, idx1] -= 1
        return self.pattern_list_for_each_plasmid[cluster_idx2, idx1], self.pattern_list_for_each_plasmid[cluster_idx1, idx2]
    def undo_remove_plasmid_connection(self, idx1, idx2):
        # must be idx1 < idx2   (あえて assersion しないよ!)
        self.unfavorable_idx_pairs_removed.remove((idx1, idx2))
        cluster_idx1 = self.cluster_idx_from_plasmid_idx[idx1]
        cluster_idx2 = self.cluster_idx_from_plasmid_idx[idx2]
        self.pattern_list_for_each_plasmid[cluster_idx1, idx2] += 1
        self.pattern_list_for_each_plasmid[cluster_idx2, idx1] += 1
    def update_available_group_for_plasmid(self, idx, available_group):
        self.update_available_group_for_plasmid_core(idx, available_group)
        # 事後処理（新たにできた lonly_connection を守る）
        print(self.unfavorable_idx_pairs_kept, "kept")
        print(self.unfavorable_idx_pairs_removed, "removed")
        self.process_lonly_connection()
    def update_available_group_for_plasmid_core(self, idx, available_group):
        plasmid_can_be_in_same_group_pre = self.available_group_for_plasmid[self.available_group_for_plasmid[:, idx], :].sum(axis=0)
        plasmid_can_be_in_same_group_post = self.available_group_for_plasmid[available_group, :].sum(axis=0)
        cluster_idx = self.cluster_idx_from_plasmid_idx[idx]
        # 前までコネクト可能だったが、このたびコネクト不可能になったペアを探索・削除
        for idx2, is_pair_lost in enumerate(plasmid_can_be_in_same_group_pre * np.logical_not(plasmid_can_be_in_same_group_post)):
            if is_pair_lost and (cluster_idx != self.cluster_idx_from_plasmid_idx[idx2]):    # 同じクラスタに入っているのは禁止
                self.remove_plasmid_connection_core(*self.sort_idx(idx, idx2))
        # アプデ
        self.available_group_for_plasmid[:, idx] = available_group
        # アプデによってさらなるアプデが可能かもしれない
        """
        # 単純な数独（完全に解かなくて良い）
        EXAMPLE (N_groups, N_clusters): 本当は dtype = bool
            1 1         1 0
            1 1     ->  1 0
            0 1         0 1
        EXAMPLE
            1 0 1       1 0 0
            1 1 0   ->  0 1 0
            0 0 1       0 0 1
        EXAMPLE
            0 1 1       0 1 0       0 1 0
            1 1 0   ->  1 1 0   ->  1 0 0
            1 1 0       1 1 0       1 0 0
            0 0 1       0 0 1       0 0 1
        """
        cluster = self.cluster_list[self.cluster_idx_from_plasmid_idx[idx]]
        available_group_sub_matrix = self.available_group_for_plasmid[:, cluster] # fancy indexing はビューではなくコピーが返される
        for row, row_sum in enumerate(available_group_sub_matrix.sum(axis=1)):
            # とある行の合計が 1 の場合、その行の中でどの列が 1 であるかを特定。
            if row_sum == 1:
                col = np.where(available_group_sub_matrix[row, :])[0]
                assert len(col) == 1
                col = col[0]    # 特定
                # 特定された列の合計が 2 以上の場合、アプデ可能である。
                if (available_group_sub_matrix[:, col]).sum() > 1:
                    plasmid_idx = cluster[col]
                    available_group_sub_matrix[:, col] = False
                    available_group_sub_matrix[row, col] = True
                    self.update_available_group_for_plasmid_core(plasmid_idx, available_group_sub_matrix[:, col])
                    break   # アプデ可能な他の列が残っていたとしても、直上の self.update_available_group_for_plasmid_core で再帰的に処理される
        # 列から処理する場合: 行の処理でいろいろアプデされてる可能性があるから、再度 available_group_sub_matrix を取得する。
        available_group_sub_matrix = self.available_group_for_plasmid[:, cluster]
        for col, col_sum in enumerate(available_group_sub_matrix.sum(axis=0)):
            # とある列の合計が 1 の場合、その列の中でどの行が 1 であるかを特定
            if col_sum == 1:
                row = np.where(available_group_sub_matrix[:, col])[0]
                assert len(row) == 1
                row = row[0]    # 特定
                # 特定された行の合計が 2 以上の場合、アプデ可能である。
                if (available_group_sub_matrix[row, :].sum() > 1):
                    # アプデする行を決定
                    for c, v in enumerate(available_group_sub_matrix[row, :]):
                        if v and (c != col):
                            break
                    else:
                        # if (available_group_sub_matrix[row, :].sum() > 1) より、ここに入るはずがない
                        raise Exception("error")
                    plasmid_idx = cluster[c]
                    available_group_sub_matrix[:, c] = False
                    available_group_sub_matrix[row, c] = True
                    self.update_available_group_for_plasmid_core(plasmid_idx, available_group_sub_matrix[:, c])
                    break   # 残っていても、直上の self.update_available_group_for_plasmid_core で再帰的に処理される
    def process_lonly_connection(self):
        unfavorable_idx_pairs_to_be_removed = []
        while True:
            unfavorable_idx_pairs_kept, unfavorable_idx_pairs_removed = self.process_lonly_connection_core()
            unfavorable_idx_pairs_to_be_removed.extend(unfavorable_idx_pairs_removed)
            if len(unfavorable_idx_pairs_kept) > 0:
                self.unfavorable_idx_pairs_kept.extend(unfavorable_idx_pairs_kept)
            else:
                break
        if len(unfavorable_idx_pairs_to_be_removed) > 0:
            for idx1, idx2 in list(set(unfavorable_idx_pairs_to_be_removed)):
                self.remove_plasmid_connection(idx1, idx2)
    def process_lonly_connection_core(self):
        unfavorable_idx_pairs_kept = []
        unfavorable_idx_pairs_removed = []
        # コネクションが 1 つだけのものを処理（所属すべきグループが決定している場合にのみ適応される: 決定していない場合、そもそもそのコネクションは使わないという選択肢もあるため）
        for plasmid_idx1, is_group_determined in enumerate(self.available_group_for_plasmid.sum(axis=0) == 1):
            # plasmid_idx1 が所属すべきグループが決定している場合
            if is_group_determined:
                for cluster_idx2, N_connection in enumerate(self.pattern_list_for_each_plasmid[:, plasmid_idx1] == 1):
                    # plasmid_idx1 とのコネクションが 1 つしかないプラスミドがあった場合
                    if N_connection == 1:
                        # そのコネクションは削除してはならない
                        executed_pair_count = 0
                        for plasmid_idx2 in self.cluster_list[cluster_idx2]:
                            plasmid_pair = self.sort_idx(plasmid_idx1, plasmid_idx2)
                            if plasmid_pair in self.unfavorable_idx_pairs_removed:
                                continue
                            else:
                                executed_pair_count += 1
                                if plasmid_pair not in self.unfavorable_idx_pairs_kept:
                                    unfavorable_idx_pairs_kept.append(plasmid_pair)
                        assert executed_pair_count == 1
                        # cluster_1 中の plasmid_idx1 以外のプラスミドについて、plasmid_idx2 との関係を断つ
                        for plasmid_idx_in_cluster1 in self.cluster_list[self.cluster_idx_from_plasmid_idx[plasmid_idx1]]:
                            if plasmid_idx_in_cluster1 == plasmid_idx1:
                                continue
                            else:
                                plasmid_pair = self.sort_idx(plasmid_idx_in_cluster1, plasmid_idx2)
                                """
                                self.process_lonly_connection で、どうしても _core が 2 度回ることが多々ある。
                                ここでself.remove_plasmid_connection_core(*plasmid_pair) してしまうと、その 2 回目の時に remove_plasmid_connection_core 内で 
                                AssertionError assert (idx1, idx2) not in self.unfavorable_idx_pairs_removed してしまう。
                                """
                                if plasmid_pair not in self.unfavorable_idx_pairs_removed:
                                    unfavorable_idx_pairs_removed.append(plasmid_pair)
        return list(set(unfavorable_idx_pairs_kept)), unfavorable_idx_pairs_removed
    # チェック関数
    def assert_available_group_for_plasmid(self):
        for idx in range(self.N_plasmids):
            available_group = self.generate_available_group(idx)
            print(idx, all(available_group == self.available_group_for_plasmid[:, idx]))
            # assert all(available_group == self.available_group_for_plasmid[:, idx])

def export_results(recommended_grouping: RecommendedGrouping, save_dir: Path):
    print("\nexporting results...")
    recommended_grouping.save(save_dir / RecommendedGrouping.file_name) # execute_pre_survey では必ず一度 save される (そのさい path が登録される) ので、再度 path を与える必要はない
    recommended_grouping.draw_heatmaps(save_dir, display_plot=False)
    print("export: DONE")


