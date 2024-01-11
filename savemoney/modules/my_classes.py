# -*- coding: utf-8 -*-

import io
import re
import tqdm
import json
import copy
import shutil
import hashlib
import zipfile
import tempfile
import contextlib
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from Bio.Seq import Seq
from pathlib import PosixPath   # 必要！
from datetime import datetime
from itertools import cycle
from collections import OrderedDict
from snapgene_reader import snapgene_file_to_dict

from ..__about__ import *

###
sans_serif_font_master = "Arial"

@contextlib.contextmanager
def fopen(filein, *args, **kwargs):
    if isinstance(filein, str) or isinstance(filein, Path):  # filename/Path
        with open(filein, *args, **kwargs) as f:
            yield f
    else:  # file-like object
        yield filein

class MyTextFormat():
    def to_text(self):
        text = ""
        for k, data_type in self.keys:
            text += f"# {k}({data_type})\n"
            v = getattr(self, k)
            if data_type == "str":
                string = str(v)
            elif data_type == "float":
                string = str(v)
            elif data_type == "ndarray":
                with io.StringIO() as s:
                    if v.dtype == np.int64:
                        np.savetxt(s, v, delimiter='\t', fmt="%d")
                    else:
                        np.savetxt(s, v, delimiter='\t', fmt='%.18e')
                    string = s.getvalue().strip()
            elif data_type == "list":
                string = "\n".join(v)
            elif data_type in ("dict", "OrderedDict"):
                string = "\n".join(f"{k}\t{v}" for k, v in v.items())
            # elif data_type == "eval":
            #     string = v.__str__()
            elif data_type in "listlist":
                string = json.dumps(v)
            elif data_type in "listPath":
                string = "\n".join(path.as_posix() for path in v)
            else:
                raise Exception(f"unsupported data type\n{type(v)}")
            text += f"{string}\n\n"
        return text
    def save(self, save_path: Path, zip=False):
        text = self.to_text()
        if not zip:
            with open(save_path, "w") as f:
                f.write(text)
        else:
            with zipfile.ZipFile(save_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                z.writestr(save_path.with_suffix(".txt").name, text)
    def load(self, load_path: Path, zip=False):
        if not zip:
            with fopen(load_path, "r") as f:
                lines = f.readlines()
        else:
            with zipfile.ZipFile(load_path, "r") as z:
                with z.open(load_path.with_suffix(".txt").name) as f:
                    lines = [l.decode("utf-8") for l in f.readlines()]
        added_keys = []
        cur_k = None
        cur_v = None
        cur_data_type = None
        for l in lines:
            if l.startswith("# "):
                if cur_k is None:   pass
                else:   self.set_attribute(cur_k, cur_v[:-2], cur_data_type)    # 改行コードが２つ入るので除く
                m = re.match(r"^(.+)\((.+)\)$", l[2:].strip("\n"))
                cur_k = m.group(1)
                cur_data_type = m.group(2)
                cur_v = ""
                added_keys.append((cur_k, cur_data_type))
            else:
                cur_v += l
        else:
            self.set_attribute(cur_k, cur_v[:-2], cur_data_type)
        return added_keys
    def set_attribute(self, cur_k: str, cur_v: str, cur_data_type: str):
        # @property のデコーレータで定義され、かつ setter が無いものは、追加しない
        if isinstance(getattr(type(self), cur_k, None), property):
            if getattr(type(self), cur_k).fset is None:
                return
        # 追加していく
        if cur_data_type == "str":
            v = cur_v
        elif cur_data_type == "ndarray":
            # format
            first_value = cur_v.split("\n")[0].split()[0]
            if re.match(r"^[0-9]\.[0-9]{18}e\+[0-9]{2}$", first_value) is not None:
                v = np.array([list(map(float, line.split())) for line in cur_v.split("\n")])
            elif re.match(r"^[0-9]+$", first_value) is not None:
                v = np.array([list(map(int, line.split())) for line in cur_v.split("\n")])
            else:
                raise Exception(f"error: {cur_v}")
        elif cur_data_type == "list":
            v = self.convert_to_number_if_possible(cur_v.split("\n"), method="all")
        elif cur_data_type == "dict":
            v = {l.split("\t")[0]:l.split("\t")[1] for l in cur_v.split("\n")}
        elif cur_data_type == "OrderedDict":
            v = OrderedDict([l.split("\t") for l in cur_v.split("\n")])
        # elif cur_data_type == "eval":
        #     v = eval(cur_v)
        elif cur_data_type == "listlist":
            v = json.loads(cur_v)
        elif cur_data_type == "listPath":
            v = list(map(Path, cur_v.split("\n")))
        elif cur_data_type == "df":
            string_io = io.StringIO(cur_v)
            v = pd.read_csv(string_io, sep="\t", index_col=0, dtype=str)
        else:
            raise Exception(f"unsupported data type\n{cur_data_type}")
        setattr(self, cur_k, v)
    def convert_to_number_if_possible(self, values, method):
        new_values = []
        for v in values:
            try:
                new_values.append(float(v))
            except:
                new_values.append(v)
        if (method == "all") and any(map(lambda x: not isinstance(x, float), new_values)):
            return values
        else:
            return new_values

class MyHeader():
    def __init__(self) -> None:
        self.header = f"{__appname__}: {__version__}\n{__comment__}"
        self.datetime = datetime.now()
    def get_version(self, key):
        m = re.search(fr"\n{key}\: ([a-z]+_[0-9\.]+)\n", "\n" + self.header + "\n")
        if m is not None:
            return m.group(1)
        else:
            return None

class MyFastQ(dict):
    maximum_q_score_allowed = 41
    def __init__(self, path=None):
        super().__init__()
        if path is not None: # for deep copy
            self.path = [Path(path)]
            with open(self.path[0].as_posix(), "r") as f:
                fastq_txt = f.readlines()
            # check
            self.N_seq, mod = divmod(len(fastq_txt), 4)
            assert mod == 0
            # register
            for i in range(self.N_seq):
                fastq_id = fastq_txt[4 * i].strip()
                seq = fastq_txt[4 * i + 1].strip()
                p = fastq_txt[4 * i + 2].strip()
                q_scores = [ord(q) - 33 for q in fastq_txt[4 * i + 3].strip()]
                q_scores = [min(q, self.maximum_q_score_allowed) for q in q_scores]
                assert p == "+"
                assert len(seq) == len(q_scores)
                self[fastq_id] = [seq, q_scores]
        else:
            self.path = None
    @property
    def combined_name_stem(self):
        return "_".join(p.stem for p in self.path)
    def get_read_lengths(self):
        return np.array([len(v[0]) for v in self.values()])
    def get_q_scores(self):
        q_scores = []
        for v in self.values():
            q_scores.extend(v[1])
        return np.array(q_scores)
    def get_new_fastq_id(self, k):
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
            new_k = self.get_new_fastq_id(k)
            self[new_k] = v
    def get_partially_rc_subset(self, query_id_rc_list):
        fastq_sub = MyFastQ()
        fastq_sub.path = self.path
        for query_id, make_it_rc in query_id_rc_list:
            if not make_it_rc:
                fastq_sub[query_id] = self[query_id]
            else:
                query_seq, q_scores = self[query_id]
                fastq_sub[query_id] = [MySeq.make_it_reverse_complement(query_seq), q_scores[::-1]]
        return fastq_sub
    @staticmethod
    def combine(fastq_list: list):
        assert len(fastq_list) > 0
        combined_fastq = copy.deepcopy(fastq_list[0])
        path = fastq_list[0].path
        if len(fastq_list) > 1:
            for fastq in fastq_list[1:]:
                combined_fastq.append(fastq)
                path.extend(fastq.path)
        combined_fastq.path = path
        return combined_fastq
    @property
    def my_hash(self):
        return hashlib.sha256(self.to_string().encode("utf-8")).hexdigest()
    def to_string(self):
        txt = ""
        for fastq_id, (seq, q_scores) in self.items():
            txt += f"{fastq_id}\n{seq}\n+\n{''.join(map(lambda x: chr(x + 33), q_scores))}\n"
        return txt#.strip()
    def export(self, save_path, overwrite=False):
        if not overwrite:
            save_path = new_file_path_wo_overlap(file_path=save_path)
        with open(save_path, "w") as f:
            f.write(self.to_string())

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

class MySeq():
    def __init__(self, seq) -> None:
        self._seq = seq.upper() # this will never be changed
        self.seq = seq.upper()
        self.offset = 0
    def set_offset(self, offset):   # accept negative values
        """
        Example: Set offset 3 for 
            AAAGGGGG    len = 8
         -> GGGGGAAA    start_idx_after_offset = len - offset
        Example: Set offset -3 for 
            GGGGGAAA    len = 8
         -> AAAGGGGG    start_idx_after_offset = (len - offset)%len
            """
        self.offset = offset
        self.seq = self.set_offset_core(self._seq, offset)
    @staticmethod
    def set_offset_core(seq, offset):
        return seq[offset:] + seq[:offset]
    def __iter__(self):
        yield from self.seq
    def __len__(self):
        return len(self.seq)
    def __str__(self) -> str:
        return self.seq
    def __getitem__(self, k):
        return self.seq[k]
    def __add__(self, k: str):  # offset が 0 じゃない場合は慎重に使って
        return self.seq + k
    def __iadd__(self, k):      # offset が 0 じゃない場合は慎重に使って
        self.seq += k
        return self
    def __delitem__(self, idx):
        if idx < 0:
            idx += len(self.seq)
        assert 0 <= idx < len(self.seq)
        self.seq = self[:idx] + self[idx+1:]
    def reverse_complement(self):
        return self.__class__(self.make_it_reverse_complement(self.seq))
    @staticmethod
    def make_it_reverse_complement(seq):
        return str(Seq(seq).reverse_complement())

class MyRefSeq(MySeq):
    allowed_snapgene_extensions = [".dna"]
    allowed_fasta_extensions = [".fa", ".fasta", ".fas"]
    @classmethod
    @property
    def allowed_plasmid_map_extensions(cls):
        allowed_plasmid_map_extensions = []
        allowed_plasmid_map_extensions.extend(cls.allowed_snapgene_extensions)
        allowed_plasmid_map_extensions.extend(cls.allowed_fasta_extensions) 
        return allowed_plasmid_map_extensions
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.suffix in self.allowed_snapgene_extensions:
            snapgene_dict = snapgene_file_to_dict(self.path.as_posix())
            # seqrecord = snapgene_file_to_seqrecord(self.path.as_posix())
            assert snapgene_dict["isDNA"]
            self.topology = snapgene_dict["dna"]["topology"]
            self.strandedness = snapgene_dict["dna"]["strandedness"]
            self.length = snapgene_dict["dna"]["length"]
            seq = snapgene_dict["seq"]
            if self.topology != "circular":
                print(f"WARNING: {self.path.name} is not circular!")
            assert self.strandedness == "double"
            assert self.length == len(seq)
        elif self.path.suffix in self.allowed_fasta_extensions:
            with open(self.path.as_posix(), 'r') as f:
                seq=''
                for line in f.readlines():
                    if line[0] != '>':
                        seq += line.strip()
            self.topology = "circular"
            self.strandedness = "double"
            self.length = len(seq)
        else:
            raise Exception(f"Unsupported type of sequence file: {self.path}")
        super().__init__(seq)
    @property
    def my_hash(self):
        return hashlib.sha256(self.seq.encode("utf-8")).hexdigest()
    def save_as(self, save_path: Path):
        assert save_path.suffix in self.allowed_fasta_extensions
        with open(save_path, "w") as f:
            f.write(f">{self.path.name}\n{self.seq}")

class AlignmentBase():
    @staticmethod
    def print_alignment(ref_seq, query_seq, my_cigar):
        try:
            ref = ""
            query = ""
            r_idx = 0
            q_idx = 0
            for i, c in enumerate(my_cigar):
                if c in "=X":
                    if c == "=":
                        assert ref_seq[r_idx] == query_seq[q_idx]
                    elif c == "X":
                        assert ref_seq[r_idx] != query_seq[q_idx]
                    ref += ref_seq[r_idx]
                    query += query_seq[q_idx]
                    r_idx += 1
                    q_idx += 1
                elif c in "I":
                    ref += "-"
                    query += query_seq[q_idx]
                    q_idx += 1
                elif c in "S":
                    ref += " "
                    query += query_seq[q_idx]
                    q_idx += 1
                elif c in "D":
                    ref += ref_seq[r_idx]
                    query += "-"
                    r_idx += 1
                elif c in "H":
                    ref += ref_seq[r_idx]
                    query += " "
                    r_idx += 1
                else:
                    raise Exception(f"unknown cigar {c}")
        except:
            print("ERROR")
            print(ref_seq)
            print(query_seq)
            print(my_cigar)
            quit()
        print(ref)
        print(query)
        print(my_cigar)
    @staticmethod
    def assert_alignment(ref_seq, query_seq, my_cigar):
        r_idx = 0
        q_idx = 0
        for i, c in enumerate(my_cigar):
            if c in "=X":
                if c == "=":
                    assert ref_seq[r_idx] == query_seq[q_idx]
                elif c == "X":
                    assert ref_seq[r_idx] != query_seq[q_idx]
                r_idx += 1
                q_idx += 1
            elif c in "IS":
                q_idx += 1
            elif c in "DH":
                r_idx += 1
            else:
                raise Exception(f"unknown cigar {c}")

class MyCigarBase():
    @staticmethod
    def cigar_to_my_cigar(cigar_str):
        return "".join([ L for N, L in re.findall('(\d+)(\D)', cigar_str) for i in range(int(N)) ])
    @staticmethod
    def generate_cigar_iter(my_cigar):
        return re.findall(r"((.)\2*)", my_cigar)

class MyTempFiles():
    def __init__(self, suffix_list: List[str]=[]) -> None:
        assert all(suffix.startswith(".") for suffix in suffix_list)
        self.suffix_list = suffix_list
        # 一時ファイルを作成、ベースのファイルパスを取得
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            self.temp_file_path_base = Path(temp_file.name)
        self.temp_file_paths = []
        # 一時ファイルが削除されるタイミングを制御可能: self.KEEP == True なら self.__del__ で、False なら self.__exit__ で制御される
        self.KEEP = False
        # atribute への登録、一時ファイルの作成
        for suffix in self.suffix_list:
            temp_file_path = self.temp_file_path_base.with_suffix(suffix)
            setattr(self, f'temp{suffix.replace(".", "_")}_file_path', temp_file_path)
            self.temp_file_paths.append(temp_file_path)
            # 空ファイル作成
            with open(temp_file_path, "wb") as f:
                pass
    # 指定したパスへの保存 (self.suffix_list を添付した形でファイルが保存される)
    def save(self, save_path_base):
        """
        example:
            self.suffix_list = [".ab", ".cd"]
            save_path_base = "/path/your_file_name.ext"
        then
            "/path/your_file_name.ext.ab"
            "/path/your_file_name.ext.cd"
        will be saved.
        """
        save_path_base = Path(save_path_base).as_posix()
        for temp_file_path in self.temp_file_paths:
            shutil.copy(temp_file_path, save_path_base + re.match(fr"{self.temp_file_path_base}(\..+)", temp_file_path.as_posix()).group(1))
    # True の場合、インスタンスが削除される (by self.__del__) まで一時ファイルが削除されない
    def keep(self, keep):
        self.KEEP = keep
    def delete_temp_files(self):
        # 一時的なファイルたちを削除
        self.temp_file_path_base.unlink()
        for temp_file_path in self.temp_file_paths:
            temp_file_path.unlink()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.KEEP:
            self.delete_temp_files()
        # Trueを返すと例外が抑制され、Falseを返すと例外が再発生
        return False
    def __del__(self):
        if self.KEEP:
            self.delete_temp_files()

class MyTQDM(tqdm.tqdm):
    def __init__(self, *keys, **kwargs):
        super().__init__(*keys, **kwargs)
        self.offset_value = 0
    def set_value(self, value):
        self.n = value + self.offset_value
        self.last_print_n = value + self.offset_value
        self.update(0)

#####################
# GENERAL FUNCTIONS #
#####################
def assert_parent_directory(dir_path: Path):
    if not dir_path.parent.exists():
        raise FileNotFoundError(f"No such file or directory: {dir_path}")

def new_dir_path_wo_overlap(dir_path_base: Path, spacing="_"):
    assert_parent_directory(dir_path_base)
    dir_path_output = dir_path_base
    i = 0
    while dir_path_output.exists():
        i += 1
        dir_path_output = dir_path_base.parent / f"{dir_path_base.name}{spacing}{i}"
    return dir_path_output
def new_file_path_wo_overlap(file_path: Path, spacing="_"):
    assert_parent_directory(file_path)
    file_path_output = file_path
    i = 0
    while file_path_output.exists():
        i += 1
        file_path_output = (file_path.parent / f"{file_path.stem}{spacing}{i}").with_suffix(file_path.suffix)
    return file_path_output

def key2argkey(key):
    if key == "mismatch_score":
        key = "mis_match_score"
    return ''.join(i[0] for i in key.split('_'))



