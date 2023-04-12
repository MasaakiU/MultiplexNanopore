# -*- coding: utf-8 -*-

import numpy as np
import io
import re
import json
import pandas as pd
import contextlib
from pathlib import Path
from pathlib import PosixPath   # 必要！
from collections import OrderedDict
from datetime import datetime

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
            elif data_type == "ndarray":
                with io.StringIO() as s:
                    np.savetxt(s, v)
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
    def save(self, save_path):
        text = self.to_text()
        with open(save_path, "w") as f:
            f.write(text)
    def load(self, load_path):
        added_keys = []
        with fopen(load_path, "r") as f:
            lines = f.readlines()
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
        if isinstance(getattr(type(self), cur_k, None), property):
            if getattr(type(self), cur_k).fset is None:
                return
        if cur_data_type == "str":
            v = cur_v
        elif cur_data_type == "ndarray":
            v = np.array([list(map(float, line.split())) for line in cur_v.split("\n")])
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
            from ast import literal_eval
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


