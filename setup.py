# -*- coding: utf-8 -*-

"""
REF
https://qiita.com/c60evaporator/items/e1ecccab07a607487dcf#テスト用pypiへのライブラリアップロード
https://zenn.dev/sikkim/articles/490f4043230b5a
https://hack.nikkei.com/blog/advent20211225/#Cythonを含んだmoduleのinstall例

$ pip install pipreqs
$ pipreqs .
-> manually pasted into INSTALL_REQUIREMENTS with some modification
$ pip install twine
$ pip install wheel
$ python setup.py sdist
-> `dist`, `savemoney.egg-info` directories will be created.
$ python setup.py bdist_wheel
-> `build` directory will be created.
$ twine upload --repository testpypi dist/*

To install parasail, 
$ brew install libtool
$ brew install automake
might be required beforehand.

"""

import re
from pathlib import Path
from setuptools import setup, find_packages, Extension

try:
    import numpy as np
except ImportError:
    from setuptools import dist
    dist.Distribution().fetch_build_eggs(['numpy>=1.26.2'])
    import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    from setuptools import dist
    dist.Distribution().fetch_build_eggs(['Cython>=3.0.7'])
    from Cython.Build import cythonize

#############################################################
#############################################################

test_version = ""
root_dir = Path(__file__).parent.resolve()
package_name = "savemoney"
sub_package_names = [f"{package_name}.{sub_package_name}" for sub_package_name in find_packages(package_name)]

class About():
    def __init__(self, about_path, keys) -> None:
        self.keys = keys
        with open(about_path) as f:
            text = f.read()
        for key in keys:
            m = re.search(fr'__{key}__\s*=\s*[\'\"](.+?)[\'\"]', text)
            setattr(self, key, m.group(1))
    def __str__(self) -> str:
        txt = ""
        for key in self.keys:
            txt += getattr(self, key) + "\n"
        return txt.strip()
    @property
    def version_pypi(self):
        return re.match(r"ver_([0-9.]+)", self.version).group(1) + test_version   # version format must be "x.x.x" etc.

about = About(
    root_dir / package_name / '__about__.py', 
    [
        "copyright", 
        "version", 
        "license", 
        "author", 
        "email", 
        "url", 
        "appname", 
        "comment", 
        "description", 
    ]
)

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

INSTALL_REQUIREMENTS = [
    "pandas>=1.5.3", 
    "parasail>=1.3.4", 
    "Pillow>=9.4.0", 
    "PuLP>=2.7.0", 
    "scipy>=1.11.4", 
    # "setuptools>=68.2.2", 
    "snapgene_reader>=0.1.20", 
    "tqdm>=4.66.1", 
    # "Bio>=1.6.0",         # ?
    "biopython>=1.83", 
    "Cython>=3.0.7", 
    "matplotlib>=3.7.1", 
    "numpy>=1.23.5", 
    # "oauth2client>=4.1.3", 
    # "pysam>=0.22.0", 
    "pyspoa>=0.2.1"
]

PYTHON_REQUIRES = ">=3.10"

setup(
    name = package_name, 
    packages = [package_name] + sub_package_names, 
    ###
    copyright = about.copyright, 
    version = about.version_pypi, 
    license = about.license, 
    author = about.author, 
    author_email = about.email, 
    maintainer = about.author, 
    maintainer_email = about.email, 
    url = about.url, 
    description = about.description, 
    ###
    long_description=long_description, 
    long_description_content_type='text/markdown',
    keywords = (
        "plasmid, "
        "whole-plasmid, "
        "sequencing, "
        "alignment, "
        "multiplexing, "
        "barcode-free, "
        "bayesian analysis, "
        "prior information, "
    ), 

    install_requires = INSTALL_REQUIREMENTS, 
    python_requires = PYTHON_REQUIRES, 
    
    classifiers = [
        'Development Status :: 4 - Beta', 
        'License :: Free for non-commercial use', 
        "Programming Language :: Cython", 
        'Programming Language :: Python', 
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.10', 
        'Topic :: Scientific/Engineering :: Bio-Informatics', 
    ],
    ext_modules = cythonize([Extension("savemoney.modules.cython_functions.alignment_functions", ["savemoney/modules/cython_functions/alignment_functions.pyx"])], compiler_directives = {"language_level": 3, "embedsignature": True}), 
    include_dirs = np.get_include(),    # cython に numpy の path を通す
    include_package_data = True         # 配布時にpyxのファイルを含める (MANIFEST.in で定義)
)



