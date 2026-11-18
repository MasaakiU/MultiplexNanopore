"""Build SAVEMONEY's Cython extension; package metadata lives in pyproject.toml."""

from pathlib import Path
from runpy import run_path

from Cython.Build import cythonize
from setuptools import Extension, setup


about = run_path(str(Path(__file__).parent / "savemoney" / "__about__.py"))

setup(
    version=about["__version__"].removeprefix("ver_"),
    ext_modules=cythonize(
        [
            Extension(
                "savemoney.modules.cython_functions.alignment_functions",
                ["savemoney/modules/cython_functions/alignment_functions.pyx"],
                language="c++",
            )
        ],
        build_dir="build/cython",
        compiler_directives={"language_level": 3, "embedsignature": True},
    ),
)
