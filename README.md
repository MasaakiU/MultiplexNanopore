<p align="center"><img src="https://github.com/MasaakiU/MultiplexNanopore/raw/master/resources/logo/SAVEMONEY_logo_with_letter.png"/></p>

*Simple Algorithm for Very Efficient Multiplexing of Oxford Nanopore Experiments for You!*

# Overview

SAVEMONEY guides researchers to mix multiple plasmids for submission as a single sample to a commercial long-read sequencing service (e.g., Oxford Nanopore Technology), reducing overall sequencing costs while maintaining fidelity of sequencing results. Following is the outline of the procedure:

- **Step 1. pre-survey** takes plasmid maps as inputs and guides users which groupings of plasmids is optimal.
- **Step 2. submit samples** according to the output of pre-survey.
- **Step 3. post-analysis** execute computational deconvolution of the obtained results, and generate a consensus sequence for each plasmid constituent within the sample mixture. This step must be run separately for each sample mixture.
- An optional third step, **Step 4. visualization of results (optional)** provides a platform for the detailed examination of the alignments and consensus generated in the post-analysis.

<p align="center"><img src="https://github.com/MasaakiU/MultiplexNanopore/raw/master/resources/figures/Fig1_20230313_margin.png" width="500"/></p>

The algorithm permits mixing of six (or potentially even more) plasmids for sequencing with Oxford Nanopore Technology (e.g., Plasmidsaurus services) and permits mixing of plasmids with as few as two base differences. For more information, please check out our publication (coming soon).

# SAVEMONEY via Google Colab!

- [SAVEMONEY](https://colab.research.google.com/github/MasaakiU/MultiplexNanopore/blob/master/colab/MultiplexNanopore.ipynb)
- SAVEMONEY_batch (coming soon!)

# SAVEMONEY for local environment

## Requirements

Verified on macOS, Linux, and Windows10

- Python 3.10 or later
- One of the following C++ compiler (though I don't know the minimum required version number)
  - [Clang 14.0.0](https://clang.llvm.org)
  - [GCC 12.2.0](https://gcc.gnu.org)
  - [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (for Windows)
- biopython>=1.81
- pandas>=2.1.4
- parasail==1.1.11    # newer version does not work with savemoney for now
- Pillow>=10.1.0
- PuLP>=2.7.0
- scipy>=1.11.4
- snapgene_reader>=0.1.20
- tqdm>=4.66.1
- Cython>=3.0.7
- matplotlib>=3.8.2
- numpy>=1.26.2

## Installation

SAVEMONEY is available via pip.

```shell
pip install savemoney
```

If installation via pip fails, please check the requirements above. If any of the package conflicts with those already present in your environment, I recommend creating a new virtual environment. 

If C++ compiler does not exist, install Xcode Command Line Tools using the following command (for macOS):

```shell
xcode-select --install
```

or download [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (for Windows).

## Quick usage

SAVEMONEY can be executed either in the python script or via command line.

### Execute SAVEMONEY in python script

To import and execute SAVEMONEY in the python script. Follow the example below:

```python
import savemoney
savemoney.pre_survey("path_to_sequence_directory", "save_directory", **kwargs)
savemoney.post_analysis("path_to_sequence_directory", "save_directory", **kwargs)
```

All of the plasmid map files with `*.dna` and `.fasta` extension (and in addition `*.fastq` files for post analysis) in the `path_to_sequence_directory` will be used for the analysis. Results will be generated in the `save_directory`. `kwargs` are optional parameters through which you can optimize the analysis:

```python
# pre-survey
kwargs = {
    'distance_threshold':   5,  # main parameter to be changed
    'number_of_groups':     1,  # main parameter to be changed
    'gap_open_penalty':     3,  # alignment parameter
    'gap_extend_penalty':   1,  # alignment parameter
    'match_score':          1,  # alignment parameter
    'mismatch_score':      -2,  # alignment parameter
}

# post-analysis
kwargs = {
    'score_threshold':    0.3,   # main parameter to be changed 
    'gap_open_penalty':     3,   # alignment parameter
    'gap_extend_penalty':   1,   # alignment parameter
    'match_score':          1,   # alignment parameter
    'mismatch_score':      -2,   # alignment parameter
    'error_rate':     0.00001,   # prior probability for Bayesian analysis
    'del_mut_rate':  0.0001/4,   # prior probability for Bayesian analysis # e.g. "A -> T, C, G, del"
    'ins_rate':       0.00001,   # prior probability for Bayesian analysis
    'window':             160,   # maximum detectable length of repetitive sequences when wrong plasmid maps are provided: if region of 80 nt is repeated adjascently two times, put the value of 160
}
```

For the meaning of these parameters, please refer to the [SAVEMONEY Google Colab page](https://colab.research.google.com/github/MasaakiU/MultiplexNanopore/blob/master/colab/MultiplexNanopore.ipynb) or the reference below.

### Execute SAVEMONEY via command line

SAVEMONEY can also be executed via command line:

```shell
python -m savemoney.pre_survey path_to_sequence_directory save_directory
python -m savemoney.post_analysis path_to_sequence_directory save_directory
```

Parameters can be specified as follows:

```shell
# pre-survey
python -m savemoney.pre_survey -h
usage: __main__.py [-h] [-gop GOP] [-gep GEP] [-ms MS] [-mms MMS] [-dt DT] [-nog NOG] plasmid_map_dir_paths save_dir_base
positional arguments:
  plasmid_map_dir_paths path to plasmid map_directory
  save_dir_base         save directory path
options:
  -h, --help            show this help message and exit
  -gop GOP              gap_open_penalty, optional, default_value = 3
  -gep GEP              gap_extend_penalty, optional, default_value = 1
  -ms MS                match_score, optional, default_value = 1
  -mms MMS              mismatch_score, optional, default_value = -2
  -dt DT                distance_threshold, optional, default_value = 5
  -nog NOG              number_of_groups, optional, default_value = 1

# post-analysis
python -m savemoney.post_analysis -h
usage: __main__.py [-h] [-gop GOP] [-gep GEP] [-ms MS] [-mms MMS] [-st ST] [-er ER] [-dmr DMR] [-ir IR] sequence_dir_paths save_dir_base
positional arguments:
  sequence_dir_paths  sequence_dir_paths
  save_dir_base       save directory path
options:
  -h, --help          show this help message and exit
  -gop GOP            gap_open_penalty, optional, default_value = 3
  -gep GEP            gap_extend_penalty, optional, default_value = 1
  -ms MS              match_score, optional, default_value = 1
  -mms MMS            mismatch_score, optional, default_value = -2
  -st ST              score_threshold, optional, default_value = 0.3
  -er ER              error_rate, optional, default_value = 0.0001
  -dmr DMR            del_mut_rate, optional, default_value = 2.5e-05
  -ir IR              ins_rate, optional, default_value = 0.0001
  -w W                window, optional, default_value = 160
```

### Output

The interpretation of output files are described on [SAVEMONEY Google Colab page](https://colab.research.google.com/github/MasaakiU/MultiplexNanopore/blob/master/colab/MultiplexNanopore.ipynb#InterpretationOfResults) in details. Other than that, you can visualize consensus alignment results by using `your_plasmid_name.ca` file generated by SAVEMONEY.

From python script: 

```python
import savempney
savemoney.show_consensus(consensus_alignment_path, center=2000, seq_range=50, offset=0)
```

From command line: 

```shell
python -m savemoney.show_consensus path_to_consensus_alignment_file
```

Parameters can be specified as follows:

```shell
python -m savemoney.show_consensus -h
usage: __main__.py [-h] [--center CENTER] [--seq_range SEQ_RANGE] [--offset OFFSET] consensus_alignment_path
positional arguments:
  consensus_alignment_path  path to consensus_alignment (*.ca) file
options:
  -h, --help            show this help message and exit
  --center CENTER       center, optional, default_value = 2000
  --seq_range SEQ_RANGE seq_range, optional, default_value = 50
  --offset OFFSET       offset, optional, default_value = 0
```

Conversion of consensus alignment results (`*.ca`) to `*.bam` and `*.fastq` format is also supported. To do this, type the following code in a python script:

```python
import savemoney
savemoney.ca2bam(consensus_alignment_path)
```

From command line, type the following commnad:

```shell
python -m savemoney.ca2bam path_to_consensus_alignment_file
```

# Reference

[Uematsu M., Baskin J. M., "Barcode-free multiplex plasmid sequencing using Bayesian analysis and nanopore sequencing." *eLife*. **2023**; 12: RP88794](https://doi.org/10.7554/eLife.88794.1)

[Slide from Weill Institute Science Workshop, May 22, 2023](https://github.com/MasaakiU/MultiplexNanopore/blob/master/resources/slides/20230522_Weill-Institute-Science-Workshop.pdf)

