# SAVEMONEY

*Simple Algorithm for Very Efficient Multiplexing of Oxford Nanopore Experiments for You!*



<p align="center"><img src="https://github.com/MasaakiU/MultiplexNanopore/raw/master/resources/logo/SAVEMONEY_logo.png" height="250"/></p>

## Overview

SAVEMONEY guides researchers to mix multiple plasmids for submission as a single sample to a commercial long-read sequencing service (e.g., Oxford Nanopore Technology), reducing overall sequencing costs while maintaining fidelity of sequencing results. Following is the outline of the procedure:

- <a href="#Step1">**Step 1. pre-survey**</a> takes plasmid maps as inputs and guide users which groupings of plasmids is optimal.
- <a href="#Step2">**Step 2. submit samples**</a> according to the output of pre-survey.
- <a href="#Step3">**Step 3. post-analysis**</a> execute computational deconvolution of the obtained results, and generate a consensus sequence for each plasmid constituent within the sample mixture. This step must be run separately for each sample mixture.
- An optional third step, <a href="#Step4">**Step 4. visualization of results (optional)**</a> provides a platform for the detailed examination of the alignments and consensus generated in the post-analysis.

<p align="center"><img src="https://github.com/MasaakiU/MultiplexNanopore/raw/master/resources/figures/Fig1_20230313_margin.png" width="500"/></p>

The algorithm permits mixing of six (or potentially even more) plasmids for sequencing with Oxford Nanopore Technology (e.g., Plasmidsaurus services) and permits mixing of plasmids with as few as two base differences. For more information, please check out our publication (coming soon).

## SAVEMONEY via Google Colab!

- [SAVEMONEY](https://colab.research.google.com/github/MasaakiU/MultiplexNanopore/blob/master/colab/MultiplexNanopore.ipynb)
- SAVEMONEY_batch (coming soon!)

## SAVEMONEY for local environment

coming soon via pip...

