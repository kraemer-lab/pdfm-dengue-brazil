# Dengue Forecasting with Foundation Models

This repository contains the code used to generate the figures and tables for the manuscript:

**Foundation models capture human and environmental context to improve dengue forecasting**

[Link to manuscript to be added upon availability.]

## Repository structure

Each figure and table directory contains scripts used to process fitted model outputs into summary data and generate the corresponding figures or tables.

The `figure_*` directories contain code for generating the main and supplementary figures. The `table_*` directories contain code for generating supplementary tables.

## Data availability

The processing scripts require the underlying study datasets and fitted model outputs, which can be substantial in size and may be subject to third-party access, redistribution or licensing restrictions. These datasets are therefore not included in this repository.

The processing scripts are retained to document how the underlying model outputs were transformed into the summary data used in the manuscript.

**Precomputed summary files are included** in the corresponding directories. These allow the figures and tables to be regenerated without access to the complete underlying datasets or fitted model outputs.

Access to the original data sources is described in the manuscript's Data and Code Availability section.

## Installation

The code was developed in Python and uses the [Thucia](https://github.com/kraemer-lab/Thucia) forecasting framework.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file specifies the required Python dependencies, including the Thucia version used for the analysis.

## Generating figures and tables

The primary script in each directory is named `generate_figX.py` or `generate_tableX.py`, where `X` identifies the corresponding figure or table.

These scripts load the precomputed summary files and generate the publication figures and tables.

For example, to regenerate Figure 1:

```bash
cd figure_1
python generate_fig1.py
```

This generates:

```text
figure_1/figures/figure1.png
figure_1/figures/figure1.eps
```

Other figures and tables can be generated similarly by running the corresponding scripts in their directories.

Some directories contain additional preparation scripts used to construct the summary data from the underlying fitted model outputs. These are provided for methodological transparency but require access to the original study datasets.

## Forecasting framework

The full forecasting framework is available separately in the [Thucia repository](https://github.com/kraemer-lab/Thucia).

## Citation

If you use this code, please cite the associated manuscript:

Brittain, J.-S., et al. *Foundation models capture human and environmental context to improve dengue forecasting.*

[Publication details to be added.]

