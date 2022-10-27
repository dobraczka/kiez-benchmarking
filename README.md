# kiez-benchmarking
Configurations and results of kiez paper

## Install dependencies
You can find the necessary depencies in the `pyproject.toml`.
If you have [poetry](https://github.com/python-poetry/poetry) installed simply execute:
```
poetry install
```

## Pre-calculated knowledge graph embeddings
The knowledge graph embeddings produced for and used in our study are available via [Zenodo](https://zenodo.org/record/6258620).

## Pre-calculated results
We make our results available in `results/max_all.csv`. 
Our plots can be created with:
```
poetry run python kiezbenchmarking/create_plots.py --use-csv output
```
We make a lot more plots available in `results/additional_report.pdf`.
To recreate this document:
```
poetry run python kiezbenchmarking/create_plots.py --extensive plot_dir
poetry run python kiezbenchmarking/create_report.py plot_dir results/
cd results
pdflatex additional_report.tex
```

## Reproduce results
It is easy to run a single experiment:
```
poetry run python kiezbenchmarking/experiment.py --embedding "AttrE" --dataset "D_W_15K_V1" --neighbors 50 faiss --candidates 100 --index-key Flat --no-gpu ls --method nicdm
```
This will automatically download any data if necessary.

You can also track your results with [wandb](https://wandb.ai/) using the `--use-wandb` flag, e.g.:

```
poetry run python kiezbenchmarking/experiment.py --embedding "AttrE" --dataset "D_W_15K_V1" --neighbors 50 --use-wandb faiss --candidates 100 --index-key Flat --no-gpu ls --method nicdm
```

This command shows you the necessary arguments to run an experiment:
```
poetry run python kiezbenchmarking/experiment.py --help
```
The individual nearest neighbor algorithm and hubness reduction method are declared via subcommand, for which you can also get help (after supplying the required arguments of the base command):
```
poetry run python kiezbenchmarking/experiment.py --embedding "AttrE" --dataset "D_W_15K_V1" --neighbors 50 faiss --help
```
or for the hubness reduction method:

```
poetry run python kiezbenchmarking/experiment.py --embedding "AttrE" --dataset "D_W_15K_V1" --neighbors 50 faiss --candidates 100 --index-key Flat --use-gpu False ls --help
```
