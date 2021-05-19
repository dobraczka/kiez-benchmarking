# kiez-benchmarking
Configurations and results of kiez paper

## Install dependencies
You can find the necessary depencies in the `pyproject.toml`.
If you have [poetry](https://github.com/python-poetry/poetry) installed simply execute:
```
poetry install
```

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
We use [SEML](https://github.com/TUM-DAML/seml) to keep track of results. If you want to reproduce our results you will have to do the same. Please refer to their instructions to set everything up.
In the `pyproject.toml` provide the path to the `kiez` library.
Install the necessary packages inside a conda env (because seml wants you to use a conda env):
```
conda create -n kiez python=3.7.1
poetry install
```
To run the experiments you would use seml:
```
# Queue the experiments
seml [db_name] add configs/[path_to_config]
# Run them
seml [db_name] run
```
Which starts a SLURM job with all the experiments and saves the results in your MongoDB using [Sacred](https://github.com/IDSIA/sacred).
