# kiez-benchmarking
Configurations and results of kiez paper

## Pre-calculated results
We make our results available in `results/max_all.csv`. 
Our plots can be created with:
```
poetry run python kiezbenchmarking/experiment/create_plots.py --use-csv output
```

## Reproduce results
We use [SEML](https://github.com/TUM-DAML/seml) to keep track of results. If you want to reproduce our results you will have to do the same. Please refer to their instructions to set everything up.
In the `pyproject.toml` provide the path to the `kiez` library.
Install the necessary packages inside a conda env (because seml wants you to use a conda env):
```
poetry install
```
To run the experiments you would use seml:
```
seml [db_name] add configs/[path_to_config]
seml [db_name] run
```
Which starts a SLURM job with all the experiments and saves the results in your MongoDB using Sacred.
