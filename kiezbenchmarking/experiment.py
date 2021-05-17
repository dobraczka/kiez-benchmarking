import copy
import logging
import os
import time

import pymongo
from kiez import NeighborhoodAlignment
from kiez.analysis import hubness_score
from kiez.evaluate.eval_metrics import hits
from kiez.io.data_loading import from_openea
from sacred import Experiment, Ingredient
from sacred.observers import MongoObserver

uri = os.environ["MONGODB_URI"]
db_name = os.environ["MONGODB_NAME"]

client2 = pymongo.MongoClient(uri, port=27017)

logging.basicConfig(filename="kiez_output.txt", level=logging.INFO)
logger = logging.getLogger("kiez")

data_ingredient = Ingredient("dataset")
ex = Experiment("kiez", ingredients=[data_ingredient])
ex.observers.append(MongoObserver(url=uri, db_name=db_name))
ex.logger = logger


@data_ingredient.capture
def load_data(emb_dir_path, kg_path):
    return from_openea(emb_dir_path, kg_path)


@ex.automain
def run(emb_dir_path, kg_path, neigh_alig_arguments, _run):
    emb1, emb2, kg1_ids, kg2_ids, gold = load_data(emb_dir_path, kg_path)
    args = copy.deepcopy(neigh_alig_arguments)
    start_time = time.time()
    align = NeighborhoodAlignment(**args)
    align.fit(emb1, emb2)
    dist, ind = align.kneighbors(return_distance=True)
    execution_time = time.time() - start_time
    _run.log_scalar("execution time", execution_time)
    res = hits(ind, gold)
    for k, k_val in res.items():
        _run.log_scalar(f"hits@{k}", k_val)
    hub_est = hubness_score(dist, k=10)
    for k, k_val in hub_est.items():
        _run.log_scalar(k, k_val)
