import copy
import logging
import os

import numpy as np
import seml
from kiez import NeighborhoodAlignment
from kiez.analysis import hubness_score
from kiez.evaluate.eval_metrics import hits
from kiez.io.data_loading import from_openea
from sacred import Experiment, Ingredient
from glob import glob

data_ingredient = Ingredient("dataset")
ex = Experiment("kiez", ingredients=[data_ingredient])
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@data_ingredient.capture
def load_data(emb_dir_path, kg_path):
    return from_openea(emb_dir_path, kg_path)


@ex.automain
def run(
    dataset_tuple,
    n_neighbors,
    algorithm,
    algorithm_params,
    hubness,
    leaf_size,
    metric,
    p,
    query_samples,
    target_samples,
    _run,
):
    result_dir = f"results/{_run.experiment_info['name']}/{_run._id}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    base_emb_dir_path, kg_path = dataset_tuple
    emb_dir_path = glob(base_emb_dir_path + "/*/")[0]
    logging.info("Received the following configuration:")
    logging.info(
        f"emb_dir_path: {emb_dir_path} , kg_path: {kg_path} , n_neighbors: {n_neighbors} , algorithm: {algorithm} , algo_params: {algorithm_params} , hubness: {hubness}, leaf_size: {leaf_size} , metric: {metric} , p: {p} "
    )
    had_to_remove = False
    if "nng" in "algorithm":
        for filename in glob("/dev/shm/skhubness*"):
            had_to_remove = True
            print(f"Removed {filename}")
            os.remove(filename)
    if had_to_remove:
        raise Exception("Had to remove nng indices")

    emb1, emb2, kg1_ids, kg2_ids, gold = load_data(emb_dir_path, kg_path)
    hub, hub_params = hubness
    if hub == "None":
        hub_algo = None
    else:
        hub_algo = hub
    algo_params = copy.deepcopy(algorithm_params)
    hubness_params = copy.deepcopy(hub_params)
    align = NeighborhoodAlignment(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        algorithm_params=algo_params,
        hubness=hub_algo,
        hubness_params=hubness_params,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
    )
    align.fit(emb1, emb2)
    dist, ind = align.kneighbors(return_distance=True)
    np.save(f"{result_dir}/dist.npy", dist)
    np.save(f"{result_dir}/ind.npy", ind)
    res = hits(ind, gold, k=[1, 5, 10, 25, 50])
    full_results = {}
    for k, k_val in res.items():
        full_results[f"hits@{k}"] = k_val
    robinhood = hubness_score(
        ind,
        query_samples=query_samples,
        target_samples=target_samples,
        k=50,
        return_value="robinhood",
    )
    full_results["robinhood"] = robinhood
    if "nng" in "algorithm":
        print(f"(Removing index from {align._index_source._index}")
        os.remove(align._index_source._index)
        print(f"(Removing index from {align._index_target._index}")
        os.remove(align._index_target._index)
    return full_results
