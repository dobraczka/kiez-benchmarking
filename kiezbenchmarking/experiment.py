import logging
from typing import Dict, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import pendulum
import pystow
import wandb
from kiez import Kiez
from kiez.evaluate.eval_metrics import hits
from kiez.hubness_reduction import (
    CSLS,
    DisSimLocal,
    HubnessReduction,
    LocalScaling,
    MutualProximity,
)
from kiez.io.data_loading import _seperate_common_embedding
from kiez.neighbors import NMSLIB, NNG, Annoy, Faiss, NNAlgorithm, SklearnNN
from sylloge import OpenEA

logger = logging.getLogger("kiez")

EMB_TO_URL = {
    "AttrE": "https://zenodo.org/record/6258620/files/AttrE.zip",
    "BootEA": "https://zenodo.org/record/6258620/files/BootEA.zip",
    "ConvE": "https://zenodo.org/record/6258620/files/ConvE.zip",
    "GCN_Align": "https://zenodo.org/record/6258620/files/GCN_Align.zip",
    "HolE": "https://zenodo.org/record/6258620/files/HolE.zip",
    "IMUSE": "https://zenodo.org/record/6258620/files/IMUSE.zip",
    "IPTransE": "https://zenodo.org/record/6258620/files/IPTransE.zip",
    "JAPE": "https://zenodo.org/record/6258620/files/JAPE.zip",
    "MultiKE": "https://zenodo.org/record/6258620/files/MultiKE.zip",
    "ProjE": "https://zenodo.org/record/6258620/files/ProjE.zip",
    "RotatE": "https://zenodo.org/record/6258620/files/RotatE.zip",
    "RSN4EA": "https://zenodo.org/record/6258620/files/RSN4EA.zip",
    "SimplE": "https://zenodo.org/record/6258620/files/SimplE.zip",
    "TransD": "https://zenodo.org/record/6258620/files/TransD.zip",
    "TransH": "https://zenodo.org/record/6258620/files/TransH.zip",
}


def _ids_to_dict(ids_df: pd.DataFrame, reorder=True) -> Dict:
    if reorder:
        ids_df = ids_df[["id", "uri"]]
    return dict(ids_df.values)


def _data_info(dataset: str) -> Tuple[str, str, str]:
    tokens = dataset.split("_")
    return "_".join(tokens[:2]), tokens[2], tokens[3]


def load_data(
    emb_name: str, dataset: str
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], Dict[int, str], Dict[str, str]]:
    graph_pair, size, version = _data_info(dataset)
    url = EMB_TO_URL[emb_name]
    ds_name = f"{graph_pair}_{size}_{version}"
    base_path = f"{emb_name}/{ds_name}"
    emb_path = f"{base_path}/ent_embeds.npy"
    kg1_ent_ids_path = f"{base_path}/kg1_ent_ids"
    kg2_ent_ids_path = f"{base_path}/kg2_ent_ids"

    with pystow.ensure_open_zip("kiez", url=url, inner_path=emb_path) as file:
        emb = np.load(file)

    csv_kwargs = dict(header=None, sep="\t", names=["uri", "id"])
    kg1_ids = _ids_to_dict(
        pystow.ensure_zip_df(
            "kiez", url=url, inner_path=kg1_ent_ids_path, read_csv_kwargs=csv_kwargs
        )
    )
    kg2_ids = _ids_to_dict(
        pystow.ensure_zip_df(
            "kiez", url=url, inner_path=kg2_ent_ids_path, read_csv_kwargs=csv_kwargs
        )
    )
    ds = OpenEA(graph_pair=graph_pair, size=size, version=version)
    ent_links = _ids_to_dict(ds.ent_links, reorder=False)
    return _seperate_common_embedding(emb, kg1_ids, kg2_ids, ent_links)


def run(
    emb_name: str,
    dataset: str,
    neighbors: int,
    algorithm: NNAlgorithm,
    hubness_reduction: HubnessReduction,
    use_wandb: bool,
):
    emb1, emb2, kg1_ids, kg2_ids, gold = load_data(emb_name, dataset)
    start_time = pendulum.now()
    kiez = Kiez(n_neighbors=neighbors, algorithm=algorithm, hubness=hubness_reduction)
    kiez.fit(emb1, emb2)
    dist, ind = kiez.kneighbors(return_distance=True)
    execution_time = pendulum.now() - start_time
    res = hits(ind, gold, k=[1, 5, 10, 25, 50])
    res = {f"hits@{k}": k_val for k, k_val in res.items()}
    click.echo(res)
    click.echo(
        f"Exection took: {execution_time.hours} hours {execution_time.minutes} minutes"
        f" {execution_time.seconds} seconds {execution_time.microseconds} microseconds"
    )
    if use_wandb:
        wandb.log(res)
        wandb.log({"time in s": execution_time.total_seconds()})


@click.group(chain=True, invoke_without_command=True)
@click.option(
    "--embedding",
    type=click.Choice(list(EMB_TO_URL.keys())),
    required=True,
)
@click.option(
    "--dataset",
    type=click.Choice(
        [
            "D_W_15K_V1",
            "D_W_15K_V2",
            "D_Y_15K_V1",
            "D_Y_15K_V2",
            "EN_DE_15K_V1",
            "EN_DE_15K_V2",
            "EN_FR_15K_V1",
            "EN_FR_15K_V2",
            "D_W_100K_V1",
            "D_W_100K_V2",
            "D_Y_100K_V1",
            "D_Y_100K_V2",
            "EN_DE_100K_V1",
            "EN_DE_100K_V2",
            "EN_FR_100K_V1",
            "EN_FR_100K_V2",
        ]
    ),
    required=True,
)
@click.option("--neighbors", type=int, required=True)
@click.option("--use-wandb/--no-wandb", type=bool, default=False)
def cli(embedding: str, dataset: str, neighbors: int, use_wandb: bool):
    pass


@cli.result_callback()
def process_pipeline(
    instances_with_args: List[Tuple[Union[NNAlgorithm, HubnessReduction], Dict]],
    embedding: str,
    dataset: str,
    neighbors: int,
    use_wandb: bool,
):
    nn_algo = None
    hubness_reduction = None
    config = {**click.get_current_context().params}
    for inst, args in instances_with_args:
        if isinstance(inst, NNAlgorithm):
            nn_algo = inst
            config.update(args)
        elif isinstance(inst, HubnessReduction):
            hubness_reduction = inst
            config.update(args)

    if use_wandb:
        wandb.init(project="kiez", config=config)
    run(
        emb_name=embedding,
        dataset=dataset,
        neighbors=neighbors,
        algorithm=nn_algo,
        hubness_reduction=hubness_reduction,
        use_wandb=use_wandb,
    )


@cli.command("faiss", help="Use the Faiss nearest neighbor algorithm")
@click.option("--candidates", type=int, required=True)
@click.option("--metric", type=click.Choice(["l2", "euclidean"], case_sensitive=False))
@click.option("--index-key", type=str, default=None)
@click.option("--index-param", type=str, default=None)
@click.option("--use-gpu/--no-gpu", type=bool, default=True)
def create_faiss(
    candidates: int, metric: str, index_key: str, index_param: str, use_gpu: bool
) -> Tuple[Faiss, Dict]:
    if metric is None:
        metric = "l2"
    return (
        Faiss(
            n_candidates=candidates,
            metric=metric,
            index_key=index_key,
            index_param=index_param,
            use_gpu=use_gpu,
        ),
        click.get_current_context().params,
    )


@cli.command("nmslib", help="Use the NMSLIB HNSW nearest neighbor algorithm")
@click.option("--candidates", type=int, required=True)
@click.option(
    "--metric",
    type=click.Choice(
        [
            "euclidean",
            "l2",
            "minkowski",
            "squared_euclidean",
            "sqeuclidean",
            "cosine",
            "cosinesimil",
        ]
    ),
)
@click.option("--m", type=int, default=16)
@click.option("--post-processing", type=int, default=2)
@click.option("--ef-construction", type=int, default=200)
@click.option("--n-jobs", type=int, default=1)
def create_nmslib(
    candidates: int,
    metric: str,
    m: int,
    post_processing: int,
    ef_construction: int,
    n_jobs: int,
) -> Tuple[NMSLIB, Dict]:
    if metric is None:
        metric = "euclidean"
    return (
        NMSLIB(
            n_candidates=candidates,
            metric=metric,
            M=m,
            post_processing=post_processing,
            ef_construction=ef_construction,
            n_jobs=n_jobs,
        ),
        click.get_current_context().params,
    )


@cli.command("nng", help="Use the NNG nearest neighbor algorithm")
@click.option("--candidates", type=int, required=True)
@click.option(
    "--metric",
    type=click.Choice(
        [
            "manhattan",
            "L1",
            "euclidean",
            "L2",
            "minkowski",
            "sqeuclidean",
            "Angle",
            "Normalized Angle",
            "Cosine",
            "Normalized Cosine",
            "Hamming",
            "Jaccard",
        ]
    ),
)
@click.option("--index-dir", type=str, default="auto")
@click.option("--edge-size-for-creation", type=int, default=80)
@click.option("--edge-size-for-search", type=int, default=40)
@click.option("--epsilon", type=float, default=0.1)
@click.option("--n-jobs", type=int, default=1)
def create_nng(
    candidates: int,
    metric: str,
    index_dir: str,
    edge_size_for_creation: int,
    edge_size_for_search,
    epsilon: float,
    n_jobs: int,
) -> Tuple[NNG, Dict]:
    if metric is None:
        metric = "euclidean"
    return (
        NNG(
            n_candidates=candidates,
            metric=metric,
            index_dir=index_dir,
            edge_size_for_creation=edge_size_for_creation,
            edge_size_for_search=edge_size_for_search,
            epsilon=epsilon,
            n_jobs=n_jobs,
        ),
        click.get_current_context().params,
    )


@cli.command("sklearn", help="Use the sci-kit learn nearest neighbor algorithm family")
@click.option("--candidates", type=int, required=True)
@click.option(
    "--algorithm", type=click.Choice(["auto", "ball_tree", "kd_tree", "brute"])
)
@click.option("--leaf-size", type=int, default=30)
@click.option(
    "--metric",
    type=click.Choice(
        ["cityblock", "cosine", "euclidean", "haversine", "l1", "l2", "manhattan"]
    ),
)
@click.option("--p", type=str, default=2)
@click.option("--n-jobs", type=str, default=None)
def create_sklearn(
    candidates: int, algorithm: str, leaf_size: int, metric: str, p: str, n_jobs: str
) -> Tuple[SklearnNN, Dict]:
    if metric is None:
        metric = "euclidean"
    return (
        SklearnNN(
            n_candidates=candidates,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            n_jobs=n_jobs,
        ),
        click.get_current_context().params,
    )


@cli.command("annoy", help="Use the Annoy nearest neighbor algorithm")
@click.option("--candidates", type=int, required=True)
@click.option(
    "--metric",
    type=click.Choice(
        ["angular", "euclidean", "manhattan", "hamming", "dot", "minkowski"]
    ),
)
@click.option("--n-trees", type=int, default=10)
@click.option("--search-k", type=int, default=-1)
@click.option("--mmap-dir", type=str, default="auto")
@click.option("--n-jobs", type=int, default=1)
def create_annoy(
    candidates: int,
    metric: str,
    n_trees: int,
    search_k: int,
    mmap_dir: str,
    n_jobs: int,
) -> Tuple[Annoy, Dict]:
    if metric is None:
        metric = "euclidean"
    return (
        Annoy(
            n_candidates=candidates,
            metric=metric,
            n_trees=n_trees,
            search_k=search_k,
            mmap_dir=mmap_dir,
            n_jobs=n_jobs,
        ),
        click.get_current_context().params,
    )


@cli.command("csls", help="Use the Cross-Domain Local Scaling hubness reduction method")
@click.option("--k", type=int, default=5)
def create_csls(k: int) -> Tuple[CSLS, Dict]:
    return CSLS(k=k), click.get_current_context().params


@cli.command("dsl", help="Use the DisSimLocal hubness reduction method")
@click.option("--k", type=int, default=5)
@click.option("--squared/--not-squared", type=bool, default=True)
def create_dsl(k: int, squared: bool) -> Tuple[DisSimLocal, Dict]:
    return DisSimLocal(k=k, squared=squared), click.get_current_context().params


@cli.command("ls", help="Use the Local Scaling hubness reduction method")
@click.option("--k", type=int, default=5)
@click.option("--method", type=click.Choice(["standard", "nicdm"]))
def create_ls(k: int, method: str) -> Tuple[LocalScaling, Dict]:
    if method is None:
        method = "standard"
    return LocalScaling(k=k, method=method), click.get_current_context().params


@cli.command("mp", help="Use the Mutual Proximity hubness reduction method")
@click.option("--method", type=click.Choice(["normal", "empiric"]))
def create_mp(k: int, method: str) -> Tuple[MutualProximity, Dict]:
    if method is None:
        method = "normal"
    return MutualProximity(method=method), click.get_current_context().params


if __name__ == "__main__":
    cli()
