import pymongo
import os
import shutil
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import argparse
from joblib import Memory
from autorank import autorank, plot_stats
from kiezbenchmarking.experiment.modified_autorank import cd_diagram

sns.set()
JOBLIB_DIR = ".joblib_cache"
FONT_SCALE = 2.5
IMPROVEMENT_FONT_SIZE = 25
memory = Memory(JOBLIB_DIR, verbose=0)

IGNORED_APPROACHES = ["AliNet", "MTransE", "RDGCN", "SEA", "TransR"]
IGNORED_HUBNESS = ["dsl squared"]
RENAMED_HUBNESS = {
    "ls nicdm": "NICDM",
    "ls standard": "LS",
    "mp empiric": "MP emp",
    "mp normal": "MP gauss",
    "csls": "CSLS",
    "dsl": "DSL",
}
RENAMED_ALGOS = {
    "hnsw": "HNSW",
    "nng": "NGT",
    "rptree": "Annoy",
    "kd_tree": "KDTree",
    "ball_tree": "BallTree",
    "brute": "Brute",
}
DS_ORDER = [
    "D-W 15K(V1)",
    "D-W 15K(V2)",
    "D-Y 15K(V1)",
    "D-Y 15K(V2)",
    "EN-DE 15K(V1)",
    "EN-DE 15K(V2)",
    "EN-FR 15K(V1)",
    "EN-FR 15K(V2)",
    "D-W 100K(V1)",
    "D-W 100K(V2)",
    "D-Y 100K(V1)",
    "D-Y 100K(V2)",
    "EN-DE 100K(V1)",
    "EN-DE 100K(V2)",
    "EN-FR 100K(V1)",
    "EN-FR 100K(V2)",
]

HUBNESS_ORDER = [
    RENAMED_HUBNESS["ls nicdm"],
    RENAMED_HUBNESS["dsl"],
    RENAMED_HUBNESS["csls"],
    RENAMED_HUBNESS["ls standard"],
    RENAMED_HUBNESS["mp normal"],
    RENAMED_HUBNESS["mp empiric"],
]


def _rename_with_dict(input, rename_dict):
    for old, new in rename_dict.items():
        input = input.replace(old, new)
    return input


@memory.cache
def get_mongo_collection():
    uri = os.environ["MONGODB_URI"]
    db_name = os.environ["MONGODB_NAME"]

    client = pymongo.MongoClient(uri, port=27017)
    db = client[db_name]
    return db["kiez"]


def check_wanted_config(config):
    algo = config["config"]["algorithm"]
    if algo == "hnsw":
        params = config["config"]["algorithm_params"]
        if params != {"M": 96, "efConstruction": 500, "n_candidates": 100}:
            return False
    elif algo == "rptree":
        params = config["config"]["algorithm_params"]
        if params != {"n_candidates": 100, "search_k": -1}:
            return False
    elif algo == "brute" and config["config"]["metric"] == "cosine":
        return False
    return True


def create_df(cursor, is_small):
    rows = []
    for d in cursor:
        if check_wanted_config(d):
            row = {}
            dataset_path = d["config"]["dataset_tuple"][1]
            emb_appr = d["config"]["dataset_tuple"][0].split("/")[6]
            if emb_appr in IGNORED_APPROACHES:
                continue
            if is_small:
                dataset = [x for x in dataset_path.split("/") if "15K" in x]
                if len(dataset) < 1:
                    continue
                dataset = dataset[0]
            else:
                dataset = [x for x in dataset_path.split("/") if "100K" in x]
                if len(dataset) < 1:
                    continue
                dataset = dataset[0]
            ds = dataset.split("_")
            row["dataset"] = f"{ds[0]}-{ds[1]} {ds[2]}({ds[3]})"
            row["algorithm"] = _rename_with_dict(
                d["config"]["algorithm"], RENAMED_ALGOS
            )
            row["algorithm_params"] = d["config"]["algorithm_params"]
            row["algorithm_params"].pop("n_candidates", None)
            row["emb_approach"] = emb_appr
            hubness = d["config"]["hubness"][0]
            hubness_params = d["config"]["hubness"][1]
            if len(hubness_params) > 0:
                if "squared" in hubness_params:
                    if hubness_params["squared"]:
                        hubness += " squared"
                elif "method" in hubness_params:
                    hubness += " " + hubness_params["method"]
            if hubness in IGNORED_HUBNESS:
                continue
            hubness = _rename_with_dict(hubness, RENAMED_HUBNESS)
            row["hubness"] = hubness
            row["metric"] = d["config"]["metric"]
            row["hits@1"] = d["result"]["hits@1"]
            row["hits@5"] = d["result"]["hits@5"]
            row["hits@10"] = d["result"]["hits@10"]
            row["hits@25"] = d["result"]["hits@25"]
            row["hits@50"] = d["result"]["hits@50"]
            row["Robin Hood"] = d["result"]["robinhood"]
            row["time in s"] = d["stats"]["real_time"]
            row["memory"] = d["stats"]["self"]["max_memory_bytes"]
            rows.append(row)
    return pd.DataFrame(rows)


def check_missing(max_all):
    possible_ds = list(max_all["dataset"].unique())
    possible_algo = list(max_all["algorithm"].unique())
    possible_hub = list(max_all["hubness"].unique())
    possible_emb = list(max_all["emb_approach"].unique())
    existing = set(
        [
            tuple(x)
            for x in max_all[
                ["dataset", "algorithm", "hubness", "emb_approach"]
            ].to_numpy()
        ]
    )
    wanted = set()
    for element in itertools.product(
        possible_ds, possible_algo, possible_hub, possible_emb
    ):
        if (
            element[2] not in IGNORED_HUBNESS
            and element[3] not in IGNORED_HUBNESS
        ):
            wanted.add(element)
    return wanted, existing, wanted - existing


def create_consistent_palette(max_all):
    my_pal = {}
    my_pal_exact = {}
    algos = max_all["algorithm"].unique()
    for a, c in zip(algos, sns.color_palette("Dark2")):
        if a == RENAMED_ALGOS["brute"]:
            my_pal_exact["Exact"] = c
        else:
            my_pal_exact[a] = c
        my_pal[a] = c
    hub_red = [
        *HUBNESS_ORDER,
        "None",
    ]
    hr_pal = {a: c for a, c in zip(hub_red, sns.color_palette())}
    return my_pal, my_pal_exact, hr_pal


def get_improvement_hubness(df, hubness, wanted_value):
    none = df[df["hubness"] == "None"].copy()
    emb = df[df["hubness"] == hubness].copy()
    new_index = ["dataset", "algorithm", "emb_approach", "metric"]
    none.set_index(new_index, inplace=True)
    emb.set_index(new_index, inplace=True)
    return ((emb[wanted_value] - none[wanted_value]) / none[wanted_value]) * 100


def create_improve_df(df, wanted_value, colname):
    hub_improved = None
    for hub in HUBNESS_ORDER:
        if hub_improved is None:
            hub_improved = (
                get_improvement_hubness(df, hub, wanted_value)
                .to_frame()
                .reset_index()
            )
            hub_improved["hubness"] = hub
        else:
            tmp = (
                get_improvement_hubness(df, hub, wanted_value)
                .to_frame()
                .reset_index()
            )
            tmp["hubness"] = hub
            hub_improved = pd.concat([hub_improved, tmp])
    return hub_improved.rename(columns={wanted_value: colname})


def get_improvement_hubness_brute(
    df, baseline, hubness, algo, new_index, wanted_value
):
    emb = df[(df["hubness"] == hubness) & (df["algorithm"] == algo)].copy()
    emb.set_index(new_index, inplace=True)
    return (
        (emb[wanted_value] - baseline[wanted_value]) / baseline[wanted_value]
    ) * 100


def create_improve_df_ann_to_brute(df, wanted_value, colname):
    baseline = df[
        (df["hubness"] == "None") & (df["algorithm"] == RENAMED_ALGOS["brute"])
    ].copy()
    new_index = ["dataset", "emb_approach"]
    baseline.set_index(new_index, inplace=True)
    hub_improved = None
    for hub in HUBNESS_ORDER:
        for algo in [
            RENAMED_ALGOS["hnsw"],
            RENAMED_ALGOS["nng"],
            RENAMED_ALGOS["rptree"],
        ]:
            if hub_improved is None:
                hub_improved = (
                    get_improvement_hubness_brute(
                        df, baseline, hub, algo, new_index, wanted_value
                    )
                    .to_frame()
                    .reset_index()
                )
                hub_improved["hubness"] = hub
                hub_improved["algorithm"] = algo
            else:
                tmp = (
                    get_improvement_hubness_brute(
                        df, baseline, hub, algo, new_index, wanted_value
                    )
                    .to_frame()
                    .reset_index()
                )
                tmp["hubness"] = hub
                tmp["algorithm"] = algo
                hub_improved = pd.concat([hub_improved, tmp])
    return hub_improved.rename(columns={wanted_value: colname})


@memory.cache
def get_from_precalculated(max_all):
    small_ds = [
        "D-W 15K(V1)",
        "D-W 15K(V2)",
        "D-Y 15K(V1)",
        "D-Y 15K(V2)",
        "EN-DE 15K(V1)",
        "EN-DE 15K(V2)",
        "EN-FR 15K(V1)",
        "EN-FR 15K(V2)",
    ]
    max_small = max_all[max_all["dataset"].isin(small_ds)]
    max_large = max_all[~max_all["dataset"].isin(small_ds)]
    hub_improved_hits_50 = create_improve_df(
        max_all, "hits@50", "% hits@50 increase"
    )
    hub_improved_hits_50_ann_to_brute = create_improve_df_ann_to_brute(
        max_all, "hits@50", "% hits@50 increase"
    )
    return (
        max_small,
        max_large,
        hub_improved_hits_50,
        hub_improved_hits_50_ann_to_brute,
    )


@memory.cache
def get_all_dfs(collection):
    small_results = collection.find(
        {"status": "COMPLETED", "config.target_samples": 15000},
        {"result": 1, "config": 1, "stats": 1},
    )
    small_result_df = create_df(small_results, True)
    large_results = collection.find(
        {"status": "COMPLETED", "config.target_samples": 100000},
        {"result": 1, "config": 1, "stats": 1},
    )
    large_result_df = create_df(large_results, False)
    max_small = small_result_df.loc[
        small_result_df.groupby(
            ["dataset", "hubness", "emb_approach", "algorithm"]
        )["hits@50"].idxmax()
    ]
    max_large = large_result_df.loc[
        large_result_df.groupby(
            ["dataset", "hubness", "emb_approach", "algorithm"]
        )["hits@50"].idxmax()
    ]
    max_all = pd.concat([max_small, max_large])
    hub_improved_hits_50 = create_improve_df(
        max_all, "hits@50", "% hits@50 increase"
    )
    hub_improved_hits_50_ann_to_brute = create_improve_df_ann_to_brute(
        max_all, "hits@50", "% hits@50 increase"
    )
    return (
        small_result_df,
        large_result_df,
        max_small,
        max_large,
        max_all,
        hub_improved_hits_50,
        hub_improved_hits_50_ann_to_brute,
    )


def plot_detailled_hubness_hits(
    df,
    save_path,
    pal=None,
    ds="D-Y 100K(V2)",
    approaches=None,
):
    if approaches is None:
        approaches = ["BootEA", "MultiKE", "SimplE"]
    order = ["None", *HUBNESS_ORDER]
    tmp = df[
        (df["algorithm"] == RENAMED_ALGOS["brute"]) & (df["dataset"] == ds)
    ]
    fig, axs = plt.subplots(ncols=3, nrows=2, sharey=True)
    sns.barplot(
        data=tmp[tmp["emb_approach"] == approaches[0]],
        x="hubness",
        order=order,
        palette=pal,
        y="Robin Hood",
        ax=axs[0, 0],
    )
    sns.barplot(
        data=tmp[tmp["emb_approach"] == approaches[1]],
        x="hubness",
        order=order,
        palette=pal,
        y="Robin Hood",
        ax=axs[0, 1],
    )
    sns.barplot(
        data=tmp[tmp["emb_approach"] == approaches[2]],
        x="hubness",
        order=order,
        palette=pal,
        y="Robin Hood",
        ax=axs[0, 2],
    )
    sns.barplot(
        data=tmp[tmp["emb_approach"] == approaches[0]],
        x="hubness",
        order=order,
        palette=pal,
        y="hits@50",
        ax=axs[1, 0],
    )
    sns.barplot(
        data=tmp[tmp["emb_approach"] == approaches[1]],
        x="hubness",
        order=order,
        palette=pal,
        y="hits@50",
        ax=axs[1, 1],
    )
    sns.barplot(
        data=tmp[tmp["emb_approach"] == approaches[2]],
        x="hubness",
        order=order,
        palette=pal,
        y="hits@50",
        ax=axs[1, 2],
    )
    axs[0, 0].set(title=approaches[0], xticklabels=[], xlabel=None)
    axs[0, 1].set(title=approaches[1], xticklabels=[], xlabel=None, ylabel=None)
    axs[0, 2].set(title=approaches[2], xticklabels=[], xlabel=None, ylabel=None)
    axs[1, 0].set(xlabel=None)
    axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=90)
    axs[1, 1].set(xlabel=None, ylabel=None)
    axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=90)
    axs[1, 2].set(xlabel=None, ylabel=None)
    axs[1, 2].set_xticklabels(axs[1, 2].get_xticklabels(), rotation=90)
    fig.savefig(save_path, bbox_inches="tight")


def large_boxplot_improvement(improv_df, save_path, pal_exact):
    sns.set(font_scale=FONT_SCALE)
    tmp = improv_df[
        ~improv_df["algorithm"].isin(
            [RENAMED_ALGOS["kd_tree"], RENAMED_ALGOS["ball_tree"]]
        )
    ]
    hits_plot = sns.catplot(
        data=tmp.replace(RENAMED_ALGOS["brute"], "Exact"),
        kind="box",
        y="% hits@50 increase",
        x="hubness",
        aspect=2,
        col_wrap=4,
        hue="algorithm",
        col="dataset",
        col_order=DS_ORDER,
        palette=pal_exact,
        showfliers=False,
    )
    hits_plot.set_titles("{col_name}")
    hits_plot.set_xlabels("")
    for ax in hits_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # ax.set_ylim(-20,20)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, fontsize=IMPROVEMENT_FONT_SIZE)
    hits_plot.savefig(save_path)
    sns.set(font_scale=1)


def boxplot_improvement_only_exact(improv_df, save_path, pal):
    sns.set(font_scale=FONT_SCALE)
    tmp = hub_improved_hits_50[
        hub_improved_hits_50["algorithm"].isin([RENAMED_ALGOS["brute"]])
    ]
    hits_plot = sns.catplot(
        data=tmp,
        kind="box",
        y="% hits@50 increase",
        x="hubness",
        aspect=2,
        col_wrap=4,
        col="dataset",
        col_order=DS_ORDER,
        order=HUBNESS_ORDER,
        palette=pal,
    )
    hits_plot.set_titles("{col_name}")
    hits_plot.set_xlabels("")
    for ax in hits_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, fontsize=IMPROVEMENT_FONT_SIZE)
    hits_plot.savefig(save_path)
    sns.set(font_scale=1)


def boxplot_improvement_ann_to_brute(improv_df, save_path, my_pal):
    sns.set(font_scale=FONT_SCALE)
    hits_plot = sns.catplot(
        data=improv_df,
        kind="box",
        y="% hits@50 increase",
        x="hubness",
        hue="algorithm",
        palette=my_pal,
        aspect=2,
        col_wrap=4,
        col="dataset",
        col_order=DS_ORDER,
        order=HUBNESS_ORDER,
    )
    hits_plot.set_titles("{col_name}")
    hits_plot.set_xlabels("")
    for ax in hits_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, fontsize=IMPROVEMENT_FONT_SIZE)
    hits_plot.savefig(save_path)
    sns.set(font_scale=1)


def create_time_or_memory_plot(
    df,
    palette,
    save_path,
    value,
    memory_division=None,
    hue_order=None,
):
    if hue_order is None:
        hue_order = [
            RENAMED_ALGOS["brute"],
            RENAMED_ALGOS["ball_tree"],
            RENAMED_ALGOS["kd_tree"],
            RENAMED_ALGOS["hnsw"],
            RENAMED_ALGOS["nng"],
            RENAMED_ALGOS["rptree"],
        ]
    sns.set(font_scale=FONT_SCALE)
    if memory_division is not None:
        df[value] = df["memory"] / memory_division
    time_plot = sns.catplot(
        data=df,
        kind="bar",
        y=value,
        x="algorithm",
        col="hubness",
        order=hue_order,
        palette=palette,
        # showfliers=False,
    )
    time_plot.set_titles("{col_name}")
    time_plot.set_xlabels("")
    for ax in time_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    time_plot.savefig(save_path)
    sns.set(font_scale=1)


def improvement_per_emb_approach(improv_df, save_path, my_pal_exact=None):
    sns.set(font_scale=FONT_SCALE)
    tmp = hub_improved_hits_50[
        ~hub_improved_hits_50["algorithm"].isin(
            [RENAMED_ALGOS["kd_tree"], RENAMED_ALGOS["ball_tree"]]
        )
    ]
    hits_plot = sns.catplot(
        data=tmp.replace(RENAMED_ALGOS["brute"], "Exact"),
        kind="strip",
        dodge=True,
        y="% hits@50 increase",
        x="hubness",
        aspect=2,
        col_wrap=4,
        hue="algorithm",
        col="emb_approach",
        palette=my_pal_exact,
    )
    hits_plot.set_titles("{col_name}")
    hits_plot.set_xlabels("")
    for ax in hits_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, fontsize=IMPROVEMENT_FONT_SIZE)
    hits_plot.savefig(save_path)
    sns.set(font_scale=1)


def cd_hits(max_all, save_path):
    matplotlib.rcParams.update({"font.size": 14})
    tmp = max_all[max_all["algorithm"] == RENAMED_ALGOS["brute"]]
    wanted = tmp.pivot_table(
        index=["dataset", "emb_approach"],
        columns=["hubness"],
        values=["hits@50"],
    )["hits@50"]
    result = autorank(wanted)
    plot_stats(result)
    plt.savefig(save_path, bbox_inches="tight")


def cd_ann_to_brute(
    max_all,
    save_path,
    value="hits@50",
    hr_to_compare=None,
    remove_ann=None,
):
    if hr_to_compare is None:
        hr_to_compare = [
            RENAMED_HUBNESS["ls nicdm"],
            RENAMED_HUBNESS["dsl"],
            RENAMED_HUBNESS["csls"],
        ]
    wanted = max_all
    nohub = wanted[
        (wanted["hubness"] == "None")
        & (wanted["algorithm"] == RENAMED_ALGOS["brute"])
    ]
    remove = [
        RENAMED_ALGOS["brute"],
        RENAMED_ALGOS["ball_tree"],
        RENAMED_ALGOS["kd_tree"],
    ]
    if remove_ann is not None:
        remove.append(remove_ann)
    bestann = wanted[
        (wanted["hubness"].isin(hr_to_compare))
        & (~wanted["algorithm"].isin(remove))
    ]
    best_ann_to_nohub_exact = pd.concat([nohub, bestann]).replace(
        RENAMED_ALGOS["brute"], "Exact"
    )
    bane = best_ann_to_nohub_exact.pivot_table(
        index=["dataset", "emb_approach"],
        columns=["algorithm", "hubness"],
        values=[value],
    )[value]
    if value == "time in s":
        order = "ascending"
    else:
        order = "descending"
    result = autorank(bane, order=order)
    cd_diagram(result, True, None, width=6, tuple_sep="/")
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--remove-cache", action="store_true", default=False)
    parser.add_argument("--use-csv", action="store_true", default=False)
    args = parser.parse_args()
    output_dir = args.output_dir
    if args.use_csv:
        max_all = pd.read_csv("results/max_all.csv", index_col=0)
        (
            max_small,
            max_large,
            hub_improved_hits_50,
            hub_improved_hits_50_ann_to_brute,
        ) = get_from_precalculated(max_all)
    else:
        if args.remove_cache:
            shutil.rmtree(JOBLIB_DIR)
            print("Removed cache")
        else:
            print("Using cache")
        collection = get_mongo_collection()
        print(f"Using {output_dir} as output")
        (
            small_result_df,
            large_result_df,
            max_small,
            max_large,
            max_all,
            hub_improved_hits_50,
            hub_improved_hits_50_ann_to_brute,
        ) = get_all_dfs(collection)
        max_all.to_csv("max_all.csv")
    print("Got dfs")
    pal, pal_exact, hr_pal = create_consistent_palette(max_all)
    plot_detailled_hubness_hits(
        max_large, f"{output_dir}/detailled_improvement.pdf", hr_pal
    )
    print("Plotted detailled_hubness_hits")
    large_boxplot_improvement(
        hub_improved_hits_50, f"{output_dir}/boxplot_improvement.pdf", pal_exact
    )
    print("Plotted large_boxplot_improvement")
    boxplot_improvement_only_exact(
        hub_improved_hits_50,
        f"{output_dir}/boxplot_improvement_only_exact.pdf",
        hr_pal,
    )
    print("Plotted large_boxplot_improvement")
    boxplot_improvement_ann_to_brute(
        hub_improved_hits_50_ann_to_brute,
        f"{output_dir}/boxplot_improvement_ann_to_brute.pdf",
        pal,
    )
    print("Plotted large_boxplot_improvement")
    create_time_or_memory_plot(
        max_small, pal, f"{output_dir}/time_small.pdf", "time in s"
    )
    print("Plotted time small")
    create_time_or_memory_plot(
        max_large, pal, f"{output_dir}/time_large.pdf", "time in s"
    )
    print("Plotted time large")
    create_time_or_memory_plot(
        max_small,
        pal,
        f"{output_dir}/memory_small.pdf",
        "memory in MB",
        1048576,
    )  # MB
    print("Plotted memory small")
    create_time_or_memory_plot(
        max_large,
        pal,
        f"{output_dir}/memory_large.pdf",
        "memory in GB",
        1073741824,
    )  # GB
    print("Plotted memory large")
    cd_hits(max_all, f"{output_dir}/cd_hubness_hits.pdf")
    print("Plotted cd diagram")
    cd_ann_to_brute(max_all, f"{output_dir}/cd_hubness_ann_to_brute.pdf")
    print("Plotted cd diagram ann to brute")
    cd_ann_to_brute(
        max_large,
        f"{output_dir}/cd_hubness_ann_to_brute_time_large_no_annoy.pdf",
        value="time in s",
        remove_ann="Annoy",
    )
    print("Plotted cd diagram ann to brute")
