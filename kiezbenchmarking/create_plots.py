import argparse
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pymongo
import seaborn as sns
from autorank import autorank, plot_stats
from matplotlib.patches import Rectangle
from tqdm import tqdm

from kiezbenchmarking.modified_autorank import cd_diagram

sns.set()
FONT_SCALE = 2.5
IMPROVEMENT_FONT_SIZE = 25

IGNORED_APPROACHES = ["AliNet", "MTransE", "RDGCN", "SEA", "TransR"]
IGNORED_HUBNESS = ["dsl squared"]
RENAMED_HUBNESS = {
    "ls nicdm": "NICDM",
    "ls standard": "LS",
    "mp empiric": "MP emp",
    "mp normal": "MP gauss",
    "csls": "CSLS",
    "dsl": "DSL",
    "dissimlocal": "DSL",
    "localscaling nicdm": "NICDM",
    "localscaling standard": "LS",
    "mutualproximity empiric": "MP emp",
    "mutualproximity normal": "MP gauss",
}
RENAMED_ALGOS = {
    "hnsw": "NMSLIB_HNSW",
    "nng": "NGT",
    "rptree": "Annoy",
    "kd_tree": "KDTree",
    "ball_tree": "BallTree",
    "brute": "Brute",
    "Faiss_HNSW32": "Faiss_HNSW",
    "Faiss_Flat": "Faiss_Brute",
    "Faiss_HNSW32_gpu": "Faiss_HNSW_gpu",
    "Faiss_Flat_gpu": "Faiss_Brute_gpu",
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
            algo_name = d["config"]["algorithm"]
            row["algorithm"] = _rename_with_dict(algo_name, RENAMED_ALGOS)
            if not algo_name == "Faiss":
                row["algorithm_params"] = d["config"]["algorithm_params"]
                row["algorithm_params"].pop("n_candidates", None)
            else:
                algo_params = {
                    "index_key": d["config"]["index_key"],
                    "use_gpu": d["config"]["use_gpu"],
                }
                index_infos = d["result"]["source_index_infos"]
                algo_info = (
                    algo_params["index_key"]
                    if index_infos == ""
                    else index_infos["index_key"]
                )
                use_gpu = "_gpu" if algo_params["use_gpu"] else ""
                row["algorithm_params"] = algo_params
                row["algorithm"] = _rename_with_dict(
                    d["config"]["algorithm"] + "_" + algo_info + use_gpu,
                    RENAMED_ALGOS,
                )
                print(row["algorithm"])
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
            if "indexing_time" in d["result"]:
                row["indexing time in s"] = d["result"]["indexing_time"]
                row["query time in s"] = d["result"]["query_time"]
                row["source_index_infos"] = d["result"]["source_index_infos"]
                row["target_index_infos"] = d["result"]["target_index_infos"]
            else:
                row["indexing time in s"] = None
                row["query time in s"] = None
                row["source_index_infos"] = None
                row["target_index_infos"] = None
            row["memory"] = d["stats"]["self"]["max_memory_bytes"]
            rows.append(row)
    return pd.DataFrame(rows)


def check_missing(max_all):
    possible_ds = list(max_all["dataset"].unique())
    possible_algo = list(max_all["algorithm"].unique())
    possible_hub = list(max_all["hubness"].unique())
    possible_emb = list(max_all["emb_approach"].unique())
    existing = {
        tuple(x)
        for x in max_all[
            ["dataset", "algorithm", "hubness", "emb_approach"]
        ].to_numpy()
    }
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


def create_consistent_palette(max_all, base_palette="Dark2"):
    my_pal = {}
    my_pal_exact = {}
    algos = max_all["algorithm"].unique()
    for a, c in zip(algos, sns.color_palette(base_palette)):
        if a == RENAMED_ALGOS["brute"]:
            my_pal_exact["Exact"] = c
        else:
            my_pal_exact[a] = c
        my_pal[a] = c
    hub_red = [
        *HUBNESS_ORDER,
        "None",
    ]
    hr_pal = dict(zip(hub_red, sns.color_palette()))
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


def create_improve_df_ann_to_brute(
    df, wanted_value, colname, compare_algos=None
):
    if not compare_algos:
        compare_algos = [
            RENAMED_ALGOS["hnsw"],
            RENAMED_ALGOS["nng"],
            RENAMED_ALGOS["rptree"],
        ]
    baseline = df[
        (df["hubness"] == "None") & (df["algorithm"] == RENAMED_ALGOS["brute"])
    ].copy()
    new_index = ["dataset", "emb_approach"]
    baseline.set_index(new_index, inplace=True)
    hub_improved = None
    for hub in HUBNESS_ORDER:
        for algo in compare_algos:
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


def get_from_precalculated(max_all, hits_value=50):
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
    improve_df = create_improve_df(
        max_all, f"hits@{hits_value}", f"% hits@{hits_value} increase"
    )
    improve_df_ann_to_brute = create_improve_df_ann_to_brute(
        max_all, f"hits@{hits_value}", f"% hits@{hits_value} increase"
    )
    return (
        max_small,
        max_large,
        improve_df,
        improve_df_ann_to_brute,
    )


def get_all_dfs(collection, hits_value=50):
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
        )[f"hits@{hits_value}"].idxmax()
    ]
    max_large = large_result_df.loc[
        large_result_df.groupby(
            ["dataset", "hubness", "emb_approach", "algorithm"]
        )[f"hits@{hits_value}"].idxmax()
    ]
    max_all = pd.concat([max_small, max_large])
    hub_improved_hits_50 = create_improve_df(
        max_all, f"hits@{hits_value}", f"% hits@{hits_value} increase"
    )
    hub_improved_hits_50_ann_to_brute = create_improve_df_ann_to_brute(
        max_all, f"hits@{hits_value}", f"% hits@{hits_value} increase"
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
    df, save_path, pal=None, ds="D-Y 100K(V2)", approaches=None, hits_value=50
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
        y=f"hits@{hits_value}",
        ax=axs[1, 0],
    )
    sns.barplot(
        data=tmp[tmp["emb_approach"] == approaches[1]],
        x="hubness",
        order=order,
        palette=pal,
        y=f"hits@{hits_value}",
        ax=axs[1, 1],
    )
    sns.barplot(
        data=tmp[tmp["emb_approach"] == approaches[2]],
        x="hubness",
        order=order,
        palette=pal,
        y=f"hits@{hits_value}",
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


def emb_approach_hits(
    max_all, save_path, pal_exact, emb_approach, hits_value=50, col_wrap=4
):
    sns.set(font_scale=FONT_SCALE)
    tmp = max_all[
        ~max_all["algorithm"].isin(
            [RENAMED_ALGOS["kd_tree"], RENAMED_ALGOS["ball_tree"]]
        )
    ]
    tmp = tmp[tmp["emb_approach"] == emb_approach]
    showlegend = col_wrap == 4
    order = ["None", *HUBNESS_ORDER]
    hits_plot = sns.catplot(
        data=tmp.replace(RENAMED_ALGOS["brute"], "Exact"),
        kind="bar",
        y=f"hits@{hits_value}",
        x="hubness",
        col_wrap=col_wrap,
        hue="algorithm",
        col="dataset",
        aspect=2,
        order=order,
        col_order=DS_ORDER,
        palette=pal_exact,
        legend=showlegend,
    )
    hits_plot.set_titles("{col_name}")
    hits_plot.set_xlabels("")
    for ax in hits_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, fontsize=IMPROVEMENT_FONT_SIZE)
    if not showlegend:
        plt.legend(
            bbox_to_anchor=(2.25, 0.3),
            loc="center right",
            fontsize="medium",
        )
    hits_plot.savefig(save_path)
    sns.set(font_scale=1)
    plt.close("all")


def large_boxplot_improvement(
    improv_df, save_path, pal_exact, hits_value=50, col_wrap=4
):
    sns.set(font_scale=FONT_SCALE)
    tmp = improv_df[
        ~improv_df["algorithm"].isin(
            [RENAMED_ALGOS["kd_tree"], RENAMED_ALGOS["ball_tree"]]
        )
    ]
    hits_plot = sns.catplot(
        data=tmp.replace(RENAMED_ALGOS["brute"], "Exact"),
        kind="box",
        y=f"% hits@{hits_value} increase",
        x="hubness",
        aspect=2,
        col_wrap=col_wrap,
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
        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, fontsize=IMPROVEMENT_FONT_SIZE)
    hits_plot.savefig(save_path)
    sns.set(font_scale=1)


def boxplot_improvement_only_exact(
    improv_df, save_path, pal, hits_value=50, col_wrap=4
):
    sns.set(font_scale=FONT_SCALE)
    showlegend = col_wrap == 4
    tmp = improv_df[improv_df["algorithm"].isin([RENAMED_ALGOS["brute"]])]
    hits_plot = sns.catplot(
        data=tmp,
        kind="box",
        y=f"% hits@{hits_value} increase",
        x="hubness",
        aspect=2,
        col_wrap=col_wrap,
        col="dataset",
        col_order=DS_ORDER,
        order=HUBNESS_ORDER,
        palette=pal,
        legend=showlegend,
    )
    hits_plot.set_titles("{col_name}")
    hits_plot.set_xlabels("")
    for ax in hits_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, fontsize=IMPROVEMENT_FONT_SIZE)
    if not showlegend:
        plt.legend(
            bbox_to_anchor=(2.25, 0.3),
            loc="center right",
            fontsize="medium",
        )
    hits_plot.savefig(save_path)
    sns.set(font_scale=1)


def boxplot_improvement_ann_to_brute(
    improv_df, save_path, my_pal, hits_value=50, col_wrap=4
):
    showlegend = col_wrap == 4
    sns.set(font_scale=FONT_SCALE)
    hits_plot = sns.catplot(
        data=improv_df,
        kind="box",
        y=f"% hits@{hits_value} increase",
        x="hubness",
        hue="algorithm",
        palette=my_pal,
        aspect=2,
        col_wrap=col_wrap,
        col="dataset",
        col_order=DS_ORDER,
        order=HUBNESS_ORDER,
        legend=showlegend,
    )
    hits_plot.set_titles("{col_name}")
    hits_plot.set_xlabels("")
    for ax in hits_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, fontsize=IMPROVEMENT_FONT_SIZE)
    if not showlegend:
        plt.legend(
            bbox_to_anchor=(2.25, 0.3),
            loc="center right",
            fontsize="medium",
        )
    hits_plot.savefig(save_path)
    sns.set(font_scale=1)


def create_time_or_memory_plot(
    df,
    palette,
    save_path,
    value,
    memory_division=None,
    hue_order=None,
    col_wrap=None,
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
        col_wrap=col_wrap,
    )
    time_plot.set_titles("{col_name}")
    time_plot.set_xlabels("")
    for ax in time_plot.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    time_plot.savefig(save_path)
    sns.set(font_scale=1)


def manipulate_palette(pal, luminosity=None, saturation=None):
    return sns.color_palette(
        [
            sns.set_hls_values(col, l=luminosity, s=saturation)
            for col in list(pal)
        ]
    )


def _stacked_bar(
    data,
    light_palette=None,
    dark_palette=None,
    x="algorithm",
    y1="time in s",
    y2="indexing time in s",
    *args,
    **kwargs,
):
    sns.barplot(data=data, x=x, y=y1, palette=light_palette, *args, **kwargs)
    plot2 = sns.barplot(
        data=data, x=x, y=y2, palette=dark_palette, *args, **kwargs
    )
    plot2.set_xticklabels(plot2.get_xticklabels(), rotation=90)
    return plot2


def create_detailled_time_plot(
    data,
    save_path,
    palette=None,
    x="algorithm",
    y1="time in s",
    y2="indexing time in s",
    y_label=None,
    col="hubness",
    legend_label1="query_time",
    legend_label2="indexing_time",
):
    if not y_label:
        y_label = y1
    if not palette:
        palette = sns.color_palette()

    dark_palette = manipulate_palette(palette, luminosity=0.3)
    light_palette = manipulate_palette(palette, luminosity=0.6)
    dark_grey = manipulate_palette(palette, luminosity=0.3, saturation=0)[0]
    light_grey = manipulate_palette(palette, luminosity=0.6, saturation=0)[0]
    g = sns.FacetGrid(data, col=col, height=5, aspect=0.8)
    g.map_dataframe(
        _stacked_bar, light_palette=light_palette, dark_palette=dark_palette
    )
    g.set_ylabels(y_label)
    g.add_legend(
        {
            legend_label1: Rectangle((0, 0), 2, 2, fill=True, color=light_grey),
            legend_label2: Rectangle((0, 0), 2, 2, fill=True, color=dark_grey),
        }
    )
    g.save_fig(save_path)


def improvement_per_emb_approach(
    improv_df, save_path, my_pal_exact=None, hits_value=50, col_wrap=4
):
    sns.set(font_scale=FONT_SCALE)
    tmp = improv_df[
        ~improv_df["algorithm"].isin(
            [RENAMED_ALGOS["kd_tree"], RENAMED_ALGOS["ball_tree"]]
        )
    ]
    hits_plot = sns.catplot(
        data=tmp.replace(RENAMED_ALGOS["brute"], "Exact"),
        kind="strip",
        dodge=True,
        y=f"% hits@{hits_value} increase",
        x="hubness",
        aspect=2,
        col_wrap=col_wrap,
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


def cd_hits(max_all, save_path, hits_value=50):
    matplotlib.rcParams.update({"font.size": 14})
    tmp = max_all[max_all["algorithm"] == RENAMED_ALGOS["brute"]]
    wanted = tmp.pivot_table(
        index=["dataset", "emb_approach"],
        columns=["hubness"],
        values=[f"hits@{hits_value}"],
    )[f"hits@{hits_value}"]
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


def create_all_plots(
    max_all,
    max_small,
    max_large,
    improvement_hits,
    improvement_hits_ann_to_brute,
    hr_pal,
    pal,
    pal_exact,
    output_dir,
    hits_value,
    hr_to_compare=None,
    remove_ann=None,
    plot_time_and_memory=True,
    col_wrap=4,
):
    plot_detailled_hubness_hits(
        max_large,
        f"{output_dir}/detailled_improvement.pdf",
        hr_pal,
        hits_value=hits_value,
    )
    print("Plotted detailled_hubness_hits")
    large_boxplot_improvement(
        improvement_hits,
        f"{output_dir}/boxplot_improvement.pdf",
        pal_exact,
        hits_value=hits_value,
        col_wrap=col_wrap,
    )
    print("Plotted large_boxplot_improvement")
    boxplot_improvement_only_exact(
        improvement_hits,
        f"{output_dir}/boxplot_improvement_only_exact.pdf",
        hr_pal,
        hits_value=hits_value,
        col_wrap=col_wrap,
    )
    print("Plotted large_boxplot_improvement")
    boxplot_improvement_ann_to_brute(
        improvement_hits_ann_to_brute,
        f"{output_dir}/boxplot_improvement_ann_to_brute.pdf",
        pal,
        hits_value=hits_value,
        col_wrap=col_wrap,
    )
    print("Plotted large_boxplot_improvement")
    if plot_time_and_memory:
        create_time_or_memory_plot(
            max_small,
            pal,
            f"{output_dir}/time_small.pdf",
            "time in s",
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
        )
        print("Plotted memory small")
        create_time_or_memory_plot(
            max_large,
            pal,
            f"{output_dir}/memory_large.pdf",
            "memory in GB",
            1073741824,
        )
        print("Plotted memory large")
    cd_hits(max_all, f"{output_dir}/cd_hubness_hits.pdf", hits_value=hits_value)
    print("Plotted cd diagram")
    cd_ann_to_brute(
        max_all,
        f"{output_dir}/cd_hubness_ann_to_brute.pdf",
        value=f"hits@{hits_value}",
        hr_to_compare=hr_to_compare,
    )
    print("Plotted cd diagram ann to brute")
    cd_ann_to_brute(
        max_large,
        f"{output_dir}/cd_hubness_ann_to_brute_time_large_no_annoy.pdf",
        value="time in s",
        remove_ann=remove_ann,
    )
    print("Plotted cd diagram ann to brute")
    plt.close("all")


def get_all_hits_improvements(max_all, ann_to_brute=True, hits_values=None):
    if hits_values is None:
        hits_values = [1, 5, 10, 25, 50]
    improvement = None
    for k in hits_values:
        if improvement is None:
            if ann_to_brute:
                improvement = create_improve_df_ann_to_brute(
                    max_all, f"hits@{k}", f"% hits@{k} increase"
                )
            else:
                improvement = create_improve_df(
                    max_all, f"hits@{k}", f"% hits@{k} increase"
                )
        else:
            if ann_to_brute:
                improvement = improvement.merge(
                    create_improve_df_ann_to_brute(
                        max_all, f"hits@{k}", f"% hits@{k} increase"
                    )
                )
            else:
                improvement = improvement.merge(
                    create_improve_df(
                        max_all, f"hits@{k}", f"% hits@{k} increase"
                    )
                )
    return improvement


def create_extensive(max_all, max_large, max_small, output_dir):
    print("Plotting a lot of plots")
    hits_values = [1, 5, 10, 25, 50]
    improvement_hits = get_all_hits_improvements(max_all, ann_to_brute=False)
    improvement_hits_ann_to_brute = get_all_hits_improvements(max_all)
    hr_to_compare = HUBNESS_ORDER
    emb_approaches = max_all["emb_approach"].unique()
    for k in hits_values:
        print(f"Starting plots for hits@{k}")
        curr_dir = f"{output_dir}/hits_at_{k}"
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        time_and_mem = k == 50
        create_all_plots(
            max_all=max_all,
            max_small=max_small,
            max_large=max_large,
            improvement_hits=improvement_hits,
            improvement_hits_ann_to_brute=improvement_hits_ann_to_brute,
            hr_pal=hr_pal,
            pal=pal,
            pal_exact=pal_exact,
            output_dir=curr_dir,
            hr_to_compare=hr_to_compare,
            plot_time_and_memory=time_and_mem,
            hits_value=k,
            col_wrap=3,
        )
        for e in tqdm(
            emb_approaches, desc=f"Plotting hits@{k} plots for approaches"
        ):
            emb_approach_hits(
                max_all,
                f"{curr_dir}/{e}.pdf",
                emb_approach=e,
                pal_exact=pal_exact,
                hits_value=k,
                col_wrap=3,
            )
        print(f"\n Done for {k} \n")
    cd_ann_to_brute(
        max_small,
        f"{output_dir}/cd_hubness_ann_to_brute_time_small.pdf",
        value="time in s",
        hr_to_compare=hr_to_compare,
    )
    cd_ann_to_brute(
        max_large,
        f"{output_dir}/cd_hubness_ann_to_brute_time_large.pdf",
        value="time in s",
        hr_to_compare=hr_to_compare,
    )
    print("\n ==== DONE ==== \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--use-mongodb", action="store_true", default=False)
    parser.add_argument("--hits", type=int, default=50)
    parser.add_argument("--extensive", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true", default=False)
    args = parser.parse_args()
    output_dir = args.output_dir
    if not args.use_mongodb:
        max_all = pd.read_csv("results/max_all.csv", index_col=0)
        if args.gpu:
            max_gpu = pd.read_csv("results/max_gpu.csv", index_col=0)
            max_all = max_gpu.append(max_all)
        (
            max_small,
            max_large,
            hub_improved_hits_50,
            hub_improved_hits_50_ann_to_brute,
        ) = get_from_precalculated(max_all, hits_value=args.hits)
    else:
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
        ) = get_all_dfs(collection, hits_value=args.hits)
        if args.gpu:
            max_all.to_csv("results/max_gpu.csv")
            tmp = pd.read_csv("results/max_all.csv", index_col=0)
            max_all = max_all.append(tmp)
            (
                max_small,
                max_large,
                hub_improved_hits_50,
                hub_improved_hits_50_ann_to_brute,
            ) = get_from_precalculated(max_all, hits_value=args.hits)
    print("Got dfs")
    pal, pal_exact, hr_pal = create_consistent_palette(max_all)
    if args.extensive:
        create_extensive(max_all, max_large, max_small, output_dir)
    else:
        create_all_plots(
            max_all=max_all,
            max_small=max_small,
            max_large=max_large,
            improvement_hits=hub_improved_hits_50,
            improvement_hits_ann_to_brute=hub_improved_hits_50_ann_to_brute,
            hr_pal=hr_pal,
            pal=pal,
            pal_exact=pal_exact,
            output_dir=output_dir,
            hr_to_compare=None,
            remove_ann="Annoy",
            plot_time_and_memory=True,
            hits_value=args.hits,
        )
