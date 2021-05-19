import argparse
from collections import OrderedDict

FILES = OrderedDict(
    [
        ("Improvement over baseline for hits@", "boxplot_improvement.pdf"),
        (
            "Critical distance diagram showing differences between hubness reduction techniques for exact NN with regards to hits@",
            "cd_hubness_hits.pdf",
        ),
        (
            "Critical distance diagram showing differences between hubness reduction techniques for ANN and baseline with regards to hits@",
            "cd_hubness_ann_to_brute.pdf",
        ),
    ]
)

T_AND_M = OrderedDict(
    [
        ("Time in seconds on 15K datasets", "time_small.pdf"),
        ("Time in seconds on 100K datasets", "time_large.pdf"),
        ("Peak memory consumption on 15K datasets", "memory_small.pdf"),
        ("Peak memory consumption on 100K datasets", "memory_large.pdf"),
        (
            "Critical distance diagram showing differences between hubness reduction techniques for ANN and baseline with regards to execution time on small datasets",
            "cd_hubness_ann_to_brute_time_small.pdf",
        ),
        (
            "Critical distance diagram showing differences between hubness reduction techniques for ANN and baseline with regards to execution time on large datasets",
            "cd_hubness_ann_to_brute_time_large.pdf",
        ),
    ]
)


def create_figure(path, caption, width=r"\textwidth"):
    return (
        r"\begin{figure}" + "\n"
        r"\centering"
        + r"\includegraphics[width="
        + width
        + "]{"
        + path
        + "}\n"
        + r"\caption{"
        + caption
        + "}"
        + r"\end{figure}"
    )


def create_report(input_dir, output_dir):
    report = (
        r"\documentclass{article}"
        + "\n"
        + r"\usepackage{graphicx}"
        + r"\usepackage{placeins}"
        + "\n"
        + r"\title{Additional Results for Kiez Benchmark}"
        + r"\date{}"
        + "\n"
        + r"\begin{document}"
        + r"\maketitle"
        + r"\section{Time and Memory}"
        + "\n"
    )
    for cap, path in T_AND_M.items():
        report += create_figure(f"{input_dir}/{path}", cap)
    report += r"\end{document}"
    for k in [1, 5, 10, 25, 50]:
        report += r"\section{Results for hits@" + str(k) + "}\n"
        for cap, path in FILES.items():
            if "hits" in cap:
                cap = f"{cap}{k}"
            if "boxplot_improvement" in path:
                report += create_figure(
                    f"{input_dir}/hits_at_{k}/{path}",
                    cap,
                    width=r"0.9\textwidth",
                )
            else:
                report += create_figure(f"{input_dir}/hits_at_{k}/{path}", cap)
        report += r"\FloatBarrier"
    report += r"\end{document}"
    with open(f"{output_dir}/additional_report.tex", "w") as out_file:
        out_file.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    create_report(args.input_dir, args.output_dir)
