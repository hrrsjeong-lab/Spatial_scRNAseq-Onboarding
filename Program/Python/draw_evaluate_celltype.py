import argparse
import collections
import itertools
import os
import tempfile
import typing
import zipfile
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import numpy
import pandas
import scanpy
import seaborn
import sklearn.metrics
import tqdm
import tqdm.contrib
import tqdm.contrib.itertools
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="INPUT hdf5 file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    input_adata = scanpy.read_h5ad(args.input, backed="r")
    print(input_adata)

    true_celltype_column = "SubclassLevel1"
    cell_type_list = sorted(set(input_adata.obs[true_celltype_column]) | set(input_adata.obs[step00.celltype_column]))
    cell_type_palette = dict(zip(cell_type_list, itertools.cycle(matplotlib.colors.XKCD_COLORS)))
    print("Celltype:", len(cell_type_list), cell_type_list)

    real_cell_type_list = sorted(set(input_adata.obs[true_celltype_column]))
    print("Real:", len(real_cell_type_list), real_cell_type_list)

    counter = collections.Counter(zip(input_adata.obs[true_celltype_column], input_adata.obs[step00.celltype_column]))
    counter_data = pandas.DataFrame(0, index=cell_type_list, columns=cell_type_list, dtype=int)
    for index, column in tqdm.contrib.itertools.product(cell_type_list, cell_type_list):
        counter_data.loc[index, column] = counter[(index, column)]
    print(counter_data)

    confusion_matrix_list = sklearn.metrics.multilabel_confusion_matrix(input_adata.obs[true_celltype_column], input_adata.obs[step00.celltype_column], labels=cell_type_list)
    confusion_matrix_data = pandas.DataFrame(index=step00.evaluation_list)
    for cell_type, confusion_matrix in tqdm.tqdm(zip(cell_type_list, confusion_matrix_list), total=len(real_cell_type_list)):
        d = step00.evaluate_confusion_matrix(confusion_matrix)
        confusion_matrix_data[cell_type] = pandas.Series(list(d.values()), index=list(d.keys()))
    else:
        d = step00.evaluate_confusion_matrix(numpy.sum(confusion_matrix_list, axis=0))
        confusion_matrix_data["ALL"] = pandas.Series(list(d.values()), index=list(d.keys()))
    score_data = pandas.melt(confusion_matrix_data.T.reset_index(), id_vars="index", value_name="Evaluation", value_vars=step00.evaluation_list).fillna(0.0)
    print(score_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

        seaborn.heatmap(data=counter_data, vmin=0, robust=True, cmap="YlOrRd", annot=True, fmt="d", annot_kws={"size": "xx-small"}, cbar=False, ax=ax)

        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
        matplotlib.pyplot.xlabel("Predicted celltype")
        matplotlib.pyplot.ylabel("Real celltype")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Count-Heatmap.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Count-Heatmap.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

        seaborn.heatmap(data=counter_data.loc[real_cell_type_list, :].div(counter_data.loc[real_cell_type_list, :].sum(axis=1), axis=0), vmin=0.0, vmax=1.0, robust=True, cmap="YlOrRd", annot=True, fmt=".2f", annot_kws={"size": "xx-small"}, cbar=False, ax=ax)

        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
        matplotlib.pyplot.xlabel("Predicted celltype")
        matplotlib.pyplot.ylabel("Real celltype")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Proportion-Heatmap.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Proportion-Heatmap.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))

        seaborn.barplot(data=score_data, x="index", order=real_cell_type_list + ["ALL"], hue="variable", hue_order=step00.evaluation_list, y="Evaluation", palette="tab10", legend="full", ax=ax)

        matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.xlabel(step00.celltype_column)
        matplotlib.pyplot.ylim(0, 1)
        matplotlib.pyplot.legend(loc="lower left")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Evaluation-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Evaluation-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
