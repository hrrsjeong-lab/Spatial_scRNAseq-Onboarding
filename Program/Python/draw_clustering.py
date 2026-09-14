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

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    cluster_list = sorted(set(input_adata.obs[step00.clustering_column]), key=int)
    cluster_palette = dict(zip(cluster_list, itertools.cycle(matplotlib.colors.XKCD_COLORS)))
    print("Clustering:", len(cluster_list), cluster_list)

    clustering_data = pandas.DataFrame(input_adata.obsm[step00.projection_key], index=input_adata.obs.index, columns=step00.projection_columns)
    for column in tqdm.tqdm([step00.clustering_column, "sample"]):
        clustering_data[column] = input_adata.obs[column]
    print(clustering_data)

    gene_list = list(input_adata.var[(input_adata.var["highly_variable"])].index)
    clustering_data = pandas.concat([clustering_data, scanpy.get.obs_df(input_adata, keys=gene_list, layer="Counts")], axis="columns", verify_integrity=True)
    print(clustering_data)

    sample_list = sorted(set(clustering_data["sample"]))
    sample_palette = dict(zip(sample_list, itertools.cycle(matplotlib.colors.TABLEAU_COLORS)))
    print("Sample:", len(sample_list), sample_list)

    counter = collections.Counter(clustering_data[[step00.clustering_column, "sample"]].itertuples(index=False, name=None))
    counter_data = pandas.DataFrame(index=sample_list, columns=cluster_list, dtype=int)
    for cluster, sample in tqdm.contrib.itertools.product(cluster_list, sample_list):
        counter_data.loc[sample, cluster] = counter[(cluster, sample)]
    counter_data = counter_data.astype(int)
    print(counter_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, sample in tqdm.contrib.tenumerate(sample_list):
            matplotlib.pyplot.bar(range(len(cluster_list)), counter_data.iloc[i, :], bottom=counter_data.iloc[:i, :].sum(axis="index"), color=sample_palette[sample], label=sample, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.clustering_column)
        matplotlib.pyplot.ylabel("Cell counts")
        matplotlib.pyplot.xticks(range(len(cluster_list)), cluster_list, rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.sample_column, loc="upper right")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.sample_column}-Count.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.sample_column}-Count.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, sample in tqdm.contrib.tenumerate(sample_list):
            matplotlib.pyplot.bar(range(len(cluster_list)), (counter_data.iloc[i, :] / counter_data.sum(axis="index")), bottom=(counter_data.iloc[:i, :].sum(axis="index") / counter_data.sum(axis="index")), color=sample_palette[sample], label=sample, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.clustering_column)
        matplotlib.pyplot.ylabel("Cell proportion")
        matplotlib.pyplot.xticks(range(len(cluster_list)), cluster_list, rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.sample_column, loc="upper right")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.sample_column}-Proportion.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.sample_column}-Proportion.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, cluster in tqdm.contrib.tenumerate(cluster_list):
            matplotlib.pyplot.bar(range(len(sample_list)), counter_data.iloc[:, i], bottom=counter_data.iloc[:, :i].sum(axis="columns"), color=cluster_palette[cluster], label=cluster, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.sample_column)
        matplotlib.pyplot.ylabel("Cell count")
        matplotlib.pyplot.xticks(range(len(sample_list)), sample_list, rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.clustering_column, loc="upper right", ncols=len(cluster_list) // 4)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.clustering_column}-Count.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.clustering_column}-Count.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, cluster in tqdm.contrib.tenumerate(cluster_list):
            matplotlib.pyplot.bar(range(len(sample_list)), (counter_data.iloc[:, i] / counter_data.sum(axis="columns")), bottom=(counter_data.iloc[:, :i].sum(axis="columns") / counter_data.sum(axis="columns")), color=cluster_palette[cluster], label=cluster, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.sample_column)
        matplotlib.pyplot.ylabel("Cell proportion")
        matplotlib.pyplot.xticks(range(len(sample_list)), sample_list, rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.clustering_column, loc="upper right", ncols=len(cluster_list) // 4)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.clustering_column}-Proportion.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.clustering_column}-Proportion.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

        seaborn.scatterplot(data=clustering_data, x=step00.projection_columns[0], y=step00.projection_columns[1], hue=step00.clustering_column, palette=cluster_palette, legend=False, rasterized=True, s=30, edgecolor=None, ax=ax)
        for cluster in tqdm.tqdm(cluster_list):
            matplotlib.pyplot.text(numpy.mean(clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[0]]), numpy.mean(clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[1]]), cluster, color="black", fontsize="x-small", horizontalalignment="center", verticalalignment="center", path_effects=step00.path_effects)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        matplotlib.pyplot.title(f"Total {len(clustering_data)} cells w/ {len(cluster_list)} clusters")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Scatter.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Scatter.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
