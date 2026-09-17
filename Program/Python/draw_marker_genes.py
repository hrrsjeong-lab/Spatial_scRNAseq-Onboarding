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

    sample_list = sorted(set(input_adata.obs[step00.sample_column]))
    sample_palette = dict(zip(sample_list, itertools.cycle(matplotlib.colors.TABLEAU_COLORS)))
    print("Sample:", len(sample_list), sample_list)

    cluster_list = sorted(set(input_adata.obs[step00.clustering_column]), key=int)
    cluster_palette = dict(zip(cluster_list, itertools.cycle(matplotlib.colors.XKCD_COLORS)))
    print("Clustering:", len(cluster_list), cluster_list)

    clustering_data = pandas.DataFrame(input_adata.obsm[step00.projection_key], index=input_adata.obs.index, columns=step00.projection_columns)
    for column in tqdm.tqdm([step00.clustering_column, step00.sample_column]):
        clustering_data[column] = input_adata.obs[column]
    print(clustering_data)

    gene_list = sorted(set().union(*input_adata.uns[step00.marker_column].values()))
    print("Gene:", len(gene_list))

    expression_data = scanpy.get.obs_df(input_adata, keys=gene_list, layer="Counts")
    print(expression_data)

    cell_type_list = sorted(set(input_adata.obs[step00.celltype_column]) & set(input_adata.uns[step00.marker_column].keys()))
    cell_type_palette = dict(zip(cell_type_list, itertools.cycle(matplotlib.colors.XKCD_COLORS)))
    print("Cell type:", len(cell_type_list), cell_type_list)

    sample_list = sorted(set(clustering_data[step00.sample_column]))
    sample_palette = dict(zip(sample_list, itertools.cycle(matplotlib.colors.TABLEAU_COLORS)))
    print("Sample:", len(sample_list), sample_list)

    cluster_score_data = input_adata.obsm[step00.celltype_column].groupby(input_adata.obs[step00.clustering_column]).mean().loc[cluster_list, cell_type_list]
    for index in tqdm.tqdm(list(cluster_score_data.index)):
        cluster_score_data.loc[index, :] = cluster_score_data.loc[index, :] / sum(cluster_score_data.loc[index, :])
    cluster_score_data = cluster_score_data.fillna(0.0)
    print(cluster_score_data)

    cell_marker_data = pandas.concat({cell_type: numpy.log1p(expression_data.loc[:, (input_adata.uns[step00.marker_column][cell_type])].sum(axis="columns")) for cell_type in cell_type_list}, axis="columns", verify_integrity=True)
    cell_marker_data[step00.celltype_column] = list(map(lambda x: cell_type_list[numpy.argmax(cell_marker_data.loc[x, :])], list(cell_marker_data.index)))
    print(cell_marker_data)

    clustering_data = pandas.concat([clustering_data, cell_marker_data], axis="columns", verify_integrity=True)
    print(clustering_data)

    counter = collections.Counter(clustering_data[[step00.clustering_column, step00.celltype_column]].itertuples(index=False, name=None))
    cluster_counter_data = pandas.DataFrame(index=cluster_list, columns=cell_type_list, dtype=int)
    for cluster, cell_type in tqdm.contrib.itertools.product(cluster_list, cell_type_list):
        cluster_counter_data.loc[cluster, cell_type] = counter[(cluster, cell_type)]
    cluster_counter_data = cluster_counter_data.astype(int)
    print(cluster_counter_data)

    counter = collections.Counter(clustering_data[[step00.sample_column, step00.celltype_column]].itertuples(index=False, name=None))
    sample_counter_data = pandas.DataFrame(index=sample_list, columns=cell_type_list, dtype=int)
    for sample, cell_type in tqdm.contrib.itertools.product(sample_list, cell_type_list):
        sample_counter_data.loc[sample, cell_type] = counter[(sample, cell_type)]
    sample_counter_data = sample_counter_data.astype(int)
    print(sample_counter_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        seaborn.heatmap(data=cluster_score_data, vmin=0.0, vmax=1.0, robust=True, cmap="YlOrRd", annot=True, fmt=".2f", annot_kws={"size": "xx-small"}, cbar=False, ax=ax)

        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
        matplotlib.pyplot.xlabel(step00.celltype_column)
        matplotlib.pyplot.ylabel(step00.clustering_column)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.celltype_column}-Score-Heatmap.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.celltype_column}-Score-Heatmap.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        seaborn.heatmap(data=cluster_counter_data, vmin=0, robust=True, cmap="YlOrRd", annot=True, fmt="d", annot_kws={"size": "xx-small"}, cbar=False, ax=ax)

        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
        matplotlib.pyplot.xlabel(step00.celltype_column)
        matplotlib.pyplot.ylabel(step00.clustering_column)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.celltype_column}-Count-Heatmap.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.celltype_column}-Count-Heatmap.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        seaborn.heatmap(data=cluster_counter_data.div(cluster_counter_data.sum(axis="columns"), axis="index"), vmin=0, robust=True, cmap="YlOrRd", annot=True, fmt=".2f", annot_kws={"size": "xx-small"}, cbar=False, ax=ax)

        matplotlib.pyplot.xticks(fontsize="xx-small")
        matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
        matplotlib.pyplot.xlabel(step00.celltype_column)
        matplotlib.pyplot.ylabel(step00.clustering_column)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.celltype_column}-Proportion-Heatmap.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.celltype_column}-Proportion-Heatmap.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        for cluster in tqdm.tqdm(cluster_list):
            cell_type = sorted(zip(cluster_score_data.loc[cluster, :], cell_type_list))[-1][1]

            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.scatterplot(data=clustering_data.sort_values(cell_type), x=step00.projection_columns[0], y=step00.projection_columns[1], hue=cell_type, palette="Reds", legend="brief", rasterized=True, s=30, edgecolor=None, ax=ax)
            step00.confidence_ellipse(clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[0]], clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[1]], ax=ax, edgecolor="black", linewidth=2.5)
            matplotlib.pyplot.text(numpy.mean(clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[0]]), numpy.mean(clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[1]]), cluster, horizontalalignment="center", verticalalignment="center", fontsize="medium", color="black", path_effects=step00.path_effects)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            matplotlib.pyplot.title(f"{cell_type} (Gene n={len(input_adata.uns[step00.marker_column][cell_type])})", fontsize="small")
            matplotlib.pyplot.legend(loc="lower left", title="")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Cluster-{cluster}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Cluster-{cluster}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

        seaborn.scatterplot(data=clustering_data, x=step00.projection_columns[0], y=step00.projection_columns[1], hue=step00.celltype_column, hue_order=cell_type_list, palette=cell_type_palette, rasterized=True, s=30, edgecolor=None, ax=ax)

        for cluster in tqdm.tqdm(cluster_list):
            step00.confidence_ellipse(clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[0]], clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[1]], ax=ax, edgecolor="black", linewidth=2.5)
            matplotlib.pyplot.text(numpy.mean(clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[0]]), numpy.mean(clustering_data.loc[(clustering_data[step00.clustering_column] == cluster), step00.projection_columns[1]]), cluster, horizontalalignment="center", verticalalignment="center", fontsize="medium", color="black", path_effects=step00.path_effects)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        matplotlib.pyplot.legend(loc="lower left", title=step00.celltype_column, title_fontsize="xx-small")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Scatter-{step00.celltype_column}.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Scatter-{step00.celltype_column}.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, cluster in tqdm.contrib.tenumerate(cluster_list):
            matplotlib.pyplot.bar(range(len(cell_type_list)), cluster_counter_data.iloc[i, :], bottom=cluster_counter_data.iloc[:i, :].sum(axis="index"), color=cluster_palette[cluster], label=cluster, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.celltype_column)
        matplotlib.pyplot.ylabel("Cell count")
        matplotlib.pyplot.xticks(range(len(cell_type_list)), cell_type_list, fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.celltype_column, loc="upper right", ncols=len(cluster_list) // 4)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.celltype_column}-{step00.clustering_column}-Count-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.celltype_column}-{step00.clustering_column}-Count-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, cluster in tqdm.contrib.tenumerate(cluster_list):
            matplotlib.pyplot.bar(range(len(cell_type_list)), (cluster_counter_data.iloc[i, :] / cluster_counter_data.sum(axis="index")), bottom=(cluster_counter_data.iloc[:i, :].sum(axis="index") / cluster_counter_data.sum(axis="index")), color=cluster_palette[cluster], label=cluster, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.celltype_column)
        matplotlib.pyplot.ylabel("Cell proportion")
        matplotlib.pyplot.xticks(range(len(cell_type_list)), cell_type_list, fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.clustering_column, loc="upper right", ncols=len(cluster_list) // 4)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.celltype_column}-{step00.clustering_column}-Proportion-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.celltype_column}-{step00.clustering_column}-Proportion-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, cell_type in tqdm.contrib.tenumerate(cell_type_list):
            matplotlib.pyplot.bar(range(len(cluster_list)), cluster_counter_data.iloc[:, i], bottom=cluster_counter_data.iloc[:, :i].sum(axis="columns"), color=cell_type_palette[cell_type], label=cell_type, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.clustering_column)
        matplotlib.pyplot.ylabel("Cell count")
        matplotlib.pyplot.xticks(range(len(cluster_list)), cluster_list, fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.celltype_column, loc="upper right", ncols=len(cell_type_list) // 4)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.clustering_column}-{step00.celltype_column}-Count-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.clustering_column}-{step00.celltype_column}-Count-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, cell_type in tqdm.contrib.tenumerate(cell_type_list):
            matplotlib.pyplot.bar(range(len(cluster_list)), (cluster_counter_data.iloc[:, i] / cluster_counter_data.sum(axis="columns")), bottom=(cluster_counter_data.iloc[:, :i].sum(axis="columns") / cluster_counter_data.sum(axis="columns")), color=cell_type_palette[cell_type], label=cell_type, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.clustering_column)
        matplotlib.pyplot.ylabel("Cell proportion")
        matplotlib.pyplot.xticks(range(len(cluster_list)), cluster_list, fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.celltype_column, loc="upper right", ncols=len(cell_type_list) // 4)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.clustering_column}-{step00.celltype_column}-Proportion-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.clustering_column}-{step00.celltype_column}-Proportion-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, cell_type in tqdm.contrib.tenumerate(cell_type_list):
            matplotlib.pyplot.bar(range(len(sample_list)), sample_counter_data.iloc[:, i], bottom=sample_counter_data.iloc[:, :i].sum(axis="columns"), color=cell_type_palette[cell_type], label=cell_type, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.sample_column)
        matplotlib.pyplot.ylabel("Cell count")
        matplotlib.pyplot.xticks(range(len(sample_list)), sample_list, fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.celltype_column, loc="upper right", ncols=len(cell_type_list) // 4)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.sample_column}-{step00.celltype_column}-Count-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.sample_column}-{step00.celltype_column}-Count-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        for i, cell_type in tqdm.contrib.tenumerate(cell_type_list):
            matplotlib.pyplot.bar(range(len(sample_list)), (sample_counter_data.iloc[:, i] / sample_counter_data.sum(axis="columns")), bottom=(sample_counter_data.iloc[:, :i].sum(axis="columns") / sample_counter_data.sum(axis="columns")), color=cell_type_palette[cell_type], label=cell_type, linewidth=0, edgecolor=None)

        matplotlib.pyplot.xlabel(step00.sample_column)
        matplotlib.pyplot.ylabel("Cell proportion")
        matplotlib.pyplot.xticks(range(len(sample_list)), sample_list, fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.legend(title=step00.celltype_column, loc="upper right", ncols=len(cell_type_list) // 4)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/{step00.sample_column}-{step00.celltype_column}-Proportion-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/{step00.sample_column}-{step00.celltype_column}-Proportion-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
