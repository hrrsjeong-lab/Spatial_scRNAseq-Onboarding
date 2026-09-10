import argparse
import itertools
import os
import tempfile
import typing
import zipfile
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import pandas
import scanpy
import seaborn
import tqdm
import tqdm.contrib
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

    gene_list = sorted(set().union(*input_adata.uns[step00.marker_column].values()) & set(input_adata.var.index))
    clustering_data = pandas.concat([clustering_data, scanpy.get.obs_df(input_adata, keys=gene_list, layer="Counts")], axis="columns", verify_integrity=True)
    print("Gene:", len(gene_list))
    print(clustering_data)

    sample_list = sorted(set(clustering_data["sample"]))
    sample_palette = dict(zip(sample_list, itertools.cycle(matplotlib.colors.TABLEAU_COLORS)))
    print("Sample:", len(sample_list), sample_list)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))

        figure_list.append(f"{directory}/Gene-scatter.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Gene-scatter.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        exit()
        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
