import argparse
import os
import tempfile
import typing
import zipfile
import matplotlib
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

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        pca_data = pandas.DataFrame(input_adata.obsm["X_pca"], index=input_adata.obs.index).iloc[:, :len(step00.projection_columns)]
        pca_data.columns = step00.projection_columns
        print(pca_data)

        selected_genes = list(input_adata.var[(input_adata.var["highly_variable"])].sort_values("dispersions_norm", ascending=False).iloc[:10, :].index)
        pca_data = pandas.concat([pca_data, scanpy.get.obs_df(input_adata, keys=selected_genes, layer="Counts")], axis="columns", verify_integrity=True, join="inner")
        print(pca_data)

        for gene in tqdm.tqdm(selected_genes):
            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.scatterplot(data=pca_data.sort_values(gene, ascending=True), x=step00.projection_columns[0], y=step00.projection_columns[1], hue=gene, palette="Reds", legend="brief", rasterized=True, s=40, edgecolor=None, ax=ax)

            matplotlib.pyplot.xlabel(f"PC-1 ({input_adata.uns[step00.pca_key]['variance_ratio'][0] * 100:.1f}%)")
            matplotlib.pyplot.ylabel(f"PC-2 ({input_adata.uns[step00.pca_key]['variance_ratio'][1] * 100:.1f}%)")
            matplotlib.pyplot.title(gene)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Scatter-{gene}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Scatter-{gene}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))
        for i, variance in tqdm.contrib.tenumerate(input_adata.uns[step00.pca_key]["variance_ratio"], start=1):
            i = i[0] + 1
            matplotlib.pyplot.text(i, variance, f"PC{i}", color="black", fontsize="x-small", horizontalalignment="center", rotation="vertical", verticalalignment="bottom", path_effects=step00.path_effects)

        matplotlib.pyplot.xticks(range(len(input_adata.uns[step00.pca_key]["variance_ratio"])), ["" for _ in input_adata.uns[step00.pca_key]["variance_ratio"]])
        matplotlib.pyplot.ylim(bottom=0, top=1.25 * max(input_adata.uns[step00.pca_key]["variance_ratio"]))
        matplotlib.pyplot.xlabel("Ranking")
        matplotlib.pyplot.ylabel("Explained variance")
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/ExplainedVariance-Hist.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/ExplainedVariance-Hist.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
