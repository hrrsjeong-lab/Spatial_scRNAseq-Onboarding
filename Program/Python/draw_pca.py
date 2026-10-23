import argparse
import os
import tempfile
import typing
import zipfile
import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import scanpy
import seaborn
import tqdm
import tqdm.contrib
import notebook_pdf
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="INPUT hdf5 file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    notebook = notebook_pdf.Notebook(title="QC filtered data", script=__file__)

    with notebook.cell("Input data"):
        input_adata = scanpy.read_h5ad(args.input)
        print(input_adata)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    pca_information = input_adata.uns[step00.pca_key]
    variance_ratio = numpy.asarray(pca_information["variance_ratio"], dtype=float)

    with notebook.cell("Variance ratio"):
        variance_data = pandas.DataFrame({"PC": numpy.arange(1, variance_ratio.size + 1), "Observed": variance_ratio * 100, "Cumulative": numpy.cumsum(variance_ratio) * 100})
        if "null_variance_ratio" in pca_information:
            variance_data["Permutation null"] = numpy.asarray(pca_information["null_variance_ratio"], dtype=float) * 100
            variance_data["Fold over null"] = variance_data["Observed"] / variance_data["Permutation null"]
        print(variance_data.to_string())

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        with notebook.cell("PCA data"):
            pca_data = pandas.DataFrame(input_adata.obsm[step00.pca_key], index=input_adata.obs.index).iloc[:, :len(step00.projection_columns)]
            pca_data.columns = step00.projection_columns
            print(pca_data)

        with notebook.cell("Variance"):
            rank_column = "dispersions_norm" if ("dispersions_norm" in input_adata.var.columns) else "variances_norm"
            selected_genes = list(input_adata.var[(input_adata.var["highly_variable"])].sort_values(rank_column, ascending=False).iloc[:5, :].index)
            pca_data = pandas.concat([pca_data, scanpy.get.obs_df(input_adata, keys=selected_genes)], axis="columns", verify_integrity=True, join="inner")
            print(pca_data)

        for gene in tqdm.tqdm(selected_genes):
            with notebook.cell(f"Scatter-{gene}"):
                fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

                seaborn.scatterplot(data=pca_data.sort_values(gene, ascending=True), x=step00.projection_columns[0], y=step00.projection_columns[1], hue=gene, palette="Reds", legend="brief", rasterized=True, s=40, edgecolor=None, ax=ax)

                matplotlib.pyplot.xlabel(f"PC-1 ({variance_ratio[0] * 100:.1f}%)")
                matplotlib.pyplot.ylabel(f"PC-2 ({variance_ratio[1] * 100:.1f}%)")
                matplotlib.pyplot.title(gene)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                matplotlib.pyplot.tight_layout()

                figure_list.append(f"{directory}/Scatter-{gene}.pdf")
                fig.savefig(figure_list[-1])
                figure_list.append(f"{directory}/Scatter-{gene}.png")
                fig.savefig(figure_list[-1])
                matplotlib.pyplot.close(fig)

        with notebook.cell("Hist variance"):
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))
            for i, variance in tqdm.contrib.tenumerate(variance_ratio, start=1):
                i = i[0] + 1
                matplotlib.pyplot.text(i, variance, f"PC{i}", color="black", fontsize="x-small", horizontalalignment="center", rotation="vertical", verticalalignment="bottom", path_effects=step00.path_effects)

            matplotlib.pyplot.xticks(range(variance_ratio.size), ["" for _ in variance_ratio])
            matplotlib.pyplot.ylim(bottom=0, top=1.25 * max(variance_ratio))
            matplotlib.pyplot.xlabel("Ranking")
            matplotlib.pyplot.ylabel("Explained variance")
            matplotlib.pyplot.grid(True)
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/ExplainedVariance-Hist.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/ExplainedVariance-Hist.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        line_columns = [column for column in ("Observed", "Permutation null") if column in variance_data.columns]
        line_data = variance_data.melt(id_vars="PC", value_vars=line_columns, var_name="Type", value_name="Explained variance (%)")

        with notebook.cell("Line variance"):
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))
            cumulative_ax = ax.twinx()

            seaborn.lineplot(data=line_data, x="PC", y="Explained variance (%)", hue="Type", hue_order=line_columns, palette={"Observed": "tab:red", "Permutation null": "tab:blue"}, linewidth=5, ax=ax)
            cumulative_ax.plot(variance_data["PC"], variance_data["Cumulative"], color="dimgray", linestyle="--", linewidth=5, label="Cumulative")

            title = f"{pca_information.get('n_features', '?')} features"
            if "signal_pcs" in pca_information:
                signal_pcs = int(pca_information["signal_pcs"])
                ax.axvline(signal_pcs + 0.5, color="black", linestyle=":", linewidth=3)
                title += f"; {signal_pcs} PCs above null ({variance_data['Cumulative'].iloc[max(signal_pcs, 1) - 1]:.1f}%)"

            ax.set_ylim(bottom=0)
            cumulative_ax.set_ylim(bottom=0, top=max(variance_data["Cumulative"]) * 1.05)
            cumulative_ax.grid(False)
            ax.set_xlabel("PC")
            cumulative_ax.set_ylabel("Cumulative explained variance (%)")
            ax.set_title(title)
            ax.legend(handles=ax.get_legend_handles_labels()[0] + cumulative_ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1] + cumulative_ax.get_legend_handles_labels()[1], loc="center right")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/ExplainedVariance-Null.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/ExplainedVariance-Null.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        figure_list.append(f"{directory}/Notebook.pdf")
        notebook.write_pdf(figure_list[-1])

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
