import argparse
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
import scipy
import seaborn
import tqdm
import step00


def annotation_size(n_columns: int, max_chars: int, width: float) -> float:
    cell_points = width * 72.0 * 0.55 / max(n_columns, 1)
    return float(numpy.clip(cell_points / (0.65 * max(max_chars, 1)), 3.0, 20.0))


def style_colorbar(ax, label: str) -> None:
    colorbar = ax.collections[0].colorbar
    colorbar.set_label(label, fontsize="x-small")
    colorbar.ax.tick_params(labelsize="xx-small")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)
    parser.add_argument("--clip", help="Symmetric z-score clipping for the color scale", type=float, default=50.0)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    assert (args.clip > 0), "Clip must be positive!!"

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    result_data = input_adata.uns[step00.neighborhood_column]
    for column in tqdm.tqdm(["Source", "Target", step00.sample_column]):
        result_data[column] = result_data[column].astype(str)
    print(result_data)

    diagnostic_data = result_data.loc[(result_data["Source"] <= result_data["Target"])].copy()
    diagnostic_data["n_pair"] = numpy.sqrt(diagnostic_data["n_source"].astype(float) * diagnostic_data["n_target"].astype(float))
    diagnostic_data["abs_zscore"] = diagnostic_data["zscore"].abs()
    diagnostic_data = diagnostic_data.loc[(diagnostic_data["n_pair"] > 0) & (numpy.isfinite(diagnostic_data["abs_zscore"]))]
    print(diagnostic_data)

    correlation, p_value = scipy.stats.spearmanr(diagnostic_data["n_pair"], diagnostic_data["abs_zscore"])
    print(f"Corr: {correlation:.3f}, p: {p_value:.2e}")

    degree_column = f"{step00.neighborhood_column}_degree"
    degree_data = input_adata.obs[[step00.sample_column, step00.celltype_column, degree_column]].copy()
    degree_data[step00.sample_column] = degree_data[step00.sample_column].astype(str)
    degree_data[step00.celltype_column] = degree_data[step00.celltype_column].astype(str)
    print(degree_data.groupby(step00.sample_column, observed=True)[degree_column].describe())

    isolated_data = degree_data.assign(Isolated=(degree_data[degree_column] == 0)).groupby(step00.sample_column, observed=True)["Isolated"].mean().rename("fraction").reset_index()
    print(isolated_data)

    cell_type_list = sorted(set(result_data["Source"]) | set(result_data["Target"]))
    cell_type_palette = dict(zip(cell_type_list, itertools.cycle(matplotlib.colors.XKCD_COLORS)))
    print("Cell type:", len(cell_type_list), cell_type_list)

    sample_list = sorted(set(result_data[step00.sample_column]))
    sample_palette = dict(zip(sample_list, itertools.cycle(matplotlib.colors.TABLEAU_COLORS)))
    print("Sample:", len(sample_list), sample_list)

    mean_degree_data = degree_data.pivot_table(index=step00.celltype_column, columns=step00.sample_column, values=degree_column, aggfunc="mean").reindex(index=cell_type_list, columns=sample_list)
    isolated_cell_data = degree_data.assign(Isolated=(degree_data[degree_column] == 0)).pivot_table(index=step00.celltype_column, columns=step00.sample_column, values="Isolated", aggfunc="mean").reindex(index=cell_type_list, columns=sample_list)
    print(mean_degree_data)
    print(isolated_cell_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="white", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        zscore_dict: typing.Dict[str, pandas.DataFrame] = dict()
        n_cells_dict: typing.Dict[str, int] = dict()

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

        seaborn.scatterplot(data=diagnostic_data, x="n_pair", y="abs_zscore", hue=step00.sample_column, hue_order=sample_list, palette=sample_palette, s=200, edgecolor=None, ax=ax)
        seaborn.regplot(data=diagnostic_data, x="n_pair", y="abs_zscore", scatter=False, logx=True, color="black", line_kws={"linewidth": 2}, ax=ax)

        matplotlib.pyplot.xscale("log")
        matplotlib.pyplot.xticks(fontsize="x-small")
        matplotlib.pyplot.yticks(fontsize="x-small")
        matplotlib.pyplot.xlabel("sqrt(n_source × n_target)", fontsize="small")
        matplotlib.pyplot.ylabel("abs(Neighborhood enrichment z-score)", fontsize="small")
        matplotlib.pyplot.legend(title=step00.sample_column, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Diagnostic-Zscore_vs_N.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Diagnostic-Zscore_vs_N.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        for sample in tqdm.tqdm(sample_list):
            sample_data = result_data.loc[(result_data[step00.sample_column] == sample)]

            zscore = sample_data.pivot(index="Source", columns="Target", values="zscore").reindex(index=cell_type_list, columns=cell_type_list)
            count = sample_data.pivot(index="Source", columns="Target", values="count").reindex(index=cell_type_list, columns=cell_type_list)
            composition = count.div(count.sum(axis="columns"), axis="index")

            n_source = sample_data.drop_duplicates("Source").set_index("Source")["n_source"].reindex(cell_type_list)
            n_cells = int(n_source.dropna().sum())
            zscore_dict[sample] = zscore
            n_cells_dict[sample] = n_cells
            y_labels = [f"{cell_type} (n={n})" for cell_type, n in zip(cell_type_list, n_source)]

            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.heatmap(zscore, cmap="RdBu_r", vmin=-args.clip, vmax=args.clip, center=0.0, fmt=".0f", annot=True, annot_kws={"fontsize": annotation_size(len(cell_type_list), 3, 18)}, square=True, linewidths=0.5, linecolor="white", cbar=True, cbar_kws={"shrink": 0.7}, ax=ax)
            style_colorbar(ax, "Neighborhood enrichment (z-score)")

            matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
            matplotlib.pyplot.yticks(fontsize="xx-small")
            matplotlib.pyplot.xlabel("Neighbor cell type", fontsize="small")
            matplotlib.pyplot.ylabel("Cell type", fontsize="small")
            matplotlib.pyplot.title(f"{sample} (n={n_cells})", fontsize="small")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Zscore-{sample}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Zscore-{sample}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.heatmap(data=composition, cmap="Blues", vmin=0.0, vmax=1.0, center=None, fmt=".2f", annot=True, annot_kws={"fontsize": annotation_size(len(cell_type_list), 4, 18)}, square=True, linewidths=0.5, linecolor="white", cbar=True, cbar_kws={"shrink": 0.7}, ax=ax)
            style_colorbar(ax, "Fraction of neighbors")

            matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
            matplotlib.pyplot.yticks(fontsize="xx-small")
            matplotlib.pyplot.xlabel("Neighbor cell type", fontsize="small")
            matplotlib.pyplot.ylabel("Cell type", fontsize="small")
            matplotlib.pyplot.title(f"{sample} (n={n_cells})", fontsize="small")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Composition-{sample}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Composition-{sample}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        diagonal_data = result_data.loc[(result_data["Source"] == result_data["Target"])].copy()

        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 24))

        seaborn.barplot(data=diagonal_data, x="Source", y="zscore", hue=step00.sample_column, order=cell_type_list, hue_order=sample_list, palette=sample_palette, ax=ax)

        matplotlib.pyplot.xlabel("Cell type", fontsize="small")
        matplotlib.pyplot.ylabel("Homotypic z-score", fontsize="small")
        matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
        matplotlib.pyplot.yticks(fontsize="x-small")
        matplotlib.pyplot.legend(title=step00.sample_column, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Homotypic-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Homotypic-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 24))

        seaborn.ecdfplot(data=degree_data, x=degree_column, hue=step00.sample_column, hue_order=sample_list, palette=sample_palette, linewidth=3, ax=ax)

        matplotlib.pyplot.xlim(0, float(numpy.quantile(degree_data[degree_column], 0.99)))
        matplotlib.pyplot.xticks(fontsize="x-small")
        matplotlib.pyplot.yticks(fontsize="x-small")
        matplotlib.pyplot.xlabel(f"Neighbors within {input_adata.uns[f'{step00.neighborhood_column}_parameters']['radius']} μm", fontsize="small")
        matplotlib.pyplot.ylabel("Cumulative fraction of cells", fontsize="small")
        seaborn.move_legend(ax, "lower right")
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Degree-ECDF.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Degree-ECDF.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        for name, data, title in [("Mean", mean_degree_data, "Mean neighborhood degree"), ("Isolated", isolated_cell_data, "Fraction of isolated cells")]:
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

            seaborn.heatmap(data=data, cmap="viridis", vmin=0.0, vmax=(1.0 if name == "Isolated" else None), annot=True, fmt=".2f", annot_kws={"fontsize": annotation_size(len(sample_list), 4, 24)}, linewidths=0.5, linecolor="white", cbar=True, cbar_kws={"shrink": 0.7}, ax=ax)
            style_colorbar(ax, "")

            matplotlib.pyplot.xticks(fontsize="xx-small", rotation="vertical")
            matplotlib.pyplot.yticks(fontsize="xx-small", rotation="horizontal")
            matplotlib.pyplot.xlabel(step00.sample_column, fontsize="small")
            matplotlib.pyplot.ylabel("Cell type", fontsize="small")
            matplotlib.pyplot.title(title, fontsize="small")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Degree-{name}-Heatmap.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Degree-{name}-Heatmap.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

        seaborn.barplot(data=isolated_data, x="fraction", y=step00.sample_column, order=sample_list, hue=step00.sample_column, hue_order=sample_list, palette=sample_palette, legend=False, orient="h", ax=ax)

        matplotlib.pyplot.xlim(0.0, 1.0)
        matplotlib.pyplot.xlabel("Fraction of isolated cells", fontsize="small")
        matplotlib.pyplot.ylabel(step00.sample_column, fontsize="small")
        matplotlib.pyplot.xticks(fontsize="x-small")
        matplotlib.pyplot.yticks(fontsize="xx-small")
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.tight_layout()
        figure_list.append(f"{directory}/Degree-Isolated.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Degree-Isolated.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
