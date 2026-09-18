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
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)
    parser.add_argument("--clip", help="Symmetric z-score clipping for the color scale", type=float, default=50.0)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    if args.clip <= 0:
        raise ValueError("Clip must be positive!!")

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    result_data = input_adata.uns[step00.neighborhood_column]
    for column in ["Source", "Target", step00.sample_column]:
        result_data[column] = result_data[column].astype(str)
    print(result_data)

    cell_type_list = sorted(set(result_data["Source"]) | set(result_data["Target"]))
    print("Cell type:", len(cell_type_list), cell_type_list)

    sample_list = sorted(set(result_data[step00.sample_column]))
    sample_palette = dict(zip(sample_list, itertools.cycle(matplotlib.colors.TABLEAU_COLORS)))
    print("Sample:", len(sample_list), sample_list)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="white", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        zscore_dict: typing.Dict[str, pandas.DataFrame] = dict()
        n_cells_dict: typing.Dict[str, int] = dict()

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

            # z-score heatmap
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

            seaborn.heatmap(zscore, cmap="RdBu_r", vmin=-args.clip, vmax=args.clip, center=0.0, fmt=".0f", annot=True, annot_kws={"fontsize": "x-small"}, square=True, linewidths=0.5, linecolor="white", cbar=True, cbar_kws={"label": "Neighborhood enrichment (z-score)", "shrink": 0.7}, ax=ax)

            matplotlib.pyplot.xticks(fontsize="x-small", rotation="vertical")
            matplotlib.pyplot.yticks(fontsize="x-small")
            matplotlib.pyplot.xlabel("Neighbor cell type")
            matplotlib.pyplot.ylabel("Cell type")
            matplotlib.pyplot.title(f"{sample} (n={n_cells})")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Zscore-{sample}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Zscore-{sample}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

            # Neighbor composition heatmap: fraction of the neighbors of a Source cell that belong to each Target type
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

            seaborn.heatmap(data=composition, cmap="Blues", vmin=0.0, vmax=1.0, center=None, fmt=".2f", annot=True, annot_kws={"fontsize": "x-small"}, square=True, linewidths=0.5, linecolor="white", cbar=True, cbar_kws={"label": "Fraction of neighbors", "shrink": 0.7}, ax=ax)

            matplotlib.pyplot.xticks(fontsize="x-small", rotation="vertical")
            matplotlib.pyplot.yticks(fontsize="x-small")
            matplotlib.pyplot.xlabel("Neighbor cell type")
            matplotlib.pyplot.ylabel("Cell type")
            matplotlib.pyplot.title(f"{sample} (n={n_cells})")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Composition-{sample}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Composition-{sample}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        # Homotypic enrichment: how strongly each cell type clusters with itself, per sample
        diagonal_data = result_data.loc[(result_data["Source"] == result_data["Target"])].copy()

        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 24))

        seaborn.barplot(data=diagonal_data, x="Source", y="zscore", hue=step00.sample_column, order=cell_type_list, hue_order=sample_list, palette=sample_palette, ax=ax)

        matplotlib.pyplot.xlabel("Cell type")
        matplotlib.pyplot.ylabel("Homotypic z-score")
        matplotlib.pyplot.setp(ax.get_xticklabels(), rotation="vertical", horizontalalignment="right")
        matplotlib.pyplot.legend(title=step00.sample_column, loc="upper right")
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Homotypic-Bar.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Homotypic-Bar.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
