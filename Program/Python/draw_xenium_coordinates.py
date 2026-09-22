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
import scipy.spatial
import seaborn
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input H5AD file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)
    parser.add_argument("--knn", help="Number of neighbors for the local density estimate", type=int, default=10)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    if args.knn < 1:
        raise ValueError("KNN must be positive!!")

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    coordinate_data = input_adata.obs[[*step00.spatial_columns, step00.sample_column, step00.celltype_column]]
    print(coordinate_data)

    cell_type_list = sorted(set(input_adata.obs[step00.celltype_column]))
    cell_type_palette = dict(zip(cell_type_list, itertools.cycle(matplotlib.colors.XKCD_COLORS)))
    print("Cell type:", len(cell_type_list), cell_type_list)

    sample_list = sorted(set(input_adata.obs[step00.sample_column]))
    sample_palette = dict(zip(sample_list, itertools.cycle(matplotlib.colors.TABLEAU_COLORS)))
    print("Sample:", len(sample_list), sample_list)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        density_list = list()

        for sample in tqdm.tqdm(sample_list):
            drawing_data = coordinate_data.loc[(coordinate_data[step00.sample_column] == sample)]
            minimum = numpy.min(drawing_data[[*step00.spatial_columns]].to_numpy()) - 500
            maximum = numpy.max(drawing_data[[*step00.spatial_columns]].to_numpy()) + 500

            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.scatterplot(data=drawing_data, x=step00.spatial_columns[0], y=step00.spatial_columns[1], hue=step00.celltype_column, hue_order=cell_type_list, palette=cell_type_palette, rasterized=True, s=5, edgecolor=None, ax=ax)

            matplotlib.pyplot.xlim(minimum, maximum)
            matplotlib.pyplot.ylim(minimum, maximum)
            matplotlib.pyplot.xlabel(f"{step00.spatial_columns[0]} (μm)")
            matplotlib.pyplot.ylabel(f"{step00.spatial_columns[1]} (μm)")
            matplotlib.pyplot.title(f"{sample} (Cell n={len(drawing_data)})")
            matplotlib.pyplot.legend(loc="upper right")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Scatter-{sample}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Scatter-{sample}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

            highlight_data = pandas.concat({cell_type: drawing_data.assign(Highlight=numpy.where(drawing_data[step00.celltype_column] == cell_type, cell_type, step00.rest_value)) for cell_type in cell_type_list}, names=["Panel"]).reset_index(level="Panel")

            g = seaborn.relplot(data=highlight_data, x=step00.spatial_columns[0], y=step00.spatial_columns[1], hue="Highlight", hue_order=[step00.rest_value] + cell_type_list, palette=cell_type_palette | {step00.rest_value: step00.rest_color}, col="Panel", col_wrap=4, kind="scatter", s=5, linewidth=0, height=6, aspect=1, legend=False, facet_kws={"sharex": True, "sharey": True})
            for ax in g.axes.flat:
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
            g.set_titles("{col_name}", fontsize=1)
            g.tight_layout()

            figure_list.append(f"{directory}/Relplot-{sample}.pdf")
            g.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Relplot-{sample}.png")
            g.savefig(figure_list[-1])
            matplotlib.pyplot.close(g.figure)

            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.histplot(data=drawing_data, x=step00.spatial_columns[0], y=step00.spatial_columns[1], stat="count", bins=100, cmap="YlOrRd", cbar=True, ax=ax)

            matplotlib.pyplot.xlim(minimum, maximum)
            matplotlib.pyplot.ylim(minimum, maximum)
            matplotlib.pyplot.xlabel(f"{step00.spatial_columns[0]} (μm)")
            matplotlib.pyplot.ylabel(f"{step00.spatial_columns[1]} (μm)")
            matplotlib.pyplot.title(f"{sample} (Cell n={len(drawing_data)})")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Hist-{sample}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Hist-{sample}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

            coordinates = drawing_data[[*step00.spatial_columns]].to_numpy()

            tree = scipy.spatial.cKDTree(coordinates)
            radius = numpy.maximum(tree.query(coordinates, k=args.knn + 1)[0][:, -1], 1e-3)
            drawing_data["Density"] = (args.knn / (numpy.pi * (radius ** 2))) * 1e6
            density_list.append(drawing_data)

            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

            points = matplotlib.pyplot.scatter(drawing_data[step00.spatial_columns[0]], drawing_data[step00.spatial_columns[1]], c=drawing_data["Density"], cmap="viridis", norm=matplotlib.colors.LogNorm(), s=15, linewidths=0, rasterized=True)
            fig.colorbar(points, ax=ax, shrink=0.7, label="Local density (cells/mm²)")

            matplotlib.pyplot.xlim(minimum, maximum)
            matplotlib.pyplot.ylim(minimum, maximum)
            matplotlib.pyplot.xlabel(f"{step00.spatial_columns[0]} (μm)")
            matplotlib.pyplot.ylabel(f"{step00.spatial_columns[1]} (μm)")
            matplotlib.pyplot.title(f"{sample} (Cell n={len(drawing_data)}, k={args.knn})")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Density-{sample}.png")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Density-{sample}.pdf")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        density_data = pandas.concat(density_list, ignore_index=True)
        density_data["log_Density"] = numpy.log10(density_data["Density"])
        print(density_data)

        fig, ax = matplotlib.pyplot.subplots(figsize=(32, 24))

        seaborn.violinplot(data=density_data, x=step00.celltype_column, y="log_Density", hue=step00.sample_column, order=cell_type_list, hue_order=sample_list, palette=sample_palette, cut=0, linewidth=2, ax=ax)

        matplotlib.pyplot.xlabel("Cell type")
        matplotlib.pyplot.ylabel("log Local density (cells/mm²)")
        matplotlib.pyplot.xticks(rotation="vertical")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Density-Violin.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Density-Violin.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
