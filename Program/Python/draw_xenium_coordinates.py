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
import scanpy
import seaborn
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input H5AD file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

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

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

        seaborn.scatterplot(data=coordinate_data, x=step00.spatial_columns[0], y=step00.spatial_columns[1], hue=step00.celltype_column, hue_order=cell_type_list, palette=cell_type_palette, style=step00.sample_column, style_order=sample_list, rasterized=True, s=30, edgecolor=None, ax=ax)

        for sample in tqdm.tqdm(sample_list):
            step00.confidence_ellipse(coordinate_data.loc[(coordinate_data[step00.sample_column] == sample), step00.spatial_columns[0]], coordinate_data.loc[(coordinate_data[step00.sample_column] == sample), step00.spatial_columns[1]], ax=ax, edgecolor="black", linewidth=2.5)
            matplotlib.pyplot.text(numpy.mean(coordinate_data.loc[(coordinate_data[step00.sample_column] == sample), step00.spatial_columns[0]]), numpy.mean(coordinate_data.loc[(coordinate_data[step00.sample_column] == sample), step00.spatial_columns[0]]), sample, horizontalalignment="center", verticalalignment="center", fontsize="small", color="black", path_effects=step00.path_effects)

        matplotlib.pyplot.xlabel(f"{step00.spatial_columns[0]} (μm)")
        matplotlib.pyplot.ylabel(f"{step00.spatial_columns[1]} (μm)")
        matplotlib.pyplot.title(f"Total (Cell n={len(coordinate_data)})")
        matplotlib.pyplot.legend(loc="upper right")
        matplotlib.pyplot.tight_layout()

        figure_list.append(f"{directory}/Scatter-{step00.celltype_column}.pdf")
        fig.savefig(figure_list[-1])
        figure_list.append(f"{directory}/Scatter-{step00.celltype_column}.png")
        fig.savefig(figure_list[-1])
        matplotlib.pyplot.close(fig)

        for sample in tqdm.tqdm(sample_list):
            drawing_data = coordinate_data.loc[(coordinate_data[step00.sample_column] == sample)]

            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.scatterplot(data=drawing_data, x=step00.spatial_columns[0], y=step00.spatial_columns[1], hue=step00.celltype_column, hue_order=cell_type_list, palette=cell_type_palette, rasterized=True, s=30, edgecolor=None, ax=ax)

            for cell_type in tqdm.tqdm(cell_type_list):
                step00.confidence_ellipse(drawing_data.loc[(drawing_data[step00.celltype_column] == cell_type), step00.spatial_columns[0]], drawing_data.loc[(drawing_data[step00.celltype_column] == cell_type), step00.spatial_columns[1]], ax=ax, edgecolor="black", linewidth=2.5)
                matplotlib.pyplot.text(numpy.mean(drawing_data.loc[(drawing_data[step00.celltype_column] == cell_type), step00.spatial_columns[0]]), numpy.mean(drawing_data.loc[(drawing_data[step00.celltype_column] == cell_type), step00.spatial_columns[1]]), cell_type, horizontalalignment="center", verticalalignment="center", fontsize="small", color="black", path_effects=step00.path_effects)

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

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
