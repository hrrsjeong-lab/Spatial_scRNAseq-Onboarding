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

        for sample in tqdm.tqdm(sample_list):
            drawing_data = coordinate_data.loc[(coordinate_data[step00.sample_column] == sample)]
            minimum = numpy.min(drawing_data[[*step00.spatial_columns]].to_numpy()) - 500
            maximum = numpy.max(drawing_data[[*step00.spatial_columns]].to_numpy()) + 500

            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.scatterplot(data=drawing_data, x=step00.spatial_columns[0], y=step00.spatial_columns[1], hue=step00.celltype_column, hue_order=cell_type_list, palette=cell_type_palette, rasterized=True, s=15, edgecolor=None, ax=ax)

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

        for sample in tqdm.tqdm(sample_list):
            drawing_data = coordinate_data.loc[(coordinate_data[step00.sample_column] == sample)]
            minimum = numpy.min(drawing_data[[*step00.spatial_columns]].to_numpy()) - 500
            maximum = numpy.max(drawing_data[[*step00.spatial_columns]].to_numpy()) + 500
            highlight_data = pandas.concat({cell_type: drawing_data.assign(Highlight=numpy.where(drawing_data[step00.celltype_column] == cell_type, cell_type, step00.rest_value)) for cell_type in cell_type_list}, names=["Panel"]).reset_index(level="Panel")

            g = seaborn.relplot(data=highlight_data, x=step00.spatial_columns[0], y=step00.spatial_columns[1], hue="Highlight", hue_order=[step00.rest_value] + cell_type_list, palette=cell_type_palette | {step00.rest_value: step00.rest_color}, col="Panel", col_wrap=4, kind="scatter", s=10, linewidth=0, height=18, aspect=1, legend=False, facet_kws={"sharex": True, "sharey": True})
            for ax in g.axes.flat:
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
            g.set_titles("{col_name}",)

            figure_list.append(f"{directory}/Relplot-{sample}.pdf")
            g.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Relplot-{sample}.png")
            g.savefig(figure_list[-1])
            matplotlib.pyplot.close(g.fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
