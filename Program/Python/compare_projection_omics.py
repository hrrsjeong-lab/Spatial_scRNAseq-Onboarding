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
import notebook_pdf
import step00

log_columns = {"variances"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Imput HDF5 file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    notebook = notebook_pdf.Notebook(title="Compare variable genes", script=__file__)

    with notebook.cell("Read data"):
        input_adata = scanpy.read_h5ad(args.input)
        print(input_adata)

    with notebook.cell("Projection"):
        projection_data = pandas.DataFrame(input_adata.obsm[step00.projection_key], index=input_adata.obs.index, columns=step00.projection_columns)
        for column in tqdm.tqdm([step00.sample_column, step00.omics_column]):
            projection_data[column] = input_adata.obs[column]
        print(projection_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        with notebook.cell("Scatter"):
            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))
            seaborn.scatterplot(data=projection_data, x=step00.projection_columns[0], y=step00.projection_columns[1], hue=step00.omics_column, hue_order=["Xenium", "scRNA"], style=step00.omics_column, legend="brief", rasterized=True, s=5, edgecolor=None, ax=ax)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            matplotlib.pyplot.title(f"Total {len(projection_data)} cells")
            matplotlib.pyplot.legend(loc="upper right", markerscale=10)
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Scatter.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Scatter.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))

        notebook.write_pdf(args.output)
