import argparse
import os
import tempfile
import typing
import zipfile
import matplotlib
import matplotlib.pyplot
import scanpy
import seaborn
import tqdm
import upsetplot
import notebook_pdf
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("xenium", help="Xenium HDF5 file", type=str)
    parser.add_argument("scrna", help="scRNA HDF5 file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.xenium, {".hdf5", ".h5"})
    step00.check_suffix(args.scrna, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    notebook = notebook_pdf.Notebook(title="QC filtered data", script=__file__)

    with notebook.cell("Input data"):
        xenium_adata = scanpy.read_h5ad(args.xenium, backed="r")
        print(xenium_adata)

        scrna_adata = scanpy.read_h5ad(args.scrna, backed="r")
        print(scrna_adata)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        with notebook.cell("Read data"):
            xenium_gene_list = list(xenium_adata.var_names.astype(str))
            print("Xenium:", len(xenium_gene_list), xenium_gene_list[:5])

            scrna_gene_list = list(scrna_adata.var_names.astype(str))
            print("scRNA:", len(scrna_gene_list), scrna_gene_list[:5])

        with notebook.cell("Upset plot"):
            fig = matplotlib.pyplot.figure(figsize=(18, 24))

            input_dict = {"Xenium": set(xenium_gene_list), "scRNA": set(scrna_gene_list)}
            upsetplot.plot(upsetplot.from_contents(input_dict), fig=fig, show_counts="%d", show_percentages=True, element_size=None)

            figure_list.append(f"{directory}/Upset.pdf")
            fig.savefig(figure_list[-1], bbox_inches="tight")
            figure_list.append(f"{directory}/Upset.png")
            fig.savefig(figure_list[-1], bbox_inches="tight")
            matplotlib.pyplot.close(fig)

        figure_list.append(f"{directory}/Notebook.pdf")
        notebook.write_pdf(figure_list[-1])

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
