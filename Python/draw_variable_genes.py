import argparse
import os
import tempfile
import typing
import zipfile
import adjustText
import matplotlib
import matplotlib.pyplot
import numpy
import scanpy
import seaborn
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)
    print(input_adata.var)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        for y in tqdm.tqdm(["dispersions", "dispersions_norm"]):
            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.scatterplot(data=input_adata.var, x="means", y=y, hue="highly_variable", hue_order=[True, False], palette={False: "gainsboro", True: "hotpink"}, legend="full", rasterized=True, s=40, edgecolor=None, ax=ax)

            total_num = len(input_adata.var)
            variable_num = len(input_adata.var[(input_adata.var["highly_variable"])])

            texts = list()
            for index, row in tqdm.tqdm(input_adata.var[(input_adata.var["highly_variable"])].sort_values(y, ascending=False).iterrows(), position=1, leave=False):
                texts.append(matplotlib.pyplot.text(row["means"], row[y], index, color="black", fontsize="xx-small", horizontalalignment="center", verticalalignment="center", path_effects=step00.path_effects))

                if len(texts) > 10:
                    break

            matplotlib.pyplot.xlabel("Mean expression")
            matplotlib.pyplot.ylabel("Variance in expression")
            matplotlib.pyplot.title(f"Highly variable: {variable_num}/{total_num} ({variable_num / total_num * 100:.1f}%)")
            matplotlib.pyplot.legend(title="Highly variable")
            matplotlib.pyplot.tight_layout()

            adjustText.adjust_text(texts, arrowprops=step00.arrowprops, ax=ax, time_lim=1)

            figure_list.append(f"{directory}/Scatter-{step00.format_filename(y)}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Scatter-{step00.format_filename(y)}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        for y in tqdm.tqdm(["dispersions", "dispersions_norm"]):
            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.histplot(data=input_adata.var, x="means", y=y, stat="count", bins=100, cmap="YlOrRd", cbar=True, ax=ax)

            matplotlib.pyplot.xlabel("Mean expression")
            matplotlib.pyplot.ylabel("Variance in expression")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/2DHist-{step00.format_filename(y)}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/2DHist-{step00.format_filename(y)}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        for x in tqdm.tqdm(["dispersions", "dispersions_norm"]):
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

            seaborn.histplot(data=input_adata.var, x=x, stat="percent", bins=100, kde=True, ax=ax)

            y = input_adata.var[x]
            s = f"Min–Max: {numpy.min(y):+.2f}–{numpy.max(y):+.2f};"
            s += f"Mean±Std: {numpy.mean(y):.2f}±{numpy.std(y):.2f}"

            matplotlib.pyplot.axvline(numpy.mean(y), color="black", linestyle="--", linewidth=2.5)
            matplotlib.pyplot.xlabel("Variance in expression")
            matplotlib.pyplot.title(s)
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Distribution-{step00.format_filename(x)}.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Distribution-{step00.format_filename(x)}.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
