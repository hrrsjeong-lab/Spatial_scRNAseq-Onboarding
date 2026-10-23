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
import notebook_pdf
import step00

log_columns = {"variances"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".zip"})

    notebook = notebook_pdf.Notebook(title="Highly variable genes", script=__file__)

    with notebook.cell("Read data"):
        input_adata = scanpy.read_h5ad(args.input)
        print(input_adata)
        print(input_adata.var)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    with notebook.cell("Values"):
        value_columns = [column for column in tqdm.tqdm(["dispersions", "dispersions_norm", "variances", "variances_norm"]) if column in input_adata.var.columns]
        print("Value columns:", value_columns)

        var_data = input_adata.var.copy()
        plot_data_dict = dict()
        for column in tqdm.tqdm(value_columns):
            plot_data_dict[column] = var_data.loc[(var_data["means"] > 0) & (var_data[column] > 0)] if (column in log_columns) else var_data
            print(column, "genes plotted:", len(plot_data_dict[column]), "/", len(var_data), "(zero mean or variance excluded on log scale)" if (column in log_columns) else "")

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        for y in tqdm.tqdm(value_columns):
            with notebook.cell(f"Scatter-{y}"):
                plot_data = plot_data_dict[y]
                fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

                seaborn.scatterplot(data=plot_data, x="means", y=y, hue="highly_variable", hue_order=[True, False], palette={False: "gainsboro", True: "hotpink"}, legend="full", rasterized=True, s=40, edgecolor=None, ax=ax)
                if y in log_columns:
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                total_num = len(var_data)
                variable_num = len(var_data[(var_data["highly_variable"])])

                texts = list()
                for index, row in tqdm.tqdm(plot_data[(plot_data["highly_variable"])].sort_values(y, ascending=False).iterrows(), position=1, leave=False):
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

        for y in tqdm.tqdm(value_columns):
            with notebook.cell(f"2DHist-{y}"):
                fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

                seaborn.histplot(data=plot_data_dict[y], x="means", y=y, stat="count", bins=100, cmap="YlOrRd", cbar=True, log_scale=(y in log_columns), ax=ax)

                matplotlib.pyplot.xlabel("Mean expression")
                matplotlib.pyplot.ylabel("Variance in expression")
                matplotlib.pyplot.tight_layout()

                figure_list.append(f"{directory}/2DHist-{step00.format_filename(y)}.pdf")
                fig.savefig(figure_list[-1])
                figure_list.append(f"{directory}/2DHist-{step00.format_filename(y)}.png")
                fig.savefig(figure_list[-1])
                matplotlib.pyplot.close(fig)

        for x in tqdm.tqdm(value_columns):
            with notebook.cell(f"Distribution-{x}"):
                fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

                seaborn.histplot(data=plot_data_dict[x], x=x, stat="percent", bins=100, kde=True, log_scale=(x in log_columns), ax=ax)

                y = plot_data_dict[x][x]
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

        figure_list.append(f"{directory}/Notebook.pdf")
        notebook.write_pdf(figure_list[-1])

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
