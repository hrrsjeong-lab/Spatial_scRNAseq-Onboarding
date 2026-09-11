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
import scipy
import seaborn
import tqdm
import tqdm.contrib.itertools
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("cell", help="Output TSV(.gz) file", type=str)
    parser.add_argument("gene", help="Output TSV(.gz) file", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.cell, {".tsv", ".tsv.gz"})
    step00.check_suffix(args.gene, {".tsv", ".tsv.gz"})
    step00.check_suffix(args.output, {".zip"})

    cell_data = pandas.read_csv(args.cell, sep="\t", index_col=0)
    cell_test_list = list(cell_data.columns)[:4]
    print("Cell test:", len(cell_test_list), cell_test_list)
    print(cell_data)

    cell_data[step00.sample_column] = list(map(lambda x: x.split("-")[-1], list(cell_data.index)))
    sample_list = sorted(set(cell_data[step00.sample_column]))
    sample_palette = dict(zip(sample_list, itertools.cycle(matplotlib.colors.TABLEAU_COLORS)))
    print("Sample:", len(sample_list), sample_list)

    sample_series = cell_data[step00.sample_column]
    sample_columns = pandas.DataFrame({sample: numpy.where(sample_series == sample, sample, step00.rest_value) for sample in tqdm.tqdm(sample_list)}, index=cell_data.index)
    cell_data = pandas.concat([cell_data, sample_columns], axis=1)
    print(cell_data)
    del sample_columns

    gene_data = pandas.read_csv(args.gene, sep="\t", index_col=0)
    gene_test_list = list(gene_data.columns)
    print("Gene test:", len(gene_test_list), gene_test_list)
    print(gene_data)

    gene_list = list(gene_data.index)
    gene_palette = dict(zip(gene_list, itertools.cycle(matplotlib.colors.XKCD_COLORS)))
    gene_data["Gene"] = gene_list
    print(gene_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)
    numpy.seterr(all="ignore")

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        for cell_test in tqdm.tqdm(cell_test_list):
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

            seaborn.histplot(data=cell_data, x=cell_test, hue=step00.sample_column, stat="count", bins=100, multiple="stack", common_norm=True, kde=True, palette=sample_palette, hue_order=sample_list, ax=ax)

            matplotlib.pyplot.xlabel(cell_test.replace("_", " "))
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test)}-Hist.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test)}-Hist.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        for cell_test in tqdm.tqdm(cell_test_list):
            g = seaborn.FacetGrid(data=cell_data, row=step00.sample_column, hue=step00.sample_column, hue_order=sample_list, aspect=5, height=5, palette=sample_palette)

            g.map(seaborn.kdeplot, cell_test, bw_adjust=0.5, clip_on=False, fill=True, linewidth=1.5)
            g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

            def label(x, color, label):
                ax = matplotlib.pyplot.gca()
                ax.text(0, 0.5, label, color="black", fontsize="x-small", horizontalalignment="left", verticalalignment="center", transform=ax.transAxes, path_effects=step00.path_effects)

            g.map(label, step00.sample_column)
            g.set_titles("")
            g.set_axis_labels("", "")
            g.figure.supxlabel(cell_test.replace("_", " "))
            g.figure.supylabel("Proportion")
            g.despine(bottom=True, left=True)
            g.tight_layout()

            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test)}-HistMutiple.pdf")
            g.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test)}-HistMutiple.png")
            g.savefig(figure_list[-1])
            matplotlib.pyplot.close(g.figure)

        for cell_test in tqdm.tqdm(cell_test_list):
            p_data = pandas.DataFrame(data=numpy.zeros((len(sample_list), len(sample_list))), index=sample_list, columns=sample_list, dtype=float)

            for sample_a, sample_b in tqdm.contrib.itertools.product(sample_list, sample_list, position=1, leave=False):
                if sample_a == sample_b:
                    p_data.loc[sample_a, sample_b] = -numpy.log10(1.0)
                    continue

                p_value = scipy.stats.mannwhitneyu(cell_data.loc[(cell_data[step00.sample_column] == sample_a), cell_test], cell_data.loc[(cell_data[step00.sample_column] == sample_b), cell_test])[1]

                if numpy.isnan(p_value):
                    p_data.loc[sample_a, sample_b] = -numpy.log10(1.0)
                else:
                    p_data.loc[sample_a, sample_b] = -numpy.log10(p_value)

            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.heatmap(data=p_data, cmap="Reds_r", robust=True, cbar=True, square=True, xticklabels=True, yticklabels=True, ax=ax)

            matplotlib.pyplot.title(cell_test.replace("_", " "))
            matplotlib.pyplot.xlabel("")
            matplotlib.pyplot.ylabel("")
            matplotlib.pyplot.xticks(rotation="vertical", fontsize="xx-small")
            matplotlib.pyplot.yticks(rotation="horizontal", fontsize="xx-small")
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test)}-Heatmap.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test)}-Heatmap.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        for cell_test in tqdm.tqdm(cell_test_list):
            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.violinplot(data=cell_data, x=step00.sample_column, order=sample_list, y=cell_test, hue=step00.sample_column, hue_order=sample_list, palette=sample_palette, inner=None, cut=0, linewidth=4, ax=ax)

            matplotlib.pyplot.xticks(rotation="vertical", fontsize="x-small")
            matplotlib.pyplot.xlabel("")
            matplotlib.pyplot.ylabel(cell_test.replace("_", " "))
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test)}-Violin.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test)}-Violin.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        for cell_test_1, cell_test_2 in tqdm.contrib.itertools.combinations(cell_test_list, r=2):
            fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

            seaborn.histplot(data=cell_data, x=cell_test_1, y=cell_test_2, stat="count", bins=100, cmap="YlOrRd", cbar=True, ax=ax)

            matplotlib.pyplot.xlabel(cell_test_1.replace("_", " "))
            matplotlib.pyplot.ylabel(cell_test_2.replace("_", " "))
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test_1)}-{step00.format_filename(cell_test_2)}-2DHist.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Cell-{step00.format_filename(cell_test_1)}-{step00.format_filename(cell_test_2)}-2DHist.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        for gene_test in tqdm.tqdm(gene_test_list):
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 18))

            seaborn.histplot(data=gene_data, x=gene_test, hue=step00.gene_column, stat="count", bins=100, multiple="stack", kde=True, palette=gene_palette, hue_order=gene_list, legend=False, linewidth=0, ax=ax)

            matplotlib.pyplot.xlabel(gene_test.replace("_", " "))
            matplotlib.pyplot.tight_layout()

            figure_list.append(f"{directory}/Gene-{step00.format_filename(gene_test)}-Hist.pdf")
            fig.savefig(figure_list[-1])
            figure_list.append(f"{directory}/Gene-{step00.format_filename(gene_test)}-Hist.png")
            fig.savefig(figure_list[-1])
            matplotlib.pyplot.close(fig)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
