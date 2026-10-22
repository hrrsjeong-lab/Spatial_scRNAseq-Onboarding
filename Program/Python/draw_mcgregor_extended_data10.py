import argparse
import itertools
import os
import tempfile
import typing
import zipfile
import anndata
import anndata.io
import h5py
import matplotlib
import matplotlib.colors
import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot
import numpy
import pandas
import scipy.sparse
import seaborn
import tqdm
import step00


def make_palette(labels: typing.List[str]) -> typing.Dict[str, str]:
    tab20 = list(matplotlib.colormaps["tab20"].colors)
    colors = (tab20[0::2] + tab20[1::2]) if (len(labels) <= 20) else list(matplotlib.colors.XKCD_COLORS.values())
    return {label: matplotlib.colors.to_hex(color) for label, color in zip(labels, itertools.cycle(colors))}


def save_figure(fig: matplotlib.figure.Figure, directory: str, name: str, figure_list: typing.List[str]) -> None:
    for suffix in ["pdf", "png"]:
        figure_list.append(f"{directory}/{name}.{suffix}")
        fig.savefig(figure_list[-1])
    matplotlib.pyplot.close(fig)


def scatter_categories(ax, coordinates: numpy.ndarray, labels: numpy.ndarray, palette: typing.Dict[str, str], size: float, rng: numpy.random.Generator) -> None:
    order = rng.permutation(len(labels))
    ax.scatter(coordinates[order, 0], coordinates[order, 1], c=[palette[label] for label in labels[order]], s=size, linewidths=0, rasterized=True)


def add_legend(ax, palette: typing.Dict[str, str], labels: typing.List[str], title: str) -> None:
    handles = [matplotlib.lines.Line2D([0], [0], marker="o", linestyle="", markersize=30, markerfacecolor=palette[label], markeredgewidth=0, label=label) for label in labels]
    ax.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=(1 if (len(labels) <= 25) else 2))


def draw_embedding(coordinates: numpy.ndarray, labels: numpy.ndarray, palette: typing.Dict[str, str], title: str, axis_name: str, size: float, directory: str, name: str, figure_list: typing.List[str]) -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 24))
    scatter_categories(ax, coordinates, labels, palette, size, numpy.random.default_rng(42))
    add_legend(ax, palette, [label for label in palette if label in set(labels)], title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{axis_name}1")
    ax.set_ylabel(f"{axis_name}2")
    fig.tight_layout()
    save_figure(fig, directory, name, figure_list)


def dot_data(adata: anndata.AnnData, groupby: str, marker_dict: typing.Dict[str, typing.List[str]]) -> pandas.DataFrame:
    genes = list(dict.fromkeys(gene for group in marker_dict for gene in marker_dict[group]))
    X = scipy.sparse.csc_matrix(adata[:, genes].X)
    groups = adata.obs[groupby].astype(str).to_numpy()
    rows = list()
    for group in marker_dict:
        mask = (groups == group)
        values = X[mask]
        rows.append(pandas.DataFrame({"Group": group, "Gene": genes, "Mean": numpy.asarray(values.mean(axis=0)).ravel(), "Fraction": numpy.asarray((values > 0).mean(axis=0)).ravel()}))
    data = pandas.concat(rows, ignore_index=True)
    data["Scaled mean"] = data.groupby("Gene")["Mean"].transform(lambda values: (values - values.min()) / ((values.max() - values.min()) or 1.0))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Step 03 HDF5 file (authors' labels, X_umap and spatial)", type=str)
    parser.add_argument("pseudobulk", help="calculate_pseudobulk_pca.py output", type=str)
    parser.add_argument("subset", help="calculate_subset_embedding.py output", type=str)
    parser.add_argument("output", help="Output ZIP file", type=str)
    parser.add_argument("--lineage-column", help="Broad lineage column", type=str, default="CellAnnotation.Level0")
    parser.add_argument("--group-column", help="Sample type column", type=str, default="TYPE")
    parser.add_argument("--umap-key", help="obsm key of the authors' UMAP", type=str, default="X_umap")
    parser.add_argument("--sections", help="Samples for the section maps (default: largest sample per group)", type=str, nargs="*", default=None)
    parser.add_argument("--point-size", help="Point size for embeddings", type=float, default=0.2)
    parser.add_argument("--section-point-size", help="Point size for section maps", type=float, default=0.5)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5", ".h5ad"})
    step00.check_suffix(args.pseudobulk, {".hdf5", ".h5", ".h5ad"})
    step00.check_suffix(args.subset, {".hdf5", ".h5", ".h5ad"})
    step00.check_suffix(args.output, {".zip"})

    with h5py.File(args.input, "r") as handle:
        obs = anndata.io.read_elem(handle["obs"])
        if f"obsm/{args.umap_key}" not in handle:
            raise KeyError(f"{args.umap_key} not in obsm: {list(handle['obsm'].keys())}")
        umap = numpy.asarray(anndata.io.read_elem(handle[f"obsm/{args.umap_key}"]), dtype=numpy.float64)
        spatial = numpy.asarray(anndata.io.read_elem(handle[f"obsm/{step00.spatial_key}"]), dtype=numpy.float64)
    print("Cells:", obs.shape[0], "| UMAP:", umap.shape, "| spatial:", spatial.shape)

    lineage_array = obs[args.lineage_column].astype(str).to_numpy()
    group_array = obs[args.group_column].astype(str).to_numpy()
    sample_array = obs[step00.sample_column].astype(str).to_numpy()
    lineage_palette = make_palette(sorted(set(lineage_array)))
    group_list = sorted(set(group_array))
    group_palette = {group: matplotlib.colors.to_hex(color) for group, color in zip(group_list, itertools.cycle(matplotlib.colormaps["tab10"].colors))}
    print(pandas.crosstab(obs[step00.sample_column], obs[args.group_column]))

    if args.sections:
        section_list = args.sections
    else:
        cell_counts = pandas.Series(sample_array).value_counts()
        section_list = [cell_counts.loc[sorted(set(sample_array[group_array == group]))].idxmax() for group in group_list]
    print("Sections:", section_list)

    pseudobulk_adata = anndata.read_h5ad(args.pseudobulk)
    variance_ratio = numpy.asarray(pseudobulk_adata.uns["pca"]["variance_ratio"])
    pca_data = pandas.DataFrame(pseudobulk_adata.obsm[step00.pca_key][:, :2], index=pseudobulk_adata.obs.index, columns=["PC1", "PC2"]).join(pseudobulk_adata.obs)
    print(pca_data)

    subset_adata = anndata.read_h5ad(args.subset)
    subtype_column = subset_adata.uns["subset_parameters"]["subtype_column"]
    subtype_array = subset_adata.obs[subtype_column].astype(str).to_numpy()
    subtype_palette = make_palette(sorted(set(subtype_array)))
    marker_dict = {group: list(genes) for group, genes in subset_adata.uns["markers"].items()}
    marker_dict = {group: marker_dict[group] for group in sorted(marker_dict)}
    dotplot_data = dot_data(subset_adata, subtype_column, marker_dict)
    print(dotplot_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="white", rc=step00.matplotlib_parameters)

    with tempfile.TemporaryDirectory() as directory:
        figure_list: typing.List[str] = list()

        draw_embedding(umap, lineage_array, lineage_palette, args.lineage_column, "UMAP", args.point_size, directory, "ED10a-UMAP-lineage", figure_list)
        draw_embedding(umap, group_array, group_palette, args.group_column, "UMAP", args.point_size, directory, "ED10b-UMAP-group", figure_list)

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
        seaborn.scatterplot(data=pca_data, x="PC1", y="PC2", hue=args.group_column, palette=group_palette, s=1000, edgecolor="black", ax=ax)
        for sample, row in pca_data.iterrows():
            ax.text(row["PC1"], row["PC2"], sample, fontsize="xx-small", ha="center", va="bottom", path_effects=step00.path_effects)
        ax.set_xlabel(f"PC1 ({variance_ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance_ratio[1] * 100:.1f}%)")
        ax.legend(title=args.group_column, loc="best")
        fig.tight_layout()
        save_figure(fig, directory, "ED10c-Pseudobulk-PCA", figure_list)

        for section in tqdm.tqdm(section_list):
            mask = (sample_array == section)
            fig, ax = matplotlib.pyplot.subplots(figsize=(32, 24))
            ax.set_facecolor("black")
            scatter_categories(ax, spatial[mask], lineage_array[mask], lineage_palette, args.section_point_size, numpy.random.default_rng(42))
            add_legend(ax, lineage_palette, [label for label in lineage_palette if label in set(lineage_array[mask])], args.lineage_column)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{section} ({group_array[mask][0]}, {int(mask.sum()):,} cells)", fontsize="small")
            fig.tight_layout()
            save_figure(fig, directory, f"ED10e-Section-{step00.format_filename(section)}", figure_list)

        draw_embedding(subset_adata.obsm["X_umap"], subtype_array, subtype_palette, subtype_column, "UMAP", args.point_size * 5, directory, "ED10f-Subset-UMAP-subtype", figure_list)
        draw_embedding(subset_adata.obsm["X_umap"], subset_adata.obs[args.group_column].astype(str).to_numpy(), group_palette, args.group_column, "UMAP", args.point_size * 5, directory, "ED10g-Subset-UMAP-group", figure_list)

        genes = list(dict.fromkeys(gene for group in marker_dict for gene in marker_dict[group]))
        fig, ax = matplotlib.pyplot.subplots(figsize=(max(24, 1.0 * len(genes) + 12), max(12, 1.0 * len(marker_dict) + 6)))
        seaborn.scatterplot(data=dotplot_data, x="Gene", y="Group", size="Fraction", sizes=(0, 1500), size_norm=(0, 1), hue="Scaled mean", palette="Reds", hue_norm=(0, 1), edgecolor="black", linewidth=0.5, legend="brief", ax=ax)
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=90, fontsize="xx-small")
        ax.set_yticks(range(len(marker_dict)))
        ax.set_yticklabels(list(marker_dict), fontsize="xx-small")
        ax.set_xlim(-0.5, len(genes) - 0.5)
        ax.set_ylim(len(marker_dict) - 0.5, -0.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        fig.tight_layout()
        save_figure(fig, directory, "ED10h-Subset-Dotplot", figure_list)

        with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for figure in tqdm.tqdm(figure_list):
                zip_file.write(figure, arcname=os.path.basename(figure))
