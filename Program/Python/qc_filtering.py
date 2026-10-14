import argparse
import re
import numpy
import pandas
import scanpy
import rapids_singlecell
import tqdm
import step00

negative_candidates = ("control_probe_counts", "nCount_ControlProbe", "nCount_NegControlProbe", "nCount_NegativeControlProbe")
area_candidates = ("cell_area", "area", "Area", "cell_area_um2")


def find_column(columns, requested, candidates):
    if requested is not None:
        return requested if (requested in columns) else None
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("cell", help="Cell QC TSV(.gz) file", type=str)
    parser.add_argument("gene", help="Gene QC TSV(.gz) file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--cpus", help="Number of CPUs to use", type=int, default=1)
    parser.add_argument("--min-counts", help="Remove cells with fewer transcripts (paper: <15)", type=int, default=15)
    parser.add_argument("--min-genes", help="Remove cells with fewer detected genes (0: off; not in paper)", type=int, default=0)
    parser.add_argument("--min-cells", help="Remove genes detected in fewer cells (0: off; not in paper)", type=int, default=0)
    parser.add_argument("--area-column", help="obs column with cell area (auto-detect if omitted)", type=str, default=None)
    parser.add_argument("--min-area", help="Remove cells with area below this (paper gives no value; off if omitted)", type=float, default=None)
    parser.add_argument("--negative-column", help="obs column with negative probe counts (auto-detect if omitted)", type=str, default=None)
    parser.add_argument("--negative-pattern", help="Regex for negative probe features in var", type=str, default=r"^NegControlProbe")
    parser.add_argument("--negative-percentile", help="Remove cells above this percentile of negative probe counts (paper: 99)", type=float, default=99.0)
    parser.add_argument("--negative-scope", help="Percentile over all cells or per sample (paper does not say)", choices=["global", "sample"], default="global")
    parser.add_argument("--reference-column", help="obs column with the authors' labels to compare against QC", type=str, default=None)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.cell, {".tsv", ".tsv.gz"})
    step00.check_suffix(args.gene, {".tsv", ".tsv.gz"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    scanpy.settings.n_jobs = args.cpus

    input_adata = scanpy.read_h5ad(args.input)
    input_adata.X = step00.safe_matrix(input_adata.X)
    print(input_adata)

    cell_data = pandas.read_csv(args.cell, sep="\t", index_col=0).loc[input_adata.obs.index, :]
    for cell_test in tqdm.tqdm(list(cell_data.columns)):
        input_adata.obs[cell_test] = cell_data[cell_test].to_numpy()

    gene_data = pandas.read_csv(args.gene, sep="\t", index_col=0).loc[input_adata.var.index, :]
    for gene_test in tqdm.tqdm(list(gene_data.columns)):
        input_adata.var[gene_test] = gene_data[gene_test].to_numpy()
    print(input_adata)

    sample_array = input_adata.obs[step00.sample_column].astype(str).to_numpy()
    negative_features = numpy.asarray([bool(re.search(args.negative_pattern, gene)) for gene in input_adata.var_names])
    negative_column = find_column(input_adata.obs.columns, args.negative_column, negative_candidates)
    area_column = find_column(input_adata.obs.columns, args.area_column, area_candidates)

    if negative_features.any():
        negative_counts = numpy.asarray(input_adata.X[:, negative_features].sum(axis=1)).ravel()
        negative_source = f"var /{args.negative_pattern}/ ({int(negative_features.sum())} features)"
    elif negative_column is not None:
        negative_counts = input_adata.obs[negative_column].to_numpy(dtype=float)
        negative_source = f"obs {negative_column}"
    else:
        negative_counts, negative_source = None, None
    print("Negative probe source:", negative_source, "Area column:", area_column)

    if negative_features.any():
        input_adata = input_adata[:, ~negative_features].copy()
    total_counts = numpy.asarray(input_adata.X.sum(axis=1)).ravel()
    n_genes = numpy.asarray((input_adata.X > 0).sum(axis=1)).ravel()

    qc_data = pandas.DataFrame({"Sample": sample_array, "Total counts": total_counts, "Genes": n_genes}, index=input_adata.obs_names)
    qc_data["Low counts"] = qc_data["Total counts"] < args.min_counts
    qc_data["Low genes"] = qc_data["Genes"] < args.min_genes

    if area_column is not None:
        qc_data["Area"] = input_adata.obs[area_column].to_numpy(dtype=float)
        print(qc_data.groupby("Sample")["Area"].describe(percentiles=[0.01, 0.05, 0.5]))
    qc_data["Small area"] = (qc_data["Area"] < args.min_area) if ((area_column is not None) and (args.min_area is not None)) else False

    qc_data["High negative"] = False
    negative_thresholds = dict()
    if negative_counts is not None:
        qc_data["Negative counts"] = negative_counts
        passed = ~(qc_data["Low counts"] | qc_data["Low genes"] | qc_data["Small area"])
        groups = [("All", passed.to_numpy())] if (args.negative_scope == "global") else [(sample, passed.to_numpy() & (sample_array == sample)) for sample in sorted(set(sample_array))]

        for name, mask in groups:
            threshold = float(numpy.percentile(negative_counts[mask], args.negative_percentile))
            negative_thresholds[name] = threshold
            target = numpy.ones_like(mask) if (name == "All") else (sample_array == name)
            qc_data.loc[target & (negative_counts > threshold), "High negative"] = True
        print("Negative probe thresholds:", negative_thresholds)
    else:
        print("WARNING: no negative probe counts found; negative probe filter NOT applied!!")

    if (area_column is None) or (args.min_area is None):
        print("WARNING: very-small-cell filter NOT applied (paper gives no area threshold)!!")

    flag_columns = ["Low counts", "Low genes", "Small area", "High negative"]
    qc_data["Removed"] = qc_data[flag_columns].any(axis=1)
    summary_data = qc_data.groupby("Sample")[flag_columns + ["Removed"]].sum()
    summary_data["Cells"] = qc_data.groupby("Sample").size()
    summary_data.loc["Total"] = summary_data.sum(axis=0)
    print(summary_data)

    if (args.reference_column is not None) and (args.reference_column in input_adata.obs.columns):
        print(pandas.crosstab(qc_data["Removed"], input_adata.obs[args.reference_column].notna().to_numpy(), rownames=["Removed by QC"], colnames=[f"{args.reference_column} present"]))

    for column in flag_columns:
        input_adata.obs[f"QC_{column.replace(' ', '_')}"] = qc_data[column].to_numpy()
    input_adata.uns["QC"] = {"min_counts": args.min_counts, "min_genes": args.min_genes, "min_cells": args.min_cells, "area_column": str(area_column), "min_area": (-1.0 if (args.min_area is None) else args.min_area), "negative_source": str(negative_source), "negative_percentile": args.negative_percentile, "negative_scope": args.negative_scope, "negative_thresholds": {k: v for k, v in negative_thresholds.items()}}

    input_adata = input_adata[~qc_data["Removed"].to_numpy()].copy()
    if args.min_cells > 0:
        scanpy.pp.filter_genes(input_adata, min_cells=args.min_cells)
    print(input_adata)

    input_adata.layers[step00.log_column] = input_adata.X.astype(float, copy=True)
    rapids_singlecell.get.anndata_to_GPU(input_adata, layer=step00.log_column)
    rapids_singlecell.pp.normalize_total(input_adata, exclude_highly_expressed=False, target_sum=10 ** 4, layer=step00.log_column)
    rapids_singlecell.pp.log1p(input_adata, layer=step00.log_column)
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
