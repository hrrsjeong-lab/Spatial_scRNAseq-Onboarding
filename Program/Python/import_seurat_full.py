import argparse
import pathlib
import typing
import anndata
import numpy
import pandas
import scipy.sparse
import tqdm
import notebook_pdf
import step00

reduction_keys = {"pca": step00.pca_key, "harmony": step00.harmony_key, "umap": step00.projection_key}
sample_candidates = [step00.sample_column, "SAMPLE_ID", "SAMPLE", "sample", "sample_id", "Sample_ID", "Donor", "orig.ident"]


def read_layer(directory: pathlib.Path, layer: str, n_genes: int, n_cells: int) -> scipy.sparse.csr_matrix:
    x = numpy.fromfile(directory / f"{layer}_x.bin", dtype="<f4")
    i = numpy.fromfile(directory / f"{layer}_i.bin", dtype="<i4")
    p = numpy.fromfile(directory / f"{layer}_p.bin", dtype="<i4")
    return scipy.sparse.csc_matrix((x, i, p), shape=(n_genes, n_cells)).T.tocsr()


def clean_frame(frame: pandas.DataFrame) -> pandas.DataFrame:
    for column in frame.columns:
        if frame[column].dtype == object:
            values = frame[column]
            frame[column] = pandas.Categorical(values.where(values.isna(), values.astype(str)))
    return frame


def describe_columns(obs: pandas.DataFrame, max_levels: int = 1000) -> str:
    return ", ".join(f"{column} ({obs[column].nunique()})" for column in obs.columns if (not pandas.api.types.is_numeric_dtype(obs[column])) and (obs[column].nunique() <= max_levels))


def choose_sample_column(obs: pandas.DataFrame, requested: typing.Optional[str]) -> str:
    if requested is not None:
        return requested
    for candidate in sample_candidates:
        if (candidate in obs.columns) and (obs[candidate].nunique() > 1):
            return candidate
    raise KeyError(f"No sample column among {sample_candidates}; pass --sample-column. Columns (levels): {describe_columns(obs)}")


def copy_label(obs: pandas.DataFrame, source: str, target: str) -> None:
    if source not in obs.columns:
        raise KeyError(f"{source} not in obs. Columns (levels): {describe_columns(obs)}")
    values = obs[source].astype(object).map(lambda value: step00.safe_celltype(str(value)), na_action="ignore")
    obs[target] = pandas.Categorical(values)
    print(f"obs['{target}'] <- obs['{source}']: {obs[target].nunique()} levels, {int(obs[target].isna().sum()):,} missing")


def log_normalize(counts: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """log1p(counts / total * 1e4): Seurat LogNormalize and qc_filtering.py (normalize_total 1e4 + log1p)"""
    totals = numpy.asarray(counts.sum(axis=1), dtype=numpy.float64).ravel()
    scale = numpy.divide(1e4, totals, out=numpy.zeros_like(totals), where=(totals > 0))
    normalized = scipy.sparse.diags(scale) @ counts.astype(numpy.float64)
    normalized = scipy.sparse.csr_matrix(normalized)
    normalized.data = numpy.log1p(normalized.data)
    return normalized.astype(numpy.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="obs.csv.gz written by export_seurat_full.R", type=str)
    parser.add_argument("output", help="Output H5AD file", type=str)
    parser.add_argument("--sample-column", help=f"meta.data column copied to obs['{step00.sample_column}'] (default: first of {sample_candidates} with >1 level)", type=str, default=None)
    parser.add_argument("--celltype-column", help=f"meta.data column copied to obs['{step00.celltype_column}'] (authors' labels)", type=str, default=None)
    parser.add_argument("--clustering-column", help=f"meta.data column copied to obs['{step00.clustering_column}']", type=str, default=None)

    args = parser.parse_args()

    step00.check_suffix(args.output, {".h5ad", ".h5", ".hdf5"})
    directory = pathlib.Path(args.input).parent

    notebook = notebook_pdf.Notebook(title="Import RDS file", script=__file__)

    with notebook.cell("Read data"):
        n_genes, n_cells = (int(value) for value in (directory / "shape.txt").read_text().split())
        genes = (directory / "genes.txt").read_text().splitlines()
        cells = (directory / "cells.txt").read_text().splitlines()
        assert (len(genes), len(cells)) == (n_genes, n_cells), f"shape {n_genes}x{n_cells} vs genes.txt {len(genes)} / cells.txt {len(cells)}"

    with notebook.cell("Cells"):
        obs = pandas.read_csv(args.input, index_col="cell_id")
        obs.index = obs.index.astype(str)
        obs.index.name = None
        obs = clean_frame(obs.loc[cells])
        sample_source = choose_sample_column(obs, args.sample_column)
        copy_label(obs, sample_source, step00.sample_column)
        if obs[step00.sample_column].isna().any():
            raise ValueError(f"{int(obs[step00.sample_column].isna().sum()):,} cells have no {sample_source}")
        if args.celltype_column is not None:
            copy_label(obs, args.celltype_column, step00.celltype_column)
        if args.clustering_column is not None:
            copy_label(obs, args.clustering_column, step00.clustering_column)
        cells_per_sample = obs[step00.sample_column].value_counts()
        print(f"{step00.sample_column}: {len(cells_per_sample)} samples; cells per sample min {cells_per_sample.min():,}, median {int(cells_per_sample.median()):,}, max {cells_per_sample.max():,}")

    with notebook.cell("Genes"):
        var = pandas.DataFrame(index=pandas.Index(genes))
        if (directory / "var.csv.gz").exists():
            feature_meta = pandas.read_csv(directory / "var.csv.gz", index_col="gene", low_memory=False)
            feature_meta.index = feature_meta.index.astype(str)
            var = var.join(feature_meta, how="left")
        variable_features = set((directory / "variable_features.txt").read_text().split()) if (directory / "variable_features.txt").exists() else set()
        var["highly_variable"] = var.index.isin(variable_features)
        var = clean_frame(var)

    with notebook.cell("Counts"):
        layer_list = (directory / "layers.txt").read_text().split()
        adata = anndata.AnnData(X=read_layer(directory, "counts", n_genes, n_cells), obs=obs, var=var)
        for layer in tqdm.tqdm(layer_list):
            if layer != "counts":
                adata.layers[step00.log_column if (layer == "data") else layer] = read_layer(directory, layer, n_genes, n_cells)

    with notebook.cell("Check cells"):
        check_cells = min(1000, adata.n_obs)
        recomputed = log_normalize(adata.X[:check_cells])
        if "data" in layer_list:
            difference = abs(adata.layers[step00.log_column][:check_cells] - recomputed).max()
            log_source = "Seurat data layer"
            print(f"layers['{step00.log_column}'] = Seurat data; max |data - log1p(CP10k)| on {check_cells:,} cells: {difference:.3g}")
            if difference > 1e-3:
                print(f"WARNING: the Seurat data layer is not log1p(CP10k) of counts (e.g. SCT or another scale factor); downstream steps treat layers['{step00.log_column}'] as log1p(CP10k)")
        else:
            adata.layers[step00.log_column] = log_normalize(adata.X)
            log_source = "log1p(CP10k) computed from counts"
            print(f"layers['{step00.log_column}'] computed as log1p(CP10k): no Seurat data layer was exported")

    with notebook.cell("Reductions"):
        reduction_list = (directory / "reductions.txt").read_text().split() if (directory / "reductions.txt").exists() else list()
        obsm_keys: typing.Dict[str, str] = dict()
        for reduction in tqdm.tqdm(reduction_list):
            embedding = pandas.read_csv(directory / f"reduction_{reduction}.csv.gz", index_col="cell_id")
            embedding.index = embedding.index.astype(str)
            embedding = embedding.reindex(adata.obs_names)
            obsm_keys[reduction] = reduction_keys.get(reduction, f"X_{reduction}")
            adata.obsm[obsm_keys[reduction]] = embedding.to_numpy(dtype=numpy.float32)
            adata.obs[f"has_{reduction}"] = embedding.notna().all(axis=1).to_numpy()
            print(f"obsm['{obsm_keys[reduction]}'] <- Seurat {reduction}: {embedding.shape[1]} dims, {int(adata.obs[f'has_{reduction}'].sum()):,}/{adata.n_obs:,} cells")

    with notebook.cell("Save"):
        adata.uns["seurat"] = {"source": str(directory), "layers": layer_list, "reductions": reduction_list, "obsm_keys": obsm_keys, "log_layer": log_source, "columns": {step00.sample_column: sample_source, step00.celltype_column: (args.celltype_column or ""), step00.clustering_column: (args.clustering_column or "")}, "object": (directory / "object.txt").read_text() if (directory / "object.txt").exists() else ""}
        sample_values = adata.X[:100].toarray()
        print("X integer-valued (first 100 cells):", bool(numpy.allclose(sample_values, numpy.round(sample_values))))
        print("Highly variable genes:", int(adata.var["highly_variable"].sum()))
        print(adata)
        adata.write_h5ad(args.output, **step00.anndata_compressions)

    notebook.write_pdf(args.output)
