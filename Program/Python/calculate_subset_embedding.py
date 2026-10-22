import argparse
import anndata
import anndata.io
import h5py
import numpy
import pandas
import scanpy
import scipy.sparse
import rapids_singlecell
import tqdm
import step00


def to_numpy(array) -> numpy.ndarray:
    return array.get() if hasattr(array, "get") else numpy.asarray(array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file (raw counts in X, logNorm layer)", type=str)
    parser.add_argument("output", help="Output HDF5 file (subset cells x marker genes)", type=str)
    parser.add_argument("--lineage-column", help="Broad lineage column in obs", type=str, default="CellAnnotation.Level0")
    parser.add_argument("--pattern", help="Regular expression on the lineage column that selects the subset", type=str, default="(?i)fibroblast|^FAS")
    parser.add_argument("--subtype-column", help="Subtype column used for colours and markers", type=str, default="CellAnnotation.Level3")
    parser.add_argument("--group-column", help="Sample-level column kept for colouring", type=str, default="TYPE")
    parser.add_argument("--n-top-genes", help="HVGs (Seurat FindVariableFeatures nfeatures)", type=int, default=2000)
    parser.add_argument("--spans", help="loess spans to try in order", type=float, nargs="+", default=[0.3, 0.5, 1.0])
    parser.add_argument("--pcs", help="Principal components", type=int, default=50)
    parser.add_argument("--dims", help="Harmony dimensions for UMAP as start:end", type=str, default="2:30")
    parser.add_argument("--neighbors", help="UMAP n.neighbors", type=int, default=30)
    parser.add_argument("--min-dist", help="UMAP min.dist", type=float, default=0.01)
    parser.add_argument("--markers", help="Top positive markers per subtype for the dot plot", type=int, default=3)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5", ".h5ad"})
    step00.check_suffix(args.output, {".hdf5", ".h5", ".h5ad"})

    with h5py.File(args.input, "r") as handle:
        obs = anndata.io.read_elem(handle["obs"])
        var = anndata.io.read_elem(handle["var"])

        lineage = obs[args.lineage_column].astype(str)
        mask = obs[args.lineage_column].notna().to_numpy() & lineage.str.contains(args.pattern, regex=True).to_numpy()
        print("Lineage labels:", lineage.value_counts().to_dict())
        print("Selected lineages:", lineage[mask].value_counts().to_dict()) 

        assert (mask.any()), f"No {args.lineage_column} label matches {args.pattern}!!"

        index = numpy.flatnonzero(mask)
        counts = scipy.sparse.csr_matrix(anndata.io.sparse_dataset(handle["X"])[index], dtype=numpy.float32)
        log_data = scipy.sparse.csr_matrix(anndata.io.sparse_dataset(handle[f"layers/{step00.log_column}"])[index], dtype=numpy.float32)

    keep_columns = [column for column in [step00.sample_column, args.group_column, args.lineage_column, args.subtype_column] if column in obs.columns]
    subset_obs = obs.iloc[index][keep_columns].copy()
    subset_obs[args.subtype_column] = pandas.Categorical(subset_obs[args.subtype_column].astype(str).map(step00.safe_celltype))
    adata = anndata.AnnData(X=counts, obs=subset_obs, var=pandas.DataFrame(index=var.index.astype(str)))
    adata.layers[step00.log_column] = log_data
    print(adata)
    print(pandas.crosstab(adata.obs[args.subtype_column], adata.obs[step00.sample_column]))

    hvg_adata = adata.copy()
    rapids_singlecell.get.anndata_to_GPU(hvg_adata)
    for span in tqdm.tqdm(args.spans):
        try:
            rapids_singlecell.pp.highly_variable_genes(hvg_adata, flavor="seurat_v3_paper", n_top_genes=args.n_top_genes, batch_key=step00.sample_column, layer=None, span=span)
            break
        except ValueError as error:
            print(f"WARNING: loess failed with span {span}: {error}")
    else:
        raise RuntimeError(f"loess failed with every span {args.spans}!!")
    feature_mask = hvg_adata.var["highly_variable"].to_numpy()
    print("HVGs:", int(feature_mask.sum()), "| loess span:", span)
    del hvg_adata

    data = log_data[:, feature_mask].toarray().astype(numpy.float64)
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    std[(std == 0)] = 1.0
    scaled_adata = anndata.AnnData(X=numpy.minimum((data - mean) / std, 10.0), obs=adata.obs[[step00.sample_column]].copy())
    del data

    rapids_singlecell.get.anndata_to_GPU(scaled_adata)
    rapids_singlecell.pp.pca(scaled_adata, n_comps=min(args.pcs, scaled_adata.n_vars - 1), mask_var=None, random_state=42)
    rapids_singlecell.pp.harmony_integrate(scaled_adata, key=step00.sample_column, basis=step00.pca_key, adjusted_basis=step00.harmony_key, correction_method="batched", random_state=42)
    adata.obsm[step00.pca_key] = to_numpy(scaled_adata.obsm[step00.pca_key]).astype(numpy.float64)
    adata.obsm[step00.harmony_key] = to_numpy(scaled_adata.obsm[step00.harmony_key]).astype(numpy.float64)
    del scaled_adata

    harmony = adata.obsm[step00.harmony_key][:, step00.parse_dims(args.dims)]
    norm = numpy.linalg.norm(harmony, axis=1)
    zero_mask = (norm == 0)
    umap_input = numpy.zeros((harmony.shape[0], harmony.shape[1] + 1), dtype=numpy.float64)
    umap_input[~zero_mask, :-1] = harmony[~zero_mask] / norm[~zero_mask, None]
    umap_input[zero_mask, -1] = 1.0
    umap_adata = anndata.AnnData(obs=pandas.DataFrame(index=adata.obs.index), obsm={"X_input": umap_input})
    rapids_singlecell.pp.neighbors(umap_adata, n_neighbors=args.neighbors, use_rep="X_input", metric="cosine", algorithm="brute", random_state=42)
    rapids_singlecell.tl.umap(umap_adata, min_dist=args.min_dist, spread=1.0, random_state=42)
    adata.obsm["X_umap"] = to_numpy(umap_adata.obsm["X_umap"]).astype(numpy.float64)
    del umap_adata

    group_sizes = adata.obs[args.subtype_column].value_counts()
    print("Subtypes with fewer than two cells (skipped):", group_sizes.loc[group_sizes < 2].to_dict())
    rapids_singlecell.tl.rank_genes_groups(adata, groupby=args.subtype_column, method="wilcoxon", use_raw=False, tie_correct=True, pts=True, mean_in_log_space=False, layer=step00.log_column, skip_empty_groups=True)
    rank_data = scanpy.get.rank_genes_groups_df(adata, group=None)
    rank_data = rank_data.loc[rank_data["logfoldchanges"] > 0].sort_values(["group", "scores"], ascending=[True, False])
    marker_dict = {str(group): frame["names"].iloc[:args.markers].tolist() for group, frame in rank_data.groupby("group", observed=True)}
    marker_list = list(dict.fromkeys(gene for genes in marker_dict.values() for gene in genes))
    print("Markers:", marker_dict)

    gene_index = adata.var.index.get_indexer(marker_list)
    output_adata = anndata.AnnData(X=log_data[:, gene_index], obs=adata.obs.copy(), var=pandas.DataFrame(index=pandas.Index(marker_list)), obsm={key: adata.obsm[key] for key in [step00.pca_key, step00.harmony_key, "X_umap"]})
    output_adata.uns["markers"] = {group: numpy.asarray(genes, dtype=object) for group, genes in marker_dict.items()}
    output_adata.uns["rank_genes_groups_table"] = rank_data.reset_index(drop=True).astype({"group": str})
    output_adata.uns["subset_parameters"] = {"lineage_column": args.lineage_column, "pattern": args.pattern, "subtype_column": args.subtype_column, "group_column": args.group_column, "n_top_genes": args.n_top_genes, "loess_span": span, "pcs": args.pcs, "dims": args.dims, "neighbors": args.neighbors, "min_dist": args.min_dist}
    print(output_adata)
    output_adata.write_h5ad(args.output, **step00.anndata_compressions)
