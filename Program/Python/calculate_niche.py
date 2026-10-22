import argparse
import anndata
import numpy
import pandas
import scanpy
import scipy.sparse
import scipy.spatial
import scipy.stats
import rapids_singlecell
import tqdm
import tqdm.contrib.itertools
import step00


def to_numpy(array) -> numpy.ndarray:
    return array.get() if hasattr(array, "get") else numpy.asarray(array)


def find_niches(counts, sample_obs: pandas.DataFrame, resolution: float, dims: int, neighbors: int) -> numpy.ndarray:
    niche_adata = anndata.AnnData(X=scipy.sparse.csr_matrix(counts, dtype=numpy.float64), obs=sample_obs.copy())
    rapids_singlecell.get.anndata_to_GPU(niche_adata)
    rapids_singlecell.pp.normalize_pearson_residuals(niche_adata, clip=10)
    rapids_singlecell.pp.pca(niche_adata, n_comps=min(30, niche_adata.n_vars - 1), mask_var=None, random_state=42)
    rapids_singlecell.pp.harmony_integrate(niche_adata, key=step00.sample_column, basis=step00.pca_key, adjusted_basis=step00.harmony_key, correction_method="batched", random_state=42)
    niche_adata.obsm[step00.niche_column] = to_numpy(niche_adata.obsm[step00.harmony_key])[:, :dims]
    rapids_singlecell.pp.neighbors(niche_adata, n_neighbors=neighbors, use_rep=step00.niche_column, method="jaccard", algorithm="cagra", random_state=42)
    rapids_singlecell.tl.louvain(niche_adata, resolution=resolution, key_added="Niche")
    return niche_adata.obs["Niche"].astype(str).to_numpy()


def niche_enrichment(niche_array: numpy.ndarray, cell_type_array: numpy.ndarray) -> pandas.DataFrame:
    table = pandas.crosstab(niche_array, cell_type_array)
    n_cells = int(table.to_numpy().sum())
    row_list = list()

    for niche, cell_type in tqdm.contrib.itertools.product(table.index, table.columns):
        a = int(table.loc[niche, cell_type])
        b = int(table.loc[niche].sum()) - a
        c = int(table[cell_type].sum()) - a
        d = n_cells - a - b - c
        odds_ratio, p_value = scipy.stats.fisher_exact([[a, b], [c, d]])
        row_list.append({"Niche": niche, step00.celltype_column: cell_type, "Estimate": odds_ratio, "p.val": p_value})

    enrichment_data = pandas.DataFrame(row_list)
    p_value = enrichment_data["p.val"].to_numpy()
    order = numpy.argsort(p_value)
    adjusted = numpy.empty_like(p_value)
    adjusted[order] = numpy.minimum(numpy.maximum.accumulate((p_value.size - numpy.arange(p_value.size)) * p_value[order]), 1.0)
    enrichment_data["FDR"] = adjusted
    return enrichment_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--label", help="Cell type column in obs", type=str, default="CellAnnotation.Level3")
    parser.add_argument("--group", help="Experimental group column in obs for co-localization", type=str, default="TYPE")
    parser.add_argument("--neighbors", help="Spatial nearest neighbours including the cell itself (Seurat k.param)", type=int, default=30)
    parser.add_argument("--resolution", help="Louvain resolution for niche clustering (spamsc FindNiches)", type=float, default=0.1)
    parser.add_argument("--dims", help="Harmony dimensions for niche clustering (spamsc FindNiches)", type=int, default=10)
    parser.add_argument("--expression", help="Also cluster aggregated neighbour expression (memory heavy)", action="store_true", default=False)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    assert (args.neighbors > 1), "Neighbors must be at least 2!!"

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    if args.label not in input_adata.obs.columns:
        raise ValueError(f"{args.label} is not in obs!!")

    labels = input_adata.obs[args.label].astype(str).astype("category")
    cell_type_list = list(labels.cat.categories)
    sample_array = input_adata.obs[step00.sample_column].astype(str).to_numpy()
    coordinates = input_adata.obs[list(step00.spatial_columns)].to_numpy(dtype=float)
    print("Cell type:", len(cell_type_list), cell_type_list)

    row_list, column_list = list(), list()
    for sample in tqdm.tqdm(sorted(set(sample_array))):
        index = numpy.flatnonzero(sample_array == sample)

        if index.size < 2:
            continue

        k = min(args.neighbors, index.size)
        _, neighbor = scipy.spatial.cKDTree(coordinates[index]).query(coordinates[index], k=k, workers=args.cpus)
        row_list.append(numpy.repeat(index, k - 1))
        column_list.append(index[neighbor[:, 1:]].ravel())

    rows, columns = numpy.concatenate(row_list), numpy.concatenate(column_list)
    adjacency = scipy.sparse.csr_matrix((numpy.ones(rows.size, dtype=numpy.float64), (rows, columns)), shape=(input_adata.n_obs, input_adata.n_obs))

    onehot = scipy.sparse.csr_matrix((numpy.ones(input_adata.n_obs, dtype=numpy.float64), (numpy.arange(input_adata.n_obs), labels.cat.codes.to_numpy())), shape=(input_adata.n_obs, len(cell_type_list)))
    neighbor_counts = numpy.asarray((adjacency @ onehot).todense(), dtype=numpy.float64)
    composition = neighbor_counts + onehot.toarray()
    print(pandas.Series(composition.sum(axis=1)).describe())
    print("Unique neighbourhood compositions:", numpy.unique(composition, axis=0).shape[0], "/", input_adata.n_obs)

    sample_obs = input_adata.obs[[step00.sample_column]].copy()
    input_adata.obs["Niche"] = pandas.Categorical(find_niches(composition, sample_obs, args.resolution, args.dims, 20))
    print(input_adata.obs["Niche"].value_counts())
    enrichment_data = niche_enrichment(input_adata.obs["Niche"].astype(str).to_numpy(), labels.to_numpy())
    print(enrichment_data)

    group_array = input_adata.obs[args.group].astype(str).to_numpy() if (args.group in input_adata.obs.columns) else numpy.full(input_adata.n_obs, "All")
    colocalization_list = list()
    for group in tqdm.tqdm(sorted(set(group_array))):
        mask = group_array == group
        identity = onehot[mask]
        neighbor = neighbor_counts[mask]
        frequency = numpy.asarray(identity.mean(axis=0)).ravel()
        covariance = numpy.asarray(identity.T @ neighbor) / mask.sum() - numpy.outer(frequency, neighbor.mean(axis=0))
        denominator = numpy.outer(numpy.sqrt(frequency * (1.0 - frequency)), neighbor.std(axis=0))
        correlation = numpy.divide(covariance, denominator, out=numpy.full(covariance.shape, numpy.nan), where=(denominator > 0))
        correlation_data = pandas.DataFrame(correlation, index=cell_type_list, columns=cell_type_list).rename_axis(step00.celltype_column).reset_index()
        colocalization_list.append(correlation_data.melt(id_vars=step00.celltype_column, var_name="Neighbor cell type", value_name="Correlation").assign(Group=group))
    colocalization_data = pandas.concat(colocalization_list, ignore_index=True)

    if args.expression:
        hvg_mask = input_adata.var["highly_variable"].to_numpy()
        expression_niche = find_niches(adjacency @ scipy.sparse.csr_matrix(input_adata.X)[:, hvg_mask], sample_obs, args.resolution, args.dims, 20)
        input_adata.obs["Niche_expression"] = pandas.Categorical(expression_niche)
        print(input_adata.obs["Niche_expression"].value_counts())

    output_adata = anndata.AnnData(obs=input_adata.obs.copy(), obsm={key: numpy.asarray(input_adata.obsm[key]) for key in [step00.projection_key] if key in input_adata.obsm})
    output_adata.obsm[step00.niche_column] = composition
    output_adata.uns["Niche_columns"] = cell_type_list
    output_adata.uns["Niche_parameters"] = {"label": args.label, "group": args.group, "neighbors": args.neighbors, "resolution": args.resolution, "dims": args.dims}
    output_adata.uns["Niche_enrichment"] = enrichment_data
    output_adata.uns["Niche_colocalization"] = colocalization_data
    print(output_adata)
    output_adata.write_h5ad(args.output, **step00.anndata_compressions)
