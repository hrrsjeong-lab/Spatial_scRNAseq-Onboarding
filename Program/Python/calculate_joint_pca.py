import argparse
import os
import anndata
import numpy
import pandas
import scanpy
import scipy.sparse
import scipy.sparse.csgraph
import sklearn.neighbors
import rapids_singlecell
import tqdm
import notebook_pdf
import step00


def to_numpy(array) -> numpy.ndarray:
    return array.get() if hasattr(array, "get") else numpy.asarray(array)


def merge_omics(adata_dict: dict[str, anndata.AnnData]) -> anndata.AnnData:
    first_adata = next(iter(adata_dict.values()))
    expressed_list = [set(adata.var_names[numpy.asarray(adata.X.sum(axis=0)).ravel() > 0]) for adata in adata_dict.values()]
    shared_genes = pandas.Index([gene for gene in first_adata.var_names if all(gene in expressed for expressed in expressed_list) and (gene not in excluded_genes)])
    variable_genes = set().union(*[adata.var_names[adata.var["highly_variable"].to_numpy(dtype=bool)] for adata in adata_dict.values()])

    joint_dict = dict()
    for name, adata in tqdm.tqdm(adata_dict.items(), total=len(adata_dict)):
        counts = scipy.sparse.csr_matrix(adata[:, shared_genes].X, dtype=numpy.float64)
        assert numpy.allclose(counts[:100].data, numpy.round(counts[:100].data)), f"X of {name} is not raw counts!!"
        joint_dict[name] = anndata.AnnData(X=counts, obs=adata.obs, var=pandas.DataFrame(index=shared_genes))

    names = list(joint_dict.keys())
    duplicated = not joint_dict[names[0]].obs_names.intersection(joint_dict[names[1]].obs_names).empty
    joint_adata = anndata.concat(joint_dict, axis="obs", join="inner", label=step00.omics_column, index_unique=("-" if duplicated else None))

    sample_sets = [set(adata.obs[step00.sample_column].astype(str)) for adata in adata_dict.values()]
    if sample_sets[0] & sample_sets[1]:
        joint_adata.obs[step00.sample_column] = pandas.Categorical(joint_adata.obs[step00.omics_column].astype(str) + ":" + joint_adata.obs[step00.sample_column].astype(str))

    totals = numpy.asarray(joint_adata.X.sum(axis=1)).ravel()
    log_data = scipy.sparse.csr_matrix(scipy.sparse.diags(1e4 / numpy.maximum(totals, 1.0)) @ joint_adata.X)
    log_data.data = numpy.log1p(log_data.data)
    joint_adata.layers[step00.log_column] = log_data
    joint_adata.var["highly_variable"] = joint_adata.var_names.isin(list(variable_genes))

    gene_data = pandas.DataFrame({"Genes": [adata.n_vars for adata in adata_dict.values()], "Expressed": [len(expressed) for expressed in expressed_list], "HVG": [int(adata.var["highly_variable"].sum()) for adata in adata_dict.values()], "Cells": [adata.n_obs for adata in adata_dict.values()]}, index=pandas.Index(names, name=step00.omics_column))
    print(gene_data)
    joint_adata.uns["merge_omics"] = {"shared_genes": shared_genes.size, "variable_genes": int(joint_adata.var["highly_variable"].sum()), "sample_prefixed": bool(sample_sets[0] & sample_sets[1]), "datasets": gene_data}
    print("Shared genes:", shared_genes.size, "| PCA features (union of HVGs):", joint_adata.uns["merge_omics"]["variable_genes"])
    return joint_adata


def scale(data: numpy.ndarray, mean: numpy.ndarray, std: numpy.ndarray) -> numpy.ndarray:
    return numpy.minimum((data - mean) / std, 10.0)


def omics_mixing(embedding_dict: dict[str, numpy.ndarray], omics: numpy.ndarray, n_cells: int, k: int = 30) -> pandas.DataFrame:
    index = numpy.sort(numpy.random.default_rng(42).choice(omics.size, min(n_cells, omics.size), replace=False))
    codes, categories = pandas.factorize(omics[index])
    cells = numpy.bincount(codes)
    mixing_data = pandas.DataFrame({"Cells": cells, "Expected": (codes.size - cells) / (codes.size - 1)}, index=pandas.Index(categories, name=step00.omics_column))
    for name, embedding in tqdm.tqdm(embedding_dict.items(), total=len(embedding_dict)):
        neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=k, n_jobs=len(os.sched_getaffinity(0))).fit(embedding[index]).kneighbors(return_distance=False)
        mixing_data[name] = numpy.bincount(codes, weights=(codes[neighbors] != codes[:, None]).mean(axis=1)) / cells
    return mixing_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("xenium", help="Xenium HDF5 file (step 04: raw counts in X, var highly_variable)", type=str)
    parser.add_argument("scrna", help="scRNA HDF5 file (step 04: raw counts in X, var highly_variable)", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--sketch-column", help="Boolean obs column with pre-defined sketch cells in both datasets (e.g. has_pca); uniform sketch per sample if omitted", type=str, default=None)
    parser.add_argument("--sketch-cells", help="Cells to sketch per sample without --sketch-column (Seurat SketchData ncells)", type=int, default=5000)
    parser.add_argument("--pcs", help="Number of PCs (Seurat RunPCA npcs)", type=int, default=50)
    parser.add_argument("--neighbors", help="Number of neighbors for kNN graph (Seurat k.param)", type=int, default=30)
    parser.add_argument("--mixing-cells", help="Random cells for the Omics mixing table", type=int, default=50000)
    parser.add_argument("--chunk", help="Cells per projection chunk", type=int, default=20000)

    args = parser.parse_args()

    step00.check_suffix(args.xenium, {".hdf5", ".h5"})
    step00.check_suffix(args.scrna, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    notebook = notebook_pdf.Notebook(title="Calculate joint PCA", script=__file__)

    with notebook.cell("Merge"):
        adata_dict = {"Xenium": scanpy.read_h5ad(args.xenium), "scRNA": scanpy.read_h5ad(args.scrna)}
        for name, adata in tqdm.tqdm(adata_dict.items(), total=len(adata_dict)):
            print(name, adata)
        input_adata = merge_omics(adata_dict)
        del adata_dict
        print(input_adata)

    with notebook.cell("HVGs"):
        feature_mask = input_adata.var["highly_variable"].to_numpy()
        data = scipy.sparse.csr_matrix(input_adata.layers[step00.log_column])[:, feature_mask].astype(numpy.float64)
        sample_array = input_adata.obs[step00.sample_column].astype(str).to_numpy()
        sample_list = sorted(set(sample_array))
        omics_array = input_adata.obs[step00.omics_column].astype(str).to_numpy()
        omics_list = sorted(set(omics_array))
        print("Features:", data.shape[1], "Samples:", len(sample_list))

    with notebook.cell("Sketch"):
        rng = numpy.random.default_rng(42)
        if args.sketch_column is not None:
            sketch_mask = input_adata.obs[args.sketch_column].astype(bool).to_numpy(copy=True)
        else:
            sketch_mask = numpy.zeros(input_adata.n_obs, dtype=bool)
            for sample in tqdm.tqdm(sample_list):
                index = numpy.flatnonzero(sample_array == sample)
                sketch_mask[index if (index.size <= args.sketch_cells) else rng.choice(index, size=args.sketch_cells, replace=False)] = True

    with notebook.cell("Fit sketch"):
        n_fit = min(int((sketch_mask & (omics_array == omics)).sum()) for omics in tqdm.tqdm(omics_list))
        for omics in tqdm.tqdm(omics_list):
            index = numpy.flatnonzero(sketch_mask & (omics_array == omics))
            sketch_mask[rng.choice(index, size=index.size - n_fit, replace=False)] = False

        input_adata.obs[step00.sketch_column] = sketch_mask
        print(pandas.crosstab(input_adata.obs[step00.omics_column], input_adata.obs[step00.sketch_column]))
        print(pandas.crosstab(input_adata.obs[step00.sample_column], input_adata.obs[step00.sketch_column]))

    with notebook.cell(""):
        sketch_data = data[sketch_mask].toarray()
        sketch_omics = omics_array[sketch_mask]
        mean, std = dict(), dict()

        scaled_data = numpy.zeros(sketch_data.shape, dtype=numpy.float64)
        for omics in tqdm.tqdm(omics_list):
            rows = (sketch_omics == omics)
            mean[omics] = sketch_data[rows].mean(axis=0)
            std[omics] = sketch_data[rows].std(axis=0, ddof=1)
            std[omics][(std[omics] <= step00.epsilon)] = 1.0
            scaled_data[rows] = scale(sketch_data[rows], mean[omics], std[omics])
        del sketch_data

    with notebook.cell("PCA"):
        sketch_adata = anndata.AnnData(X=scaled_data)
        rapids_singlecell.get.anndata_to_GPU(sketch_adata)
        rapids_singlecell.pp.pca(sketch_adata, n_comps=args.pcs, mask_var=None, random_state=42)

    with notebook.cell("Scale"):
        null_data = scaled_data.astype(numpy.float64)
        del scaled_data
        numpy.random.default_rng(42).permuted(null_data, axis=0, out=null_data)
        null_adata = anndata.AnnData(X=null_data)
        rapids_singlecell.get.anndata_to_GPU(null_adata)
        rapids_singlecell.pp.pca(null_adata, n_comps=args.pcs, mask_var=None, random_state=42)

    with notebook.cell("PCA variance"):
        loadings = to_numpy(sketch_adata.varm["PCs"]).astype(numpy.float64)
        variance = to_numpy(sketch_adata.uns["pca"]["variance"])
        variance_ratio = to_numpy(sketch_adata.uns["pca"]["variance_ratio"])
        null_variance_ratio = to_numpy(null_adata.uns["pca"]["variance_ratio"])
        del sketch_adata, null_adata

        above_null = variance_ratio > null_variance_ratio
        signal_pcs = int(above_null.size if above_null.all() else numpy.argmin(above_null))
        variance_data = pandas.DataFrame({"Eigenvalue": variance, "Ratio": variance_ratio, "Cumulative": numpy.cumsum(variance_ratio), "Null ratio": null_variance_ratio, "Fold over null": variance_ratio / null_variance_ratio}, index=[f"PC{i + 1}" for i in range(variance.size)])
        print("Features:", data.shape[1], "Total variance:", variance[0] / variance_ratio[0], "PCs above permutation null:", signal_pcs)
        print(variance_data.to_string(float_format=lambda x: f"{x:.4f}"))

    with notebook.cell("PCA embedding"):
        pca_embedding = numpy.zeros((input_adata.n_obs, args.pcs), dtype=numpy.float64)
        for omics in tqdm.tqdm(omics_list):
            rows = numpy.flatnonzero(omics_array == omics)
            for start in tqdm.tqdm(range(0, rows.size, args.chunk), desc=omics):
                index = rows[start:(start + args.chunk)]
                pca_embedding[index] = scale(data[index].toarray(), mean[omics], std[omics]) @ loadings

    with notebook.cell("Harmony"):
        harmony_adata = anndata.AnnData(obs=input_adata.obs[[step00.omics_column, step00.sample_column]].copy(), obsm={step00.pca_key: pca_embedding})
        rapids_singlecell.pp.harmony_integrate(harmony_adata, key=[step00.omics_column, step00.sample_column], basis=step00.pca_key, adjusted_basis=step00.harmony_key, correction_method="batched", random_state=42)
        harmony_embedding = to_numpy(harmony_adata.obsm[step00.harmony_key]).astype(numpy.float64)
        del harmony_adata

    with notebook.cell("Add obsm"):
        input_adata.obsm[step00.pca_key] = pca_embedding
        input_adata.obsm[step00.harmony_key] = harmony_embedding
        input_adata.obsm[step00.sketch_key] = numpy.where(sketch_mask[:, None], harmony_embedding, numpy.nan)
        input_adata.uns[step00.pca_key] = {"variance": variance, "variance_ratio": variance_ratio, "null_variance_ratio": null_variance_ratio, "signal_pcs": signal_pcs, "n_features": int(data.shape[1]), "fit_cells_per_omics": n_fit, "sketch_method": (args.sketch_column if (args.sketch_column is not None) else f"Uniform {args.sketch_cells}"), "harmony_keys": [step00.omics_column, step00.sample_column]}

    with notebook.cell("Mixing"):
        mixing_data = omics_mixing({"PCA": pca_embedding[:, :30], "Harmony": harmony_embedding[:, :30]}, omics_array, args.mixing_cells)
        input_adata.uns[f"{step00.omics_column}_mixing"] = mixing_data
        print(mixing_data)
        print(input_adata)

    with notebook.cell("Neighbors"):
        rapids_singlecell.pp.neighbors(input_adata, n_neighbors=args.neighbors, use_rep=step00.harmony_key, algorithm="cagra", random_state=42)

        connectivities = input_adata.obsp["connectivities"].copy()
        connectivities.eliminate_zeros()
        n_components, components = scipy.sparse.csgraph.connected_components(connectivities, directed=False)
        component_size = numpy.bincount(components)[components]
        input_adata.obs["kNN_component_size"] = component_size
        print("kNN components:", n_components, "Largest:", numpy.sort(numpy.bincount(components))[::-1][:5])
        print("Cells in components < 100:", (component_size < 100).sum())

    with notebook.cell("Save"):
        print(input_adata)
        input_adata.write_h5ad(args.output, **step00.anndata_compressions)

    notebook.write_pdf(args.output)
