import argparse
import anndata
import numpy
import pandas
import scanpy
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
import rapids_singlecell
import tqdm
import step00


def to_numpy(array) -> numpy.ndarray:
    return array.get() if hasattr(array, "get") else numpy.asarray(array)


def count_sketch(n_rows: int, n_columns: int, rng: numpy.random.Generator) -> scipy.sparse.csr_matrix:
    return scipy.sparse.csr_matrix((rng.choice([-1.0, 1.0], size=n_columns), (rng.integers(0, n_rows, size=n_columns), numpy.arange(n_columns))), shape=(n_rows, n_columns))


def leverage_score(matrix: scipy.sparse.csr_matrix, n_total: int, rng: numpy.random.Generator, n_sketch: int = 5000, eps: float = 0.5) -> numpy.ndarray:
    matrix = matrix[:, numpy.flatnonzero(matrix.getnnz(axis=0) > 0)].astype(numpy.float64)
    n_cells, n_features = matrix.shape

    if n_cells < 1.5 * n_sketch:
        u, _, _ = scipy.sparse.linalg.svds(matrix, k=min(50, min(matrix.shape) - 1), random_state=42)
        return (u ** 2).sum(axis=1)

    n_sketch = int(min(max(n_sketch, 1.1 * n_features), n_total))
    r = numpy.linalg.qr(numpy.asarray((count_sketch(n_sketch, n_cells, rng) @ matrix).todense()), mode="r")
    n_dims = int(numpy.floor(4 * numpy.log(n_total) / ((eps ** 2) / 2 - (eps ** 3) / 3)))
    s = int(numpy.ceil(numpy.sqrt(n_dims)))
    jl = rng.choice([-1.0, 0.0, 1.0], size=(n_features, n_dims), p=[1 / (2 * s), 1 - 1 / s, 1 / (2 * s)])
    z = numpy.asarray(matrix @ (numpy.linalg.pinv(r) @ jl))
    return (z ** 2).sum(axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--sketch-cells", help="Cells to sketch per sample (Seurat SketchData ncells)", type=int, default=5000)
    parser.add_argument("--sketch-column", help="Boolean obs column with pre-defined sketch cells (e.g. has_pca); leverage-score sketch if omitted", type=str, default=None)
    parser.add_argument("--pcs", help="Number of PCs (Seurat RunPCA npcs)", type=int, default=50)
    parser.add_argument("--neighbors", help="Number of neighbors for kNN graph (Seurat k.param)", type=int, default=20)
    parser.add_argument("--min-component", help="Remove cells in kNN components smaller than this (0 keeps all)", type=int, default=0)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    feature_mask = input_adata.var["highly_variable"].to_numpy()
    data = scipy.sparse.csr_matrix(input_adata.layers[step00.log_column])[:, feature_mask].astype(numpy.float64)
    sample_array = input_adata.obs[step00.sample_column].astype(str).to_numpy()
    sample_list = sorted(set(sample_array))
    print("Features:", data.shape[1], "Samples:", len(sample_list))

    if args.sketch_column is not None:
        sketch_mask = input_adata.obs[args.sketch_column].astype(bool).to_numpy()
    else:
        rng = numpy.random.default_rng(123)
        sketch_mask = numpy.zeros(input_adata.n_obs, dtype=bool)

        for sample in tqdm.tqdm(sample_list):
            index = numpy.flatnonzero(sample_array == sample)

            if index.size <= args.sketch_cells:
                sketch_mask[index] = True
                continue

            score = leverage_score(data[index], input_adata.n_obs, rng, n_sketch=args.sketch_cells)
            sketch_mask[rng.choice(index, size=args.sketch_cells, replace=False, p=score / score.sum())] = True

    input_adata.obs[step00.sketch_column] = sketch_mask
    print(pandas.crosstab(input_adata.obs[step00.sample_column], input_adata.obs[step00.sketch_column]))

    sketch_data = data[sketch_mask].toarray()
    mean = sketch_data.mean(axis=0)
    std = sketch_data.std(axis=0, ddof=1)
    std[(std == 0)] = 1.0
    sketch_adata = anndata.AnnData(X=numpy.minimum((sketch_data - mean) / std, 10.0).astype(numpy.float64), obs=input_adata.obs.loc[sketch_mask, [step00.sample_column]].copy())

    rapids_singlecell.get.anndata_to_GPU(sketch_adata)
    rapids_singlecell.pp.pca(sketch_adata, n_comps=args.pcs, mask_var=None, random_state=42)
    rapids_singlecell.pp.harmony_integrate(sketch_adata, key=step00.sample_column, basis=step00.pca_key, adjusted_basis=step00.harmony_key, correction_method="batched", random_state=42)
    loadings = to_numpy(sketch_adata.varm["PCs"])
    sketch_harmony = to_numpy(sketch_adata.obsm[step00.harmony_key])
    print(sketch_adata)

    feature_sketch = count_sketch(int(round(0.8 * data.shape[1])), data.shape[1], numpy.random.default_rng(42)).T.tocsr()
    sketch_position = numpy.full(input_adata.n_obs, -1)
    sketch_position[sketch_mask] = numpy.arange(sketch_mask.sum())
    pca_embedding = numpy.zeros((input_adata.n_obs, args.pcs), dtype=numpy.float64)
    harmony_embedding = numpy.zeros((input_adata.n_obs, args.pcs), dtype=numpy.float64)

    for sample in tqdm.tqdm(sample_list):
        index = numpy.flatnonzero(sample_array == sample)
        atom = index[sketch_mask[index]]

        assert (atom.size > 0), f"No sketch cells in {sample}!!"

        pca_embedding[index] = numpy.minimum((data[index].toarray() - mean) / std, 10.0) @ loadings
        transform = numpy.linalg.pinv((data[atom] @ feature_sketch).toarray()) @ sketch_harmony[sketch_position[atom]]
        harmony_embedding[index] = (data[index] @ feature_sketch) @ transform

    input_adata.obsm[step00.pca_key] = pca_embedding
    input_adata.obsm[step00.harmony_key] = harmony_embedding
    input_adata.obsm[step00.sketch_key] = numpy.where(sketch_mask[:, None], sketch_harmony[numpy.maximum(sketch_position, 0)], numpy.nan).astype(numpy.float64)
    input_adata.uns[step00.pca_key] = {"variance_ratio": to_numpy(sketch_adata.uns["pca"]["variance_ratio"]), "sketch_cells": args.sketch_cells}
    print(input_adata)

    rapids_singlecell.pp.neighbors(input_adata, n_neighbors=args.neighbors, use_rep=step00.harmony_key, algorithm="cagra", random_state=42)

    connectivities = input_adata.obsp["connectivities"].copy()
    connectivities.eliminate_zeros()
    n_components, components = scipy.sparse.csgraph.connected_components(connectivities, directed=False)
    component_size = numpy.bincount(components)[components]
    input_adata.obs["kNN_component_size"] = component_size
    print("kNN components:", n_components, "Largest:", numpy.sort(numpy.bincount(components))[::-1][:5])
    print("Cells in components < 100:", (component_size < 100).sum())

    if args.min_component > 0:
        input_adata = input_adata[component_size >= args.min_component].copy()
        rapids_singlecell.pp.neighbors(input_adata, n_neighbors=args.neighbors, use_rep=step00.harmony_key, algorithm="cagra", random_state=42)
        print("Removed cells:", (component_size < args.min_component).sum())

    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
