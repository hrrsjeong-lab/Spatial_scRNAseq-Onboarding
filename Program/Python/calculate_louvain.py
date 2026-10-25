import argparse
import anndata
import numpy
import pandas
import scanpy
import scipy.sparse
import tqdm
import rapids_singlecell
import step00


def to_numpy(array) -> numpy.ndarray:
    return array.get() if hasattr(array, "get") else numpy.asarray(array)


def nearest_neighbors(reference: numpy.ndarray, query: numpy.ndarray, k: int):
    try:
        import cuml.neighbors
        model = cuml.neighbors.NearestNeighbors(n_neighbors=k).fit(reference)
    except ImportError:
        import sklearn.neighbors
        model = sklearn.neighbors.NearestNeighbors(n_neighbors=k).fit(reference)
    distances, indices = model.kneighbors(query)
    return to_numpy(distances), to_numpy(indices)


def seurat_snn(embedding: numpy.ndarray, k: int, prune: float = 1 / 15) -> scipy.sparse.csr_matrix:
    n_cells = embedding.shape[0]
    _, indices = nearest_neighbors(embedding, embedding, k)
    knn = scipy.sparse.csr_matrix((numpy.ones(n_cells * k), indices.ravel().astype(numpy.int64), numpy.arange(0, n_cells * k + 1, k)), shape=(n_cells, n_cells))
    snn = (knn @ knn.T).tocsr()
    snn.data = snn.data / (2 * k - snn.data)
    snn.data[snn.data < prune] = 0.0
    snn.setdiag(0.0)
    snn.eliminate_zeros()
    return snn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--neighbors", help="Number of neighbors for the sketch SNN graph (Seurat k.param)", type=int, default=30)
    parser.add_argument("--transfer-neighbors", help="Number of sketch neighbors for label transfer (Seurat TransferSketchLabels k)", type=int, default=50)
    parser.add_argument("--max-clusters", help="Maximum number of clusters", type=int, default=30)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    sketch_mask = input_adata.obs[step00.sketch_column].to_numpy(dtype=bool)
    sketch_adata = anndata.AnnData(obs=input_adata.obs.loc[sketch_mask, [step00.sample_column]].copy(), obsm={step00.harmony_key: numpy.asarray(input_adata.obsm[step00.sketch_key])[sketch_mask, 0:30]})
    snn = seurat_snn(sketch_adata.obsm[step00.harmony_key], args.neighbors)
    print("SNN graph:", snn.shape, "edges per cell:", snn.nnz / snn.shape[0], "min weight:", snn.data.min())

    resolutions = [1.0, 0.8, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01]
    resolution_columns = {r: f"{step00.clustering_column}_{r}" for r in resolutions}
    rapids_singlecell.tl.louvain(sketch_adata, resolution=resolutions, key_added=step00.clustering_column, adjacency=snn)
    summary_data = pandas.DataFrame([{"Resolution": r, "Clusters": sketch_adata.obs[resolution_columns[r]].nunique()} for r in resolutions])
    print(summary_data)

    candidate_data = summary_data.loc[(summary_data["Clusters"] <= args.max_clusters)]
    assert (not candidate_data.empty), f"No resolution gives at most {args.max_clusters} clusters!!"

    selected = float(candidate_data["Resolution"].max())
    print("Selected resolution:", selected)

    sketch_labels = sketch_adata.obs[resolution_columns[selected]].astype(str).to_numpy()
    label_list = numpy.asarray(sorted(set(sketch_labels)))
    codes = numpy.searchsorted(label_list, sketch_labels)

    harmony_full = numpy.asarray(input_adata.obsm[step00.harmony_key])[:, 0:30]
    distances, indices = nearest_neighbors(harmony_full[sketch_mask], harmony_full, args.transfer_neighbors)
    ratio = numpy.divide(distances, distances[:, -1:], out=numpy.zeros_like(distances), where=(distances[:, -1:] > 0))
    weights = 1.0 - numpy.exp(-(1.0 - ratio) / 4.0)
    weights = weights / weights.sum(axis=1, keepdims=True)
    probability = numpy.zeros((input_adata.n_obs, label_list.size), dtype=numpy.float64)
    rows = numpy.arange(input_adata.n_obs)

    for j in tqdm.trange(indices.shape[1]):
        probability[rows, codes[indices[:, j]]] += weights[:, j]

    labels = pandas.Series(label_list[probability.argmax(axis=1)], index=input_adata.obs_names)

    order = list(labels.value_counts().index)
    mapping = {cluster: str(i) for i, cluster in enumerate(order)}
    input_adata.obs[step00.clustering_column] = pandas.Categorical(labels.map(mapping), categories=[str(i) for i in range(len(order))])
    input_adata.obs[f"{step00.clustering_column}_score"] = probability.max(axis=1)
    input_adata.obs[f"{step00.clustering_column}_sketch"] = pandas.Series(sketch_labels, index=sketch_adata.obs_names).map(mapping).reindex(input_adata.obs_names).astype("category")
    input_adata.uns[step00.clustering_column] = {"selected_resolution": selected, "n_clusters": {str(r): int(n) for r, n in zip(summary_data["Resolution"], summary_data["Clusters"])}, "dims": "0:30", "neighbors": args.neighbors, "prune": 1/15, "transfer_dims": "0:30", "transfer_neighbors": args.transfer_neighbors}

    print(input_adata.obs[step00.clustering_column].value_counts())
    print(input_adata.obs[f"{step00.clustering_column}_score"].describe())
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
