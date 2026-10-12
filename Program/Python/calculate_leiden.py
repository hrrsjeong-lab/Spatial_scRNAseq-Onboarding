import argparse
import numpy
import pandas
import scipy.spatial
import scanpy
import rapids_singlecell
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--max-clusters", help="Maximum number of clusters", type=int, default=30)
    parser.add_argument("--min-size", help="Merge clusters smaller than this into the nearest large cluster (0 disables)", type=int, default=10 ** 3)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    resolutions = [1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01]
    rapids_singlecell.tl.leiden(input_adata, resolution=resolutions, random_state=42, key_added="Leiden")

    summary_data = pandas.DataFrame([{"Resolution": r, "Clusters": input_adata.obs[f"Leiden_{r}"].nunique(), "Major clusters": int((input_adata.obs[f"Leiden_{r}"].value_counts() >= args.min_size).sum())} for r in resolutions])
    print(summary_data)

    candidate_data = summary_data.loc[(summary_data["Major clusters"] > 0) & (summary_data["Major clusters"] <= args.max_clusters)]
    if candidate_data.empty:
        raise ValueError(f"No resolution gives at most {args.max_clusters} major clusters!!")
    selected = float(candidate_data["Resolution"].max())
    print("Selected resolution:", selected)

    labels = input_adata.obs[f"Leiden_{selected}"].astype(str)
    sizes = labels.value_counts()
    is_minor = ~labels.isin(sizes.index[sizes >= args.min_size]).to_numpy()
    if is_minor.any():
        embedding = numpy.asarray(input_adata.obsm[step00.harmony_key])
        centroid_data = pandas.DataFrame(embedding[~is_minor]).groupby(labels.to_numpy()[~is_minor]).mean()
        distances = scipy.spatial.distance.cdist(embedding[is_minor], centroid_data.to_numpy())
        labels.iloc[numpy.flatnonzero(is_minor)] = centroid_data.index.to_numpy()[distances.argmin(axis=1)]
    print("Merged cells:", is_minor.sum())

    order = list(labels.value_counts().index)
    input_adata.obs[step00.clustering_column] = pandas.Categorical(labels.map({cluster: str(i) for i, cluster in enumerate(order)}), categories=[str(i) for i in range(len(order))])
    input_adata.obs["Merged_small_cluster"] = is_minor
    input_adata.obs.drop(columns=[f"Leiden_{r}" for r in resolutions], inplace=True)

    input_adata.uns["Leiden"]["selected_resolution"] = selected
    input_adata.uns["Leiden"]["min_size"] = args.min_size
    input_adata.uns["Leiden"]["n_clusters"] = {str(r): int(n) for r, n in zip(summary_data["Resolution"], summary_data["Clusters"])}
    input_adata.uns["Leiden"]["n_major_clusters"] = {str(r): int(n) for r, n in zip(summary_data["Resolution"], summary_data["Major clusters"])}

    print(input_adata.obs[step00.clustering_column].value_counts())
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
