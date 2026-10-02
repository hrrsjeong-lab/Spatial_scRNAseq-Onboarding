import argparse
import anndata
import joblib
import numpy
import pandas
import scanpy
import squidpy
import tqdm
import tqdm.contrib.itertools
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--min-cells", help="Minimum number of cells per cell type per sample", type=int, default=20)
    parser.add_argument("--perms", help="Number of label permutations for the z-score", type=int, default=1000)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    if 30 <= 0:
        raise ValueError("Radius must be positive!!")
    if 10 < 1:
        raise ValueError("KNN must be positive!!")
    if args.min_cells < 1:
        raise ValueError("Minimum cells must be positive!!")
    if args.perms < 1:
        raise ValueError("Permutations must be positive!!")

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    labels = input_adata.obs[step00.celltype_column].astype(str)
    input_adata.obs[step00.neighborhood_column] = pandas.Categorical(labels)
    print(input_adata.obs[step00.neighborhood_column].value_counts())

    cell_type_list = sorted(set(labels))
    print("Celltype:", len(cell_type_list), cell_type_list)

    light_adata = anndata.AnnData(obs=input_adata.obs[[step00.sample_column, step00.neighborhood_column]].copy())
    light_adata.obs[step00.sample_column] = light_adata.obs[step00.sample_column].astype(str).astype("category")
    light_adata.obsm[step00.spatial_key] = input_adata.obs[list(step00.spatial_columns)].to_numpy(dtype=float)

    squidpy.gr.spatial_neighbors_radius(light_adata, radius=30, spatial_key=step00.spatial_key, library_key=step00.sample_column, key_added=step00.spatial_key, n_jobs=args.cpus)
    print(light_adata)

    degree = numpy.asarray(light_adata.obsp[step00.connectivity_key].sum(axis=1)).ravel()
    light_adata.obs[f"{step00.neighborhood_column}_degree"] = degree
    print(f"Isolated cells: {int((degree == 0).sum())} / {light_adata.n_obs}; median degree: {numpy.median(degree):.1f}")

    sample_list = sorted(set(light_adata.obs[step00.sample_column]))
    print("Sample:", len(sample_list), sample_list)

    result_list = list()
    for sample in tqdm.tqdm(sample_list):
        sample_adata = light_adata[(light_adata.obs[step00.sample_column] == sample)].copy()

        sample_labels = sample_adata.obs[step00.neighborhood_column].astype(str)
        counts = sample_labels.value_counts()
        sample_cell_type_list = sorted(counts.index[counts >= args.min_cells])
        dropped_cell_type_list = sorted(set(counts.index) - set(sample_cell_type_list))
        sample_categories = sample_cell_type_list + ([step00.rest_value] if dropped_cell_type_list else [])
        sample_adata.obs[step00.neighborhood_column] = pandas.Categorical(sample_labels.where(sample_labels.isin(sample_cell_type_list), step00.rest_value), categories=sample_categories)
        print(f"> {sample}: kept {len(sample_cell_type_list)} cell types with >= {args.min_cells} cells; pooled into {step00.rest_value}: {dropped_cell_type_list}")

        with joblib.parallel_config(max_nbytes=None):
            squidpy.gr.nhood_enrichment(sample_adata, cluster_key=step00.neighborhood_column, connectivity_key=step00.connectivity_key, n_perms=args.perms, seed=42, n_jobs=args.cpus, show_progress_bar=False)

        result = sample_adata.uns[f"{step00.neighborhood_column}_nhood_enrichment"]
        zscore = pandas.DataFrame(result["zscore"], index=sample_categories, columns=sample_categories)
        count = pandas.DataFrame(result["count"], index=sample_categories, columns=sample_categories)

        for source, target in tqdm.contrib.itertools.product(sample_cell_type_list, sample_cell_type_list, position=1, leave=False):
            result_list.append((sample, source, target, float(zscore.loc[source, target]), int(count.loc[source, target]), int(counts[source]), int(counts[target])))

    result_data = pandas.DataFrame(result_list, columns=[step00.sample_column, "Source", "Target", "zscore", "count", "n_source", "n_target"])
    print(result_data)

    input_adata.obs[f"{step00.neighborhood_column}_degree"] = light_adata.obs[f"{step00.neighborhood_column}_degree"].to_numpy()
    input_adata.obsm[step00.spatial_key] = light_adata.obsm[step00.spatial_key]
    for key in tqdm.tqdm(list(light_adata.obsp.keys())):
        input_adata.obsp[key] = light_adata.obsp[key]
    input_adata.uns[step00.neighborhood_column] = result_data
    input_adata.uns[f"{step00.neighborhood_column}_parameters"] = {"celltype": step00.celltype_column, "radius": 30, "knn": 10, "min_cells": args.min_cells, "perms": args.perms}
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
