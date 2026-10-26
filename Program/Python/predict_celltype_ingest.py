import argparse
import anndata
import anndata.io
import h5py
import numpy
import pandas
import scanpy
import notebook_pdf
import step00


def read_obs_column(path: str, column: str, names: pandas.Index, suffix: str) -> pandas.Series:
    # obs[column] of another HDF5 file, aligned to names; anndata.concat(index_unique="-") adds f"-{Omics}" to the names
    with h5py.File(path, "r") as handle:
        obs = anndata.io.read_elem(handle["obs"])

    assert (column in obs.columns), f"{column} not in obs of {path}: {list(obs.columns)}"
    values = obs[column].astype(object).reindex(names)
    if values.isna().all():
        values = obs[column].astype(object).reindex(names.str.removesuffix(suffix))
        values.index = names
    return values


def knn_vote(indices: numpy.ndarray, codes: numpy.ndarray, n_labels: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    neighbor_codes = codes[indices]
    counts = numpy.stack([(neighbor_codes == code).sum(axis=1) for code in range(n_labels)], axis=1)
    best = counts.argmax(axis=1)
    return best, counts[numpy.arange(best.size), best] / indices.shape[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Joint HDF5 file (calculate_joint_pca.py output with obs Omics and obsm X_pca_harmony)", type=str)
    parser.add_argument("reference", help="scRNA HDF5 file with the reference labels in obs", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--label", help="Label column in the reference obs", type=str, default=step00.celltype_column)
    parser.add_argument("--truth", help="Xenium HDF5 file with the authors' labels in obs (optional, for a crosstab)", type=str, default="")
    parser.add_argument("--truth-column", help="Label column in --truth", type=str, default="CellAnnotation.Level0")
    parser.add_argument("--dims", help="First dimensions of X_pca_harmony used for kNN (calculate_umap.py uses 30)", type=int, default=30)
    parser.add_argument("--neighbors", help="Reference neighbours per query cell", type=int, default=30)
    parser.add_argument("--min-fraction", help="Cells whose majority label has a smaller neighbour fraction become Unassigned (0: never)", type=float, default=0.0)
    parser.add_argument("--distance-quantile", help="Cells farther from their reference neighbours than this quantile of reference-to-reference distances become Unassigned (0: never)", type=float, default=0.0)
    parser.add_argument("--umap", help="Also place query cells on a reference UMAP (Ingest.map_embedding)", action="store_true", default=False)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.reference, {".hdf5", ".h5", ".h5ad"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    if args.truth:
        step00.check_suffix(args.truth, {".hdf5", ".h5", ".h5ad"})
    
    assert ((0.0 <= args.min_fraction <= 1.0) and (0.0 <= args.distance_quantile < 1.0)), "--min-fraction must be in [0, 1] and --distance-quantile in [0, 1)"

    notebook = notebook_pdf.Notebook(title="Predict cell types with ingest", script=__file__)
    predicted_column = f"{args.label}_ingest"
    fraction_column = f"{args.label}_ingest_fraction"
    distance_column = f"{args.label}_ingest_distance"

    with notebook.cell("Read", "Only obs and the joint Harmony embedding are read; the expression matrix is not needed."):
        with h5py.File(args.input, "r") as handle:
            obs = anndata.io.read_elem(handle["obs"])
            embedding = numpy.asarray(handle["obsm"][step00.harmony_key][:, :args.dims], dtype=numpy.float64)
        omics_array = obs[step00.omics_column].astype(str).to_numpy()
        print(obs[step00.omics_column].value_counts())
        print("Embedding:", embedding.shape)

    with notebook.cell("Reference labels", "Labels come from the scRNA file; cells without a label are left out of the reference."):
        reference_mask = (omics_array == "scRNA")
        labels = read_obs_column(args.reference, args.label, obs.index[reference_mask], "-scRNA")
        keep = labels.notna().to_numpy()
        reference_index = numpy.flatnonzero(reference_mask)[keep]
        query_index = numpy.flatnonzero(omics_array == "Xenium")
        categories = pandas.Categorical(labels[keep].astype(str).map(step00.safe_celltype))
        print(pandas.Series(categories).value_counts().to_frame("Reference cells"))
        print("Reference:", reference_index.size, "| unlabelled reference cells:", int((~keep).sum()), "| query:", query_index.size)

    with notebook.cell("Reference graph", "Ingest needs scanpy neighbors (equal neighbours per row) on the reference; UMAP only with --umap."):
        reference_adata = anndata.AnnData(obs=pandas.DataFrame({args.label: categories}, index=obs.index[reference_index]), obsm={"X_ingest": embedding[reference_index]})
        query_adata = anndata.AnnData(obs=pandas.DataFrame(index=obs.index[query_index]), obsm={"X_ingest": embedding[query_index]})
        scanpy.pp.neighbors(reference_adata, n_neighbors=args.neighbors, use_rep="X_ingest", random_state=42)
        if args.umap:
            scanpy.tl.umap(reference_adata, random_state=42)
        print(reference_adata)

    with notebook.cell("Ingest", "Query cells are searched in the reference kNN index; same space as calculate_umap.py (X_pca_harmony, first --dims dimensions)."):
        ingest = scanpy.tl.Ingest(reference_adata)
        ingest.fit(query_adata)
        ingest.neighbors(k=args.neighbors)
        if args.umap:
            ingest.map_embedding("umap")
        ingest.to_adata(inplace=True)
        print(query_adata)

    with notebook.cell("Label vote", "Majority of the k neighbours as in Ingest.map_labels, the fraction of neighbours that agree, and the mean distance to them (large for cell types missing from the reference)."):
        best, fraction = knn_vote(ingest._indices, categories.codes.astype(numpy.int64), len(categories.categories))  # Ingest keeps the neighbour indices and distances only as private attributes
        distance = ingest._distances.mean(axis=1)
        reference_distance = numpy.asarray(reference_adata.obsp["distances"].sum(axis=1)).ravel() / (args.neighbors - 1)
        distance_cut = numpy.quantile(reference_distance, args.distance_quantile) if (args.distance_quantile > 0) else numpy.inf
        print(pandas.DataFrame({"Reference to reference": pandas.Series(reference_distance).describe(percentiles=[0.5, 0.9, 0.99]), "Query to reference": pandas.Series(distance).describe(percentiles=[0.5, 0.9, 0.99])}))

        predicted = numpy.asarray(categories.categories)[best].astype(object)
        predicted[(fraction < args.min_fraction) | (distance > distance_cut)] = "Unassigned"
        prediction_data = pandas.DataFrame({predicted_column: predicted, fraction_column: fraction, distance_column: distance}, index=query_adata.obs_names)
        print(prediction_data[predicted_column].value_counts().to_frame("Query cells"))
        print(prediction_data.groupby(predicted_column)[fraction_column].describe())

    with notebook.cell("Truth", "Crosstab of the predictions against the authors' Xenium labels (rows: authors, columns: ingest)."):
        tables = dict()
        if args.truth:
            truth = read_obs_column(args.truth, args.truth_column, query_adata.obs_names, "-Xenium").fillna("Unlabelled").astype(str).map(step00.safe_celltype)
            crosstab = pandas.crosstab(truth, prediction_data[predicted_column]).rename_axis(index=args.truth_column, columns=predicted_column)
            tables["Crosstab"] = crosstab
            tables["Best_match"] = pandas.DataFrame({"Top ingest label": crosstab.idxmax(axis=1), "Fraction": crosstab.max(axis=1) / crosstab.sum(axis=1), "Cells": crosstab.sum(axis=1)}).sort_values("Cells", ascending=False)
            print(tables["Best_match"].to_string(float_format=lambda x: f"{x:.3f}"))
        else:
            print("No --truth file")

    with notebook.cell("Save"):
        output_obs = obs[[column for column in [step00.omics_column, step00.sample_column, step00.sketch_column] if column in obs.columns]].copy()
        output_obs[predicted_column] = pandas.Series(dtype=object, index=output_obs.index)
        output_obs.loc[obs.index[reference_index], predicted_column] = numpy.asarray(categories.astype(str), dtype=object)
        output_obs.loc[prediction_data.index, predicted_column] = prediction_data[predicted_column]
        output_obs[predicted_column] = pandas.Categorical(output_obs[predicted_column])
        output_obs[fraction_column] = numpy.nan
        output_obs.loc[prediction_data.index, fraction_column] = prediction_data[fraction_column]
        output_obs[distance_column] = numpy.nan
        output_obs.loc[prediction_data.index, distance_column] = prediction_data[distance_column]

        if args.truth:
            output_obs.loc[truth.index, args.truth_column] = truth

        output_adata = anndata.AnnData(obs=output_obs, obsm={step00.harmony_key: embedding})
        if args.umap:
            projection = numpy.full((obs.shape[0], 2), numpy.nan)
            projection[reference_index] = reference_adata.obsm["X_umap"]
            projection[query_index] = query_adata.obsm["X_umap"]
            output_adata.obsm[step00.projection_key] = projection
        output_adata.uns["ingest"] = {key: value for key, value in vars(args).items() if key not in ("input", "output")} | {"reference_cells": int(reference_index.size), "query_cells": int(query_index.size), "use_rep": f"{step00.harmony_key}[:, :{args.dims}]", "distance_cut": float(distance_cut)} | tables
        print(output_adata)
        output_adata.write_h5ad(args.output, **step00.anndata_compressions)

    notebook.write_pdf(args.output)
