import argparse
import typing
import anndata
import anndata.io
import h5py
import numpy
import pandas
import scanpy
import step00
from predict_celltype_ingest import read_obs

unassigned_value = "Unassigned"
parameter_keys = ["label", "reference", "dims", "neighbors", "min_fraction", "distance_quantile", "use_rep"]


def read_ingest(path: str) -> typing.Tuple[typing.Dict[str, typing.Any], pandas.DataFrame]:
    with h5py.File(path, "r") as handle:
        assert ("ingest" in handle["uns"]), f"uns ingest not in {path}"
        parameters = {key: anndata.io.read_elem(handle["uns"]["ingest"][key]) for key in parameter_keys if key in handle["uns"]["ingest"]}
        key = f"{parameters.get('label', step00.celltype_column)}_ingest"
        assert (key in handle["obsm"]), f"obsm {key} not in {path}: rerun step 13 with the current predict_celltype_ingest.py"
        votes = anndata.io.read_elem(handle["obsm"][key])

    assert isinstance(votes, pandas.DataFrame), f"obsm {key} of {path} is not a DataFrame: {type(votes)}"
    return parameters, votes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Step 07 HDF5 file of the Xenium cells (the select_marker_genes.py input)", type=str)
    parser.add_argument("ingest", help="Step 13 HDF5 file (predict_celltype_ingest.py output with obsm <label>_ingest)", type=str)
    parser.add_argument("output", help="Output HDF5 file with obs Celltype and obsm Celltype as select_marker_genes.py writes them", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.ingest, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    parameters, vote_data = read_ingest(args.ingest)
    label = str(parameters.get("label", step00.celltype_column))
    predicted_column, fraction_column, distance_column = f"{label}_ingest", f"{label}_ingest_fraction", f"{label}_ingest_distance"
    print("Ingest parameters:", parameters)

    # Xenium cells of step 13 under the step 07 names (calculate_joint_pca.py adds "-Xenium" only when the names overlap)
    ingest_obs = read_obs(args.ingest)
    xenium_mask = (ingest_obs[step00.omics_column].astype(str) == "Xenium").to_numpy()
    ingest_obs = ingest_obs.loc[xenium_mask, [predicted_column, fraction_column, distance_column]]
    vote_data = vote_data.iloc[numpy.flatnonzero(xenium_mask)]

    names = pandas.Index(ingest_obs.index.astype(str))
    stripped = names.str.removesuffix("-Xenium")
    if stripped.isin(input_adata.obs_names).sum() > names.isin(input_adata.obs_names).sum():
        names = stripped
    assert (not names.has_duplicates), f"Duplicated Xenium cell names in {args.ingest}"
    ingest_obs.index = names
    vote_data.index = names

    matched = input_adata.obs_names.isin(names)
    print("Step 07 cells with an ingest label:", int(matched.sum()), "of", input_adata.n_obs)
    print("Step 13 Xenium cells absent from step 07:", int((~names.isin(input_adata.obs_names)).sum()), "of", names.size)
    assert matched.any(), f"No cell of {args.input} found in {args.ingest}"

    # Same keys as select_marker_genes.py: obs Celltype (one label per cell) and obsm Celltype (cells x labels, columns sorted)
    celltype = ingest_obs[predicted_column].astype(object).reindex(input_adata.obs_names).fillna(unassigned_value).astype(str)
    input_adata.obs[step00.celltype_column] = pandas.Categorical(celltype)
    input_adata.obs[fraction_column] = ingest_obs[fraction_column].astype(float).reindex(input_adata.obs_names).to_numpy()
    input_adata.obs[distance_column] = ingest_obs[distance_column].astype(float).reindex(input_adata.obs_names).to_numpy()
    input_adata.obsm[step00.celltype_column] = vote_data.loc[:, sorted(vote_data.columns)].reindex(input_adata.obs_names).astype(numpy.float64)
    print(input_adata.obs[step00.celltype_column].value_counts())
    print(input_adata.obsm[step00.celltype_column])

    input_adata.uns[f"{step00.celltype_column}_source"] = {"method": "Ingest", "ingest": args.ingest, "matched_cells": int(matched.sum()), "unassigned_cells": int((celltype == unassigned_value).sum())} | {key: (value.item() if isinstance(value, numpy.generic) else value) for key, value in parameters.items()}
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
