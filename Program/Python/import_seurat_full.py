import argparse
import pathlib
import anndata
import numpy
import pandas
import scipy.sparse
import step00


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="obs.csv.gz written by export_seurat_full.R", type=str)
    parser.add_argument("output", help="Output H5AD file", type=str)
    parser.add_argument("--data-layer", help="AnnData layer name for the Seurat data layer", type=str, default=step00.log_column)

    args = parser.parse_args()

    step00.check_suffix(args.output, {".h5ad", ".h5", ".hdf5"})
    directory = pathlib.Path(args.input).parent

    n_genes, n_cells = (int(value) for value in (directory / "shape.txt").read_text().split())
    genes = (directory / "genes.txt").read_text().splitlines()
    cells = (directory / "cells.txt").read_text().splitlines()
    if (len(genes), len(cells)) != (n_genes, n_cells):
        raise ValueError(f"shape {n_genes}x{n_cells} vs genes.txt {len(genes)} / cells.txt {len(cells)}")

    obs = pandas.read_csv(args.input, index_col="cell_id", low_memory=False)
    obs.index = obs.index.astype(str)
    obs.index.name = None
    obs = clean_frame(obs.loc[cells])

    var = pandas.DataFrame(index=pandas.Index(genes))
    if (directory / "var.csv.gz").exists():
        feature_meta = pandas.read_csv(directory / "var.csv.gz", index_col="gene", low_memory=False)
        feature_meta.index = feature_meta.index.astype(str)
        var = var.join(feature_meta, how="left")
    variable_features = set((directory / "variable_features.txt").read_text().split()) if (directory / "variable_features.txt").exists() else set()
    var["highly_variable"] = var.index.isin(variable_features)
    var = clean_frame(var)

    layer_list = (directory / "layers.txt").read_text().split()
    adata = anndata.AnnData(X=read_layer(directory, "counts", n_genes, n_cells), obs=obs, var=var)
    for layer in layer_list:
        if layer != "counts":
            adata.layers[args.data_layer if (layer == "data") else layer] = read_layer(directory, layer, n_genes, n_cells)

    reduction_list = (directory / "reductions.txt").read_text().split() if (directory / "reductions.txt").exists() else list()
    for reduction in reduction_list:
        embedding = pandas.read_csv(directory / f"reduction_{reduction}.csv.gz", index_col="cell_id")
        embedding.index = embedding.index.astype(str)
        embedding = embedding.reindex(adata.obs_names)
        adata.obsm[f"X_{reduction}"] = embedding.to_numpy(dtype=numpy.float32)
        adata.obs[f"has_{reduction}"] = embedding.notna().all(axis=1).to_numpy()
        print(f"obsm['X_{reduction}']: {embedding.shape[1]} dims, {int(adata.obs[f'has_{reduction}'].sum()):,}/{adata.n_obs:,} cells")

    adata.uns["seurat"] = {"source": str(directory), "layers": layer_list, "reductions": reduction_list, "object": (directory / "object.txt").read_text() if (directory / "object.txt").exists() else ""}
    sample_values = adata.X[:100].toarray()
    print("X integer-valued (first 100 cells):", bool(numpy.allclose(sample_values, numpy.round(sample_values))))
    print("Highly variable genes:", int(adata.var["highly_variable"].sum()))
    print(adata)
    adata.write_h5ad(args.output, **step00.anndata_compressions)
