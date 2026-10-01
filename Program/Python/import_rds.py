import argparse
import pathlib
import anndata
import h5py
import numpy
import pandas
import scipy.sparse
import tqdm
import step00


def read_counts(d):
    d = pathlib.Path(d)
    with h5py.File(d / "counts_csc.h5", "r") as f:
        x, i, p = f["x"][:], f["i"][:], f["p"][:]
        n_genes, n_cells = (int(v) for v in tqdm.tqdm(f["shape"][:]))

    X = scipy.sparse.csc_matrix((x.astype(numpy.float32), i, p), shape=(n_genes, n_cells)).T.tocsr()
    genes = (d / "genes.txt").read_text().splitlines()
    cells = (d / "cells.txt").read_text().splitlines()

    if (len(genes), len(cells)) != (n_genes, n_cells):
        raise ValueError(f"shape {n_genes}x{n_cells} vs genes.txt {len(genes)} / cells.txt {len(cells)}")

    return X, genes, cells


def build(d, layer_key=step00.log_column, spatial_key=step00.spatial_key):
    d = pathlib.Path(d)
    X, genes, cells = read_counts(d)

    obs = pandas.read_parquet(d / "obs.parquet").set_index("cell_id")
    obs.index.name = None
    if obs.index.duplicated().any():
        raise ValueError("duplicated cell_id in obs.parquet")

    if list(obs.index) != cells:
        missing = pandas.Index(cells).difference(obs.index)
        if len(missing):
            raise ValueError(f"{len(missing)} matrix cells are missing from obs.parquet")
        obs = obs.loc[cells]

    adata = anndata.AnnData(X=X, obs=obs, var=pandas.DataFrame(index=pandas.Index(genes)))
    adata.layers[layer_key] = adata.X.copy()

    for fp in tqdm.tqm(sorted(d.glob("obsm_*.parquet"))):
        r = fp.stem.replace("obsm_", "", 1)
        emb = pandas.read_parquet(fp).set_index("cell_id").reindex(adata.obs_names)
        adata.obsm[f"X_{r}"] = emb.to_numpy(dtype=numpy.float32)
        adata.obs[f"has_{r}"] = emb.notna().all(axis=1).to_numpy()
        print(f"obsm['X_{r}']: {emb.shape[1]} dims, {int(adata.obs[f'has_{r}'].sum()):,}/{adata.n_obs:,} cells")

    sc_fp = d / "spatial_centroids.parquet"
    if sc_fp.exists():
        xy = pandas.read_parquet(sc_fp)
        xy = xy.drop_duplicates("cell").set_index("cell").reindex(adata.obs_names)
        if "fov" in xy:
            adata.obs["fov"] = pandas.Categorical(xy["fov"])
        src = "spatial_centroids.parquet"
    elif {"x", "y"} <= set(adata.obs.columns):
        xy, src = adata.obs[["x", "y"]], "obs x, y"
    else:
        xy, src = None, None

    if xy is not None:
        adata.obsm[spatial_key] = xy[["x", "y"]].to_numpy(dtype=numpy.float64)
        n_ok = int(numpy.isfinite(adata.obsm[spatial_key]).all(axis=1).sum())
        print(f"obsm['{spatial_key}'] from {src}: {n_ok:,}/{adata.n_obs:,} cells")

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input directory ", type=str)
    parser.add_argument("output", help="Output H5AD file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.output, {".h5ad", ".h5", ".hdf5"})

    adata = build(args.input)
    adata.write_h5ad(args.output,**step00.anndata_compressions)
