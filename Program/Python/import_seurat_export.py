import argparse
import pathlib
import anndata
import numpy
import pandas
import scipy.sparse
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="obs.csv.gz written by export_seurat_reference.R", type=str)
    parser.add_argument("output", help="Output H5AD file", type=str)

    args = parser.parse_args()

    step00.check_suffix(args.output, {".h5ad", ".h5", ".hdf5"})
    directory = pathlib.Path(args.input).parent

    n_genes, n_cells = (int(v) for v in (directory / "shape.txt").read_text().split())
    x = numpy.fromfile(directory / "counts_x.bin", dtype="<f4")
    i = numpy.fromfile(directory / "counts_i.bin", dtype="<i4")
    p = numpy.fromfile(directory / "counts_p.bin", dtype="<i4")
    X = scipy.sparse.csc_matrix((x, i, p), shape=(n_genes, n_cells)).T.tocsr()

    genes = (directory / "genes.txt").read_text().splitlines()
    cells = (directory / "cells.txt").read_text().splitlines()
    if (len(genes), len(cells)) != (n_genes, n_cells):
        raise ValueError(f"shape {n_genes}x{n_cells} vs genes.txt {len(genes)} / cells.txt {len(cells)}")

    obs = pandas.read_csv(args.input, index_col="cell_id", low_memory=False)
    obs.index.name = None
    obs = obs.loc[cells]

    adata = anndata.AnnData(X=X, obs=obs, var=pandas.DataFrame(index=pandas.Index(genes)))
    print(adata)
    adata.write_h5ad(args.output, **step00.anndata_compressions)
