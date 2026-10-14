import argparse
import numpy
import scanpy
import umap
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--dims", help="Number of Harmony dimensions (Seurat RunUMAP dims)", type=int, default=30)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    sketch_mask = input_adata.obs[step00.sketch_column].to_numpy(dtype=bool)
    model = umap.UMAP(n_neighbors=30, n_components=2, metric="cosine", min_dist=0.3, spread=1.0, n_jobs=args.cpus)
    model.fit(numpy.asarray(input_adata.obsm[step00.sketch_key])[sketch_mask, :args.dims])
    input_adata.obsm[step00.projection_key] = model.transform(numpy.asarray(input_adata.obsm[step00.harmony_key])[:, :args.dims]).astype(numpy.float64)
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
