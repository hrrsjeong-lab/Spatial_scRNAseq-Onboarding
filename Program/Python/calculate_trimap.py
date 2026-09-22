import argparse
import numpy
import scanpy
import trimap
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    scanpy.settings.n_jobs = args.cpus

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    projection_input = numpy.ascontiguousarray(input_adata.obsm[step00.harmony_key], dtype=numpy.float32)
    print("TriMap input:", projection_input.shape, projection_input.dtype)

    trimap_operator = trimap.TRIMAP(n_dims=2, n_inliers=10, n_outliers=5, n_random=5, distance="euclidean", n_iters=400, apply_pca=False, verbose=True)
    input_adata.obsm[step00.projection_key] = trimap_operator.fit_transform(projection_input)
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
