import argparse
import numpy
import phate
import scanpy
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
    print("PHATE input:", projection_input.shape, projection_input.dtype)

    phate_operator = phate.PHATE(n_components=2, knn=5, decay=15, n_landmark=args.landmark, t="auto", gamma=1.0, n_pca=None, knn_dist="euclidean", mds_dist="euclidean", mds="metric", n_jobs=args.cpus, random_state=42, verbose=1)
    input_adata.obsm[step00.projection_key] = phate_operator.fit_transform(projection_input)
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
