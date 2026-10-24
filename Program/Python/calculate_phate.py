import argparse
import numpy
import pandas
import phate
import scanpy
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--scrna", help="scRNA HDF5 file", type=str, default="")
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    scanpy.settings.n_jobs = args.cpus

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    if args.scrna:
        step00.check_suffix(args.scrna, {".hdf5", ".h5"})
        scrna_adata = scanpy.read_h5ad(args.scrna)
        print(scrna_adata)

        input_adata = scanpy.concat([input_adata, scrna_adata], axis="obs", join="inner", label=step00.omics_column, keys=["Xenium", "scRNA"])
        print(input_adata)

    sketch_mask = input_adata.obs[step00.sketch_column].to_numpy(dtype=bool)
    sample_array = input_adata.obs[step00.sample_column].astype(str).to_numpy()
    harmony_full = numpy.asarray(input_adata.obsm[step00.harmony_key], dtype=numpy.float64)[:, :30]
    print(pandas.DataFrame(harmony_full))

    finite_mask = numpy.isfinite(harmony_full).all(axis=1)
    if not finite_mask.all():
        print(pandas.Series(sample_array[~finite_mask]).value_counts().to_string())
        raise ValueError(f"{int((~finite_mask).sum())} cells have NaN/Inf in {step00.harmony_key}!!")

    norm = numpy.linalg.norm(harmony_full, axis=1)
    zero_mask = (norm == 0)
    input_adata.obs["zero_input"] = zero_mask
    print("Cells with an all-zero input:", int(zero_mask.sum()))

    if zero_mask.any():
        print(pandas.Series(sample_array[zero_mask]).value_counts().to_string())

    input_adata.obsm[step00.projection_key] = phate.PHATE(n_components=2, knn=5, decay=15, t="auto", gamma=1.0, n_pca=None, knn_dist="euclidean", mds_dist="euclidean", mds="metric", n_jobs=args.cpus, random_state=42, verbose=True).fit_transform(harmony_full.astype(numpy.float64))
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
