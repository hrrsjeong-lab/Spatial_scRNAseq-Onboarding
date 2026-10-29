import argparse
import gc
import shutil
import anndata
import anndata.io
import h5py
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
    parser.add_argument("--knn-max", help="PHATE knn_max: most neighbours per cell in the alpha-decay kernel", type=int, default=30)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)
    if (args.knn_max < 0):
        raise ValueError("--knn-max must be >= 0!!")

    assert args.knn_max > 5, "--knn-max must be >= 0!!"

    scanpy.settings.n_jobs = args.cpus

    if args.scrna:
        step00.check_suffix(args.scrna, {".hdf5", ".h5"})
        input_adata = scanpy.read_h5ad(args.input)
        print(input_adata)
        scrna_adata = scanpy.read_h5ad(args.scrna)
        print(scrna_adata)

        input_adata = scanpy.concat([input_adata, scrna_adata], axis="obs", join="inner", label=step00.omics_column, keys=["Xenium", "scRNA"])
        del scrna_adata
        print(input_adata)

        obs = input_adata.obs
        harmony_full = numpy.asarray(input_adata.obsm[step00.harmony_key], dtype=numpy.float64)[:, :30].copy()
    else:
        input_adata = None
        with h5py.File(args.input, "r") as handle:
            print("Input:", args.input, "| obsm:", list(handle["obsm"].keys()), "| layers:", (list(handle["layers"].keys()) if ("layers" in handle) else list()))
            obs = anndata.io.read_elem(handle["obs"])
            harmony_full = numpy.asarray(handle["obsm"][step00.harmony_key][:, :30], dtype=numpy.float64)
        print("Cells:", len(obs), "| obs columns:", list(obs.columns))

    sketch_mask = obs[step00.sketch_column].to_numpy(dtype=bool)
    sample_array = obs[step00.sample_column].astype(str).to_numpy()
    print(pandas.DataFrame(harmony_full))

    finite_mask = numpy.isfinite(harmony_full).all(axis=1)
    if not finite_mask.all():
        print(pandas.Series(sample_array[~finite_mask]).value_counts().to_string())
        raise ValueError(f"{int((~finite_mask).sum())} cells have NaN/Inf in {step00.harmony_key}!!")

    norm = numpy.linalg.norm(harmony_full, axis=1)
    zero_mask = (norm == 0)
    obs["zero_input"] = zero_mask
    print("Cells with an all-zero input:", int(zero_mask.sum()))

    if zero_mask.any():
        print(pandas.Series(sample_array[zero_mask]).value_counts().to_string())

    phate_operator = phate.PHATE(n_components=2, knn=5, decay=15, t="auto", gamma=1.0, n_pca=None, knn_dist="euclidean", knn_max=args.knn_max, mds_dist="euclidean", mds="metric", n_jobs=args.cpus, random_state=42, verbose=True)
    projection = numpy.asarray(phate_operator.fit_transform(harmony_full), dtype=numpy.float64)
    print("Kernel non-zeros per cell:", phate_operator.graph.kernel.nnz / harmony_full.shape[0], "| optimal t:", phate_operator.optimal_t)
    del phate_operator, harmony_full
    gc.collect()

    if input_adata is not None:
        input_adata.obsm[step00.projection_key] = projection
        print(input_adata)
        input_adata.write_h5ad(args.output, **step00.anndata_compressions)
    else:
        shutil.copyfile(args.input, args.output)
        with h5py.File(args.output, "r+") as handle:
            anndata.io.write_elem(handle, "obs", obs, dataset_kwargs=step00.anndata_compressions)
            anndata.io.write_elem(handle["obsm"], step00.projection_key, projection, dataset_kwargs=step00.anndata_compressions)
            print("Output:", args.output, "| obs columns:", list(obs.columns), "| obsm:", list(handle["obsm"].keys()))
