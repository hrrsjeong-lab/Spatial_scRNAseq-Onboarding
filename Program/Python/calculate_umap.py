import argparse
import anndata
import numpy
import pandas
import scanpy
import step00


def to_numpy(array) -> numpy.ndarray:
    return array.get() if hasattr(array, "get") else numpy.asarray(array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--dims", help="Harmony dimensions as start:end (Seurat RunUMAP dims)", type=str, default="1:30")
    parser.add_argument("--fit", help="Fit UMAP on sketch cells and project the rest, or fit on all cells", choices=["sketch", "full"], default="sketch")
    parser.add_argument("--backend", help="UMAP implementation; rapids (GPU) only with --fit full", choices=["umap-learn", "rapids"], default="umap-learn")
    parser.add_argument("--init", help="Embedding initialisation (uwot and Seurat default: spectral)", choices=["spectral", "random"], default="spectral")
    parser.add_argument("--n-neighbors", help="Seurat RunUMAP n.neighbors", type=int, default=30)
    parser.add_argument("--metric", help="Seurat RunUMAP metric", type=str, default="cosine")
    parser.add_argument("--min-dist", help="Seurat RunUMAP min.dist", type=float, default=0.3)
    parser.add_argument("--spread", help="Seurat RunUMAP spread", type=float, default=1.0)
    parser.add_argument("--n-epochs", help="Seurat RunUMAP n.epochs (0: 200, the uwot default for large data)", type=int, default=0)
    parser.add_argument("--learning-rate", help="Seurat RunUMAP learning.rate", type=float, default=1.0)
    parser.add_argument("--set-op-mix-ratio", help="Seurat RunUMAP set.op.mix.ratio (umap-learn only)", type=float, default=1.0)
    parser.add_argument("--local-connectivity", help="Seurat RunUMAP local.connectivity (umap-learn only)", type=int, default=1)
    parser.add_argument("--repulsion-strength", help="Seurat RunUMAP repulsion.strength (umap-learn only)", type=float, default=1.0)
    parser.add_argument("--negative-sample-rate", help="Seurat RunUMAP negative.sample.rate", type=int, default=5)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    if (args.backend == "rapids") and (args.fit != "full"):
        raise ValueError("--backend rapids requires --fit full!!")

    dims = step00.parse_dims(args.dims)
    n_epochs = 200 if (args.n_epochs <= 0) else args.n_epochs

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata, flush=True)

    sketch_mask = input_adata.obs[step00.sketch_column].to_numpy(dtype=bool)
    sample_array = input_adata.obs[step00.sample_column].astype(str).to_numpy()
    harmony_full = numpy.asarray(input_adata.obsm[step00.harmony_key], dtype=numpy.float64)[:, dims]
    print("UMAP input:", harmony_full.shape, "backend:", args.backend, "fit:", args.fit, flush=True)

    finite_mask = numpy.isfinite(harmony_full).all(axis=1)
    if not finite_mask.all():
        print(pandas.Series(sample_array[~finite_mask]).value_counts().to_string(), flush=True)
        raise ValueError(f"{int((~finite_mask).sum())} cells have NaN/Inf in {step00.harmony_key}!!")

    norm = numpy.linalg.norm(harmony_full, axis=1)
    zero_mask = (norm == 0)
    input_adata.obs["UMAP_zero_input"] = zero_mask
    print("Cells with an all-zero UMAP input:", int(zero_mask.sum()), flush=True)
    if zero_mask.any():
        print(pandas.Series(sample_array[zero_mask]).value_counts().to_string(), flush=True)

    if args.backend == "rapids":
        import rapids_singlecell

        umap_input = harmony_full
        if args.metric == "cosine":
            umap_input = numpy.zeros((harmony_full.shape[0], harmony_full.shape[1] + 1), dtype=numpy.float64)
            umap_input[~zero_mask, :-1] = harmony_full[~zero_mask] / norm[~zero_mask, None]
            umap_input[zero_mask, -1] = 1.0

        umap_adata = anndata.AnnData(obs=pandas.DataFrame(index=input_adata.obs_names), obsm={"X_input": umap_input.astype(numpy.float32)})
        rapids_singlecell.pp.neighbors(umap_adata, n_neighbors=args.n_neighbors, use_rep="X_input", metric=args.metric, algorithm="brute", random_state=42)
        print("Neighbors done", flush=True)
        rapids_singlecell.tl.umap(umap_adata, min_dist=args.min_dist, spread=args.spread, maxiter=n_epochs, alpha=args.learning_rate, negative_sample_rate=args.negative_sample_rate, init_pos=args.init, random_state=42)
        input_adata.obsm[step00.projection_key] = to_numpy(umap_adata.obsm["X_umap"]).astype(numpy.float64)
        del umap_adata, umap_input
    else:
        import umap

        model = umap.UMAP(n_neighbors=args.n_neighbors, n_components=2, metric=args.metric, min_dist=args.min_dist, spread=args.spread, n_epochs=n_epochs, learning_rate=args.learning_rate, set_op_mix_ratio=args.set_op_mix_ratio, local_connectivity=args.local_connectivity, repulsion_strength=args.repulsion_strength, negative_sample_rate=args.negative_sample_rate, init=args.init, n_jobs=args.cpus, verbose=True)
        if args.fit == "full":
            input_adata.obsm[step00.projection_key] = model.fit_transform(harmony_full).astype(numpy.float64)
        else:
            model.fit(numpy.asarray(input_adata.obsm[step00.sketch_key])[sketch_mask, dims])
            input_adata.obsm[step00.projection_key] = model.transform(harmony_full).astype(numpy.float64)

    input_adata.uns[step00.projection_key] = {key: value for key, value in vars(args).items() if key not in ("input", "output", "cpus")} | {"n_epochs": n_epochs, "zero_input_cells": int(zero_mask.sum())}
    print(input_adata, flush=True)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
