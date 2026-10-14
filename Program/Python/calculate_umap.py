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
    parser.add_argument("--n-neighbors", help="Seurat RunUMAP n.neighbors", type=int, default=30)
    parser.add_argument("--metric", help="Seurat RunUMAP metric", type=str, default="cosine")
    parser.add_argument("--min-dist", help="Seurat RunUMAP min.dist", type=float, default=0.3)
    parser.add_argument("--spread", help="Seurat RunUMAP spread", type=float, default=1.0)
    parser.add_argument("--n-epochs", help="Seurat RunUMAP n.epochs (0: library default)", type=int, default=0)
    parser.add_argument("--learning-rate", help="Seurat RunUMAP learning.rate", type=float, default=1.0)
    parser.add_argument("--set-op-mix-ratio", help="Seurat RunUMAP set.op.mix.ratio", type=float, default=1.0)
    parser.add_argument("--local-connectivity", help="Seurat RunUMAP local.connectivity", type=int, default=1)
    parser.add_argument("--repulsion-strength", help="Seurat RunUMAP repulsion.strength", type=float, default=1.0)
    parser.add_argument("--negative-sample-rate", help="Seurat RunUMAP negative.sample.rate", type=int, default=5)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    sketch_mask = input_adata.obs[step00.sketch_column].to_numpy(dtype=bool)
    model = umap.UMAP(n_neighbors=args.n_neighbors, n_components=2, metric=args.metric, min_dist=args.min_dist, spread=args.spread, n_epochs=(None if (args.n_epochs <= 0) else args.n_epochs), learning_rate=args.learning_rate, set_op_mix_ratio=args.set_op_mix_ratio, local_connectivity=args.local_connectivity, repulsion_strength=args.repulsion_strength, negative_sample_rate=args.negative_sample_rate, n_jobs=args.cpus)
    model.fit(numpy.asarray(input_adata.obsm[step00.sketch_key])[sketch_mask, :args.dims])
    input_adata.obsm[step00.projection_key] = model.transform(numpy.asarray(input_adata.obsm[step00.harmony_key])[:, :args.dims]).astype(numpy.float64)
    input_adata.uns[step00.projection_key] = {key: value for key, value in vars(args).items() if key not in ("input", "output", "cpus")}
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
