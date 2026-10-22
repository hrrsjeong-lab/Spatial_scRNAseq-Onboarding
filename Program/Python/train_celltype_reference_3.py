import argparse
import celltypist
import numpy
import scanpy
import rapids_singlecell
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="scRNA-seq H5AD file with raw counts in X", type=str)
    parser.add_argument("reference", help="Xenium HDF5 file providing the gene panel", type=str)
    parser.add_argument("output", help="Output PKL file", type=str)
    parser.add_argument("--label", help="Label column in obs", type=str, required=True)
    parser.add_argument("--cpus", help="CPUs to use", type=int, default=1)
    parser.add_argument("--feature-selection", help="celltypist feature_selection (auto: only when more genes than --top-genes)", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--top-genes", help="celltypist top_genes for feature selection", type=int, default=300)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5ad", ".h5"})
    step00.check_suffix(args.reference, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".pkl"})
    step00.check_cpus(args.cpus)

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    reference_adata = scanpy.read_h5ad(args.reference, backed="r")
    gene_set = set(reference_adata.var.index)
    reference_adata.file.close()

    sample_values = input_adata.X[:100].toarray()
    assert numpy.allclose(sample_values, numpy.round(sample_values)), "X is not raw counts"

    input_adata = input_adata[input_adata.obs[args.label].notna(), input_adata.var.index.isin(gene_set)].copy()
    print("Panel genes:", len(gene_set), "Overlap:", input_adata.n_vars)

    input_adata.obs[args.label] = input_adata.obs[args.label].astype(str).map(step00.safe_celltype)
    print(input_adata.obs[args.label].value_counts())

    target_sum = 10 ** 4
    rapids_singlecell.pp.normalize_total(input_adata, target_sum=target_sum)
    scanpy.pp.log1p(input_adata)
    print(input_adata)

    feature_selection = (args.feature_selection == "on") or ((args.feature_selection == "auto") and (input_adata.n_vars > args.top_genes))
    print("Feature selection:", feature_selection, "| genes:", input_adata.n_vars, "| top_genes:", args.top_genes)

    model = celltypist.train(input_adata, labels=args.label, use_SGD=True, n_jobs=args.cpus, feature_selection=feature_selection, top_genes=args.top_genes)
    print(model)

    model.description["normalization_genes"] = list(input_adata.var.index)
    model.description["target_sum"] = target_sum
    model.write(args.output)
