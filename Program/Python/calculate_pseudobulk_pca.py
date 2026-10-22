import argparse
import anndata
import anndata.io
import h5py
import numpy
import pandas
import pydeseq2.dds
import scipy.sparse
import sklearn.decomposition
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file with raw counts in X", type=str)
    parser.add_argument("output", help="Output HDF5 file (samples x genes)", type=str)
    parser.add_argument("--group-column", help="Sample-level obs column for colouring (e.g. TYPE)", type=str, default="TYPE")
    parser.add_argument("--meta-columns", help="Other sample-level obs columns to keep", type=str, nargs="*", default=["LOCATION"])
    parser.add_argument("--top-genes", help="Most variable VST genes for PCA (DESeq2 plotPCA ntop)", type=int, default=1000)
    parser.add_argument("--chunk", help="Rows read from X per step", type=int, default=200000)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5", ".h5ad"})
    step00.check_suffix(args.output, {".hdf5", ".h5", ".h5ad"})

    with h5py.File(args.input, "r") as handle:
        obs = anndata.io.read_elem(handle["obs"])
        var_names = pandas.Index(anndata.io.read_elem(handle["var"]).index.astype(str))
        X = anndata.io.sparse_dataset(handle["X"])

        sample_array = obs[step00.sample_column].astype(str).to_numpy()
        sample_list = sorted(set(sample_array))
        indicator = scipy.sparse.csr_matrix((numpy.ones(obs.shape[0]), (pandas.Index(sample_list).get_indexer(sample_array), numpy.arange(obs.shape[0]))), shape=(len(sample_list), obs.shape[0]))

        counts = numpy.zeros((len(sample_list), len(var_names)), dtype=numpy.float64)
        for start in tqdm.trange(0, obs.shape[0], args.chunk):
            stop = min(start + args.chunk, obs.shape[0])
            counts += (indicator[:, start:stop] @ scipy.sparse.csr_matrix(X[start:stop])).toarray()

    assert numpy.allclose(counts, numpy.round(counts)), "X is not raw counts"
    print("Pseudobulk:", counts.shape, "| cells per sample:", dict(zip(sample_list, numpy.asarray(indicator.sum(axis=1)).ravel().astype(int))))

    meta_columns = [column for column in [args.group_column] + args.meta_columns if column in obs.columns]
    metadata = obs.assign(**{step00.sample_column: sample_array}).groupby(step00.sample_column, observed=True)[meta_columns].agg(lambda values: "|".join(sorted(set(values.astype(str)))))
    metadata = metadata.loc[sample_list]
    metadata["Cells"] = numpy.asarray(indicator.sum(axis=1)).ravel().astype(int)
    for column in tqdm.tqdm(meta_columns):
        if metadata[column].str.contains("|", regex=False).any():
            print(f"WARNING: {column} is not constant within some samples")
    print(metadata)

    count_data = pandas.DataFrame(numpy.round(counts).astype(numpy.int64), index=sample_list, columns=var_names)
    count_data = count_data.loc[:, count_data.sum(axis=0) > 0]
    dds = pydeseq2.dds.DeseqDataSet(counts=count_data, metadata=metadata.assign(Intercept=1)[["Intercept"]], design="~1", quiet=True)
    dds.vst(use_design=False)
    vst_data = pandas.DataFrame(numpy.asarray(dds.layers["vst_counts"]), index=sample_list, columns=count_data.columns)
    metadata["Size factor"] = numpy.asarray(dds.obs["size_factors"] if ("size_factors" in dds.obs.columns) else dds.obsm["size_factors"]).ravel()

    variance = vst_data.var(axis=0, ddof=1)
    top_genes = variance.sort_values(ascending=False).index[:args.top_genes]
    pca = sklearn.decomposition.PCA(n_components=min(len(sample_list), len(top_genes)))
    coordinates = pca.fit_transform(vst_data[top_genes].to_numpy())
    pca_data = pandas.DataFrame(coordinates[:, :2], index=sample_list, columns=["PC1", "PC2"]).join(metadata)
    print("Variance ratio PC1, PC2:", pca.explained_variance_ratio_[:2].round(4).tolist())
    print(pca_data)

    output_adata = anndata.AnnData(X=vst_data.to_numpy(dtype=numpy.float32), obs=metadata, var=pandas.DataFrame({"highly_variable": vst_data.columns.isin(top_genes), "variance": variance.to_numpy()}, index=vst_data.columns))
    output_adata.layers["counts"] = count_data.to_numpy(dtype=numpy.float32)
    output_adata.obsm[step00.pca_key] = coordinates.astype(numpy.float64)
    output_adata.uns["pca"] = {"variance_ratio": pca.explained_variance_ratio_, "top_genes": args.top_genes, "group_column": args.group_column}
    print(output_adata)
    output_adata.write_h5ad(args.output, **step00.anndata_compressions)
