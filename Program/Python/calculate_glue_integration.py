import argparse
import functools
import os
import tempfile
import anndata
import networkx
import numpy
import pandas
import scanpy
import scipy.sparse
import scglue
import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing
import tqdm
import rapids_singlecell
import notebook_pdf
import step00
from calculate_joint_pca import merge_omics, omics_mixing, scale, to_numpy


def patch_scglue_for_pandas3() -> None:
    if int(pandas.__version__.split(".")[0]) < 3:
        return
    aggregate_obs = scglue.data.aggregate_obs

    @functools.wraps(aggregate_obs)
    def aggregate_obs_str(adata, by, *args, **kwargs):
        adata.obs[by] = adata.obs[by].astype(str).astype(object)
        return aggregate_obs(adata, by, *args, **kwargs)

    estimate_balancing_weight = scglue.data.estimate_balancing_weight

    @functools.wraps(estimate_balancing_weight)
    def estimate_balancing_weight_writable(*args, **kwargs):
        to_numpy = pandas.Series.to_numpy
        pandas.Series.to_numpy = lambda self, *to_numpy_args, **to_numpy_kwargs: numpy.array(to_numpy(self, *to_numpy_args, **to_numpy_kwargs))
        try:
            return estimate_balancing_weight(*args, **kwargs)
        finally:
            pandas.Series.to_numpy = to_numpy

    scglue.data.aggregate_obs = aggregate_obs_str
    scglue.models.estimate_balancing_weight = estimate_balancing_weight_writable


def modality_pca(data: scipy.sparse.csr_matrix, train_mask: numpy.ndarray, n_comps: int, chunk: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    train_data = data[train_mask].toarray()
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0, ddof=1)
    std[(std == 0)] = 1.0
    pca = sklearn.decomposition.PCA(n_components=n_comps, svd_solver="randomized", random_state=42).fit(scale(train_data, mean, std))
    del train_data
    embedding = numpy.vstack([pca.transform(scale(data[start:start + chunk].toarray(), mean, std)) for start in tqdm.trange(0, data.shape[0], chunk)])
    return embedding.astype(numpy.float64), pca.explained_variance_ratio_


def shared_gene_consistency(embedding: numpy.ndarray, omics: numpy.ndarray, counts: scipy.sparse.csr_matrix, n_meta_list: tuple = (10, 20, 50, 100, 200)) -> pandas.DataFrame:
    rows = list()
    for n_meta in n_meta_list:
        if n_meta > min(int((omics == name).sum()) for name in ["Xenium", "scRNA"]):
            continue
        labels = sklearn.cluster.MiniBatchKMeans(n_clusters=n_meta, random_state=0, n_init=3).fit_predict(embedding)
        profiles, present = dict(), dict()
        for name in tqdm.tqdm(["Xenium", "scRNA"]):
            mask = (omics == name)
            membership = scipy.sparse.csr_matrix((numpy.ones(int(mask.sum())), (labels[mask], numpy.arange(int(mask.sum())))), shape=(n_meta, int(mask.sum())))
            summed = numpy.asarray((membership @ counts[mask]).todense(), dtype=numpy.float64)
            present[name] = numpy.asarray(membership.sum(axis=1)).ravel() > 0
            profiles[name] = numpy.log1p(summed / numpy.maximum(summed.sum(axis=1, keepdims=True), 1.0) * 1e4)
        common = present["Xenium"] & present["scRNA"]
        x_data = profiles["Xenium"][common] - profiles["Xenium"][common].mean(axis=0)
        y_data = profiles["scRNA"][common] - profiles["scRNA"][common].mean(axis=0)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            correlation = (x_data * y_data).sum(axis=0) / numpy.sqrt((x_data ** 2).sum(axis=0) * (y_data ** 2).sum(axis=0))
        rows.append({"Metacells": n_meta, "Common metacells": int(common.sum()), "Genes": int(numpy.isfinite(correlation).sum()), "Mean r": float(numpy.nanmean(correlation)), "Median r": float(numpy.nanmedian(correlation))})
    return pandas.DataFrame(rows)


def add_pipeline_keys(adata: anndata.AnnData) -> pandas.DataFrame:
    train_mask = adata.obs[step00.sketch_column].to_numpy(dtype=bool)
    normalized = sklearn.preprocessing.normalize(numpy.asarray(adata.obsm[step00.glue_column], dtype=numpy.float64))
    rotation = sklearn.decomposition.PCA(n_components=normalized.shape[1], svd_solver="full").fit(normalized[train_mask])
    ordered = rotation.transform(normalized).astype(numpy.float64)

    adata.obsm[step00.pca_key] = ordered
    adata.obsm[step00.harmony_key] = ordered.copy()
    adata.obsm[step00.sketch_key] = numpy.where(train_mask[:, None], ordered, numpy.nan)
    adata.uns[step00.pca_key] = {"variance": rotation.explained_variance_.astype(numpy.float64), "variance_ratio": rotation.explained_variance_ratio_.astype(numpy.float64), "n_features": int(adata.var["highly_variable"].sum()), "fit_cells": int(train_mask.sum()), "source": f"Full-rank PCA rotation of L2-normalised {step00.glue_column}"}
    return pandas.DataFrame({"Ratio": rotation.explained_variance_ratio_, "Cumulative": numpy.cumsum(rotation.explained_variance_ratio_)}, index=[f"PC{i + 1}" for i in range(normalized.shape[1])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("xenium", help="Xenium HDF5 file (step 04: raw counts in X, var highly_variable)", type=str)
    parser.add_argument("scrna", help="scRNA HDF5 file (step 04: raw counts in X, var highly_variable)", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--sketch-column", help="Boolean obs column with the training cells in both datasets (e.g. has_pca); uniform sample per sample if omitted", type=str, default=None)
    parser.add_argument("--sketch-cells", help="Training cells per sample without --sketch-column", type=int, default=5000)
    parser.add_argument("--pcs", help="PCs per modality used as GLUE encoder input (GLUE tutorial: 100)", type=int, default=100)
    parser.add_argument("--latent-dim", help="GLUE latent dimensions (GLUE default: 50)", type=int, default=50)
    parser.add_argument("--max-epochs", help="GLUE max_epochs (0: GLUE automatic value)", type=int, default=0)
    parser.add_argument("--encode-batch", help="Cells per batch when encoding all cells", type=int, default=4096)
    parser.add_argument("--xenium-label", help="Xenium obs column for the figure", type=str, default="CellAnnotation.Level0")
    parser.add_argument("--mixing-cells", help="Random cells for the Omics mixing table", type=int, default=50000)
    parser.add_argument("--chunk", help="Cells per projection chunk", type=int, default=20000)

    args = parser.parse_args()

    step00.check_suffix(args.xenium, {".hdf5", ".h5"})
    step00.check_suffix(args.scrna, {".hdf5", ".h5"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})

    notebook = notebook_pdf.Notebook(title="Integrate Xenium and scRNA with GLUE", script=__file__)

    with notebook.cell("Merge"):
        adata_dict = {"Xenium": scanpy.read_h5ad(args.xenium), "scRNA": scanpy.read_h5ad(args.scrna)}
        label_dict = {"Xenium": args.xenium_label, "scRNA": step00.celltype_column}

        for name, adata in tqdm.tqdm(adata_dict.items(), total=len(adata_dict)):
            print(name, adata)

        label_values = {omics: adata.obs[label_dict[omics]].astype(str).to_numpy() for omics, adata in adata_dict.items() if label_dict[omics] in adata.obs.columns}
        input_adata = merge_omics(adata_dict)
        del adata_dict

        input_adata.X = input_adata.X.astype(numpy.float64)
        input_adata.layers[step00.log_column] = input_adata.layers[step00.log_column].astype(numpy.float64)
        omics_array = input_adata.obs[step00.omics_column].astype(str).to_numpy()
        sample_array = input_adata.obs[step00.sample_column].astype(str).to_numpy()
        feature_mask = input_adata.var["highly_variable"].to_numpy(dtype=bool)
        for omics, values in tqdm.tqdm(label_values.items(), total=len(label_values)):
            column = pandas.Series(numpy.nan, index=input_adata.obs_names, dtype=object)
            column[(omics_array == omics)] = values
            input_adata.obs[label_dict[omics]] = pandas.Categorical(column)
        print(input_adata)

    with notebook.cell("Training cells"):
        rng = numpy.random.default_rng(42)
        if args.sketch_column is not None:
            train_mask = input_adata.obs[args.sketch_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
        else:
            train_mask = numpy.zeros(input_adata.n_obs, dtype=bool)
            for sample in tqdm.tqdm(sorted(set(sample_array))):
                index = numpy.flatnonzero(sample_array == sample)
                train_mask[index if (index.size <= args.sketch_cells) else rng.choice(index, size=args.sketch_cells, replace=False)] = True
        input_adata.obs[step00.sketch_column] = train_mask
        print(pandas.crosstab(input_adata.obs[step00.omics_column], input_adata.obs[step00.sketch_column]))

    with notebook.cell("Encoder input"):
        pca_embedding = numpy.zeros((input_adata.n_obs, args.pcs), dtype=numpy.float64)
        variance_dict = dict()
        log_data = scipy.sparse.csr_matrix(input_adata.layers[step00.log_column])[:, feature_mask]
        for omics in tqdm.tqdm(["Xenium", "scRNA"]):
            rows = numpy.flatnonzero(omics_array == omics)
            pca_embedding[rows], variance_dict[omics] = modality_pca(log_data[rows], train_mask[rows], args.pcs, args.chunk)
            print(omics, "PC1-5 variance ratio:", numpy.round(variance_dict[omics][:5], 4), "| cumulative:", round(float(variance_dict[omics].sum()), 4))
        del log_data

    with notebook.cell("Guidance graph"):
        modality_dict = dict()
        for omics in tqdm.tqdm(["Xenium", "scRNA"]):
            rows = numpy.flatnonzero(omics_array == omics)
            obs = pandas.DataFrame(index=input_adata.obs_names[rows])
            if step00.sample_column:
                obs[step00.sample_column] = input_adata.obs[step00.sample_column].astype(str).to_numpy()[rows]
            modality_dict[omics] = anndata.AnnData(X=scipy.sparse.csr_matrix(input_adata.X[rows]), obs=obs, var=input_adata.var[["highly_variable"]].copy(), obsm={step00.pca_key: pca_embedding[rows]})
            scglue.models.configure_dataset(modality_dict[omics], "NB", use_highly_variable=True, use_rep="X_pca", use_batch=(step00.sample_column or None))

        guidance = networkx.MultiDiGraph()
        guidance.add_edges_from([(gene, gene, {"weight": 1.0, "sign": 1, "type": "self"}) for gene in input_adata.var_names])
        scglue.graph.check_graph(guidance, modality_dict.values())
        guidance_hvf = guidance.subgraph(input_adata.var_names[feature_mask]).copy()
        print("Graph vertices:", guidance.number_of_nodes(), "| HVF vertices:", guidance_hvf.number_of_nodes())

    with notebook.cell("Train GLUE"):
        train_dict = {omics: adata[train_mask[(omics_array == omics)]].copy() for omics, adata in modality_dict.items()}
        for omics, adata in tqdm.tqdm(train_dict.items(), total=len(train_dict)):
            print(omics, "training cells:", adata.n_obs)
        patch_scglue_for_pandas3()

        with tempfile.TemporaryDirectory() as directory:
            fit_kws = {"directory": directory} | ({"max_epochs": args.max_epochs} if (args.max_epochs > 0) else dict())
            glue = scglue.models.fit_SCGLUE(train_dict, guidance_hvf, init_kws={"latent_dim": args.latent_dim, "random_seed": 42}, fit_kws=fit_kws)
        glue.save(f"{os.path.splitext(args.output)[0]}.dill")

    with notebook.cell("Encode"):
        glue_embedding = numpy.zeros((input_adata.n_obs, args.latent_dim), dtype=numpy.float64)

        for omics, adata in tqdm.tqdm(modality_dict.items(), total=len(modality_dict)):
            glue_embedding[omics_array == omics] = glue.encode_data(omics, adata, batch_size=args.encode_batch)

        input_adata.obsm[step00.glue_column] = glue_embedding
        input_adata.obsm["X_pca_glue_input"] = pca_embedding
        feature_embedding = pandas.DataFrame(glue.encode_graph(guidance_hvf), index=glue.vertices)
        input_adata.varm[step00.glue_column] = feature_embedding.reindex(input_adata.var_names).to_numpy(dtype=numpy.float64)
        consistency_data = shared_gene_consistency(glue_embedding[train_mask], omics_array[train_mask], scipy.sparse.csr_matrix(input_adata.X)[train_mask][:, feature_mask])
        print(consistency_data)

    with notebook.cell("Pipeline keys"):
        rotation_data = add_pipeline_keys(input_adata)
        print(rotation_data.reindex(["PC2", "PC10", "PC30"]))

    with notebook.cell("Projection"):
        umap_adata = anndata.AnnData(obs=pandas.DataFrame(index=input_adata.obs_names), obsm={"X_input": sklearn.preprocessing.normalize(glue_embedding)})
        rapids_singlecell.pp.neighbors(umap_adata, use_rep="X_input", algorithm="brute", random_state=42)
        rapids_singlecell.tl.umap(umap_adata, random_state=42)
        input_adata.obsm[step00.projection_key] = to_numpy(umap_adata.obsm["X_umap"]).astype(numpy.float64)
        del umap_adata

    with notebook.cell("Mixing"):
        mixing_data = omics_mixing({"GLUE": sklearn.preprocessing.normalize(glue_embedding)}, omics_array, args.mixing_cells)
        print(mixing_data)

    with notebook.cell("Save"):
        input_adata.uns["glue"] = {key: value for key, value in vars(args).items() if key not in ("xenium", "scrna", "output")} | {"training_cells": pandas.Series(omics_array[train_mask]).value_counts().to_dict(), "features": int(feature_mask.sum()), "pca_variance_ratio": {omics: variance for omics, variance in variance_dict.items()}, "shared_gene_consistency": consistency_data, "mixing": mixing_data, "model": f"{os.path.splitext(args.output)[0]}.dill"}
        print(input_adata)
        input_adata.write_h5ad(args.output, **step00.anndata_compressions)

    notebook.write_pdf(args.output)
