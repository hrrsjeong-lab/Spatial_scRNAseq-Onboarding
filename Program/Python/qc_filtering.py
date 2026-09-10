import argparse
import numpy
import pandas
import scanpy
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input HDF5 file", type=str)
    parser.add_argument("gene", help="Gene QC TSV(.gz) file", type=str)
    parser.add_argument("cell", help="Gene QC TSV(.gz) file", type=str)
    parser.add_argument("output", help="Output HDF5 file", type=str)
    parser.add_argument("--cpus", help="Number of CPUs to use", type=int, default=1)
    parser.add_argument("--percentile", help="Percentile threshold", type=int, default=5)

    args = parser.parse_args()

    step00.check_suffix(args.input, {".hdf5", ".h5"})
    step00.check_suffix(args.gene, {".tsv", ".tsv.gz"})
    step00.check_suffix(args.cell, {".tsv", ".tsv.gz"})
    step00.check_suffix(args.output, {".hdf5", ".h5"})
    step00.check_cpus(args.cpus)

    if not (0 < args.percentile < 50):
        raise ValueError("Percentile must be (0, 50)!!")

    scanpy.settings.n_jobs = args.cpus

    input_adata = scanpy.read_h5ad(args.input)
    print(input_adata)

    array = numpy.asarray(input_adata.X)
    array = numpy.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array[(array < 0)] = 0.0
    input_adata.X = array
    print(input_adata)

    scanpy.pp.filter_cells(input_adata, min_genes=200)
    print(input_adata)

    scanpy.pp.filter_genes(input_adata, min_cells=3)
    print(input_adata)

    cell_data = pandas.read_csv(args.cell, sep="\t", index_col=0)
    cell_data = cell_data.loc[input_adata.var.index, :]
    print(cell_data)

    for cell_test in tqdm.tqdm(list(cell_data.columns)):
        input_adata.var[cell_test] = cell_data[cell_test].to_numpy()
    print(input_adata)

    gene_data = pandas.read_csv(args.gene, sep="\t", index_col=0)
    gene_data = gene_data.loc[input_adata.obs.index, :]
    print(gene_data)

    for gene_test in tqdm.tqdm(list(gene_data.columns)):
        input_adata.obs[gene_test] = gene_data[gene_test].to_numpy()
    print(input_adata)

    input_adata.layers[step00.log_column] = input_adata.layers["Counts"].astype(float, copy=True)

    scanpy.pp.normalize_total(input_adata, exclude_highly_expressed=True, target_sum=10 ** 4, layer=step00.log_column, inplace=True)
    print(input_adata)

    scanpy.pp.log1p(input_adata, layer=step00.log_column)
    print(input_adata)
    input_adata.write_h5ad(args.output, **step00.anndata_compressions)
