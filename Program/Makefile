# Variables
SHELL = /bin/bash -O globstar
.SHELLFLAGS = -o pipefail -c
CPUS = 10
MEMS = 100
MEM_PER_CPU = $(shell echo '$(MEMS)/$(CPUS)' | bc)

# Programs
SLURM = srun --chdir=$(abspath .) --cpus-per-task=$(CPUS) --export=ALL --job-name="$(PROJECT)_$(@F)" --mem=$(MEMS)G --mail-user="jaewoong@gist.ac.kr" --mail-type=FAIL
GPUS = --gpus=1
PYTHON = /home/Live/jaewoong/Spatial_scRNAseq-Onboarding/.venv/bin/python3 -B -u -X faulthandler
R = /usr/bin/env RENV_CONFIG_SYNCHRONIZED_CHECK=FALSE /usr/bin/Rscript
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))
product = $(foreach x,$1,$(addprefix $x.,$2))

# References
HUMAN_FASTA = /home/Share/References/chm13/chm13v2.0.maskedY.fasta
MATRIX_XLSX = Metadata/GSE311609_series_matrix.xlsx

# Constants
PROJECT = Spatial
CANCERS = Test Crohn #Kidney Lung_5K Lung Breast
OPTIMAL = Crohn
DEGS = 01_Seurat 02_Cell_ranger
PROJECTIONS = 01_umap 02_tsne 03_phate 04_trimap
CLUSTERINGS = 01_Leiden 02_Louvain
CELLTYPES = 07_CRC 08_Crohn 09_McGregor2025_scRNA_all 10_McGregor2025_Xenium480

SEURAT_K = 20
SEURAT_PRUNE = 0.0666666667
SEURAT_RESOLUTION = 0.7
TRANSFER_K = 50

MCGREGOR_XENIUM_ASSAY =
MCGREGOR_XENIUM_LABEL = CellAnnotation.Level0.Factor
MCGREGOR_XENIUM_PER_LABEL = 5000

EVALUATE_TRUE_COLUMN = CellAnnotation.Level3
EVALUATE_MARKER_TABLE = Output/08_Marker_genes/References/McGregor2025_Xenium480_Level0_markers.tsv

MCGREGOR_LINEAGE_COLUMN = CellAnnotation.Level0
MCGREGOR_SUBTYPE_COLUMN = CellAnnotation.Level3
MCGREGOR_GROUP_COLUMN = TYPE
MCGREGOR_SUBSET_PATTERN = (?i)fibroblast|^FAS
MCGREGOR_SECTIONS =

# Options
LOGGING = 1>$(abspath $@).log 2>&1
LOGGING_DIR = 1>$(abspath $(@D)).log 2>&1
LOGGING_TEE = 2>&1 | tee $(abspath $@).log

all: step00 test optimal step01 step02 step03 step04 step05 step06 step07 step08 step09 step10 step11 step12
	@date | mail -s "[$(PROJECT)] Making $@ has been finished on $(shell hostname)" "jaewoong@gist.ac.kr"
.PHONY: all

clean:
.PHONY: clean

optimal: step01-optimal step02-optimal step03-optimal step04-optimal step05-optimal step06-optimal step07-optimal step08-optimal step09-optimal step10-optimal step11-optimal step12-optimal
	@date | mail -s "[$(PROJECT)] Making $@ has been finished on $(shell hostname)" "jaewoong@gist.ac.kr"
.PHONY: optimal

test: step01-test step02-test step03-test step04-test step05-test step06-test step07-test step08-test step09-test step10-test step11-test step12-test
	@date | mail -s "[$(PROJECT)] Making $@ has been finished on $(shell hostname)" "jaewoong@gist.ac.kr"
.PHONY: test

# Step 00 (Authors' Seurat parameters and objects)
Output/00_Authors:
	mkdir -p $@

Output/00_Authors/Crohn.parameters.txt: R/dump_seurat_parameters.R /home/Store/PMC12804086/Xenium/xenium5k.seurat.RDS | Output/00_Authors
	$(SLURM) $(R) $(abspath $^ $@) $(LOGGING)

Output/00_Authors/McGregor2025_Fibroblasts/obs.csv.gz: R/export_seurat_full.R /home/Store/PMC12804086/scRNA-seq/scrna_seq_meta_analysis_fibroblasts.RDS | Output/00_Authors
	$(SLURM) $(R) $(abspath $^) $(abspath $(@D)) --assay=RNA_CLEAN --layers=counts,data --reductions=all $(LOGGING_DIR)

Output/00_Authors/McGregor2025_Fibroblasts.hdf5: Python/import_seurat_full.py Output/00_Authors/McGregor2025_Fibroblasts/obs.csv.gz
	$(SLURM) $(PYTHON) $(abspath $^ $@) --celltype-column "CellAnnotation.Level2" $(LOGGING)

Output/00_Authors/McGregor2025_scRNA_all/obs.csv.gz: R/export_seurat_full.R /home/Store/PMC12804086/scRNA-seq/scrna_seq_all_cells-6eGUr1.RDS | Output/00_Authors
	$(SLURM) $(R) $(abspath $^) $(abspath $(@D)) --layers=counts,data --reductions=all $(LOGGING_DIR)

Output/00_Authors/McGregor2025_scRNA_all.hdf5: Python/import_seurat_full.py Output/00_Authors/McGregor2025_scRNA_all/obs.csv.gz
	$(SLURM) $(PYTHON) $(abspath $^ $@) --celltype-column "active.ident" $(LOGGING_TEE)

step00: Output/00_Authors/Crohn.parameters.txt Output/00_Authors/McGregor2025_Fibroblasts.hdf5 Output/00_Authors/McGregor2025_scRNA_all.hdf5
.PHONY: step00

# Step 01 (Merge input)
Output/01_Merging:
	mkdir -p $@

Output/01_Merging/%.hdf5: Python/merge_input.py Metadata/%.xlsx | Output/01_Merging
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/01_Merging/Crohn.hdf5: Python/import_rds.py /home/Store/PMC12804086/Xenium_export | Output/01_Merging
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step01-test: Output/01_Merging/Test.hdf5
step01-optimal: Output/01_Merging/$(OPTIMAL).hdf5
step01: $(addprefix Output/01_Merging/,$(addsuffix .hdf5,$(CANCERS)))
.PHONY: step01-test step01-optimal step01

# Step 02 (Number of expressed genes)
Output/02_QC_expressed_genes:
	mkdir -p $@

Output/02_QC_expressed_genes/Figures:
	mkdir -p $@

Output/02_QC_expressed_genes/%.cell.tsv.gz: Python/qc_expressed_genes.py Output/01_Merging/%.hdf5 | Output/02_QC_expressed_genes
	$(SLURM) $(PYTHON) $(abspath $^ $@) --target "cell" $(LOGGING)

Output/02_QC_expressed_genes/%.gene.tsv.gz: Python/qc_expressed_genes.py Output/01_Merging/%.hdf5 | Output/02_QC_expressed_genes
	$(SLURM) $(PYTHON) $(abspath $^ $@) --target "gene" $(LOGGING)

Output/02_QC_expressed_genes/Figures/%.zip: Python/draw_expressed_genes.py Output/02_QC_expressed_genes/%.cell.tsv.gz Output/02_QC_expressed_genes/%.gene.tsv.gz | Output/02_QC_expressed_genes/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step02-test: Output/02_QC_expressed_genes/Test.cell.tsv.gz Output/02_QC_expressed_genes/Test.gene.tsv.gz Output/02_QC_expressed_genes/Figures/Test.zip
step02-optimal: Output/02_QC_expressed_genes/$(OPTIMAL).cell.tsv.gz Output/02_QC_expressed_genes/$(OPTIMAL).gene.tsv.gz Output/02_QC_expressed_genes/Figures/$(OPTIMAL).zip
step02: $(addprefix Output/02_QC_expressed_genes/,$(addsuffix .cell.tsv.gz,$(CANCERS))) $(addprefix Output/02_QC_expressed_genes/,$(addsuffix .gene.tsv.gz,$(CANCERS))) $(addprefix Output/02_QC_expressed_genes/Figures/,$(addsuffix .zip,$(CANCERS)))
.PHONY: step02-test step02-optimal step02

# Step 03 (QC filtered data)
Output/03_QC_filtering:
	mkdir -p $@

Output/03_QC_filtering/Figures:
	mkdir -p $@

Output/03_QC_filtering/%.hdf5: Python/qc_filtering.py Output/01_Merging/%.hdf5 Output/02_QC_expressed_genes/%.cell.tsv.gz Output/02_QC_expressed_genes/%.gene.tsv.gz | Output/03_QC_filtering
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) --min-counts 0 --reference-column "CellAnnotation.Level3" $(LOGGING)

Output/03_QC_filtering/Figures/%.zip: Python/compare_genes.py Output/01_Merging/%.hdf5 Output/00_Authors/McGregor2025_scRNA_all.hdf5 | Output/03_QC_filtering/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step03-test: Output/03_QC_filtering/Test.hdf5 Output/03_QC_filtering/Figures/Test.zip
step03-optimal: Output/03_QC_filtering/$(OPTIMAL).hdf5 Output/03_QC_filtering/Figures/$(OPTIMAL).zip
step03: $(addprefix Output/03_QC_filtering/,$(addsuffix .hdf5,$(CANCERS)))
.PHONY: step03-test step03-optimal step03

# Step 04 (Highly variable genes)
Output/04_Highly_variable_genes:
	mkdir -p $@

Output/04_Highly_variable_genes/Figures:
	mkdir -p $@

Output/04_Highly_variable_genes/%.01_Seurat.hdf5: Python/highly_variable_genes.py Output/03_QC_filtering/%.hdf5 | Output/04_Highly_variable_genes
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --flavor "seurat_v3_paper" --n-top-genes 2000 --subset-column has_pca $(LOGGING)

Output/04_Highly_variable_genes/%.02_Cell_ranger.hdf5: Python/highly_variable_genes.py Output/03_QC_filtering/%.hdf5 | Output/04_Highly_variable_genes
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --flavor "cell_ranger" $(LOGGING)

Output/04_Highly_variable_genes/McGregor2025_scRNA_all.01_Seurat.hdf5: Python/highly_variable_genes.py Output/00_Authors/McGregor2025_scRNA_all.hdf5 | Output/04_Highly_variable_genes
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --flavor "seurat_v3_paper" --n-top-genes 2000 --subset-column has_pca $(LOGGING)

Output/04_Highly_variable_genes/Figures/%.zip: Python/draw_variable_genes.py Output/04_Highly_variable_genes/%.hdf5 | Output/04_Highly_variable_genes/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/04_Highly_variable_genes/Figures/%.Compare.zip: Python/compare_variable_genes.py Output/04_Highly_variable_genes/%.hdf5 Output/04_Highly_variable_genes/McGregor2025_scRNA_all.01_Seurat.hdf5 | Output/04_Highly_variable_genes/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step04-test: Output/04_Highly_variable_genes/Test.01_Seurat.hdf5 Output/04_Highly_variable_genes/Figures/Test.01_Seurat.zip Output/04_Highly_variable_genes/McGregor2025_scRNA_all.01_Seurat.hdf5 Output/04_Highly_variable_genes/Figures/McGregor2025_scRNA_all.01_Seurat.zip Output/04_Highly_variable_genes/Figures/Test.01_Seurat.Compare.zip
step04-optimal: Output/04_Highly_variable_genes/$(OPTIMAL).01_Seurat.hdf5 Output/04_Highly_variable_genes/Figures/$(OPTIMAL).01_Seurat.zip Output/04_Highly_variable_genes/McGregor2025_scRNA_all.01_Seurat.hdf5 Output/04_Highly_variable_genes/Figures/McGregor2025_scRNA_all.01_Seurat.zip
step04: $(addprefix Output/04_Highly_variable_genes/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(DEGS)))) $(addprefix Output/04_Highly_variable_genes/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(DEGS))))
.PHONY: step04-test step04-optimal step04

# Step 05 (PCA)
Output/05_PCA:
	mkdir -p $@

Output/05_PCA/Figures:
	mkdir -p $@

Output/05_PCA/%.hdf5: Python/calculate_pca.py Output/04_Highly_variable_genes/%.hdf5 | Output/05_PCA
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --pcs 50 --sketch-column 'has_pca' $(LOGGING)

Output/05_PCA/%.Compare.hdf5: Python/calculate_joint_pca.py Output/04_Highly_variable_genes/%.hdf5 Output/04_Highly_variable_genes/McGregor2025_scRNA_all.01_Seurat.hdf5 | Output/05_PCA
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --pcs 50 --sketch-column 'has_pca' $(LOGGING)

Output/05_PCA/Figures/%.zip: Python/draw_pca.py Output/05_PCA/%.hdf5 | Output/05_PCA/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step05-test: Output/05_PCA/Test.01_Seurat.hdf5 Output/05_PCA/Figures/Test.01_Seurat.zip Output/05_PCA/Test.01_Seurat.Compare.hdf5
step05-optimal: Output/05_PCA/$(OPTIMAL).01_Seurat.hdf5 Output/05_PCA/Figures/$(OPTIMAL).01_Seurat.zip Output/05_PCA/$(OPTIMAL).01_Seurat.Compare.hdf5
step05: $(addprefix Output/05_PCA/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(DEGS)))) $(addprefix Output/05_PCA/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(DEGS))))
.PHONY: step05-test step05-optimal step05

# Step 06 (Dimensional reduction)
Output/06_Dimensional_reduction:
	mkdir -p $@

Output/06_Dimensional_reduction/Figures:
	mkdir -p $@

Output/06_Dimensional_reduction/%.01_umap.hdf5: Python/calculate_umap.py Output/05_PCA/%.hdf5 | Output/06_Dimensional_reduction
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/06_Dimensional_reduction/%.02_tsne.hdf5: Python/calculate_tsne.py Output/05_PCA/%.hdf5 | Output/06_Dimensional_reduction
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/06_Dimensional_reduction/%.03_phate.hdf5: Python/calculate_phate.py Output/05_PCA/%.hdf5 | Output/06_Dimensional_reduction
	$(SLURM) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/06_Dimensional_reduction/%.04_trimap.hdf5: Python/calculate_trimap.py Output/05_PCA/%.hdf5 | Output/06_Dimensional_reduction
	$(SLURM) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/06_Dimensional_reduction/Figures/%.zip: Python/compare_projection_omics.py Output/06_Dimensional_reduction/%.hdf5 | Output/06_Dimensional_reduction/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step06-test: Output/06_Dimensional_reduction/Test.01_Seurat.01_umap.hdf5 Output/06_Dimensional_reduction/Test.01_Seurat.Compare.01_umap.hdf5 Output/06_Dimensional_reduction/Figures/Test.01_Seurat.Compare.01_umap.zip
step06-optimal: Output/06_Dimensional_reduction/$(OPTIMAL).01_Seurat.01_umap.hdf5 Output/06_Dimensional_reduction/$(OPTIMAL).01_Seurat.Compare.01_umap.hdf5 Output/06_Dimensional_reduction/Figures/$(OPTIMAL).01_Seurat.Compare.01_umap.zip
step06: $(addprefix Output/06_Dimensional_reduction/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(call product,$(DEGS),$(PROJECTIONS)))))
.PHONY: step06-test step06-optimal step06

# Step 07 (Clustering)
Output/07_Clustering:
	mkdir -p $@

Output/07_Clustering/Figures:
	mkdir -p $@

Output/07_Clustering/%.01_Leiden.hdf5: Python/calculate_leiden.py Output/06_Dimensional_reduction/%.hdf5 | Output/07_Clustering
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/07_Clustering/%.02_Louvain.hdf5: Python/calculate_louvain.py Output/06_Dimensional_reduction/%.hdf5 | Output/07_Clustering
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/07_Clustering/Figures/%.zip: python/draw_clustering.py Output/07_Clustering/%.hdf5 | Output/07_Clustering/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step07-test: Output/07_Clustering/Test.01_Seurat.01_umap.02_Louvain.hdf5 Output/07_Clustering/Figures/Test.01_Seurat.01_umap.02_Louvain.zip
step07-optimal: Output/07_Clustering/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.hdf5 Output/07_Clustering/Figures/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.zip
step07: $(addprefix Output/07_Clustering/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(CLUSTERINGS)))))) $(addprefix Output/07_Clustering/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(CLUSTERINGS))))))
.PHONY: step07-test step07-optimal step07

# Step 08 (Marker genes)
Output/08_Marker_genes/References:
	mkdir -p $@

Output/08_Marker_genes/Figures:
	mkdir -p $@

Output/08_Marker_genes/References/01_Human_Lung_Atlas.pkl: | Output/08_Marker_genes/References
	wget "https://celltypist.cog.sanger.ac.uk/models/Human_Lung_Sikkema/v2/Human_Lung_Atlas.pkl" -O $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/02_Cells_Lung_Airway.pkl: | Output/08_Marker_genes/References
	wget "https://celltypist.cog.sanger.ac.uk/models/Lung_Airway_Madissoon/v4/Cells_Lung_Airway.pkl" -O $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/03_Immune_All_High.pkl: | Output/08_Marker_genes/References
	wget "https://celltypist.cog.sanger.ac.uk/models/Pan_Immune_CellTypist/v2/Immune_All_High.pkl" -O $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/04_Immune_All_Low.pkl: | Output/08_Marker_genes/References
	wget "https://celltypist.cog.sanger.ac.uk/models/Pan_Immune_CellTypist/v2/Immune_All_Low.pkl" -O $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/Bilous2026_Lung5K.h5ad: | Output/08_Marker_genes/References
	wget "https://datasets.cellxgene.cziscience.com/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad" -O $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/05_Bilous2026_Lung5K.pkl: Python/train_celltype_reference.py Output/08_Marker_genes/References/Bilous2026_Lung5K.h5ad Output/07_Clustering/$(OPTIMAL).01_Seurat.01_umap.01_Leiden.hdf5
	$(SLURM) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/08_Marker_genes/References/Lake2025_Kidney.h5ad: | Output/08_Marker_genes/References
	# wget "https://datasets.cellxgene.cziscience.com/06b8d9c2-0bde-48bd-88d9-730993fc95e2.h5ad" -O $(abspath $@) $(LOGGING)
	wget "https://datasets.cellxgene.cziscience.com/19aab2ef-654a-4047-8f60-3529c041851d.h5ad" -O $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/06_Lake2025_Kidney.pkl: Python/train_celltype_reference_2.py Output/08_Marker_genes/References/Lake2025_Kidney.h5ad
	$(SLURM) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/08_Marker_genes/References/07_CRC.pkl: | Output/08_Marker_genes/References
	wget "https://celltypist.cog.sanger.ac.uk/models/Colorectal_Cancer_Lee/v1/Human_Colorectal_Cancer.pkl" -O $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/08_Crohn.pkl: | Output/08_Marker_genes/References
	wget "https://celltypist.cog.sanger.ac.uk/models/Intestinal_Tract_Elmentaite/v1/Cells_Intestinal_Tract.pkl" -O $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/09_McGregor2025_scRNA_all.pkl: Python/train_celltype_reference_3.py Output/00_Authors/McGregor2025_scRNA_all.hdf5 Output/03_QC_filtering/Crohn.hdf5 | Output/08_Marker_genes/References
	$(SLURM) $(PYTHON) $(abspath $^ $@) --label "Celltype" --cpus $(CPUS) $(LOGGING)

Output/08_Marker_genes/References/McGregor2025_Xenium480/obs.csv.gz: R/export_seurat_reference.R /home/Store/PMC12804086/Xenium/xenium_spatial_transcriptomics.RDS | Output/08_Marker_genes/References
	$(SLURM) $(R) $(abspath $^) $(abspath $(@D)) --label=$(MCGREGOR_XENIUM_LABEL) --max-per-label=$(MCGREGOR_XENIUM_PER_LABEL) --seed=42 $(LOGGING_DIR)

Output/08_Marker_genes/References/McGregor2025_Xenium480.h5ad: Python/import_seurat_export.py Output/08_Marker_genes/References/McGregor2025_Xenium480/obs.csv.gz
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/References/10_McGregor2025_Xenium480.pkl: Python/train_celltype_reference_3.py Output/08_Marker_genes/References/McGregor2025_Xenium480.h5ad Output/03_QC_filtering/Crohn.hdf5
	$(SLURM) $(PYTHON) $(abspath $^ $@) --label "$(MCGREGOR_XENIUM_LABEL)" --cpus $(CPUS) $(LOGGING)

Output/08_Marker_genes/%.01_Human_Lung_Atlas.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/01_Human_Lung_Atlas.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.02_Cells_Lung_Airway.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/02_Cells_Lung_Airway.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.03_Immune_All_High.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/03_Immune_All_High.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.04_Immune_All_Low.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/04_Immune_All_Low.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.05_Bilous2026_Lung5K.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/05_Bilous2026_Lung5K.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.07_CRC.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/07_CRC.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.08_Crohn.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/08_Crohn.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.10_McGregor2025_Xenium480.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/10_McGregor2025_Xenium480.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.09_McGregor2025_scRNA_all.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/09_McGregor2025_scRNA_all.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/Figures/%.zip: Python/draw_marker_genes.py Output/08_Marker_genes/%.hdf5 | Output/08_Marker_genes/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step08-test: Output/08_Marker_genes/References/09_McGregor2025_scRNA_all.pkl Output/08_Marker_genes/Test.01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.hdf5 Output/08_Marker_genes/Figures/Test.01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.zip
step08-optimal: Output/08_Marker_genes/References/09_McGregor2025_scRNA_all.pkl Output/08_Marker_genes/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.hdf5 Output/08_Marker_genes/Figures/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.zip
step08: $(addprefix Output/08_Marker_genes/References/,$(addsuffix .pkl,$(CELLTYPES))) $(addprefix Output/08_Marker_genes/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(call product,$(CLUSTERINGS),$(CELLTYPES))))))) $(addprefix Output/08_Marker_genes/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(call product,$(CLUSTERINGS),$(CELLTYPES)))))))
.PHONY: step08-test step08-optimal step08

# Step 09 (Xenium)
Output/09_Xenium/Figures:
	mkdir -p $@

Output/09_Xenium/Figures/%.zip: Python/draw_xenium_coordinates.py Output/08_Marker_genes/%.hdf5 | Output/09_Xenium/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step09-test: Output/09_Xenium/Figures/Test.01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.zip
step09-optimal: Output/09_Xenium/Figures/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.zip
step09: $(addprefix Output/09_Xenium/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(call product,$(CLUSTERINGS),$(CELLTYPES)))))))
.PHONY: step09-test step09-optimal step09

# Step 10 (Neighborhood enrichment)
Output/10_Neighborhood:
	mkdir -p $@

Output/10_Neighborhood/Figures:
	mkdir -p $@

Output/10_Neighborhood/%.hdf5: Python/calculate_nhood_enrichment.py Output/08_Marker_genes/%.hdf5 | Output/10_Neighborhood
	$(SLURM) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) --min-cells 20 --perms 1000 $(LOGGING)

Output/10_Neighborhood/Figures/%.zip: Python/draw_nhood_enrichment.py Output/10_Neighborhood/%.hdf5 | Output/10_Neighborhood/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step10-test: Output/10_Neighborhood/Test.01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.hdf5 Output/10_Neighborhood/Figures/Test.01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.zip
step10-optimal: Output/10_Neighborhood/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.hdf5 Output/10_Neighborhood/Figures/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.zip
step10: $(addprefix Output/10_Neighborhood/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(call product,$(CLUSTERINGS),$(CELLTYPES))))))) $(addprefix Output/10_Neighborhood/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(call product,$(CLUSTERINGS),$(CELLTYPES)))))))
.PHONY: step10-test step10-optimal step10

# Step 11 (Evaluate celltype)
Output/11_Evaluate_celltype:
	mkdir -p $@

Output/11_Evaluate_celltype/Figures:
	mkdir -p $@

Output/08_Marker_genes/References/McGregor2025_Xenium480_Level0_markers.tsv: | Output/08_Marker_genes/References
	wget "https://data.mendeley.com/public-files/datasets/64fkdfcpzb/files/ac2e8611-ee1e-488a-9555-e5361c06940a/file_downloaded" -O $(abspath $@) $(LOGGING)

Output/11_Evaluate_celltype/%.09_McGregor2025_scRNA_all.hdf5: Python/evaluate_marker_genes.py Output/08_Marker_genes/%.09_McGregor2025_scRNA_all.hdf5 Output/08_Marker_genes/References/09_McGregor2025_scRNA_all.pkl Output/08_Marker_genes/References/McGregor2025_Xenium480.h5ad | Output/11_Evaluate_celltype $(EVALUATE_MARKER_TABLE)
	$(SLURM) $(PYTHON) $(abspath $(word 1,$^) $(word 2,$^) $@) --model $(abspath $(word 3,$^)) --reference $(abspath $(word 4,$^)) --true-column "$(EVALUATE_TRUE_COLUMN)" --marker-table $(abspath $(EVALUATE_MARKER_TABLE)) $(LOGGING)

Output/11_Evaluate_celltype/%.10_McGregor2025_Xenium480.hdf5: Python/evaluate_marker_genes.py Output/08_Marker_genes/%.10_McGregor2025_Xenium480.hdf5 Output/08_Marker_genes/References/10_McGregor2025_Xenium480.pkl Output/08_Marker_genes/References/McGregor2025_Xenium480.h5ad | Output/11_Evaluate_celltype $(EVALUATE_MARKER_TABLE)
	$(SLURM) $(PYTHON) $(abspath $(word 1,$^) $(word 2,$^) $@) --model $(abspath $(word 3,$^)) --reference $(abspath $(word 4,$^)) --true-column "$(EVALUATE_TRUE_COLUMN)" --marker-table $(abspath $(EVALUATE_MARKER_TABLE)) $(LOGGING)

Output/11_Evaluate_celltype/Figures/%.zip: Python/draw_evaluate_marker_genes.py Output/11_Evaluate_celltype/%.hdf5 | Output/11_Evaluate_celltype/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/11_Evaluate_celltype/%.09_McGregor2025_scRNA_all.hdf5: Python/evaluate_marker_genes.py Output/08_Marker_genes/%.09_McGregor2025_scRNA_all.hdf5 Output/08_Marker_genes/References/09_McGregor2025_scRNA_all.pkl Output/00_Authors/McGregor2025_scRNA_all.hdf5 | Output/11_Evaluate_celltype $(EVALUATE_MARKER_TABLE)
	$(SLURM) $(PYTHON) $(abspath $(word 1,$^) $(word 2,$^) $@) --model $(abspath $(word 3,$^)) --reference $(abspath $(word 4,$^)) --true-column "CellAnnotation.Level0" --marker-table $(abspath $(EVALUATE_MARKER_TABLE)) $(LOGGING)

step11-test: Output/08_Marker_genes/References/McGregor2025_Xenium480_Level0_markers.tsv Output/11_Evaluate_celltype/Test.01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.hdf5 Output/11_Evaluate_celltype/Figures/Test.01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.zip
step11-optimal: Output/08_Marker_genes/References/McGregor2025_Xenium480_Level0_markers.tsv Output/11_Evaluate_celltype/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.hdf5 Output/11_Evaluate_celltype/Figures/$(OPTIMAL).01_Seurat.01_umap.02_Louvain.09_McGregor2025_scRNA_all.zip
step11: step11-test step11-optimal
.PHONY: step11-test step11-optimal step11

# Step 12 (McGregor 2025 Extended Data Fig. 10)
Output/12_McGregor_figures:
	mkdir -p $@

Output/12_McGregor_figures/Figures:
	mkdir -p $@

Output/12_McGregor_figures/%.pseudobulk.hdf5: Python/calculate_pseudobulk_pca.py Output/03_QC_filtering/%.hdf5 | Output/12_McGregor_figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) --group-column "$(MCGREGOR_GROUP_COLUMN)" --top-genes 1000 $(LOGGING)

Output/12_McGregor_figures/%.fibroblasts.hdf5: Python/calculate_subset_embedding.py Output/03_QC_filtering/%.hdf5 | Output/12_McGregor_figures
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --lineage-column "$(MCGREGOR_LINEAGE_COLUMN)" --pattern '$(MCGREGOR_SUBSET_PATTERN)' --subtype-column "$(MCGREGOR_SUBTYPE_COLUMN)" --group-column "$(MCGREGOR_GROUP_COLUMN)" --n-top-genes 2000 --pcs $(SEURAT_NPCS) --dims $(UMAP_DIMS) --neighbors $(UMAP_NEIGHBORS) --min-dist $(UMAP_MIN_DIST) $(LOGGING)

Output/12_McGregor_figures/Figures/%.ED10.zip: Python/draw_mcgregor_extended_data10.py Output/03_QC_filtering/%.hdf5 Output/12_McGregor_figures/%.pseudobulk.hdf5 Output/12_McGregor_figures/%.fibroblasts.hdf5 | Output/12_McGregor_figures/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) --lineage-column "$(MCGREGOR_LINEAGE_COLUMN)" --group-column "$(MCGREGOR_GROUP_COLUMN)" $(if $(MCGREGOR_SECTIONS),--sections $(MCGREGOR_SECTIONS),) $(LOGGING)

step12-test: Output/12_McGregor_figures/Test.pseudobulk.hdf5 Output/12_McGregor_figures/Test.fibroblasts.hdf5 Output/12_McGregor_figures/Figures/Test.ED10.zip
step12-optimal: Output/12_McGregor_figures/$(OPTIMAL).pseudobulk.hdf5 Output/12_McGregor_figures/$(OPTIMAL).fibroblasts.hdf5 Output/12_McGregor_figures/Figures/$(OPTIMAL).ED10.zip
step12: step12-test step12-optimal
.PHONY: step12-test step12-optimal step12

# Step 13 (Cell type prediction with ingest)
Output/13_Ingest:
	mkdir -p $@

Output/13_Ingest/Figures:
	mkdir -p $@

Output/13_Ingest/%.hdf5: Python/predict_celltype_ingest.py Output/05_PCA/%.Compare.hdf5 Output/00_Authors/McGregor2025_scRNA_all.hdf5 Output/04_Highly_variable_genes/%.hdf5 | Output/13_Ingest
	$(SLURM) $(PYTHON) $(abspath $(word 1,$^) $(word 2,$^) $(word 3,$^) $@) --truth $(abspath $(word 4,$^)) --umap $(LOGGING)

Output/13_Ingest/Figures/%.zip: Python/draw_marker_genes_2.py Output/13_Ingest/%.hdf5 | Output/13_Ingest/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step13-test: Output/13_Ingest/Test.01_Seurat.hdf5 Output/13_Ingest/Figures/Test.01_Seurat.zip
step13-optimal: Output/13_Ingest/$(OPTIMAL).01_Seurat.hdf5 Output/13_Ingest/Figures/$(OPTIMAL).01_Seurat.zip
step13: step13-test step13-optimal
.PHONY: step13-test step13-optimal step13
