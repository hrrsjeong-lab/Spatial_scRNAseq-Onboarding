# Variables
SHELL = /bin/bash -O globstar
CPUS = 10
MEMS = 100
MEM_PER_CPU = $(shell echo '$(MEMS)/$(CPUS)' | bc)

# Programs
SLURM = srun --chdir=$(abspath .) --cpus-per-task=$(CPUS) --export=ALL --job-name="scRNA_$(@F)" --mem=$(MEMS)G --mail-user="jaewoong@gist.ac.kr" --mail-type=FAIL
GPUS = --gpus=1
PYTHON = /home/Live/jaewoong/Spatial_scRNAseq-Onboarding/.venv/bin/python3 -B
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))
product = $(foreach x,$1,$(addprefix $x.,$2))

# References
HUMAN_FASTA = /home/Share/References/chm13/chm13v2.0.maskedY.fasta
MATRIX_XLSX = Metadata/GSE311609_series_matrix.xlsx
CANCERS = Test Lung_5K# Lung Breast
DEGS = 01_Seurat 02_Cell_ranger
PROJECTIONS = 01_umap 02_tsne 03_phate 04_trimap
CLUSTERINGS = 01_Leiden 02_Louvain
CELLTYPES = 01_Human_Lung_Atlas 02_Cells_Lung_Airway 03_Immune_All_High 04_Immune_All_Low 05_Bilous2026_Lung5K_Level2

# Options
LOGGING = 1>$(abspath $@).log 2>&1
LOGGING_DIR = 1>$(abspath $(@D)).log 2>&1
LOGGING_TEE = 2>&1 | tee $(abspath $@).log

all: test optimal step01 step02 step03 step04 step05 step06 step07 step08 step09
.PHONY: all

clean:
.PHONY: clean

optimal: step01-optimal step02-optimal step03-optimal step04-optimal step05-optimal step06-optimal step07-optimal step08-optimal step09-optimal
.PHONY: optimal

test: step01-test step02-test step03-test step04-test step05-test step06-test step07-test step08-test step09-test
.PHONY: test

# Step 01 (Merge input)
Output/01_Merging:
	mkdir -p $@

Output/01_Merging/%.hdf5: Python/merge_input.py Metadata/%.xlsx | Output/01_Merging
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step01-test: Output/01_Merging/Test.hdf5
step01-optimal: Output/01_Merging/Lung_5K.hdf5
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
step02-optimal: Output/02_QC_expressed_genes/Lung_5K.cell.tsv.gz Output/02_QC_expressed_genes/Lung_5K.gene.tsv.gz Output/02_QC_expressed_genes/Figures/Lung_5K.zip
step02: $(addprefix Output/02_QC_expressed_genes/,$(addsuffix .cell.tsv.gz,$(CANCERS))) $(addprefix Output/02_QC_expressed_genes/,$(addsuffix .gene.tsv.gz,$(CANCERS))) $(addprefix Output/02_QC_expressed_genes/Figures/,$(addsuffix .zip,$(CANCERS)))
.PHONY: step02-test step02-optimal step02

# Step 03 (QC filtered data)
Output/03_QC_filtering:
	mkdir -p $@

Output/03_QC_filtering/%.hdf5: Python/qc_filtering.py Output/01_Merging/%.hdf5 Output/02_QC_expressed_genes/%.cell.tsv.gz Output/02_QC_expressed_genes/%.gene.tsv.gz | Output/03_QC_filtering
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

step03-test: Output/03_QC_filtering/Test.hdf5
step03-optimal: Output/03_QC_filtering/Lung_5K.hdf5
step03: $(addprefix Output/03_QC_filtering/,$(addsuffix .hdf5,$(CANCERS)))
.PHONY: step03-test step03-optimal step03

# Step 04 (Highly variable genes)
Output/04_Highly_variable_genes:
	mkdir -p $@

Output/04_Highly_variable_genes/Figures:
	mkdir -p $@

Output/04_Highly_variable_genes/%.01_Seurat.hdf5: Python/highly_variable_genes.py Output/03_QC_filtering/%.hdf5 | Output/04_Highly_variable_genes
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --flavor "seurat" $(LOGGING)

Output/04_Highly_variable_genes/%.02_Cell_ranger.hdf5: Python/highly_variable_genes.py Output/03_QC_filtering/%.hdf5 | Output/04_Highly_variable_genes
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --flavor "cell_ranger" $(LOGGING)

Output/04_Highly_variable_genes/Figures/%.zip: Python/draw_variable_genes.py Output/04_Highly_variable_genes/%.hdf5 | Output/04_Highly_variable_genes/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step04-test: Output/04_Highly_variable_genes/Test.01_Seurat.hdf5 Output/04_Highly_variable_genes/Figures/Test.01_Seurat.zip
step04-optimal: Output/04_Highly_variable_genes/Lung_5K.01_Seurat.hdf5 Output/04_Highly_variable_genes/Figures/Lung_5K.01_Seurat.zip
step04: $(addprefix Output/04_Highly_variable_genes/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(DEGS)))) $(addprefix Output/04_Highly_variable_genes/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(DEGS))))
.PHONY: step04-test step04-optimal step04

# Step 05 (PCA)
Output/05_PCA:
	mkdir -p $@

Output/05_PCA/Figures:
	mkdir -p $@

Output/05_PCA/%.hdf5: Python/calculate_pca.py Output/04_Highly_variable_genes/%.hdf5 | Output/05_PCA
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/05_PCA/Figures/%.zip: Python/draw_pca.py Output/05_PCA/%.hdf5 | Output/05_PCA/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step05-test: Output/05_PCA/Test.01_Seurat.hdf5 Output/05_PCA/Figures/Test.01_Seurat.zip
step05-optimal: Output/05_PCA/Lung_5K.01_Seurat.hdf5 Output/05_PCA/Figures/Lung_5K.01_Seurat.zip
step05: $(addprefix Output/05_PCA/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(DEGS)))) $(addprefix Output/05_PCA/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(DEGS))))
.PHONY: step05-test step05-optimal step05

# Step 06 (Dimensional reduction)
Output/06_Dimensional_reduction:
	mkdir -p $@

Output/06_Dimensional_reduction/%.01_umap.hdf5: Python/calculate_umap.py Output/05_PCA/%.hdf5 | Output/06_Dimensional_reduction
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/06_Dimensional_reduction/%.02_tsne.hdf5: Python/calculate_tsne.py Output/05_PCA/%.hdf5 | Output/06_Dimensional_reduction
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/06_Dimensional_reduction/%.03_phate.hdf5: Python/calculate_phate.py Output/05_PCA/%.hdf5 | Output/06_Dimensional_reduction
	$(SLURM) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/06_Dimensional_reduction/%.04_trimap.hdf5: Python/calculate_trimap.py Output/05_PCA/%.hdf5 | Output/06_Dimensional_reduction
	$(SLURM) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

step06-test: Output/06_Dimensional_reduction/Test.01_Seurat.01_umap.hdf5
step06-optimal: Output/06_Dimensional_reduction/Lung_5K.01_Seurat.01_umap.hdf5
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

Output/07_Clustering/Figures/%.zip: Python/draw_clustering.py Output/07_Clustering/%.hdf5 | Output/07_Clustering/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step07-test: Output/07_Clustering/Test.01_Seurat.01_umap.01_Leiden.hdf5 Output/07_Clustering/Figures/Test.01_Seurat.01_umap.01_Leiden.zip
step07-optimal: Output/07_Clustering/Lung_5K.01_Seurat.01_umap.01_Leiden.hdf5 Output/07_Clustering/Figures/Lung_5K.01_Seurat.01_umap.01_Leiden.zip
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

Output/08_Marker_genes/References/Bilous2026_Lung5K_Level2.h5ad: | Output/08_Marker_genes/References
	wget "https://datasets.cellxgene.cziscience.com/4ca422ae-e4ad-4884-9073-7c3799cd9143.h5ad" -o $(abspath $@) $(LOGGING)

Output/08_Marker_genes/References/05_Bilous2026_Lung5K_Level2.pkl: Python/train_celltype_reference.py Output/08_Marker_genes/References/Bilous2026_Lung5K_Level2.h5ad Output/07_Clustering/Lung_5K.01_Seurat.01_umap.01_Leiden.hdf5
	$(SLURM) $(PYTHON) $(abspath $^ $@) --cpus $(CPUS) $(LOGGING)

Output/08_Marker_genes/%.01_Human_Lung_Atlas.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/01_Human_Lung_Atlas.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.02_Cells_Lung_Airway.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/02_Cells_Lung_Airway.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.03_Immune_All_High.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/03_Immune_All_High.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.04_Immune_All_Low.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/04_Immune_All_Low.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/%.05_Bilous2026_Lung5K_Level2.hdf5: Python/select_marker_genes.py Output/07_Clustering/%.hdf5 Output/08_Marker_genes/References/05_Bilous2026_Lung5K_Level2.pkl
	$(SLURM) $(GPUS) $(PYTHON) $(abspath $^ $@) $(LOGGING)

Output/08_Marker_genes/Figures/%.zip: Python/draw_marker_genes.py Output/08_Marker_genes/%.hdf5 | Output/08_Marker_genes/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step08-test: Output/08_Marker_genes/References/05_Bilous2026_Lung5K_Level2.pkl Output/08_Marker_genes/Test.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.hdf5 Output/08_Marker_genes/Figures/Test.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.zip
step08-optimal: Output/08_Marker_genes/References/05_Bilous2026_Lung5K_Level2.pkl Output/08_Marker_genes/Lung_5K.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.hdf5 Output/08_Marker_genes/Figures/Lung_5K.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.zip
step08: $(addprefix Output/08_Marker_genes/References/,$(addsuffix .pkl,$(CELLTYPES))) $(addprefix Output/08_Marker_genes/,$(addsuffix .hdf5,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(call product,$(CLUSTERINGS),$(CELLTYPES))))))) $(addprefix Output/08_Marker_genes/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(call product,$(CLUSTERINGS),$(CELLTYPES)))))))
.PHONY: step08-test step08-optimal step08

# Step 09 (Xenium)
Output/09_Xenium/Figures:
	mkdir -p $@

Output/09_Xenium/Figures/%.zip: Python/draw_xenium_coordinates.py Output/08_Marker_genes/%.hdf5 | Output/09_Xenium/Figures
	$(SLURM) $(PYTHON) $(abspath $^ $@) $(LOGGING)

step09-test: Output/09_Xenium/Figures/Test.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.zip
step09-optimal: Output/09_Xenium/Figures/Lung_5K.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.zip
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

step10-test: Output/10_Neighborhood/Test.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.hdf5 Output/10_Neighborhood/Figures/Test.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.zip
step10-optimal: Output/10_Neighborhood/Lung_5K.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.hdf5 Output/10_Neighborhood/Figures/Lung_5K.01_Seurat.01_umap.01_Leiden.05_Bilous2026_Lung5K_Level2.zip
step10: $(addprefix Output/10_Neighborhood/Figures/,$(addsuffix .zip,$(call product,$(CANCERS),$(call product,$(DEGS),$(call product,$(PROJECTIONS),$(call product,$(CLUSTERINGS),$(CELLTYPES)))))))
.PHONY: step10-test step10-optimal step10
