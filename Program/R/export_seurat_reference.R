args <- commandArgs(trailingOnly = TRUE)
input_rds <- args[1]
output_dir <- args[2]

library(SeuratObject)
library(Matrix)

obj <- readRDS(input_rds)
print(obj)

assay <- DefaultAssay(obj)
cat("Assays:", paste(Assays(obj), collapse = ", "), "/ default:", assay, "\n")
assay_object <- obj[[assay]]
cat("Assay class:", class(assay_object)[1], "\n")
cat("Layers:", paste(Layers(assay_object), collapse = ", "), "\n")

if (inherits(assay_object, "Assay5") && any(grepl("^counts\\.", Layers(assay_object)))) {
    assay_object <- JoinLayers(assay_object)
    cat("Layers after JoinLayers:", paste(Layers(assay_object), collapse = ", "), "\n")
}

if (!("counts" %in% Layers(assay_object))) stop("No counts layer in assay ", assay)

counts <- LayerData(assay_object, layer = "counts")
if (!inherits(counts, "dgCMatrix")) counts <- as(counts, "dgCMatrix")
cat("Counts:", nrow(counts), "genes x", ncol(counts), "cells; nnz", length(counts@x), "\n")
cat("Integer-valued:", all(counts@x == round(counts@x)), "\n")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
write_binary <- function(values, file_name) {
    con <- file(file.path(output_dir, file_name), "wb")
    writeBin(values, con, size = 4, endian = "little")
    close(con)
}
write_binary(as.integer(counts@i), "counts_i.bin")
write_binary(as.integer(counts@p), "counts_p.bin")
write_binary(as.numeric(counts@x), "counts_x.bin")
writeLines(c(as.character(nrow(counts)), as.character(ncol(counts))), file.path(output_dir, "shape.txt"))
writeLines(rownames(counts), file.path(output_dir, "genes.txt"))
writeLines(colnames(counts), file.path(output_dir, "cells.txt"))

stopifnot(!anyDuplicated(colnames(counts)), all(colnames(counts) %in% rownames(obj@meta.data)))
cat("Meta order identical:", identical(colnames(counts), rownames(obj@meta.data)), "\n")
meta <- obj@meta.data[colnames(counts), , drop = FALSE]
n_count <- Matrix::colSums(counts)
for (column in grep("^nCount_", colnames(meta), value = TRUE)) {
    cat(column, "matches colSums(counts):", mean(abs(meta[[column]] - n_count) < 1e-6), "\n")
}
meta <- cbind(cell_id = rownames(meta), meta)
utils::write.csv(meta, gzfile(file.path(output_dir, "obs.csv.gz")), row.names = FALSE)

for (column in colnames(obj@meta.data)) {
    values <- obj@meta.data[[column]]
    if ((is.character(values) || is.factor(values)) && (length(unique(values)) <= 100)) {
        cat("==", column, "==\n")
        print(sort(table(values, useNA = "ifany"), decreasing = TRUE))
    }
}
