args <- commandArgs(trailingOnly = TRUE)
input_rds <- args[1]
output_dir <- args[2]

options <- list(assay = NA_character_, label = NA_character_, "max-per-label" = NA_character_, seed = "42")
for (argument in args[-(1:2)]) {
    key_value <- regmatches(argument, regexec("^--([^=]+)=(.*)$", argument))[[1]]
    if ((length(key_value) != 3) || !(key_value[2] %in% names(options))) stop("Unknown argument: ", argument)
    options[[key_value[2]]] <- key_value[3]
}
max_per_label <- as.integer(options[["max-per-label"]])

library(SeuratObject)
library(Matrix)

obj <- readRDS(input_rds)
print(obj)

assay <- if (is.na(options$assay)) DefaultAssay(obj) else options$assay
cat("Assays (cells):", paste0(Assays(obj), " (", vapply(Assays(obj), function(a) as.numeric(ncol(obj[[a]])), numeric(1)), ")", collapse = ", "), "/ default:", DefaultAssay(obj), "/ used:", assay, "\n")
if (!(assay %in% Assays(obj))) stop("Assay ", assay, " not found")
if (inherits(obj[[assay]], "SCTAssay")) warning("SCT assay counts are corrected counts, not raw counts")
if (is.na(options$assay) && (ncol(obj[[assay]]) < ncol(obj))) stop("Default assay ", assay, " holds ", ncol(obj[[assay]]), " of ", ncol(obj), " cells; pass --assay= with the full assay")

cells <- colnames(obj[[assay]])
if (!is.na(options$label)) {
    if (!(options$label %in% colnames(obj@meta.data))) {
        print(vapply(obj@meta.data, function(x) as.numeric(length(unique(x))), numeric(1)))
        stop("Label column ", options$label, " not found in meta.data (levels per column printed above)")
    }
    labels <- as.character(obj@meta.data[cells, options$label])
    cells <- cells[!is.na(labels)]
    labels <- labels[!is.na(labels)]
    cat("Cells with a label:", length(cells), "/", ncol(obj[[assay]]), "| labels:", length(unique(labels)), "\n")
    if (!is.na(max_per_label)) {
        set.seed(as.integer(options$seed))
        cells <- unlist(lapply(split(cells, labels), function(x) if (length(x) > max_per_label) sample(x, max_per_label) else x), use.names = FALSE)
        cat("Cells after at most", max_per_label, "per label (seed", options$seed, "):", length(cells), "\n")
    }
}

meta_all <- obj@meta.data
assay_object <- obj[[assay]]
rm(obj)
invisible(gc())
if (length(cells) < ncol(assay_object)) assay_object <- subset(assay_object, cells = cells)
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

stopifnot(!anyDuplicated(colnames(counts)), all(colnames(counts) %in% rownames(meta_all)))
cat("Meta order identical:", identical(colnames(counts), rownames(meta_all)), "\n")
meta <- meta_all[colnames(counts), , drop = FALSE]
n_count <- Matrix::colSums(counts)
for (column in grep("^nCount_", colnames(meta), value = TRUE)) {
    cat(column, "matches colSums(counts):", mean(abs(meta[[column]] - n_count) < 1e-6), "\n")
}
meta <- cbind(cell_id = rownames(meta), meta)
utils::write.csv(meta, gzfile(file.path(output_dir, "obs.csv.gz")), row.names = FALSE)

for (column in setdiff(colnames(meta), "cell_id")) {
    values <- meta[[column]]
    if ((is.character(values) || is.factor(values)) && (length(unique(values)) <= 100)) {
        cat("==", column, "==\n")
        print(sort(table(values, useNA = "ifany"), decreasing = TRUE))
    }
}
