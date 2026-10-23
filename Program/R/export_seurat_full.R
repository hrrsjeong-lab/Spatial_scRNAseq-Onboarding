args <- commandArgs(trailingOnly = TRUE)
input_rds <- args[1]
output_dir <- args[2]

options <- list(assay = NA_character_, layers = "counts,data", reductions = "all")
for (argument in args[-(1:2)]) {
    key_value <- regmatches(argument, regexec("^--([^=]+)=(.*)$", argument))[[1]]
    if ((length(key_value) != 3) || !(key_value[2] %in% names(options))) stop("Unknown argument: ", argument)
    options[[key_value[2]]] <- key_value[3]
}
layer_list <- unique(c("counts", strsplit(options$layers, ",")[[1]]))

suppressPackageStartupMessages({library(SeuratObject); library(Matrix)})

obj <- readRDS(input_rds)
print(obj)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
writeLines(capture.output(print(obj)), file.path(output_dir, "object.txt"))

assay <- if (is.na(options$assay)) DefaultAssay(obj) else options$assay
cat("Assays (cells):", paste0(Assays(obj), " (", vapply(Assays(obj), function(a) as.numeric(ncol(obj[[a]])), numeric(1)), ")", collapse = ", "), "/ default:", DefaultAssay(obj), "/ used:", assay, "\n")
if (!(assay %in% Assays(obj))) stop("Assay ", assay, " not found")
if (inherits(obj[[assay]], "SCTAssay")) warning("SCT assay counts are corrected counts, not raw counts; pass --assay=RNA for raw counts")
if (is.na(options$assay) && (ncol(obj[[assay]]) < ncol(obj))) stop("Default assay ", assay, " holds ", ncol(obj[[assay]]), " of ", ncol(obj), " cells; pass --assay= with the full assay")

# Reductions and cell metadata must be taken before the object is dropped
reduction_list <- if (options$reductions == "all") Reductions(obj) else if (options$reductions %in% c("", "none")) character(0) else strsplit(options$reductions, ",")[[1]]
missing_reductions <- setdiff(reduction_list, Reductions(obj))
if (length(missing_reductions) > 0) stop("Reductions not found: ", paste(missing_reductions, collapse = ", "), " (available: ", paste(Reductions(obj), collapse = ", "), ")")
embedding_list <- lapply(setNames(reduction_list, reduction_list), function(name) Embeddings(obj[[name]]))
meta_all <- obj@meta.data
if (!("active.ident" %in% colnames(meta_all))) meta_all$active.ident <- Idents(obj)[rownames(meta_all)]  # labels set by Idents()/RenameIdents() live outside meta.data
assay_object <- obj[[assay]]
rm(obj)
invisible(gc())

cat("Assay class:", class(assay_object)[1], "| layers:", paste(Layers(assay_object), collapse = ", "), "\n")
if (inherits(assay_object, "Assay5") && any(grepl("^(counts|data)\\.", Layers(assay_object)))) {
    assay_object <- JoinLayers(assay_object)
    cat("Layers after JoinLayers:", paste(Layers(assay_object), collapse = ", "), "\n")
}
if (!("counts" %in% Layers(assay_object))) stop("No counts layer in assay ", assay)

write_binary <- function(values, file_name) {
    con <- file(file.path(output_dir, file_name), "wb")
    writeBin(values, con, size = 4, endian = "little")
    close(con)
}

counts <- LayerData(assay_object, layer = "counts")
if (!inherits(counts, "dgCMatrix")) counts <- as(counts, "dgCMatrix")
cat("counts:", nrow(counts), "genes x", ncol(counts), "cells; nnz", length(counts@x), "| integer-valued:", all(counts@x == round(counts@x)), "\n")
exported_layers <- character(0)
for (layer in layer_list) {
    if (!(layer %in% Layers(assay_object))) {
        warning("Layer ", layer, " not in assay ", assay, "; skipped")
        next
    }
    matrix <- if (layer == "counts") counts else LayerData(assay_object, layer = layer)
    if (!identical(dim(matrix), dim(counts))) stop("Layer ", layer, " has dim ", paste(dim(matrix), collapse = " x "), " but counts has ", paste(dim(counts), collapse = " x "))
    if (!identical(rownames(matrix), rownames(counts)) || !identical(colnames(matrix), colnames(counts))) matrix <- matrix[rownames(counts), colnames(counts)]
    if (!inherits(matrix, "dgCMatrix")) matrix <- as(matrix, "dgCMatrix")
    if ((layer != "counts") && identical(matrix@p, counts@p) && identical(matrix@i, counts@i) && isTRUE(all.equal(matrix@x, counts@x))) {
        warning("Layer ", layer, " equals counts (never normalized); skipped")
        next
    }
    write_binary(as.integer(matrix@i), paste0(layer, "_i.bin"))
    write_binary(as.integer(matrix@p), paste0(layer, "_p.bin"))
    write_binary(as.numeric(matrix@x), paste0(layer, "_x.bin"))
    exported_layers <- c(exported_layers, layer)
    cat("Layer", layer, "exported; nnz", length(matrix@x), "| max", max(matrix@x), "\n")
}
writeLines(exported_layers, file.path(output_dir, "layers.txt"))
writeLines(c(as.character(nrow(counts)), as.character(ncol(counts))), file.path(output_dir, "shape.txt"))
writeLines(rownames(counts), file.path(output_dir, "genes.txt"))
writeLines(colnames(counts), file.path(output_dir, "cells.txt"))

# Gene-level information
variable_features <- tryCatch(VariableFeatures(assay_object), error = function(e) character(0))
writeLines(as.character(variable_features), file.path(output_dir, "variable_features.txt"))
feature_meta <- tryCatch(assay_object[[]], error = function(e) data.frame())
if (ncol(feature_meta) > 0) utils::write.csv(cbind(gene = rownames(feature_meta), feature_meta), gzfile(file.path(output_dir, "var.csv.gz")), row.names = FALSE)
cat("VariableFeatures:", length(variable_features), "| feature meta columns:", ncol(feature_meta), "\n")

# Cell-level information
stopifnot(!anyDuplicated(colnames(counts)), all(colnames(counts) %in% rownames(meta_all)))
meta <- meta_all[colnames(counts), , drop = FALSE]
utils::write.csv(cbind(cell_id = rownames(meta), meta), gzfile(file.path(output_dir, "obs.csv.gz")), row.names = FALSE)
for (column in colnames(meta)) {
    values <- meta[[column]]
    if ((is.character(values) || is.factor(values)) && (length(unique(values)) <= 100)) {
        cat("==", column, "==\n")
        print(sort(table(values, useNA = "ifany"), decreasing = TRUE))
    }
}

for (name in names(embedding_list)) {
    embedding <- embedding_list[[name]]
    cells <- intersect(colnames(counts), rownames(embedding))
    utils::write.csv(cbind(data.frame(cell_id = cells), as.data.frame(embedding[cells, , drop = FALSE])), gzfile(file.path(output_dir, paste0("reduction_", name, ".csv.gz"))), row.names = FALSE)
    cat("Reduction", name, ":", length(cells), "/", ncol(counts), "cells x", ncol(embedding), "dims\n")
}
writeLines(as.character(reduction_list), file.path(output_dir, "reductions.txt"))
