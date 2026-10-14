args <- commandArgs(trailingOnly = TRUE)
input_rds <- args[1]
output_txt <- args[2]

suppressPackageStartupMessages(library(SeuratObject))
obj <- readRDS(input_rds)

show_value <- function(x) {
    if (is.null(x)) return("NULL")
    if (is.function(x) || is.language(x)) return(paste0("<", class(x)[1], ">"))
    if (is.list(x)) return(paste0("list(", paste(names(x), collapse = ", "), ")"))
    if (length(x) > 10) return(paste0("<", class(x)[1], " length ", length(x), ": ", paste(head(x, 5), collapse = ", "), ", ..., ", format(tail(x, 1)), ">"))
    paste(format(x), collapse = ", ")
}

sink(output_txt, split = TRUE)

cat("== Object ==\n")
print(obj)
cat("Built with SeuratObject:", as.character(obj@version), "\n")
for (assay in Assays(obj)) {
    variable_features <- tryCatch(VariableFeatures(obj[[assay]]), error = function(e) character(0))
    cat("Assay", assay, "|", class(obj[[assay]])[1], "| features:", nrow(obj[[assay]]), "| layers:", paste(Layers(obj[[assay]]), collapse = ", "), "| VariableFeatures:", length(variable_features), "\n")
}

cat("\n== Commands ==\n")
if (length(obj@commands) == 0) cat("obj@commands is empty\n")
for (name in names(obj@commands)) {
    command <- obj@commands[[name]]
    cat("###", name, "|", format(command@time.stamp), "| assay:", show_value(command@assay.used), "\n")
    cat(command@call.string, sep = "\n")
    for (parameter in names(command@params)) cat("    ", parameter, " = ", show_value(command@params[[parameter]]), "\n", sep = "")
}

cat("\n== Reductions ==\n")
for (name in Reductions(obj)) {
    reduction <- obj[[name]]
    cat("###", name, "| embeddings:", paste(dim(Embeddings(reduction)), collapse = " x "), "| key:", Key(reduction), "| assay:", show_value(DefaultAssay(reduction)), "| stdev:", length(reduction@stdev), "| misc:", paste(names(reduction@misc), collapse = ", "), "\n")
    for (entry in setdiff(names(reduction@misc), "model")) {
        value <- reduction@misc[[entry]]
        if (!is.list(value) && (length(value) <= 10)) cat("    misc$", entry, " = ", show_value(value), "\n", sep = "")
    }
    if (length(reduction@stdev) > 0) cat("    stdev[1:5] = ", show_value(head(reduction@stdev, 5)), "\n", sep = "")
    model <- reduction@misc$model
    if (!is.null(model)) {
        for (parameter in names(model)) {
            value <- model[[parameter]]
            if (parameter == "metric") cat("    model$metric = ", show_value(names(value)), "\n", sep = "")
            else if (!is.list(value) && (length(value) <= 10)) cat("    model$", parameter, " = ", show_value(value), "\n", sep = "")
        }
    }
}

cat("\n== Graphs ==\n")
for (name in Graphs(obj)) {
    graph <- obj[[name]]
    off_diagonal <- graph@x[graph@i != rep(seq_len(ncol(graph)) - 1, diff(graph@p))]
    cat(name, "| dim:", paste(dim(graph), collapse = " x "), "| non-zero per cell:", round(length(graph@x) / nrow(graph), 2), "| min off-diagonal weight:", if (length(off_diagonal)) signif(min(off_diagonal), 6) else NA, "\n")
}
cat("\n== Neighbors ==\n")
print(names(obj@neighbors))

cat("\n== meta.data ==\n")
print(colnames(obj@meta.data))
for (column in grep("res\\.|snn|cluster|leverage|area|control|negative|blank|codeword|nCount|nFeature", colnames(obj@meta.data), value = TRUE, ignore.case = TRUE)) {
    values <- obj@meta.data[[column]]
    if (is.numeric(values)) cat(column, ": numeric |", paste(names(summary(values)), signif(summary(values), 4), sep = "=", collapse = " "), "\n")
    else cat(column, ": ", length(unique(values)), " levels\n", sep = "")
}

cat("\n== tools ==\n")
print(names(obj@tools))

transfer <- obj@tools$TransferSketchLabels
if (!is.null(transfer$full_sketch.nn)) {
    nn <- transfer$full_sketch.nn
    cat("TransferSketchLabels full_sketch.nn | cells:", nrow(Indices(nn)), "| k:", ncol(Indices(nn)), "\n")
    for (entry in names(nn@alg.info)) cat("    alg.info$", entry, " = ", show_value(nn@alg.info[[entry]]), "\n", sep = "")
}

sink()
