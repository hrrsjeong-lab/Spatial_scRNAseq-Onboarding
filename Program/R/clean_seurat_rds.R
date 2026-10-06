args <- commandArgs(trailingOnly = TRUE)
input_rds <- args[1]
output_rds <- args[2]
commands_txt <- args[3]

suppressPackageStartupMessages(library(SeuratObject))
obj <- readRDS(input_rds)

find_unsupported <- function(x, path) {
    if (is.function(x) || is.language(x)) {
        cat(path, ":", class(x)[1], "\n")
        return(invisible(NULL))
    }
    if (isS4(x)) {
        for (s in methods::slotNames(x)) find_unsupported(methods::slot(x, s), paste0(path, "@", s))
    } else if (is.list(x) && (length(x) <= 10000)) {
        nm <- names(x)
        for (i in seq_along(x)) find_unsupported(x[[i]], paste0(path, "[[", if (!is.null(nm) && nzchar(nm[i])) shQuote(nm[i]) else i, "]]"))
    }
    invisible(NULL)
}

cat("== Unsupported objects before cleaning ==\n")
find_unsupported(obj, "obj")

sink(commands_txt)
for (n in names(obj@commands)) {
    cat("###", n, "\n")
    cat(obj@commands[[n]]@call.string, "\n")
    print(obj@commands[[n]]@params)
    cat("\n")
}
sink()

obj@commands <- list()

cat("== Unsupported objects after cleaning ==\n")
find_unsupported(obj, "obj")
saveRDS(obj, output_rds:w)
