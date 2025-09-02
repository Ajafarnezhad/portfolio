#' @title TCGA-COAD Survival and Gene Expression Analysis
#' 
#' @description
#' This script performs comprehensive analysis on TCGA-COAD (Colon Adenocarcinoma) data, including:
#' - Downloading and processing clinical and gene expression data from TCGA.
#' - Performing differential expression, survival analysis, and multivariate Cox regression.
#' - Conducting Weighted Gene Co-expression Network Analysis (WGCNA) to identify modules correlated with CDKN2A expression.
#' - Enrichment analysis using KEGG, GO, and other pathways.
#' - Generating visualizations for survival curves, risk scores, and heatmaps.
#' 
#' @details
#' The script assumes access to TCGA data via TCGAbiolinks and requires various R packages for analysis.
#' Data is filtered, normalized, and analyzed for associations with survival and molecular subtypes.
#' Results are saved as CSV files and RDS objects for further use.
#' 
#' @author Amirhossein Jafarnezhad
#' @date September 02, 2025
#' @version 1.0.0
#' 
#' @dependencies
#' - TCGAbiolinks
#' - survminer
#' - survival
#' - SummarizedExperiment
#' - tidyverse
#' - DESeq2
#' - WGCNA
#' - CorLevelPlot
#' - gridExtra
#' - clusterProfiler
#' - org.Hs.eg.db
#' 
#' @usage
#' Run the script in an R environment with the required packages installed.
#' Ensure internet access for downloading TCGA data.

# Load required libraries ------------------------------------------------------
library(TCGAbiolinks)      # For querying and downloading TCGA data
library(survminer)         # For survival curve visualization
library(survival)          # For survival analysis functions
library(SummarizedExperiment)  # For handling summarized experiment data
library(tidyverse)         # For data manipulation and visualization
library(DESeq2)            # For differential expression analysis
library(WGCNA)             # For weighted gene co-expression network analysis
library(CorLevelPlot)      # For correlation level plots
library(gridExtra)         # For arranging multiple plots
library(clusterProfiler)   # For pathway enrichment analysis
library(org.Hs.eg.db)      # Human gene annotation database

# Section 1: Clinical Metadata Processing -------------------------------------
# Query and process clinical data for TCGA-COAD patients

# Retrieve clinical metadata from TCGA-COAD
clinic_coad <- GDCquery_clinic("TCGA-COAD")

# Identify columns related to survival status
which(colnames(clinic_coad) %in% c("vital_status", "days_to_last_follow_up", "days_to_death"))

# Display survival-related columns
clinic_coad[, which(colnames(clinic_coad) %in% c("vital_status", "days_to_last_follow_up", "days_to_death"))]

# Tabulate vital status
table(clinic_coad$vital_status)

# Create a binary deceased indicator
clinic_coad$deceased <- ifelse(clinic_coad$vital_status == "Alive", FALSE, TRUE)

# Tabulate deceased status
table(clinic_coad$deceased)

# Calculate overall survival time
clinic_coad$overall_survival <- ifelse(clinic_coad$vital_status == "Alive", 
                                       clinic_coad$days_to_last_follow_up, 
                                       clinic_coad$days_to_death)

# Display overall survival
clinic_coad$overall_survival

# Check length of submitter IDs
nchar(clinic_coad$submitter_id)

# Extract substrings from coldata rownames (assuming coldata is defined later)
substr(rownames(coldata), 1, 12)

# Display rownames and colnames of coldata
rownames(coldata)
colnames(coldata)

# Make rownames unique by substring and set as names
rownames(coldata) <- make.names(substr(rownames(coldata), 1, 12), unique = TRUE)

# Merge clinical data with selected coldata columns
clinic_coad <- cbind(clinic_coad, 
                     coldata[make.names(clinic_coad$submitter_id), 
                             c("paper_MSI_status", "paper_methylation_subtype", "paper_expression_subtype")])

# Tabulate various clinical features
table(clinic_coad$stage)
table(clinic_coad$paper_MSI_status)
table(clinic_coad$paper_methylation_subtype)
table(clinic_coad$paper_expression_subtype)
table(clinic_coad$primary_diagnosis)
table(coldata$primary_diagnosis)

# Simplify stage by removing A/B/C suffixes
clinic_coad$stage <- gsub("[ABC]", "", clinic_coad$ajcc_pathologic_stage)

# Section 2: Gene Expression Data Processing ----------------------------------
# Query, download, and prepare gene expression data from TCGA-COAD

# Query TCGA-COAD for RNA-Seq gene expression data
query_coad_all <- GDCquery(project = "TCGA-COAD", 
                           data.category = "Transcriptome Profiling", 
                           experimental.strategy = "RNA-Seq",
                           workflow.type = "STAR - Counts", 
                           data.type = "Gene Expression Quantification", 
                           sample.type = "Primary Tumor", 
                           access = "open")

# Get query results
output_coad <- getResults(query_coad_all)

# Tabulate sample types
table(output_coad$sample_type)

# Download the queried data
GDCdownload(query_coad_all)

# Prepare the data as a SummarizedExperiment object
tcga_coad_data <- GDCprepare(query_coad_all, summarizedExperiment = TRUE)

# Extract unstranded count matrix
coad_matrix <- assay(tcga_coad_data, "unstranded")

# Extract gene metadata
gene_metadata <- as.data.frame(rowData(tcga_coad_data))

# Extract column data (sample metadata)
coldata <- as.data.frame(colData(tcga_coad_data))

# Create DESeqDataSet from count matrix
dds <- DESeqDataSetFromMatrix(countData = coad_matrix, 
                              colData = coldata, 
                              design = ~1)

# Filter low-count genes (rows with sum < 10)
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep, ]

# Perform variance stabilizing transformation (VST)
vsd <- vst(dds, blind = FALSE)

# Extract VST-transformed matrix
coad_matrix_vst <- assay(vsd)

# Display dimensions of VST matrix
dim(coad_matrix_vst)

# WARNING: The following line references undefined 'dds_norm'. Likely meant 'vsd'.
# coad_matrix_vst <- assay(dds_norm)  # Commenting out to avoid error; replace with 'coad_matrix_vst <- assay(vsd)' if intended.

# Section 3: Import and Process Cytoscape Results ----------------------------
# Load significant genes from Cytoscape export and prepare subset matrix

# Read significant correlations from CSV
sig_cytoscape <- read_csv("sigsurcor.csv")

# Extract gene names
cyto_genes <- sig_cytoscape$name

# Subset gene metadata to Cytoscape genes
gene_metadata_masked <- gene_metadata[gene_metadata$gene_name %in% cyto_genes, 
                                      c("gene_id", "gene_name")]

# Subset VST matrix to masked genes
coad_matrix_vst2 <- coad_matrix_vst[rownames(coad_matrix_vst) %in% rownames(gene_metadata_masked), ]

# Check rowname matching
table(rownames(coad_matrix_vst2) == rownames(gene_metadata_masked))

# Set rownames to gene names
rownames(coad_matrix_vst2) <- gene_metadata_masked$gene_name

# Transpose matrix and add case ID
coad_matrix_vst2 <- as.data.frame(t(coad_matrix_vst2))
coad_matrix_vst2 <- cbind(as.data.frame(gsub("-01.*", "", rownames(coad_matrix_vst2))), 
                          coad_matrix_vst2)
colnames(coad_matrix_vst2)[1] <- "case_id"

# Merge with clinical data
coad_cyto <- merge(coad_matrix_vst2, 
                   clinic_coad[, c("submitter_id", "deceased", "overall_survival", 
                                   "stage", "paper_MSI_status", "paper_methylation_subtype", 
                                   "paper_expression_subtype")], 
                   by.x = 'case_id', by.y = 'submitter_id')

# Relocate survival columns
coad_cyto <- relocate(coad_cyto, c(deceased, overall_survival), .after = case_id)

# Tabulate expression subtypes
table(coad_cyto$paper_expression_subtype)

# Section 4: Cox Regression Analysis ------------------------------------------
# Perform univariate and multivariate Cox proportional hazards regression

# Example univariate Cox for ANLN gene
res.cox <- coxph(Surv(overall_survival, deceased) ~ ANLN, data = coad_cyto)
res.cox

# Prepare formulas for all Cytoscape genes
covariates <- cyto_genes
univ_formulas <- sapply(covariates, 
                        function(x) as.formula(paste('Surv(overall_survival, deceased) ~', x)))

# Fit univariate models
univ_models <- lapply(univ_formulas, function(x) { coxph(x, data = coad_cyto) })

# Extract results from models
univ_results <- lapply(univ_models, function(x) {
  x <- summary(x)
  p.value <- signif(x$wald["pvalue"], digits = 2)
  wald.test <- signif(x$wald["test"], digits = 2)
  beta <- signif(x$coef[1], digits = 2)
  HR <- signif(x$coef[2], digits = 2)
  HR.confint.lower <- signif(x$conf.int[, "lower .95"], 2)
  HR.confint.upper <- signif(x$conf.int[, "upper .95"], 2)
  HR <- paste0(HR, " (", HR.confint.lower, "-", HR.confint.upper, ")")
  res <- c(beta, HR, wald.test, p.value)
  names(res) <- c("beta", "HR (95% CI for HR)", "wald.test", "p.value")
  return(res)
})

# Transpose and convert to data frame
res <- t(as.data.frame(univ_results, check.names = FALSE))
res <- as.data.frame(res)

# Filter significant results (p < 0.05) and sort by beta
res_filtered <- res[res$p.value < 0.05, ]
res_filtered <- res_filtered[order(res_filtered$beta, decreasing = TRUE), ]

# Save univariate results
write.csv(res_filtered, "univariate.csv")

# Multivariate Cox with selected genes
multi_res.cox <- coxph(Surv(overall_survival, deceased) ~ GPX3 + CDKN2A + MAD2L1 + TPX2 + CCNB1 + 
                         DIAPH3 + SULT1B1 + SLC4A4 + AXIN2 + UGT2A3 + CLCA4, 
                       data = coad_cyto)
multi_res.cox

# Simplified multivariate with top genes
multi_res.cox <- coxph(Surv(overall_survival, deceased) ~ GPX3 + CDKN2A, 
                       data = coad_cyto)

# WARNING: 'cox_as_data_frame' is not defined. Using broom::tidy as a substitute.
# multi_res.cox <- cox_as_data_frame(multi_res.cox)
multi_res.cox_df <- broom::tidy(multi_res.cox)
write.csv(multi_res.cox_df, "multivariate.csv")

# Calculate risk score based on CDKN2A coefficient
coad_cyto$risk_score <- multi_res.cox_df$estimate[2] * coad_cyto$CDKN2A

# Relocate risk score column
coad_cyto <- relocate(coad_cyto, risk_score, .after = case_id)

# Section 5: Survival Analysis and Subgroup Filtering -------------------------
# Prepare data for survival plots and filter by subgroups

# Select relevant columns for survival analysis
g <- coad_cyto[, c("case_id", "deceased", "overall_survival", "CDKN2A", "stage",
                   "paper_MSI_status", "paper_methylation_subtype", "paper_expression_subtype")]

# Display dimensions
dim(g)

# Tabulate subgroups
table(g$stage)
table(g$paper_MSI_status)
table(g$paper_methylation_subtype)
table(g$paper_expression_subtype)

# Calculate risk score (using hardcoded coefficient; consider parameterizing)
g <- cbind(g, g$CDKN2A * 0.14157)
colnames(g)[9] <- "Risk_Score"

# Median risk score
median(g$Risk_Score)

# Assign risk groups
g$Risk <- ifelse(g$Risk_Score >= median(g$Risk_Score), "High-Risk", "Low-Risk")

# Example subgroup filtering (uncomment as needed)
# g <- g[g$stage %in% c("Stage I", "Stage II"), ]
# g <- g[g$stage %in% c("Stage III", "Stage IV"), ]

# MSI status filtering examples
# table(g$paper_MSI_status)
# g <- g[g$paper_MSI_status %in% c("MSI-H"), ]
# g <- g[g$paper_MSI_status %in% c("MSI-L"), ]
# g <- g[g$paper_MSI_status %in% c("MSS"), ]
# g <- g[g$paper_MSI_status %in% c("MSI-H", "MSI-L"), ]

# Methylation subtype filtering examples
# table(g$paper_methylation_subtype)
# g <- g[g$paper_methylation_subtype %in% c("CIMP.H"), ]
# g <- g[g$paper_methylation_subtype %in% c("CIMP.L"), ]
# g <- g[g$paper_methylation_subtype %in% c("CIMP.L", "CIMP.H"), ]

# Expression subtype filtering examples
# table(g$paper_expression_subtype)
# g <- g[g$paper_expression_subtype %in% c("CIN"), ]
# g <- g[g$paper_expression_subtype %in% c("Invasive"), ]
# g <- g[g$paper_expression_subtype %in% c("MSI/CIMP"), ]

# Fit survival model by risk group
fit <- survfit(Surv(overall_survival, deceased) ~ Risk, data = g)

# Calculate median survival times
med_time <- surv_median(fit)

# Cox model for hazard ratio
cox_mod <- coxph(Surv(overall_survival, deceased) ~ Risk, data = g)
HR <- (exp(coef(cox_mod))) ** (-1)  # Inverse HR for low vs high?

# Generate survival plot
surv_plot <- ggsurvplot(fit, data = g, pval = TRUE, risk.table = TRUE, 
                        title = "Overall survival subtype CIMP", 
                        conf.int = FALSE, xlab = "Time (Days)",
                        ggtheme = theme_classic2(base_size = 16, base_family = "Arial"),
                        legend.title = "Group", font.family = "Arial")

# Add HR annotation to plot
surv_plot$plot <- surv_plot$plot + annotate("text", x = 0, y = 0.1,
                                            label = paste0("HR = ", round(HR, 2)),
                                            size = 4.5, hjust = -0.2)

# Display plot
surv_plot

# Save dataset as RDS
saveRDS(g, "g.RDS")

# Sort by risk score for visualization
g <- g[order(g$Risk_Score), ]

# Dot plot of risk scores
ggplot(g, aes(x = case_id, y = Risk_Score)) +
  geom_point(size = 3, color = "blue") +
  labs(x = "Patients (Increasing Risk Score)", y = "Risk Score") +
  ggtitle("Dot Plot of Risk Scores for Patients")

# Section 6: WGCNA Analysis ---------------------------------------------------
# Perform Weighted Gene Co-expression Network Analysis focused on CDKN2A

# Prepare count matrix for WGCNA
coad_matrix  # Original count matrix
dim(coad_matrix)
wcoad_matrix_vst <- coad_matrix  # WARNING: Should this be 'coad_matrix_vst' instead?

dim(wcoad_matrix_vst)

# Check for good samples and genes
gsg <- goodSamplesGenes(t(wcoad_matrix_vst))
summary(gsg)
gsg$allOK
table(gsg$goodGenes)
table(gsg$goodSamples)

# Filter to good genes
wcoad_matrix_vst <- wcoad_matrix_vst[gsg$goodGenes == TRUE, ]

# Hierarchical clustering of samples
htree <- hclust(dist(t(wcoad_matrix_vst)), method = "average")
plot(htree)

# PCA for outlier detection
pca <- prcomp(t(wcoad_matrix_vst))
pca.dat <- pca$x
pca.var <- pca$sdev^2
pca.var.percent <- round(pca.var / sum(pca.var) * 100, digits = 2)
pca.dat <- as.data.frame(pca.dat)

# Plot PCA
ggplot(pca.dat, aes(PC1, PC2)) + 
  geom_point() + 
  geom_text(label = rownames(pca.dat)) +
  labs(x = paste0('PC1: ', pca.var.percent[1], ' %'),
       y = paste0('PC2: ', pca.var.percent[2], ' %'))

# Exclude outlier samples
samples_excluded <- c('TCGA-A6-5659-01B-04R-A277-07', 'TCGA-A6-5661-01B-05R-2302-07')
data.subset <- wcoad_matrix_vst[, !(colnames(wcoad_matrix_vst) %in% samples_excluded)]

# Subset phenoData accordingly
phenoData <- coldata[rownames(coldata) %in% colnames(data.subset), ]

# Verify matching
all(colnames(data.subset) == rownames(phenoData))

# Create DESeqDataSet from subset
dds <- DESeqDataSetFromMatrix(countData = data.subset, colData = phenoData, design = ~1)
dim(dds)

# Filter low-count and low-mean genes
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep, ]
keep <- rowMeans(counts(dds)) >= 1
dds <- dds[keep, ]

# VST normalization
dds_norm <- vst(dds)
wgcna_count_norm <- assay(dds_norm) %>% t()
dim(wgcna_count_norm)

# Save normalized counts
saveRDS(wgcna_count_norm, "wgcna_count_norm.RDS")

# Power selection for WGCNA
power_w <- 1:20
sft <- pickSoftThreshold(data = wgcna_count_norm, powerVector = power_w, 
                         networkType = "signed", verbose = 5)

# Save SFT results
saveRDS(sft, "sft.RDS")

# Extract fit indices
sft.data <- sft$fitIndices

# WARNING: 'temp_cor' is not defined. Defining it here for WGCNA compatibility.
temp_cor <- cor
cor <- WGCNA::cor

# Plot scale-free topology fit
a1 <- ggplot(sft.data, aes(Power, SFT.R.sq, label = Power)) +
  geom_point() +
  geom_text(nudge_y = 0.1) +
  geom_hline(yintercept = 0.9, color = 'red') +
  labs(x = 'Power', y = 'Scale free topology model fit, signed') + 
  theme_classic()

# Plot mean connectivity
a2 <- ggplot(sft.data, aes(Power, mean.k., label = Power)) +
  geom_point() +
  geom_text(nudge_y = 0.1) +
  labs(x = 'Power', y = 'Mean Connectivity') + 
  theme_classic()

# Arrange plots
grid.arrange(a1, a2, ncol = 2)

# Alternative plotting with base R
par(mfrow = c(1, 2))
cex1 <- 0.9
plot(sft$fitIndices[, 1], -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2], 
     xlab = "Soft Threshold (power)", ylab = "Scale Free Topology Model Fit, signed R^2", 
     type = "n", main = "Scale independence")
text(sft$fitIndices[, 1], -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2], 
     labels = power_w, cex = cex1, col = "red")
abline(h = 0.90, col = "red")

plot(sft$fitIndices[, 1], sft$fitIndices[, 5], xlab = "Soft Threshold (power)", 
     ylab = "Mean Connectivity", type = "n", main = "Mean connectivity")
text(sft$fitIndices[, 1], sft$fitIndices[, 5], labels = power_w, cex = cex1, col = "red")
abline(h = 0.90, col = "red")

# Ensure numeric data for WGCNA
wgcna_count_norm[] <- sapply(wgcna_count_norm, as.numeric)

# Restore WGCNA correlation function
soft_power <- 13
cor <- WGCNA::cor

dim(wgcna_count_norm)

# Build blockwise modules
bwnet <- blockwiseModules(wgcna_count_norm, TOMType = 'signed', power = soft_power, 
                          mergeCutHeight = 0.25, numericLabels = FALSE, 
                          randomSeed = 1234, verbose = 3, maxBlockSize = 17000)

# Extract module labels and colors
moduleLabels <- bwnet$colors
moduleColors <- labels2colors(bwnet$colors)
MEs <- bwnet$MEs
geneTree <- bwnet$dendrograms[[1]]

# Plot dendrograms with colors
sizeGrWindow(12, 9)
mergedColors <- labels2colors(bwnet$colors)
plotDendroAndColors(bwnet$dendrograms[[1]], mergedColors[bwnet$blockGenes[[1]]], 
                    "Module colors", dendroLabels = FALSE, hang = 0.03, 
                    addGuide = TRUE, guideHang = 0.05)

plotDendroAndColors(bwnet$dendrograms[[2]], mergedColors[bwnet$blockGenes[[2]]], 
                    "Module colors", dendroLabels = FALSE, hang = 0.03, 
                    addGuide = TRUE, guideHang = 0.05)

# Prepare trait data based on CDKN2A
datTraits <- as.data.frame(wgcna_count_norm[, "ENSG00000147889.18"])
datTraits <- cbind(datTraits, datTraits$`wgcna_count_norm[, "ENSG00000147889.18"]` * 0.14157)
colnames(datTraits) <- c("CDKN2A", "Risk Score")
datTraits$Risk <- ifelse(datTraits$`Risk Score` >= median(datTraits$`Risk Score`), "High-Risk", "Low-Risk")
datTraits$bin.Risk <- ifelse(datTraits$`Risk Score` >= median(datTraits$`Risk Score`), 1, 0)
datTraits <- datTraits[, -3]
colnames(datTraits)[3] <- "Risk"

# Module eigengenes and correlations
nGenes <- ncol(wgcna_count_norm)
nSamples <- nrow(wgcna_count_norm)
MEs0 <- moduleEigengenes(wgcna_count_norm, moduleColors)$eigengenes
MEs <- orderMEs(MEs0)
moduleTraitCor <- cor(MEs, datTraits, use = "p")
moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nSamples)

# Prepare text matrix for heatmap
textMatrix <- paste(signif(moduleTraitCor, 2), "\n(", 
                    signif(moduleTraitPvalue, 1), ")", sep = "")
dim(textMatrix) <- dim(moduleTraitCor)

# Plot module-trait heatmap
par(mar = c(6, 8.5, 3, 3))
labeledHeatmap(Matrix = moduleTraitCor, xLabels = names(datTraits), 
               yLabels = names(MEs), ySymbols = names(MEs), 
               colorLabels = FALSE, colors = greenWhiteRed(50), 
               textMatrix = textMatrix, setStdMargins = FALSE, 
               cex.text = 0.8, zlim = c(-1, 1), 
               main = "Module-trait relationships")

# Section 7: Enrichment Analysis ----------------------------------------------
# Perform GSEA and enrichment on WGCNA modules correlated with CDKN2A

# Tabulate module assignments
table(bwnet$colors)

# Load DEGs (assuming file exists)
degs <- read.csv("DEGs.csv")

# Extract genes from blue (positive) and green (negative) modules
blue_pos <- names(bwnet$colors[bwnet$colors == "blue"])
green_neg <- names(bwnet$colors[bwnet$colors == "green"])

# Module eigengenes
module <- bwnet$MEs

# Subset gene metadata for modules
blue_pos <- gene_metadata[gene_metadata$gene_id %in% blue_pos, c(5, 7)]
green_neg <- gene_metadata[gene_metadata$gene_id %in% green_neg, c(5, 7)]

# Check for NAs
table(is.na(blue_pos$gene_id))
table(is.na(green_neg$gene_id))

# Calculate Spearman correlations with CDKN2A
blue_pos <- cbind(blue_pos, cor(wgcna_count_norm[, blue_pos$gene_id], 
                                wgcna_count_norm[, "ENSG00000147889.18"], method = "spearman"))
colnames(blue_pos)[3] <- "CDKN2A_Cor"
blue_pos <- blue_pos[order(blue_pos$CDKN2A_Cor, decreasing = TRUE), ]

green_neg <- cbind(green_neg, cor(wgcna_count_norm[, green_neg$gene_id], 
                                  wgcna_count_norm[, "ENSG00000147889.18"], method = "spearman"))
colnames(green_neg)[3] <- "CDKN2A_Cor"
green_neg <- green_neg[order(green_neg$CDKN2A_Cor, decreasing = TRUE), ]

# Map to Entrez IDs
blue_pos$entrez <- mapIds(org.Hs.eg.db, keys = gsub("\\.(\\d+)", "", blue_pos$gene_id), 
                          keytype = "ENSEMBL", column = "ENTREZID")
blue_pos <- na.omit(blue_pos)

# Prepare ranked gene list for GSEA
gene_rank <- blue_pos$CDKN2A_Cor
names(gene_rank) <- blue_pos$entrez

# GSEA KEGG
kk2 <- gseKEGG(geneList = gene_rank, organism = 'hsa', minGSSize = 3, 
               pvalueCutoff = 0.05, verbose = TRUE, keyType = "ncbi-geneid")

# Additional GSEA examples
a <- gseWP(gene_rank, organism = "Homo sapiens")
View(a@result)

gsePathway(gene_rank, pvalueCutoff = 0.2, pAdjustMethod = "BH", 
           verbose = FALSE, ont = "MF")

a <- gseGO(gene_rank, OrgDb = org.Hs.eg.db, ont = "BP", minGSSize = 10)
View(a@result)

b <- enrichKEGG(names(gene_rank), keyType = "ncbi-geneid")
View(b@result)

# Make readable and save
a <- setReadable(a, 'org.Hs.eg.db', 'ENTREZID')
write.table(a@result, "gsea_res.csv")

# GSEA plot
gseaplot2(a, geneSetID = 1:5)

# Section 8: Additional Visualizations ----------------------------------------
# Risk score, survival time, and heatmap visualizations

# Prepare sorted data
g2 <- g[order(g$case_id), ]  # Fixed: Changed sort() to order()
g2 <- cbind(sort(rownames(wgcna_count_norm)), g)
colnames(g2)[c(1, 6)] <- c("id", "risk_score")
g2$Risk <- ifelse(g2$risk_score >= median(g2$risk_score), "High risk", "Low risk")
g2$Status <- ifelse(g2$deceased == TRUE, "Dead", "Alive")

# Risk score dot plot
ggplot(g2, aes(x = reorder(id, risk_score), y = risk_score, color = Risk)) +
  geom_point(size = 2) +
  theme_bw() +
  theme(panel.background = element_rect(fill = "white"),
        panel.border = element_rect(color = "black", size = 1),
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) +
  geom_vline(xintercept = nrow(g2) / 2, linetype = "dashed", color = "black", size = 0.5) +
  scale_color_manual(values = c("darkgreen", "red")) +
  xlab("Patients (increasing risk score)") +
  ylab("Risk score")

# Survival time dot plot
ggplot(na.omit(g2), aes(x = reorder(id, risk_score), y = overall_survival, color = Status)) +
  geom_point(size = 2) +
  theme_bw() +
  theme(panel.background = element_rect(fill = "white"),
        panel.border = element_rect(color = "black", size = 1),
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) +
  geom_vline(xintercept = nrow(g2) / 2, linetype = "dashed", color = "black", size = 0.5) +
  scale_color_manual(values = c("darkgreen", "red")) +
  xlab("Patients (increasing risk score)") +
  ylab("Survival time (Days)")

# Heatmap of CDKN2A and risk score
library(pheatmap)
g3 <- t(g2[, c("CDKN2A", "risk_score")])
rownames(g3) <- "CDKN2A"
colnames(g3) <- g2$id
annot <- as.data.frame(g2[, "Risk"])
rownames(annot) <- g2$id
colnames(annot) <- "Risk"
color <- colorRampPalette(c("red", "black", "green"))(50)

pheatmap(g3, cluster_rows = FALSE, color = color, scale = "row", 
         annotation_col = annot, show_colnames = FALSE)