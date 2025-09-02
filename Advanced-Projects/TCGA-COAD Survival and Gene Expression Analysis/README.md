# TCGA-COAD Analysis Pipeline

[![R Version](https://img.shields.io/badge/R-4.0%2B-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/Ajafarnezhad/portfolio)](https://github.com/Ajafarnezhad/portfolio/issues)
[![Stars](https://img.shields.io/github/stars/Ajafarnezhad/portfolio?style=social)](https://github.com/Ajafarnezhad/portfolio)

## Overview

Welcome to the **TCGA-COAD Analysis Pipeline**! This repository provides a comprehensive R script for analyzing Colon Adenocarcinoma (COAD) data from The Cancer Genome Atlas (TCGA). It covers everything from data download and preprocessing to advanced analyses like survival modeling, Weighted Gene Co-expression Network Analysis (WGCNA), and pathway enrichment.

Whether you're a bioinformatician, researcher, or student exploring cancer genomics, this pipeline helps uncover insights into gene expression patterns, survival predictors, and molecular subtypes in COAD.

### Key Features
- **Data Acquisition**: Automatically queries and downloads TCGA-COAD clinical and RNA-Seq data using TCGAbiolinks.
- **Preprocessing**: Normalizes gene expression with DESeq2's VST and filters low-quality samples/genes.
- **Survival Analysis**: Performs univariate/multivariate Cox regression and generates Kaplan-Meier plots with risk stratification.
- **WGCNA**: Identifies co-expression modules correlated with key genes like CDKN2A.
- **Enrichment Analysis**: Conducts GSEA on KEGG, GO, and WikiPathways to reveal biological insights.
- **Visualizations**: Includes PCA, dendrograms, heatmaps, and risk score plots for intuitive data exploration.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/Ajafarnezhad/portfolio.git
   cd portfolio/Advanced-Projects/tcga-coad-analysis
   ```

2. **Install Dependencies**:
   Install required R packages via CRAN/Bioconductor:
   ```r
   install.packages(c("tidyverse", "survminer", "survival", "gridExtra", "pheatmap", "broom"))
   BiocManager::install(c("TCGAbiolinks", "SummarizedExperiment", "DESeq2", "WGCNA", "clusterProfiler", "org.Hs.eg.db"))
   ```
   Note: The `CorLevelPlot` package may require installation from a specific source (e.g., GitHub). Check documentation or contact the author for details.

3. **Run the Script**:
   Open `analysis.R` in RStudio and execute it step-by-step. Ensure internet access for TCGA data download.

## Usage

### Quick Start
Run the entire script:
```r
source("analysis.R")
```

### Customization
- **Subgroup Analysis**: Uncomment filtering lines in Section 5 to focus on specific stages, MSI statuses, etc.
- **Gene Focus**: Modify `cyto_genes` or `covariates` to analyze different gene sets.
- **Output Files**:
  - `univariate.csv`: Univariate Cox results.
  - `multivariate.csv`: Multivariate Cox results.
  - `gsea_res.csv`: GSEA results.
  - RDS files for reusable data objects.

### Example Outputs
- **Survival Plot**: Kaplan-Meier curves stratified by risk groups.
- **Risk Score Dot Plot**: Visualizes patient risk distribution.
- **Module-Trait Heatmap**: Shows WGCNA module correlations with traits like CDKN2A risk score.

![Example Survival Plot](survival_plot.png)
*Note: Generate the survival plot using `ggsave("survival_plot.png", surv_plot$plot)` and upload it to the repository.*

## Contributing

We welcome contributions! Fork the repo, make improvements (e.g., bug fixes, new features), and submit a pull request. Please follow standard R coding conventions and add tests where possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on open-source tools like TCGAbiolinks, DESeq2, and WGCNA.
- Data sourced from [TCGA](https://portal.gdc.cancer.gov/).
- Inspired by cancer genomics research communities.

For questions, open an issue or contact [aiamirjd@gmail.com]. Happy analyzing! 🚀