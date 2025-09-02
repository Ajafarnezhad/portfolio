# Lung Cancer Gene Expression and Survival Analysis

[![R Version](https://img.shields.io/badge/R-4.0%2B-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/Ajafarnezhad/portfolio)](https://github.com/Ajafarnezhad/portfolio/issues)
[![Stars](https://img.shields.io/github/stars/Ajafarnezhad/portfolio?style=social)](https://github.com/Ajafarnezhad/portfolio)

## Overview

Welcome to the **Lung Cancer Gene Expression and Survival Analysis Pipeline**! This repository provides a comprehensive R script for analyzing lung cancer data, integrating RNA-seq and microarray gene expression data with clinical covariates. The pipeline performs differential expression analysis, survival modeling, network analysis with WGCNA, and pathway analysis using GSVA, focusing on tumor vs. normal, KRAS mutation status, and gender-specific differences.

This pipeline is designed for bioinformaticians, researchers, and students exploring lung cancer genomics and survival predictors.

### Key Features
- **Data Integration**: Merges RNA-seq and microarray data (e.g., TCGA, CPTAC, GEO datasets) with voom and quantile normalization.
- **Preprocessing**: Handles missing values, encodes categorical variables, and removes batch effects using ComBat.
- **Differential Expression**: Identifies DEGs for Tumor vs. Normal, KRAS mutant vs. wildtype, and gender-specific tumorigenesis using limma.
- **Survival Analysis**: Performs Cox regression with adjustments for stage, age, gender, and KRAS mutation status.
- **Network Analysis**: Uses WGCNA to identify survival-associated gene modules and exports networks to Cytoscape.
- **Pathway Analysis**: Conducts GSVA on oncogenic signatures (MSigDB C6) to explore pathway activity.
- **Visualizations**: Includes volcano plots, Venn diagrams, UMAP plots, survival curves, and heatmaps.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   cd portfolio/Advanced-Projects/lung-cancer-analysis
   ```

2. **Install Dependencies**:
   Install required R packages via CRAN/Bioconductor:
   ```r
   install.packages(c("readxl", "httpgd", "matrixStats", "dplyr", "preprocessCore", "caret", "fastDummies", "survival", "RegParallel", "survminer", "ggplot2", "ggrepel", "ggvenn", "sva", "umap", "ggfortify", "WGCNA", "ppcor", "GSVA", "msigdbr", "ComplexHeatmap", "circlize", "broom"))
   BiocManager::install(c("GEOquery", "org.Hs.eg.db", "limma"))
   ```

3. **Prepare Input Files**:
   - The script uses `file.choose()` to select Excel files for RNA-seq (`s1`, `s2`, `s3`), microarray (`s6`, `s7`), and covariate data (`s8`, `s9`, `s10`, `s13`, `s14`). Replace with explicit file paths or include sample data in the repository.
   - Ensure GEO platforms (`GPL15048`, `GPL96`) are accessible via `GEOquery`.

4. **Run the Script**:
   Open `lung_cancer_analysis.R` in RStudio and execute step-by-step. Ensure internet access for GEO data download.

## Usage

### Quick Start
Run the entire script:
```r
source("lung_cancer_analysis.R")
```

### Customization
- **Input Files**: Replace `file.choose()` with specific file paths for reproducibility.
- **Subgroup Analysis**: Modify `selected_vars_expr` filters to focus on specific covariates (e.g., stage, gender).
- **Output Files**:
  - `survival_analysis_demogr_lung.txt`: Demographic survival analysis results.
  - `DEGs_Tumor_Normal_adjusted_gender_age.txt`: Tumor vs. Normal DEGs.
  - `DEGs_Tumor_mutant.txt`: KRAS mutant vs. wildtype DEGs.
  - `gender_relevant_tumorgeneisis.csv`: Gender-specific DEGs.
  - `male_degs.csv`, `female_degs.csv`: Gender-specific DEGs.
  - `survival_analysis_lung_adjust_kras_gender_age_stage.csv`: Gene-level survival analysis.
  - `survival_analysis_lung_pathway_adjust_gender_age_stage.csv`: Pathway-level survival analysis.
  - `cytoscape_coexp.txt`, `cytoscape_tom.txt`: WGCNA network files.
  - `survival_data_lung_gene_demo.csv`, `survival_data_lung_pathway_demo.csv`: Merged survival data.
  - Plots: `survival_plot.png`, `volcano_tumor_normal.png`, `volcano_kras_mutant.png`, `volcano_gender_tumor.png`, `venn_degs_male_female.png`.

### Example Outputs
- **Survival Plot**: Kaplan-Meier curves for KRAS mutation status.
- **Volcano Plots**: DEGs for Tumor vs. Normal, KRAS mutant vs. wildtype, and gender-specific differences.
- **Venn Diagram**: Overlap of significant DEGs between male and female subgroups.
- **Heatmap**: Pathway activity (GSVA) with sample annotations.

![Example Survival Plot](survival_plot.png)
*Note: Generate plots using `ggsave()` and upload to the repository.*

## Contributing

We welcome contributions! Fork the repo, make improvements (e.g., bug fixes, new features), and submit a pull request. Please follow standard R coding conventions and add tests where possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on open-source tools like limma, WGCNA, GSVA, and GEOquery.
- Data sourced from TCGA, CPTAC, and GEO.
- Inspired by cancer genomics research communities.

For questions, open an issue or contact [aiamirjd@gmail.com]. Happy analyzing! 🚀