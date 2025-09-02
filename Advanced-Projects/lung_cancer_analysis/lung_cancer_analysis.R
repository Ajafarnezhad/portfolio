#' @title Lung Cancer Gene Expression and Survival Analysis
#' 
#' @description
#' This script performs comprehensive analysis on lung cancer RNA-seq and microarray data, including:
#' - Loading and merging gene expression data from RNA-seq and microarray datasets.
#' - Normalizing data using voom and quantile normalization.
#' - Processing demographic and clinical covariates.
#' - Performing differential expression analysis (Tumor vs. Normal, KRAS mutant vs. wildtype, gender-specific).
#' - Conducting survival analysis with Cox regression, adjusting for covariates.
#' - Network analysis using WGCNA to identify survival-associated gene modules.
#' - Pathway analysis using GSVA to explore oncogenic signatures.
#' - Generating visualizations like volcano plots, Venn diagrams, UMAP, and heatmaps.
#' 
#' @details
#' The script assumes input Excel files for gene expression and covariates. Ensure these files are available or specify paths.
#' Results are saved as CSV files for DEGs, survival analysis, and pathway analysis.
#' 
#' @author Amirhossein Jafarnezhad
#' @date September 02, 2025
#' @version 1.0.0
#' 
#' @dependencies
#' - readxl
#' - httpgd
#' - GEOquery
#' - org.Hs.eg.db
#' - matrixStats
#' - limma
#' - dplyr
#' - preprocessCore
#' - caret
#' - fastDummies
#' - survival
#' - RegParallel
#' - survminer
#' - ggplot2
#' - ggrepel
#' - ggvenn
#' - sva
#' - umap
#' - ggfortify
#' - WGCNA
#' - ppcor
#' - GSVA
#' - msigdbr
#' - ComplexHeatmap
#' - circlize
#' - broom
#' 
#' @usage
#' Run the script in an R environment with required packages installed.
#' Ensure input files (Excel, GEO data) are available in the working directory.

# Load required libraries ------------------------------------------------------
library(readxl)
library(httpgd)
library(GEOquery)
hgd()
library(org.Hs.eg.db)
library(matrixStats)
library(limma)
library(dplyr)
library(preprocessCore)
library(caret)
library(fastDummies)
library(survival)
library(RegParallel)
library(survminer)
library(ggplot2)
library(ggrepel)
library(ggvenn)
library(sva)
library(umap)
library(ggfortify)
library(WGCNA)
library(ppcor)
library(GSVA)
library(msigdbr)
library(ComplexHeatmap)
library(circlize)
library(broom)

getwd()

# Loading gene expression data (RNA-seq) addresses
s1 = file.choose()
s2 = file.choose()
s3 = file.choose()

# Loading gene expression data (microarray) addresses
s6 = file.choose()
s7 = file.choose()

# Loading demographic data (RNA-seq) addresses
s8 = file.choose()
s9 = file.choose()
s10 = file.choose()

# Loading demographic data (microarray) addresses
s13 = file.choose()
s14 = file.choose()

# Normalizing s1 dataframe with voom
df_s1 = read_excel(s1)
df_s1 = as.data.frame(df_s1)

View(df_s1[rowSums(is.na(df_s1))>0,])

df_s1 = df_s1[rowSums(is.na(df_s1))==0,]

str(df_s1)

df_s1 = df_s1 %>% group_by(Hugo_Symbol) %>% summarise(across(where(is.numeric),median))
df_s1 = as.data.frame(df_s1)

rownames(df_s1) = df_s1$Hugo_Symbol

df_s1 = df_s1[,-1]

df_s1 = df_s1 %>% mutate_all(as.numeric)

df_s1_hivar = df_s1[rowMads(as.matrix(df_s1))>0,]

nomalized = voom(log2(df_s1_hivar+1),plot = TRUE)

# Preprocessing of microarray data (merging microarray)
gpl_names = c('GPL15048','GPL96')

gpl_tables = lapply(gpl_names,FUN = function(x) Table(getGEO(x,destdir = getwd())))
names(gpl_tables)=gpl_names

merge_gplmicro = function(mic,gpl){
  df_s6 = read_excel(mic)
  df_s6 = as.data.frame(df_s6)
  
  col_namsym = colnames(gpl)[grep('Symbol',colnames(gpl))]
  
  gpl = gpl[,c(colnames(gpl)[1],col_namsym)]
  
  df_s6 = merge(df_s6,gpl, by.x = colnames(df_s6)[1],by.y=colnames(gpl)[1])
  
  df_s6[df_s6==''] = NA
  
  df_s6 = df_s6[!is.na(df_s6[[col_namsym]]),]
  df_s6 = df_s6[,-1]
  
  df_s6 = df_s6[!duplicated(df_s6[[col_namsym]]),]
  
  rownames(df_s6) = df_s6[[col_namsym]]
  
  df_s6 = df_s6[,-which(colnames(df_s6)==col_namsym)]
  
  return(df_s6)
}

df_s6 = merge_gplmicro(s6,gpl_tables[[1]])
df_s7 = merge_gplmicro(s7,gpl_tables[[2]])

df_micro = merge(df_s6,df_s7,by=0)
rownames(df_micro) = df_micro$Row.names
df_micro = df_micro[,-1]

# Merging all RNA-seq expression data
handle_charac = function(x){
  out = read_excel(x)
  out = as.data.frame(out)
  
  df = out %>% mutate_at(vars(-colnames(out)[1]),as.numeric) %>% as.data.frame()
  return(df)
}

file_paths = c(s2,s3)
list_dataf = lapply(file_paths,FUN = function(x) handle_charac(x))

merg_data = list_dataf[[1]]

for (i in 2:length(list_dataf)){
  merg_data = merge(merg_data, list_dataf[[i]], by.x=colnames(merg_data)[1], all=TRUE,
                    by.y=colnames(list_dataf[[i]])[1])
}

merg_data = merge(nomalized$E, merg_data, all = TRUE,
                  by.x = 0, by.y = colnames(merg_data)[1])

merg_data = merg_data %>%
  group_by(Row.names) %>%
  summarise(across(where(is.numeric), median)) %>% as.data.frame()

rownames(merg_data) = merg_data$Row.names
merg_data = merg_data[,-1]

# Merging covariates
cov_path = c(s8,s9,s10,s13,s14)

read_cov = function(x){
  df = read_excel(x)
  df[[1]] = tolower(df[[1]])
  df[[1]] = gsub('_',' ', df[[1]])
  return(df)
}

covariates_df = Reduce(f = function(x,y) merge(x,y,by.x=colnames(x)[1],by.y=colnames(y)[1],all=TRUE),
                       x=lapply(cov_path,FUN = function(x) read_cov(x)))

# Different types of batches in RNA-seq and microarray data
col_tmp <- gsub("^([A-Za-z0-9]+).*", "\\1", colnames(covariates_df))
col_pat <- sub("(.*[A-Za-z]\\d{3})\\d*$", "\\1", col_tmp)
ordered_col_table <- table(col_pat)[unique(col_pat)]

rna_seq_batches_numbers <- c(
  ordered_col_table[grep('TCGA', unique(col_pat))],
  c3l = sum(ordered_col_table[grep('(C3.*|X11LU)', unique(col_pat))])
)

microarray_batches_numbers <- c(
  ordered_col_table[grep('GSM185', unique(col_pat))],
  ordered_col_table[grep('GSM167', unique(col_pat))]
)

microarray_batches <- rep((length(rna_seq_batches_numbers) + 1):(length(rna_seq_batches_numbers) + length(microarray_batches_numbers)), microarray_batches_numbers)
rna_seq_batches <- rep(1:length(rna_seq_batches_numbers), rna_seq_batches_numbers)

covariates_df <- covariates_df[, unlist(lapply(names(ordered_col_table), function(pattern) {
  grep(pattern, colnames(covariates_df))
}))]

rownames(covariates_df) <- covariates_df[[1]]
covariates_df <- covariates_df[,-1]
covariates_df["batch", ] <- c(rna_seq_batches, microarray_batches)

# Summary of missing values in covariates_df
missing_summary <- rowSums(is.na(covariates_df)) * 100 / ncol(covariates_df)
missing_summary <- sort(missing_summary, decreasing = TRUE)

barplot(missing_summary, main = "Missing values in covariates_df",
        xlab = "Percentage missing", ylab = "Variables", las = 2, col = "skyblue", horiz = FALSE)

missing_tissue = colnames(covariates_df)[which(is.na(covariates_df["tissue",]))]
covariates_df["tissue",missing_tissue] = "Normal"

# Select desirable variables
selected_vars <- covariates_df[c("os months", "os status", "age", "gender",
                                "stage", "kras mutation", "race", "organization name", "tissue", "batch"), ]

rowSums(is.na(selected_vars))
rowSums(is.na(selected_vars[,grep("^(TCGA)-.*",colnames(selected_vars))]))
selected_vars <- selected_vars[, colSums(is.na(selected_vars[-7,])) == 0]

# Convert categorical variables to numerical
unique(as.vector(selected_vars["os status", ]))
os_status <- selected_vars["os status", ]
selected_vars["os status", ] <- ifelse(os_status %in% c("Alive", "0:LIVING"), 0, ifelse(is.na(os_status), NA, 1))

unique(as.vector(selected_vars["gender", ]))
selected_vars["gender", ] <- ifelse(selected_vars["gender", ] %in% c("Male","MALE"), 1, 0)

unique(as.vector(selected_vars["kras mutation", ]))
selected_vars["kras mutation", ] <- ifelse(selected_vars["kras mutation", ] %in% c("WT", "NO"), 0, 
                                          ifelse(!is.na(selected_vars["kras mutation", ]), 1, NA))

unique(as.vector(selected_vars["stage", ]))
stage_fun = function(x){
  if (is.na(x)){
    return(NA)
  } else {
    if (grepl("pN0",x) & grepl("pT1",x)){
      return(1)
    } else if (grepl("pN1",x) | (grepl("pT2",x) & grepl("pN0",x))){
      return(2)
    } else if (grepl("pN2",x) | (grepl("pT4",x))){
      return(3)
    } else if (grepl("M1",x)){
      return(4)
    } else if (x %in% c("1", "1A", "1B")){
      return(1)
    } else if (x %in% c("2", "2A", "2B")){
      return(2)
    } else if (x %in% c("3", "3A", "3B")){
      return(3)
    } else if (x == "4"){
      return(4)
    } else {
      return(NA)
    }
  }
}

selected_vars["stage", ] <- sapply(selected_vars["stage", ], stage_fun)
selected_vars <- selected_vars[,!is.na(selected_vars["stage", ])]

unlist(unique(as.vector(selected_vars["tissue", ])))
selected_vars["tissue", ] <- ifelse(selected_vars["tissue", ] == "Normal", 0, 1)

unlist(unique(as.vector(selected_vars['race',])))
selected_vars['race',] <- ifelse(selected_vars['race',] %in% c("Unknown","UNKNOWN"), 2,
                                ifelse(selected_vars['race', ] == "Not Reported", 0, 
                                       ifelse(is.na(selected_vars['race',]), NA, 1)))

unlist(unique(as.vector(selected_vars['organization name',])))
selected_vars['organization name',] <- ifelse(grepl('^National',selected_vars['organization name',]),1,0)

rowSums(is.na(selected_vars))

selected_vars <- selected_vars %>% t() %>% as.data.frame(stringsAsFactors = FALSE) %>% 
  mutate_all(as.numeric)

colSums(is.na(selected_vars))

# Visualizing demographic descriptive statistics
colnames(selected_vars) <- gsub(" ", "_", colnames(selected_vars))
selected_vars_demodes = selected_vars %>%
  mutate_at(vars(-os_months, -age), as.factor)

View(summary(selected_vars_demodes))

barplot(table(selected_vars_demodes$batch), main = "tumorVsnormal_distribution", col='cyan')
hist(selected_vars_demodes$os_months, main = "overall_survival_months_distribution", col='coral')

selected_vars = selected_vars[!is.na(selected_vars['os_months']),]

# One-hot encoding race variable
selected_vars[, "race"] <- as.factor(selected_vars$race)
selected_vars_dummy <- dummy_cols(
  selected_vars, 
  select_columns = "race", 
  remove_first_dummy = FALSE, 
  remove_selected_columns = TRUE
)
rownames(selected_vars_dummy) <- rownames(selected_vars)

# Normalize age
hist(selected_vars_dummy$age)
preproc_ <- preProcess(as.data.frame(selected_vars_dummy$age), method = c("center", "scale"))
scaled_age <- predict(preproc_, as.data.frame(selected_vars_dummy$age))
selected_vars_dummy$age = scaled_age[,1]
hist(selected_vars_dummy$age)

# Survival analysis on demographic variables
selected_vars_dummy_demosurv = selected_vars_dummy
colSums(is.na(selected_vars_dummy_demosurv))
selected_vars_dummy_demosurv = na.omit(selected_vars_dummy_demosurv)
colnames(selected_vars_dummy_demosurv) = sub(' ','_',colnames(selected_vars_dummy_demosurv))

if (length(selected_vars_dummy_demosurv$os_months) != length(selected_vars_dummy_demosurv$os_status)) {
  stop("Error: 'os_months' and 'os_status' columns have different lengths.")
}

res <- RegParallel(
  data = selected_vars_dummy_demosurv,
  formula = "Surv(os_months, os_status) ~ [*] + stage + age",
  FUN = function(formula, data){
    coxph(formula = formula,
          data = data,
          ties = 'breslow',
          singular.ok = TRUE)},
  FUNtype = 'coxph',
  variables = colnames(selected_vars_dummy_demosurv)[-c(1,2,3,5,8)],
  blocksize = 1,
  cores = 8,
  nestedParallel = FALSE,
  conflevel = 95,
  p.adjust = 'BH')

res <- res[!is.na(res$P),]
write.table(res, 'survival_analysis_demogr_lung.txt', quote = FALSE, sep = '\t', row.names = FALSE)

# Plotting survival analysis results
survplotdata <- selected_vars_dummy_demosurv[,c('os_months','os_status','age','gender','batch','stage','kras_mutation','race_0','race_1','race_2')]
survplotdata[survplotdata$batch==3 | survplotdata$batch==4,'batch'] = 2

model = coxph(Surv(os_months, os_status) ~ stage + age + kras_mutation, 
              data = survplotdata[,c('os_months','os_status','age','stage','batch','gender','kras_mutation','race_2')])

summary(model)

test_data = data.frame(age = mean(survplotdata$age), stage = survplotdata$stage[3], kras_mutation = c(0,1))
model_km = survfit(model, newdata = test_data)

hr <- summary(model)$coefficients[, "exp(coef)"][3]
p_value <- summary(model)$coefficients[, "Pr(>|z|)"][3]
annotation_text <- paste0("HR: ", round(hr, 2), "\nP-value: ", format.pval(p_value, digits = 2))

plot <- ggsurvplot(model_km, data = test_data, conf.int = TRUE,
                   palette = c("cyan3", "coral3"),
                   legend.labs = c("KRAS WT", "KRAS MUT"),
                   ggtheme = theme_minimal())
plot$plot <- plot$plot + annotate("text", x = 10, y = 0.2, label = annotation_text, size = 5, hjust = 0)
print(plot)

# Save survival plot
ggsave("survival_plot.png", plot$plot, width = 8, height = 6)

# Preprocessing merg_data
merg_data_smiss = merg_data[rowSums(is.na(merg_data)) / ncol(merg_data) < 0.001, ]
merg_data_smiss = merg_data_smiss[,colSums(is.na(merg_data_smiss)) / nrow(merg_data_smiss) < 0.01]
merg_data_smiss = merg_data_smiss[!duplicated(merg_data_smiss), ]
any(is.na(merg_data_smiss))

# Preprocessing df_micro
any(is.na(df_micro))

# Preprocess covariates_df for gene expression analysis
unlist(unique(as.vector(covariates_df['tissue',])))
covariates_df['kras mutation',covariates_df['tissue',]=='Normal'] = 'WT'

selected_vars_expr <- covariates_df[c("age", "stage", "gender", "tissue", "batch", "kras mutation"), ]
rowSums(is.na(selected_vars_expr))

selected_vars_expr["gender", ] <- ifelse(selected_vars_expr["gender", ] == "Male", 1, 0)
selected_vars_expr["stage", ] <- sapply(selected_vars_expr["stage", ], stage_fun)
selected_vars_expr["tissue", ] <- ifelse(selected_vars_expr["tissue", ] == "Normal", 0, 1)
selected_vars_expr["kras mutation", ] <- ifelse(selected_vars_expr["kras mutation", ] %in% c('NO','WT'), 0, 
                                               ifelse(is.na(selected_vars_expr["kras mutation", ]), NA, 1))

selected_vars_expr <- selected_vars_expr %>% t() %>% as.data.frame(stringsAsFactors = FALSE) %>% 
  mutate_all(as.numeric)

View(summary(selected_vars_expr))

hist(selected_vars_expr$age)
preproc_ <- preProcess(as.data.frame(selected_vars_expr$age), method = c("center", "scale"))
selected_vars_expr$age <- predict(preproc_, as.data.frame(selected_vars_expr$age))[[1]]
hist(selected_vars_expr$age)

# Normalization of RNA-seq based on microarray distribution
boxplot(df_micro[,400:600])
matrix_micro = as.matrix(df_micro)
normalized_micro = normalize.quantiles(log2(matrix_micro+1),keep.names = TRUE)
boxplot(normalized_micro[,400:600])

matrix_rna = as.matrix(merg_data_smiss)
normalized_rna=normalize.quantiles.use.target(matrix_rna, target = as.vector(normalized_micro))
boxplot(normalized_rna[,300:500])

rownames(normalized_rna) = rownames(merg_data_smiss)
colnames(normalized_rna) = colnames(merg_data_smiss)
dev.off()

normalized_rna <- as.data.frame(normalized_rna)
normalized_micro <- as.data.frame(normalized_micro)

# Merging all gene expression dataframes
gene_expr_data = merge(normalized_rna, normalized_micro, by = 0, all = FALSE)
rownames(gene_expr_data) = gene_expr_data$Row.names
gene_expr_data = gene_expr_data[,-1]

gene_expr_data_smiss = gene_expr_data %>% t() %>% as.data.frame(stringsAsFactors = FALSE) %>% 
  mutate_all(as.numeric)
gene_expr_data_smiss = gene_expr_data_smiss[colnames(covariates_df),]

covariates_df["tissue", ] <- ifelse(covariates_df["tissue", ] == "Normal", "Normal", 'Tumor')

# Batch effect removal
data_umap = umap(gene_expr_data_smiss)
colors_batches=factor(as.character(covariates_df['batch',]))
colors_y= factor(as.character(covariates_df['tissue',]))

ggplot(data_umap$layout, aes(x=data_umap$layout[,1],y=data_umap$layout[,2],color=colors_batches))+
  geom_point()+scale_color_discrete()

gene_expr_data_smiss = gene_expr_data_smiss %>% t() %>% as.data.frame(stringsAsFactors = FALSE)
colnames(selected_vars_expr) = sub(' ','_',colnames(selected_vars_expr))

tumor_samples_expr = gene_expr_data_smiss[,selected_vars_expr[,'tissue'] != 0]
tumor_samples_cov = selected_vars_expr[selected_vars_expr[,'tissue'] != 0,]
tumor_samples_cov = tumor_samples_cov[!is.na(tumor_samples_cov$kras_mutation),]
tumor_samples_expr = tumor_samples_expr[,rownames(tumor_samples_cov)]

colSums(is.na(tumor_samples_cov))
mod = model.matrix(~as.factor(tumor_samples_cov$kras_mutation) + as.factor(tumor_samples_cov$gender))
tumor_samples_sbatch = ComBat(dat = tumor_samples_expr, batch = tumor_samples_cov$batch, mod = mod)

alsamp = merge(as.data.frame(tumor_samples_sbatch), 
               gene_expr_data_smiss[,selected_vars_expr[,'tissue'] == 0], by = 0, all = FALSE)
rownames(alsamp) = alsamp$Row.names
alsamp = alsamp[,-1]

selected_vars_expr = selected_vars_expr[colnames(alsamp),]
boxplot(tumor_samples_sbatch[,300:500])

data_umap = umap(t(alsamp))
colors_batches=factor(selected_vars_expr$batch, labels=c('tcga','cptac','gse720_micro'))
colors_y= factor(selected_vars_expr$tissue,labels = c('Normal','Tumor'))
table(colors_y)

ggplot(data_umap$layout, aes(x=data_umap$layout[,1],y=data_umap$layout[,2],color=colors_batches))+
  geom_point()+scale_color_discrete()

# Differentially expressed genes analysis (Tumor vs Normal)
selected_vars_expr1 = selected_vars_expr[!is.na(selected_vars_expr$age),]
alsamp1 = alsamp[,rownames(selected_vars_expr1)]

design = model.matrix(~0 + as.factor(selected_vars_expr1$tissue) + 
                     as.factor(selected_vars_expr1$gender) + selected_vars_expr1$age)
colnames(design) = c('Normal','Tumor','Male','Age')
fit = lmFit(alsamp1,design)
contrast_matrix = makeContrasts(Tumor_Normal = Tumor - Normal, levels = design)
fit2 = contrasts.fit(fit, contrast_matrix)
fit2 = eBayes(fit2)
results_limma = topTable(fit2, coef = 'Tumor_Normal', number = Inf, adjust.method = 'BH')
write.table(results_limma, 'DEGs_Tumor_Normal_adjusted_gender_age.txt', quote = FALSE, sep = '\t', row.names = TRUE)

results_limma$color='Over expressed'
results_limma[results_limma$logFC<(-0.4),'color']='Under expressed'
results_limma[((results_limma$logFC>(-0.4)) & (results_limma$logFC<0.4)) | (results_limma$adj.P.Val>0.001),'color'] = 'Filtered out'

ggplot(results_limma, aes(x=logFC, y=-log10(adj.P.Val),col=color,main='Volcano Plot'))+
  geom_point()+theme_classic()+scale_color_manual(values = c('gray','coral2','cyan3'))+
  geom_hline(yintercept=-log10(0.001),linetype='dashed')+geom_vline(xintercept=-0.4,linetype='dashed')+
  geom_vline(xintercept=0.4,linetype='dashed')+
  labs(x=expression('LogFC'),y=expression('-Log10(Adjusted p-value)'),colour='Genes')+
  geom_text_repel(max.overlaps=Inf,label = ifelse(results_limma$logFC>5.0 | results_limma$logFC<(-4.0), 
                                                  rownames(results_limma),""),box.padding = unit(0.25,'line'),size=3)
ggsave("volcano_tumor_normal.png", width = 8, height = 6)

# Differentially expressed genes (KRAS mutant vs wildtype tumors)
selected_vars_expr2 = selected_vars_expr[selected_vars_expr$tissue == 1,]
alsamp_kras = alsamp[,rownames(selected_vars_expr2)]
design2 = model.matrix(~0 + as.factor(selected_vars_expr2$kras_mutation) + 
                      as.factor(selected_vars_expr2$gender) + selected_vars_expr2$age)
colnames(design2) = c('wild','mut','male','age')
fit = lmFit(alsamp_kras,design2)
contrast_matrix2 = makeContrasts(T_mutVsT_wild = mut - wild, levels = design2)
fit2 = contrasts.fit(fit,contrast_matrix2)
fit2 = eBayes(fit2)
results2_limma = topTable(fit2, coef = 'T_mutVsT_wild', number=Inf, adjust.method = 'BH')
write.table(results2_limma, 'DEGs_Tumor_mutant.txt', quote = FALSE, sep = '\t', row.names = TRUE)

results2_limma$color='Over expressed'
results2_limma[results2_limma$logFC<(-0.1),'color']='Under expressed'
results2_limma[((results2_limma$logFC>(-0.1)) & (results2_limma$logFC<0.1)) | (results2_limma$adj.P.Val>0.001),'color'] = 'Filtered out'

ggplot(results2_limma, aes(x=logFC, y=-log10(adj.P.Val),col=color,main='Volcano Plot'))+
  geom_point()+theme_classic()+scale_color_manual(values = c('gray','coral2','cyan3'))+
  geom_hline(yintercept=-log10(0.001),linetype='dashed')+geom_vline(xintercept=-0.1,linetype='dashed')+
  geom_vline(xintercept=0.1,linetype='dashed')+
  labs(x=expression('LogFC'),y=expression('-Log10(Adjusted p-value)'),colour='Genes')+
  geom_text_repel(max.overlaps=Inf,label = ifelse(results2_limma$logFC>0.5 | results2_limma$logFC<(-0.41), 
                                                  rownames(results2_limma),""),box.padding = unit(0.25,'line'),size=3)
ggsave("volcano_kras_mutant.png", width = 8, height = 6)

# Differentially expressed genes for gender-relevant pathways in tumorigenesis
selected_vars_expr$gentum_int = interaction(selected_vars_expr$tissue,selected_vars_expr$gender)
design3 = model.matrix(~ 0 + gentum_int + selected_vars_expr$age, data = selected_vars_expr)
colnames(design3) = c('normal_female','tumor_female','normal_male','tumor_male','age')
fit = lmFit(alsamp,design3)
contrast_matrix3 = makeContrasts(tumor_mVStumor_fem = (tumor_male - normal_male) - (tumor_female - normal_female), 
                                 levels=design3)
fit2 = contrasts.fit(fit,contrast_matrix3)
fit2 = eBayes(fit2)
results3_limma = topTable(fit2, coef='tumor_mVStumor_fem', number=Inf, adjust.method='fdr')
write.csv(results3_limma, file = "gender_relevant_tumorgeneisis.csv", quote = FALSE, row.names=TRUE)

results3_limma$color='Over expressed'
results3_limma[results3_limma$logFC<(-0.1),'color']='Under expressed'
results3_limma[((results3_limma$logFC>(-0.1)) & (results3_limma$logFC<0.1)) | (results3_limma$P.Val>0.001),'color'] = 'Filtered out'
results3_limma$logFC <- pmax(pmin(results3_limma$logFC, 5), -5)
results3_limma$P.Val <- pmax(results3_limma$P.Val, 1e-10)

ggplot(results3_limma, aes(x=logFC, y=-log10(P.Val),col=color,main='Volcano Plot'))+
  geom_point()+theme_classic()+scale_color_manual(values = c('gray','coral2','cyan3'))+
  geom_hline(yintercept=-log10(0.001),linetype='dashed')+geom_vline(xintercept=-0.1,linetype='dashed')+
  geom_vline(xintercept=0.1,linetype='dashed')+
  labs(x=expression('LogFC'),y=expression('-Log10(p-value)'),colour='Genes')+
  geom_text_repel(max.overlaps=Inf,label = ifelse(results3_limma$logFC>0.4 | results3_limma$logFC<(-0.4), 
                                                  rownames(results3_limma),""),box.padding = unit(0.25,'line'),size=3)
ggsave("volcano_gender_tumor.png", width = 8, height = 6)

# Subgroup DEGs analysis for male adjusted for age
alsamp3 = alsamp[,rownames(selected_vars_expr)[selected_vars_expr$gender == 1]]
selected_vars_expr3 = selected_vars_expr[selected_vars_expr$gender == 1,]
design4 = model.matrix(~0 + as.factor(selected_vars_expr3$tissue) + selected_vars_expr3$age)
colnames(design4) = c('Normal', 'Tumor', 'age')
fit = lmFit(alsamp3,design4)
contrast_matrix4 = makeContrasts(Tumor_Normal = Tumor - Normal, levels = design4)
fit2 = contrasts.fit(fit,contrast_matrix4)
fit2 = eBayes(fit2)
results4_limma = topTable(fit2, coef='Tumor_Normal', number=Inf, adjust.method='fdr')
write.csv(results4_limma, 'male_degs.csv', quote = FALSE, row.names = TRUE)

# Subgroup analysis for female adjusted for age
alsamp4 = alsamp[,rownames(selected_vars_expr)[selected_vars_expr$gender == 0]]
selected_vars_expr4 = selected_vars_expr[selected_vars_expr$gender == 0,]
design5 = model.matrix(~0 + as.factor(selected_vars_expr4$tissue) + selected_vars_expr4$age)
colnames(design5) = c('Normal', 'Tumor', 'age')
fit = lmFit(alsamp4,design5)
contrast_matrix5 = makeContrasts(Tumor_Normal = Tumor - Normal, levels = design5)
fit2 = contrasts.fit(fit,contrast_matrix5)
fit2 = eBayes(fit2)
results5_limma = topTable(fit2, coef='Tumor_Normal', number=Inf, adjust.method='fdr')
write.csv(results5_limma, 'female_degs.csv', quote = FALSE, row.names = TRUE)

# Venn diagram between significant DEGs of male and female
sig_male_degs <- rownames(results4_limma[results4_limma$adj.P.Val < 0.001 & (results4_limma$logFC > 0.4 | results4_limma$logFC < -0.4), ])
sig_female_degs <- rownames(results5_limma[results5_limma$adj.P.Val < 0.001 & (results5_limma$logFC > 0.4 | results5_limma$logFC < -0.4), ])
venn_data <- list(Male = sig_male_degs, Female = sig_female_degs)
venn_plot <- ggvenn(venn_data, fill_color = c("cyan3", "coral3"), stroke_size = 0.5, set_name_size = 6, text_size = 5) +
  ggtitle("Significant DEGs (P < 0.001, FC > 0.4 or FC < -0.4)") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))
print(venn_plot)
ggsave("venn_degs_male_female.png", venn_plot, width = 8, height = 6)

# Survival analysis of gene expression data
selected_vars_dummy = selected_vars_dummy[,-12]
selected_vars_dummy = na.omit(selected_vars_dummy)
selected_vars_dummy_gensurv = selected_vars_dummy[,-c(7,8,9,10,11)]
colnames(selected_vars_dummy_gensurv) = sub(' ','_',colnames(selected_vars_dummy_gensurv))

alsamp_surv = alsamp %>% t() %>% as.data.frame(stringsAsFactors = FALSE) %>% 
  mutate_all(as.numeric)
surv_data_lung = merge(selected_vars_dummy_gensurv,alsamp_surv,by = 0, all = FALSE)
rownames(surv_data_lung) = surv_data_lung$Row.names
surv_data_lung = surv_data_lung[,-1]

surv_data_lung2 = surv_data_lung
colnames(surv_data_lung2)[(ncol(selected_vars_dummy_gensurv)+1):ncol(surv_data_lung)] = paste0('gene',seq(1:ncol(alsamp_surv)))
res <- RegParallel(
  data = surv_data_lung2,
  formula = "Surv(os_months, os_status) ~ [*] + stage + age + gender + kras_mutation",
  FUN = function(formula, data){
    coxph(formula = formula,
          data = data,
          ties = 'breslow',
          singular.ok = TRUE)},
  FUNtype = 'coxph',
  variables = colnames(surv_data_lung2)[-c(1,2,3,4,5,6)],
  blocksize = 2000,
  cores = 8,
  nestedParallel = FALSE,
  conflevel = 95,
  p.adjust = 'BH')
res <- res[!is.na(res$P),]
dic_key = data.frame('Variable' = colnames(surv_data_lung2)[(ncol(selected_vars_dummy_gensurv)+1):ncol(surv_data_lung)], 
                     'genes'=colnames(alsamp_surv))
res = merge(dic_key,res,by = 'Variable',all.x = TRUE)
write.csv(res,'survival_analysis_lung_adjust_kras_gender_age_stage.csv',quote = FALSE,row.names = FALSE)

# Plotting survival analysis results
survplotdata <- surv_data_lung[,c('os_months','os_status','age','gender','stage','kras_mutation','HDGF','RAF1', 'ADRB2','KRAS')]
model = coxph(Surv(os_months, os_status) ~ KRAS + stage + age + gender + kras_mutation, 
              data = survplotdata[,c('os_months','os_status','age','stage','gender','kras_mutation','KRAS')])
test_data = data.frame(age = mean(survplotdata$age), stage = survplotdata$stage[3], kras_mutation = 1, gender=1,
                       KRAS = quantile(survplotdata$KRAS)[c(2, 4)])
model_km = survfit(model, newdata = test_data)
hr <- summary(model)$coefficients[, "exp(coef)"][1]
p_value <- summary(model)$coefficients[, "Pr(>|z|)"][1]
annotation_text <- paste0("HR: ", round(hr, 2), "\nP-value: ", format.pval(p_value, digits = 2))
plot <- ggsurvplot(model_km, data = test_data, conf.int = TRUE,
                   palette = c("cyan3", "coral3"),
                   legend.labs = c("KRAS 25% quantile expression", "KRAS 75% quantile expression"),
                   ggtheme = theme_minimal())
plot$plot <- plot$plot + annotate("text", x = 10, y = 0.2, label = annotation_text, size = 5, hjust = 0)
print(plot)
ggsave("survival_kras_plot.png", plot$plot, width = 8, height = 6)

# Network analysis of survival affecting genes
WGCNA::enableWGCNAThreads(nThreads = 7)
sig_surv_genes = res[((res$Variable == res$Term) & (res$P.adjust < 0.001)),]
surv_data_lung_net = surv_data_lung[,sig_surv_genes$genes]
surv_data_lung_net = surv_data_lung_net[,colMads(as.matrix(surv_data_lung_net))>0.3]

mic_cor = function(y_gene,y_cov){
  library(ppcor)
  x = colnames(y_gene)
  genes_combination = combn(x,m = 2,simplify = FALSE)
  res_mat = matrix(0,nrow=length(x),ncol=length(x),dimnames = list(x,x))
  for (i in genes_combination){
    i1 = i[1]
    j = i[2]
    test = pcor.test(y_gene[,i1],y_gene[,j],y_cov[,c('gender','stage','age')],method = 'pearson')
    if ((test$p.value<0.001) & (test$estimate>0.3 | test$estimate<(-0.3))){
      res_mat[i1,j] = test$estimate
    }
  }
  res_mat[lower.tri(res_mat)] = t(res_mat)[lower.tri(res_mat)]
  diag(res_mat) = 1
  return(res_mat)
}

t1=Sys.time()
sim=mic_cor(surv_data_lung_net,surv_data_lung)
t2=Sys.time()
t2-t1
write.csv(sim,'corelation_matrix.csv',quote = FALSE,sep = ',')

sim = as.matrix(read.csv('corelation_matrix.csv',header = TRUE,sep = ',',quote = ''))
rownames(sim)=sim[,'X']
sim=sim[,-1]
sim = matrix(as.numeric(sim),nrow = nrow(sim),ncol = ncol(sim),dimnames = list(row=rownames(sim),col=colnames(sim)))

t1=Sys.time()
scale_free=pickSoftThreshold.fromSimilarity(similarity = abs(sim),powerVector = c(seq(1,100,1)),verbose = 5)
t2=Sys.time()
t2-t1

dev.off()
plot(scale_free$fitIndices$Power,-sign(scale_free$fitIndices$slope)*scale_free$fitIndices$SFT.R.sq,type = 'n')
text(scale_free$fitIndices$Power,-sign(scale_free$fitIndices$slope)*scale_free$fitIndices$SFT.R.sq,labels = scale_free$fitIndices$Power,col='red')
abline(h=0.9)
plot(scale_free$fitIndices$Power,scale_free$fitIndices$mean.k.)
text(scale_free$fitIndices$Power,scale_free$fitIndices$mean.k.,labels = scale_free$fitIndices$Power,col='red',adj = c(1,1))
abline(h=100)

adjacency_matrix = adjacency.fromSimilarity(similarity = sim,power = 50)
exportNetworkToCytoscape(adjacency_matrix,edgeFile = 'cytoscape_coexp.txt',weighted = TRUE,threshold = 0.0)

tom_sim = TOMsimilarity(adjacency_matrix)
colnames(tom_sim)=rownames(sim)
rownames(tom_sim)=rownames(sim)
exportNetworkToCytoscape(tom_sim,edgeFile = 'cytoscape_tom.txt',weighted = TRUE,threshold = 0.0)

tom2manu = read.table('cytoscape_coexp.txt',header = TRUE,sep = '\t')
tom2manu = cbind(tom2manu, sign = 0)
for (i in 1:nrow(tom2manu)){
  source = tom2manu[i,'fromNode']
  target = tom2manu[i,'toNode']
  if (sim[source,target]<0){
    tom2manu[(tom2manu$fromNode==source) & (tom2manu$toNode==target),'sign'] = -1
  } else if (sim[source,target]>0){
    tom2manu[(tom2manu$fromNode==source) & (tom2manu$toNode==target),'sign'] = 1
  }
}
write.table(tom2manu,file = 'cytoscape_coexp.txt',quote = FALSE,sep = '\t',row.names = FALSE)

# Gene set variation analysis
sets_lung = msigdbr(species = "Homo sapiens", category = 'C6')
gene_sets = split(sets_lung$gene_symbol, sets_lung$gs_name)
surv_data_lung_t = surv_data_lung[7:ncol(surv_data_lung)] %>% t()
gsva_results <- gsva(surv_data_lung_t, gene_sets, method = "gsva", kcdf = "Gaussian", verbose = TRUE)
gsva_results = gsva_results %>% t() %>% as.data.frame(stringsAsFactors = FALSE)
surv_path = merge(selected_vars_dummy_gensurv, gsva_results, by=0, all=FALSE)
rownames(surv_path) = surv_path$Row.names
surv_path = surv_path[,-1]

surv_path2 = surv_path
colnames(surv_path2)[7:ncol(surv_path)] = paste0('path',seq(1:(ncol(surv_path)-6)))
res <- RegParallel(
  data = surv_path2,
  formula = "Surv(os_months, os_status) ~ [*] + stage + age + gender",
  FUN = function(formula, data){
    coxph(formula = formula,
          data = data,
          ties = 'breslow',
          singular.ok = TRUE)},
  FUNtype = 'coxph',
  variables = colnames(surv_path2)[-c(1,2,3,4,5,6)],
  blocksize = 20,
  cores = 8,
  nestedParallel = FALSE,
  conflevel = 95,
  p.adjust = 'BH')
res <- res[!is.na(res$P),]
dic_key = data.frame('Variable' = colnames(surv_path2)[(7:ncol(surv_path2))], 
                     'paths'=colnames(surv_path)[(7:ncol(surv_path))])
res = merge(dic_key,res,by = 'Variable',all.x = TRUE)
write.csv(res,'survival_analysis_lung_pathway_adjust_gender_age_stage.csv',quote = FALSE,row.names = FALSE)

# Clustering samples and pathways based on gsva_results
res_selected = res[(res$Variable == res$Term) & (res$P.adjust < 0.001), ]
res_selected = res_selected[order(res_selected$HR,decreasing = TRUE),]
selected_pathways <- gsva_results[, res_selected$paths]
col_fun <- colorRamp2(c(-1, 0, 1), c("blue", "white", "red"))
metadata <- selected_vars_dummy_gensurv[, c("gender", "kras_mutation",'stage','age')]
metadata <- metadata[rownames(selected_pathways),]
metadata <- metadata %>% mutate_at(vars(),as.factor) %>% as.data.frame(stringsAsFactors=FALSE)
sample_annotation <- rowAnnotation(
  df = metadata,
  col = list(
    Gender = c("0" = "red", "1" = "blue"),
    KRASStatus = c("0" = "orange", "1" = "green"),
    Stage = c("1" = 'red', '2'='blue', '3'='orange', '4'='green'),
    Age = metadata$age
  )
)
Heatmap(as.matrix(selected_pathways),
        name = "Pathway Activity",
        row_names_side = "left",
        column_names_side = "bottom",
        col = col_fun,
        clustering_distance_rows = "euclidean",
        clustering_distance_columns = "euclidean",
        column_names_rot = 45,
        show_column_dend = TRUE,
        left_annotation = sample_annotation,
        show_row_names = FALSE,
        heatmap_legend_param = list(
          title = "Pathway Activity",
          at = c(-1, 0, 1),
          labels = c("Low", "Intermediate", "High")
        ))
dev.off()