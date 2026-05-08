################################################################################
### Project:   Strategies for auxiliary variable inclusion in missing data 
#              imputation: a comparison of inclusive, selection-based, and 
#              Machine Learning methods
### Author:    LM de Groot 
### Date:      7 May 2026
################################################################################

################################################################################
# DATA GENERATION
################################################################################

# ---------------------------------------------
# 0.1. LOAD REQUIRED PACKAGES
# ---------------------------------------------
library(dplyr)
library(glmnet)
library(MASS)
library(mice)
library(missForest)
library(pROC)
library(psfmi)
library(VIM)

# ---------------------------------------------
# 0.2. DEFINE SIMULATION PARAMETERS
# ---------------------------------------------
n_iterations <- 1000   
n <- 1000    
# n <- 200

TRUE_BETA <- c(
  "(Intercept)" = 0,
  "x1"  = 0.5, 
# "x1_squared" = 0.1,
  "x2"  = -0.7, 
#  "x2_squared" = -0.2,
  "x3"  = -0.5, 
  "x4"  = 0.7,
  "x52" = -0.5, 
  "x53" = 0.7, 
  "x62" = 0.5, 
  "x63" = -0.7
)

all_complete_datasets <- list()
result_names <- c(paste0("b_", names(TRUE_BETA)), 
                  paste0("se_", names(TRUE_BETA)), 
                  paste0("p_", names(TRUE_BETA)),
                  paste0("cov_", names(TRUE_BETA)),
                  "auc")

results_estimates <- matrix(NA, nrow = n_iterations, ncol = length(result_names))
colnames(results_estimates) <- result_names

# #############################################
# START LOOP
# #############################################

for (i in 1:n_iterations) {
  if(i %% 100 == 0) cat("Running iteration:", i, "out of", n_iterations, "\n")
  set.seed(2011 + i)
  
  # ---------------------------------------------
  # 1. GENERATE LATENT BASIS (Z)
  # ---------------------------------------------
  Z <- mvrnorm(n, mu = rep(0, 6), Sigma = diag(6)) 
  colnames(Z) <- paste0("z", 1:6)
  
  # ---------------------------------------------
  # 2. CREATE OBSERVED PREDICTORS (X)
  # ---------------------------------------------
  df_X <- data.frame(
    x1 = Z[,1],                      
    x2 = Z[,2],                     
    x3 = as.numeric(Z[,3] > 0),     
    x4 = as.numeric(Z[,4] > 0),      
    x5 = as.factor(ntile(Z[,5], 3)), 
    x6 = as.factor(ntile(Z[,6], 3))  
  )
  
  # ---------------------------------------------
  # 3. CREATE AUXILIARY VARIABLES (A)
  # ---------------------------------------------
  n_aux <- 44
  A_matrix <- matrix(NA, nrow = n, ncol = n_aux)
  colnames(A_matrix) <- paste0("a", 7:50)
  
  for(j in 1:n_aux) {
    n_targets <- sample(1:3, 1, prob = c(0.6, 0.3, 0.1)) 
    targets <- sample(1:6, n_targets) 
    rho_values <- rep(0, 6)
    rho_values[targets] <- sample(c(0.5, 0.3), n_targets, replace = TRUE)
    residual_sd <- sqrt(1 - sum(rho_values^2))
    A_matrix[,j] <- (Z %*% rho_values) + residual_sd * rnorm(n)
  }
  
  # ---------------------------------------------
  # 4. GENERATE OUTCOME (Y)
  # ---------------------------------------------
  X_final <- model.matrix(~ x1 + x2 + x3 + x4 + x5 + x6, data = df_X)
# X_final <- model.matrix(~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 + x5 + x6, data = df_X)  
  colnames(X_final) <- names(TRUE_BETA) 
  
  linpred <- X_final %*% TRUE_BETA
  prob    <- 1 / (1 + exp(-linpred))
  y_outcome <- rbinom(n, 1, prob)
  df <- cbind(data.frame(y_outcome), df_X, as.data.frame(A_matrix))
  all_complete_datasets[[i]] <- df  
  
  
  # ---------------------------------------------
  # 5. COMPLETE DATA ANALYSIS (VALIDATION)
  # ---------------------------------------------
  fit_complete <- glm(y_outcome ~ x1 + x2 + x3 + x4 + x5 + x6, 
                      data = df, family = binomial)
# fit_complete <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 + x5 + x6, data = df, family = binomial)
# rownames(coef_table)[rownames(coef_table) == "I(x1^2)"] <- "x1_squared"
# rownames(coef_table)[rownames(coef_table) == "I(x2^2)"] <- "x2_squared"
  
  summary_c <- summary(fit_complete)$coefficients
  b_est  <- summary_c[names(TRUE_BETA), 1]
  se_est <- summary_c[names(TRUE_BETA), 2]
  p_val  <- summary_c[names(TRUE_BETA), 4]
  lower_ci <- b_est - 1.96 * se_est
  upper_ci <- b_est + 1.96 * se_est
  coverage <- as.numeric(TRUE_BETA >= lower_ci & TRUE_BETA <= upper_ci)
  probs <- predict(fit_complete, type = "response")
  current_auc <- as.numeric(roc(df$y_outcome, probs, quiet = TRUE)$auc)
  results_estimates[i, ] <- c(b_est, se_est, p_val, coverage, current_auc)
  
  
} #END LOOP

#save.image("1_datageneration.Rdata")



################################################################################
# MAR
################################################################################
# load("1_datageneration.Rdata")

n_iterations <- 1000
all_mar_datasets <- list()

# ---------------------------------------------
# 1. MAR DATA FUNCTION
# ---------------------------------------------
make_MAR_original <- function(df, prop_incomplete = 0.6) {
  df_miss <- df

  aux_vars <- paste0("a", 7:50) 
  vars_target <- paste0("x", 1:6)
  
  n <- nrow(df)
  complete_ids <- sample(1:n, size = floor(n / 2), replace = FALSE)
  incomplete_ids <- setdiff(1:n, complete_ids)
    target_offset <- log(prop_incomplete / (1 - prop_incomplete))
  max_p_miss_incomplete <- 0.8
  min_p_miss_incomplete <- 0.4 
  beta_range <- c(-0.5, 0.5)
    df_numeric <- df
  df_numeric$x5 <- as.numeric(as.character(df$x5))
  df_numeric$x6 <- as.numeric(as.character(df$x6))
  cor_matrix <- cor(df_numeric[, vars_target], df_numeric[, aux_vars])
  
  for (var in vars_target) {
    cor_with_aux <- abs(cor_matrix[var, ])
    sorted_aux <- names(sort(cor_with_aux, decreasing = TRUE))
    n_aux <- sample(15:30, 1) 
    n_strong <- ceiling(n_aux * 0.85)
    strong_aux <- sorted_aux[1:min(length(sorted_aux), 20)] 
    weak_aux <- sorted_aux[(min(length(sorted_aux), 21)):length(sorted_aux)]
    selected_strong <- sample(strong_aux, min(n_strong, length(strong_aux)), replace = FALSE)
    selected_weak <- sample(weak_aux, (n_aux - length(selected_strong)), replace = FALSE)
    selected_aux <- c(selected_strong, selected_weak)
    
    X_mat <- model.matrix(
      as.formula(paste("~ y_outcome +", paste(selected_aux, collapse = " + "), "-1")),
      data = df_miss[incomplete_ids, ]
    )
    
    betas_miss <- runif(ncol(X_mat), beta_range[1], beta_range[2])
    linpred_base <- X_mat %*% betas_miss
    
    current_mean <- mean(linpred_base)
    linpred <- linpred_base + (target_offset - current_mean)
    
    p_miss <- plogis(linpred)
    p_miss <- pmin(p_miss, max_p_miss_incomplete)
    p_miss <- pmax(p_miss, min_p_miss_incomplete)
    
    miss <- rbinom(length(p_miss), 1, prob = p_miss) == 1
    df_miss[incomplete_ids[miss], var] <- NA
  }
  
  return(df_miss)
}

# ---------------------------------------------
# 2. MAKE MAR 
# ---------------------------------------------
for (i in 1:n_iterations) {
  if(i %% 100 == 0) cat("Processing MAR iteration:", i, "\n")
  set.seed(2011 + i)
  df_complete <- all_complete_datasets[[i]]
  df_mar <- make_MAR_original(df = df_complete, prop_incomplete = 0.6)
  all_mar_datasets[[i]] <- df_mar
}
#save.image("2_makemar.Rdata")




################################################################################
# IMPUTATION
################################################################################
# ---------------------------------------------
# 0.1. PREPARATION
# ---------------------------------------------
n_iterations <- 1000
backup_file <- "3_IMPUTATION_CHECKPOINT.RData"

save_objects <- c(
  "imputed_datasets_final", "imputed_datasets_quickpred", "imputed_datasets_lasso",
  "imputed_datasets_includeall", "imputed_datasets_rf", "imputed_datasets_missForest",
  "imputed_datasets_knn", "imputed_datasets_svm",
  "coefs_final", "coefs_quickpred", "coefs_lasso", "coefs_includeall",
  "coefs_rf", "coefs_missForest", "coefs_knn", "coefs_svm",
  "results_auc", "pred_matrices_quickpred_results", "pred_matrices_lasso_results"
)

if (file.exists(backup_file)) {
  load(backup_file)
  start_iteration <- max(which(!is.na(results_auc[, "final"]))) + 1
  cat("--- CHECKPOINT FOUND --- restart at iteration:", start_iteration, "\n")
} else {
  start_iteration <- 1
  cat("--- NO BACKUP FOUND --- start at iteration 1.\n")

  coefs_final <- list(); coefs_quickpred <- list(); coefs_lasso <- list()
  coefs_includeall <- list(); coefs_rf <- list(); coefs_missForest <- list()
  coefs_knn <- list(); coefs_svm <- list()
  
  imputed_datasets_final <- list(); imputed_datasets_quickpred <- list()
  imputed_datasets_lasso <- list(); imputed_datasets_includeall <- list()
  imputed_datasets_rf <- list(); imputed_datasets_missForest <- list()
  imputed_datasets_knn <- list(); imputed_datasets_svm <- list()
  
  all_methods <- c("final", "quickpred", "lasso", "includeall", "rf", "missForest", "knn", "svm")
  results_auc <- matrix(NA, nrow = n_iterations, ncol = length(all_methods), 
                        dimnames = list(NULL, all_methods))
  
  pred_matrices_quickpred_results <- list()
  pred_matrices_lasso_results <- list()
}

TRUE_BETA_NUMERIC <- c(
  0,   
  0.5, 
# 0.1,  
  -0.7,
# -0.2,
  -0.5, 
  0.7, 
  -0.5, 
  0.7, 
  0.5, 
  -0.7  
)

TRUE_BETA_NAMES <- c(
  "(Intercept)", "x1", "x2", "x3", "x4", "x52", "x53", "x62", "x63"
)
# TRUE_BETA_NAMES <- c("(Intercept)", "x1", "x1_squared", "x2", "x2_squared", "x3", "x4", "x52", "x53", "x62", "x63"

TRUE_BETA <- TRUE_BETA_NUMERIC
names(TRUE_BETA) <- TRUE_BETA_NAMES
true_coefs <- TRUE_BETA
result_names_b <- names(TRUE_BETA)
result_names_se <- paste0("se_", names(TRUE_BETA))
all_param_names <- c(result_names_b, result_names_se)
n_params_total <- length(all_param_names)


create_collapsed_imputed_set <- function(mice_obj) {
  original <- mice_obj$data
  m <- mice_obj$m
  completes <- lapply(1:m, function(j) complete(mice_obj, j))
  result <- original
  
  for (col in colnames(original)) {
    all_values <- sapply(completes, function(df) df[[col]])
    if (is.null(dim(all_values))) {
      all_values <- matrix(all_values, ncol = m)
    }
    if (is.numeric(original[[col]])) {
      result[[col]] <- rowMeans(all_values)
    } else if (is.factor(original[[col]]) && length(levels(original[[col]])) == 2) {
      modes <- apply(all_values, 1, function(x) {
        ux <- unique(x)
        ux[which.max(tabulate(match(x, ux)))]
      })
      result[[col]] <- factor(modes, levels = levels(original[[col]]))
    } else if (is.factor(original[[col]])) {
      modes <- apply(all_values, 1, function(x) {
        ux <- unique(x)
        ux[which.max(tabulate(match(x, ux)))]
      })
      result[[col]] <- factor(modes, levels = levels(original[[col]]))
    } else {
      result[[col]] <- original[[col]]
    }
  }
  
  return(result)
}




# #############################################
# START LOOP
# #############################################
for (i in start_iteration:n_iterations) {
  
  cat("\nRunning iteration:", i, "of", n_iterations) 
  
  df_mar <- all_mar_datasets[[i]]
  current_seed <- 2011 +i
  N_IMP <- 30
  set.seed(current_seed)
  
  
  # ------------------------------------------
  # METHOD 1: MICE - FINAL MODEL 
  # ------------------------------------------
  
  # 1. DATA PREPARATION
  df_subset <- df_mar[, c("y_outcome", "x1", "x2", "x3", "x4", "x5", "x6")] 
  df_subset$x3 <- factor(df_subset$x3, levels = c(0, 1))
  df_subset$x4 <- factor(df_subset$x4, levels = c(0, 1))
  df_subset$x5 <- factor(df_subset$x5)
  df_subset$x6 <- factor(df_subset$x6)
  meth <- make.method(df_subset)
  
  # 2. IMPUTATION
  imp_final <- mice(df_subset, m = N_IMP, method = meth, seed = current_seed, print = FALSE)
  
  # 3. ANALYSIS AND POOLING
  fit_final <- with(data = imp_final, glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6, family = binomial))
# fit_final <- with(data = imp_final, glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6, family = binomial))
  pooled_final <- pool(fit_final)
  summary_final <- summary(pooled_final)
  
  # 4. SAVE COEFFS
  coefs_final[[i]] <- summary_final[, c("term", "estimate", "std.error")]
  
  # 5. AUC 
  auc_values_final <- list()
  auc_ses_final <- list()
  
  for (j in 1:N_IMP) { 
    current_data <- complete(imp_final, j)
    current_data$x3 <- factor(current_data$x3, levels = c(0, 1))
    current_data$x4 <- factor(current_data$x4, levels = c(0, 1))
    current_data$x5 <- factor(current_data$x5)
    current_data$x6 <- factor(current_data$x6)
    
    # Fit model
    model <- glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6, data = current_data, family = "binomial")
#   model <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6, data = current_data, family = "binomial")
    preds_final <- predict(model, type = "response")
    
    # AUC & SE
    roc_obj <- pROC::roc(response = current_data$y_outcome, predict = preds_final, 
                         levels = c(0,1), direction = "<", ci = T)
    
    auc_values_final[[j]] <- roc_obj$auc
    auc_ses_final[[j]] <- sqrt(var(roc_obj))
  }
  
  # Pool AUC 
  pooled_auc_result_final <- pool_auc(est_auc = unlist(auc_values_final), est_se = unlist(auc_ses_final), nimp = N_IMP)
  auc_final_pooled <- as.numeric(pooled_auc_result_final[,"C-statistic"])
  results_auc[i, "final"] <- auc_final_pooled
  
  # 6. IMPUTED DATASET
  imputed_datasets_final[[i]] <- create_collapsed_imputed_set(imp_final) 
  
  # ------------------------------------------
  # PREPARATION
  # ------------------------------------------ 
  
  # 0.1. DEFINE VARIABLE TYPES
  vars_cont <- c("x1", "x2")
  vars_dicho <- c("x3", "x4")
  vars_cat <- c("x5", "x6")
  all_vars <- c(vars_cont, vars_dicho, vars_cat)
  
  
  # 0.2. PREPARATION
  df_full <- df_mar[, c("y_outcome", paste0("x", 1:6), paste0("a", 7:50))]
  df_full$x3 <- factor(df_full$x3, levels = c(0, 1)) 
  df_full$x4 <- factor(df_full$x4, levels = c(0, 1))
  df_full$x5 <- factor(df_full$x5)
  df_full$x6 <- factor(df_full$x6)
  
  
  # 0.3. DEFINE IMPUTATION METHODS
  meth_full <- make.method(df_full)
  ini <- mice(df_full, maxit = 0)
  meth_full <- ini$method 
  vars_with_na <- names(df_full)[colSums(is.na(df_full)) > 0]
  for (v in vars_with_na) {
    if (is.numeric(df_full[[v]])) { 
      meth_full[v] <- "pmm" 
    } else if (is.factor(df_full[[v]]) && nlevels(df_full[[v]]) == 2) { 
      meth_full[v] <- "logreg" 
    } else if (is.factor(df_full[[v]]) && nlevels(df_full[[v]]) > 2) { 
      meth_full[v] <- "polyreg" 
    }
  }
  
  # ------------------------------------------
  # METHOD 2: MICE - QUICKPRED
  # ------------------------------------------    
  # 1. PREDICTOR MATRIX W/ QUICKPRED 
  pred_matrix_quickpred <- quickpred(df_full, mincor = 0.2, include = c(var_y, all_vars))
  pred_matrices_quickpred_results[[i]] <- pred_matrix_quickpred
  
  
  # 2. IMPUTATION
  imp_quickpred <- mice(df_full, 
                        m = N_IMP, 
                        method = meth_full, 
                        predictorMatrix = pred_matrix_quickpred, 
                        seed = current_seed, 
                        print = FALSE)
  
  
  # 3. ANALYSIS AND POOLING 
  fit_quickpred <- with(data = imp_quickpred, glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6, family = binomial))  
# fit_quickpred <- with(data = imp_quickpred, glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,, family = binomial)) 
  pooled_quickpred <- pool(fit_quickpred)
  summary_quickpred <- summary(pooled_quickpred)
  
  
  # 4. SAVE COEFFS
  coefs_quickpred[[i]] <- summary_quickpred[, c("term", "estimate", "std.error")]
  
  
  # 5. AUC
  auc_values_quickpred <- list()
  auc_ses_quickpred <- list()
  
  for (j in 1:N_IMP) {
    current_data <- complete(imp_quickpred, j)
    current_data$x3 <- factor(current_data$x3, levels = c(0, 1))
    current_data$x4 <- factor(current_data$x4, levels = c(0, 1))
    current_data$x5 <- factor(current_data$x5)
    current_data$x6 <- factor(current_data$x6)
    
    model <- glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6,, data = current_data, family = "binomial")
#   model <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,, data = current_data, family = "binomial")  
    preds_quickpred <- predict(model, type = "response")
    
    roc_obj <- pROC::roc(response = current_data$y_outcome, predict = preds_quickpred,
                         levels = c(0,1), direction = "<", ci = T)
    
    auc_values_quickpred[[j]] <- roc_obj$auc
    auc_ses_quickpred[[j]] <- sqrt(var(roc_obj))
  }
  
  # Pool 
  pooled_auc_result_quickpred <- pool_auc(est_auc = unlist(auc_values_quickpred), est_se = unlist(auc_ses_quickpred), nimp = N_IMP)
  auc_quickpred_pooled <- as.numeric(pooled_auc_result_quickpred[,"C-statistic"])
  results_auc[i, "quickpred"] <- auc_quickpred_pooled
  
  
  # 6. IMPUTED DATASET
  imputed_datasets_quickpred[[i]] <- create_collapsed_imputed_set(imp_quickpred)
  
  # ------------------------------------------
  # METHOD 3: MICE - LASSO
  # ------------------------------------------     
  
  # 1. PREDICTOR MATRIX W/ LASSO
  pred_matrix_lasso <- matrix(0, ncol = ncol(df_full), nrow = ncol(df_full),
                              dimnames = list(colnames(df_full), colnames(df_full)))  
  
  # Lasso function for target variable
  fit_lasso_for_target <- function(data, target) {
    predictors <- setdiff(names(data), target)
    relevant_cols <- c(target, predictors)
    data_subset <- data[, relevant_cols, drop = FALSE]
    complete_cases_subset <- complete.cases(data_subset)
    
    X <- data_subset[complete_cases_subset, predictors, drop = FALSE]
    y <- data_subset[complete_cases_subset, target]
    
    X_matrix <- as.matrix(X)
    zero_var_cols <- apply(X_matrix, 2, function(col) length(unique(col)) < 2)
    X_matrix_filtered <- X_matrix[, !zero_var_cols, drop = FALSE]
    

    if (ncol(X_matrix_filtered) == 0) {
      message(paste("Warning: No predictors left for:", target, "after Zero Variance filter."))
      return(NULL)
    }
    
    X_matrix <- X_matrix_filtered
    
    if (is.factor(y)) {
      if (nlevels(y) == 2) {
        family_type <- "binomial"
        y_glmnet <- as.numeric(y) - 1
        if (length(unique(y_glmnet)) < 2 || any(table(y_glmnet) < 8)) return(NULL)
      } else if (nlevels(y) == 3) {
        family_type <- "multinomial"
        y_glmnet <- y
        if (length(unique(y_glmnet)) < nlevels(y) || any(table(y_glmnet) < 8)) return(NULL)
      } else {
        return(NULL)
      }
    } else if (is.numeric(y)) {
      family_type <- "gaussian"
      y_glmnet <- y
    } else {
      return(NULL)
    }
    
    if (is.null(family_type) || is.null(y_glmnet)) return(NULL)

      cvfit <- tryCatch({
      cv.glmnet(x = X_matrix, y = y_glmnet, alpha = 1, nfolds = 5,
                family = family_type, maxit = 1e6,
                lambda.min.ratio = 0.05)
    }, error = function(e) {
      message(paste("Error for variable", target, ":", e$message))
      return(NULL)
    })
    
    
    if (is.null(cvfit)) return(NULL)
    
    coef_lasso <- coef(cvfit, s = "lambda.min")
    
    if (family_type == "multinomial") {
      selected_vars_list <- lapply(coef_lasso, function(x) {
        rownames(x)[x[, 1] != 0]
      })
      selected_vars <- unique(unlist(selected_vars_list))
    } else {
      selected_vars <- rownames(coef_lasso)[coef_lasso[, 1] != 0]
    }
    
    setdiff(selected_vars, "(Intercept)")
  }
  
  # Loop over target vars with NA's, fill predMatrix
  for (target in vars_with_na) {
    selected_vars <- fit_lasso_for_target(df_full, target)
    if (!is.null(selected_vars) && length(selected_vars) > 0) {
      valid_selected_vars <- intersect(selected_vars, colnames(df_full))
      if (length(valid_selected_vars) > 0) {
        pred_matrix_lasso[target, valid_selected_vars] <- 1
      }
    }
    pred_matrix_lasso[target, target] <- 0
  }
  
  #Force y and x in de imputation model
  pred_matrix_lasso[vars_with_na, c(var_y, all_vars)] <- 1
  pred_matrices_lasso_results[[i]] <- pred_matrix_lasso
  
  # 2. IMPUTATION
  imp_lasso <- mice(df_full, 
                    m = N_IMP, 
                    method = meth_full, 
                    predictorMatrix = pred_matrix_lasso, 
                    seed = current_seed, 
                    print = FALSE)
  
  
  # 3. ANALYSIS AND POOLING 
  fit_lasso <- with(data = imp_lasso, glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6, family = binomial))
# fit_lasso <- with(data = imp_lasso, glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6, family = binomial))  
  pooled_lasso <- pool(fit_lasso)
  summary_lasso <- summary(pooled_lasso)
  
  
  # 4. SAVE COEFFS
  coefs_lasso[[i]] <- summary_lasso[, c("term", "estimate", "std.error")]
  
  
  # 5. AUC
  auc_values_lasso <- list()
  auc_ses_lasso <- list()
  
  for (j in 1:N_IMP) {
    current_data <- complete(imp_lasso, j)
    current_data$x3 <- factor(current_data$x3, levels = c(0, 1))
    current_data$x4 <- factor(current_data$x4, levels = c(0, 1))
    current_data$x5 <- factor(current_data$x5, levels = levels(df_full$x5))
    current_data$x6 <- factor(current_data$x6, levels = levels(df_full$x6))
    
    model <- glm(y_outcome ~ x1 +  x2 + x3 + x4 +x5 + x6,, data = current_data, family = "binomial")
#   model <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,, data = current_data, family = "binomial")
    preds_lasso <- predict(model, type = "response")
    
    roc_obj <- pROC::roc(response = current_data$y_outcome, predict = preds_lasso,
                         levels = c(0,1), direction = "<", ci = T)
    
    auc_values_lasso[[j]] <- roc_obj$auc
    auc_ses_lasso[[j]] <- sqrt(var(roc_obj))
  }
  
  # Pool 
  pooled_auc_result_lasso <- pool_auc(est_auc = unlist(auc_values_lasso), est_se = unlist(auc_ses_lasso), nimp = N_IMP)
  auc_lasso_pooled <- as.numeric(pooled_auc_result_lasso[,"C-statistic"])
  results_auc[i, "lasso"] <- auc_lasso_pooled
  
  
  # 6. IMPUTED DATASET
  imputed_datasets_lasso[[i]] <- create_collapsed_imputed_set(imp_lasso)   
  
  
  # ------------------------------------------
  # METHOD 4: MICE - INCLUDE ALL
  # ------------------------------------------
  
  # 1. PREDICTOR MATRIX
  pred_matrix_includeall <- make.predictorMatrix(df_full)
  
  # 2. IMPUTATION
  imp_includeall <- mice(df_full, 
                         m = N_IMP,              
                         method = meth_full,     
                         predictorMatrix = pred_matrix_includeall, 
                         seed = current_seed,
                         printFlag = FALSE)
  
  # 3. ANALYSIS AND POOLING 
  fit_includeall <- with(data = imp_includeall, glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6,, family = binomial))
# fit_includeall <- with(data = imp_includeall, glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,, family = binomial))  
  pooled_includeall <- pool(fit_includeall)
  summary_includeall <- summary(pooled_includeall)
  
  
  # 4. SAVE COEFFS
  coefs_includeall[[i]] <- summary_includeall[, c("term", "estimate", "std.error")]
  
  
  # 5. AUC
  auc_values_includeall <- list()
  auc_ses_includeall <- list()
  
  for (j in 1:N_IMP) {
    current_data <- complete(imp_includeall, j)
    current_data$x3 <- factor(current_data$x3, levels = c(0, 1))
    current_data$x4 <- factor(current_data$x4, levels = c(0, 1))
    current_data$x5 <- factor(current_data$x5, levels = levels(df_full$x5))
    current_data$x6 <- factor(current_data$x6, levels = levels(df_full$x6))
    
    model <- glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6, data = current_data, family = "binomial")
#   model <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6, data = current_data, family = "binomial")    
    preds_includeall <- predict(model, type = "response")
    
    roc_obj <- pROC::roc(response = current_data$y_outcome, predict = preds_includeall,
                         levels = c(0,1), direction = "<", ci = T)
    
    auc_values_includeall[[j]] <- roc_obj$auc
    auc_ses_includeall[[j]] <- sqrt(var(roc_obj))
  }
  
  # Pool 
  pooled_auc_result_includeall <- pool_auc(est_auc = unlist(auc_values_includeall), est_se = unlist(auc_ses_includeall), nimp = N_IMP)
  auc_includeall_pooled <- as.numeric(pooled_auc_result_includeall[,"C-statistic"])
  results_auc[i, "includeall"] <- auc_includeall_pooled
  
  # 6. IMPUTED DATASET
  imputed_datasets_includeall[[i]] <- create_collapsed_imputed_set(imp_includeall)
  
  
  # ---------------------------------------------
  # METHOD 5: MICE RF
  # ---------------------------------------------
  # 1. PREDICTOR MATRIX AND METHODS
  meth_full_rf <- make.method(df_full)
  for (v in names(meth_full_rf)) {
    if (meth_full_rf[v] != "") {
      meth_full_rf[v] <- "rf"
    }
  }
  
  pred_matrix_rf <- make.predictorMatrix(df_full)
  
  
  # 2. IMPUTATION
  imp_rf <- mice(df_full, method = meth_full_rf, predictorMatrix = pred_matrix_rf, m = 30, current_seed = current_seed)
  
  
  # 3. ANALYSIS AND POOLING
  fit_rf <- with(data = imp_rf, glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6, family = binomial))
# fit_rf <- with(data = imp_rf, glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,, family = binomial))
  pooled_rf <- pool(fit_rf)
  summary_rf <- summary(pooled_rf)
  
  
  # 4. SAVE COEFFS
  coefs_rf[[i]] <- summary_rf[,c("term", "estimate", "std.error")]
  
  
  # 5. AUC
  auc_values_rf <- list()
  auc_ses_rf <- list()
  
  for (j in 1:N_IMP) {
    current_data <- complete(imp_rf, j)
    current_data$x3 <- factor(current_data$x3, levels = c(0, 1))
    current_data$x4 <- factor(current_data$x4, levels = c(0, 1))
    current_data$x5 <- factor(current_data$x5, levels = levels(df_full$x5))
    current_data$x6 <- factor(current_data$x6, levels = levels(df_full$x6))
    
    model <- glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6,, data = current_data, family = "binomial")
#   model <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,, data = current_data, family = "binomial")
    preds_rf <- predict(model, type = "response")
    
    roc_obj <- pROC::roc(response = current_data$y_outcome, predict = preds_rf,
                         levels = c(0,1), direction = "<", ci = T)
    
    auc_values_rf[[j]] <- roc_obj$auc
    auc_ses_rf[[j]] <- sqrt(var(roc_obj))
  }
  
  # Pool 
  pooled_auc_result_rf <- pool_auc(est_auc = unlist(auc_values_rf), est_se = unlist(auc_ses_rf), nimp = N_IMP)
  auc_rf_pooled <- as.numeric(pooled_auc_result_rf[,"C-statistic"])
  results_auc[i, "rf"] <- auc_rf_pooled
  
  
  # 6. IMPUTED DATASET
  imputed_datasets_rf[[i]] <- create_collapsed_imputed_set(imp_rf)
  
  
  
  # ---------------------------------------------
  # METHOD 6: missForest
  # ---------------------------------------------
  
  # 2. IMPUTATION
  imputation_results <- missForest(df_full)
  imp_missForest <- imputation_results$ximp
  
  
  # 3. ANALYSIS
  fit_missForest <- glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6,
                        data = imp_missForest,
                        family = binomial
  )
# fit_missForest <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,,
#                       data = imp_missForest,
#                        family = binomial
#  )
  summary_missForest <- summary(fit_missForest)
  
  
  # 4. SAVE COEFS
  coefs_missForest[[i]] <- summary_missForest$coefficients[,1:2]
  
  
  # 5. AUC
  preds_missForest <- predict(fit_missForest, newdata = imp_missForest, type = "response")
  roc_obj <- roc(response = imp_missForest$y_outcome, predict = preds_missForest, 
                 levels = c(0,1), direction = "<", ci = T)
  auc_missForest <- as.numeric(roc_obj$auc)
  results_auc[i, "missForest"] <- auc_missForest
  
  # 6. IMPUTED DATASET
  imputed_datasets_missForest[[i]] <- imp_missForest
  
  
  
  # ---------------------------------------------
  # METHOD 7: kNN
  # ---------------------------------------------
  
  # 2. IMPUTATION
  imp_knn <- kNN(df_full, k = 5, imp_var = FALSE)
  
  
  # 3. ANALYSIS
  fit_knn <- glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6,
                 data = imp_knn,
                 family = binomial
  )
#  fit_knn <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,
#                 data = imp_knn,
#                 family = binomial
#  )
  summary_knn <- summary(fit_knn)
  
  
  # 4. SAVE COEFS
  coefs_knn[[i]] <- summary_knn$coefficients[,1:2]
  
  
  # 5. AUC
  preds_knn <- predict(fit_knn, newdata = imp_knn, type = "response")
  roc_obj <- roc(response = imp_knn$y_outcome, predict = preds_knn, 
                 levels = c(0,1), direction = "<", ci = T)
  auc_knn <- as.numeric(roc_obj$auc)
  results_auc[i, "knn"] <- auc_knn
  
  # 6. IMPUTED DATASET
  imputed_datasets_knn[[i]] <- imp_knn
  
  # ---------------------------------------------
  # METHOD 8: SVM
  # ---------------------------------------------
  
  # PREDICTOR MATRIX AND METHODS
  impute_svm_var <- function(data, target_var, predictor_vars, current_seed_svm) {
    set.seed(current_seed_svm)
    
    complete_idx <- which(!is.na(data[[target_var]]))
    incomplete_idx <- which(is.na(data[[target_var]]))
    
    if (length(incomplete_idx) == 0) {
      return(data)
    }
    
    # select train and pred data
    train_data <- data[complete_idx, c(target_var, predictor_vars), drop = FALSE]
    pred_data <- data[incomplete_idx, predictor_vars, drop = FALSE]
    train_data <- train_data[complete.cases(train_data), ]
    
    if (nrow(train_data) == 0) {
      warning(paste("No complete cases for", target_var, "to train SVM"))
      return(data)
    }
    
    # Method    
    if (is.factor(train_data[[target_var]])) {
      model_type <- "C-classification"
      train_data[[target_var]] <- droplevels(train_data[[target_var]]) 
    } else if (is.numeric(train_data[[target_var]])) {
      model_type <- "eps-regression"
    } else {
      stop("Target variable must be factor or numeric")
    }
    
    svm_model <- svm(
      formula = as.formula(paste(target_var, "~ .")),
      data = train_data,
      type = model_type,
      kernel = "radial" 
    )
    
    preds <- predict(svm_model, newdata = pred_data)
    
    
    data[incomplete_idx, target_var] <- preds
    
    if (is.factor(train_data[[target_var]])) {
      data[[target_var]] <- factor(data[[target_var]], levels = levels(train_data[[target_var]]))
    }
    
    return(data)
  }
  
  
  impute_svm_iterative <- function(data, impute_vars, max_iter = 5, base_seed_svm) {
    data_imp <- data 
    
    # initially fill with mean and modus
    message("Performing initial imputation of all missing values...")
    for (col_name in names(data_imp)) {
      if (any(is.na(data_imp[[col_name]]))) {
        if (is.numeric(data_imp[[col_name]])) {
          data_imp[is.na(data_imp[[col_name]]), col_name] <- mean(data_imp[[col_name]], na.rm = TRUE)
        } else if (is.factor(data_imp[[col_name]])) {
          tab <- table(data_imp[[col_name]])
          mode_val <- names(tab)[which.max(tab)]
          data_imp[is.na(data_imp[[col_name]]), col_name] <- factor(mode_val, levels = levels(data_imp[[col_name]]))
        } 
      }
    }
    message("Initial imputation complete.")
    
    # All predictor vars
    all_cols_in_data <- names(data_imp) 
    potential_predictors_pool <- all_cols_in_data 
    
    
    for (iter_svm in 1:max_iter) {
      message(paste("Running SVM Imputation Iteration:", iter_svm))
      
      
      for (var in impute_vars) {
        current_predictor_vars <- setdiff(potential_predictors_pool, var)
        
        data_imp <- impute_svm_var(data_imp, var, current_predictor_vars,
                                   current_seed_svm = base_seed_svm + iter_svm + which(impute_vars == var))
      }
    }
    
    return(data_imp)
  }
  
  # vars to be imputed
  impute_vars <- paste0("x", 1:6)
  
  # 2. IMPUTATION
  imp_svm <- impute_svm_iterative(df_full, impute_vars, max_iter = 5, base_seed_svm = current_seed)
  
  # 3. ANALAYSIS
  fit_svm <- glm(y_outcome ~ x1 + x2 + x3 + x4 +x5 + x6,
                 data = imp_svm, family = binomial
  )
# fit_svm <- glm(y_outcome ~ x1 + I(x1^2) + x2 + I(x2^2) + x3 + x4 +x5 + x6,
#                 data = imp_svm, family = binomial
#  )
  summary_svm <- summary(fit_svm)
  
  # 4. SAVE COEFS
  coefs_svm[[i]] <- summary_svm$coefficients[,1:2]
  
  # 5. AUC
  preds_svm <- predict(fit_svm, newdata = imp_svm, type = "response")
  roc_obj <- roc(response = imp_svm$y, predict = preds_svm, 
                 levels = c(0,1), direction = "<", ci = T)
  auc_svm <- as.numeric(roc_obj$auc)
  results_auc[i, "svm"] <- auc_svm
  
  # 6. IMPUTED DATASET
  imputed_datasets_svm[[i]] <- imp_svm
  
  # Sla op na iedere 5 of 10 iteraties (of elke iteratie als het erg traag is)
  if (i %% 10 == 0 || i == n_iterations) {
    cat(">>> Backup opslaan bij iteratie", i, "...\n")
    save(list = intersect(save_objects, ls()), file = backup_file)
  }
  
}

#save.image("3_imputation.RData")  


################################################################################
# RESULTS
################################################################################
#load("3_imputation.RData")

# --------------------------
# 0.LISTS FOR SAVING 
# ---------------------------
vars_cont <- c("x1", "x2")       
vars_dicho <- c("x3", "x4")         
vars_cat <- c("x5", "x6")  

true_values <- data.frame(
  term = c("x1", "x2", "x31", "x41", "x52", "x53", "x62", "x63" ),
  true_coefs = c(0.5, -0.7, -0.5, 0.7, -0.5, 0.7, 0.5, -0.7)
)
# true_values <- data.frame(
# term = c("x1", "x1_squared", "x2", "x2_squared", "x31", "x41", "x52", "x53", "x62", "x63" ),
# true_coefs = c(0.5, 0.1, -0.7, -0.2, -0.5, 0.7, -0.5, 0.7, 0.5, -0.7)
#)

list_of_imputation_results <- list(
  final = imputed_datasets_final,   
  quickpred = imputed_datasets_quickpred,
  lasso = imputed_datasets_lasso,
  includeall = imputed_datasets_includeall,
  rf = imputed_datasets_rf,
  missForest = imputed_datasets_missForest,
  knn = imputed_datasets_knn,
  svm = imputed_datasets_svm
)


# ---------------------------------------------
# 1. ACCURACY OF IMPUTED VARIABLES
# ---------------------------------------------

# 1. NRMSE and PFC function
eval_run_imputation <- function(imputed_df, df_complete, mask_missing,
                                vars_cont, vars_dicho, vars_cat) {
  
  all_vars <- c(vars_cont, vars_dicho, vars_cat)
  results <- vector("list", length(all_vars))
  
  k <- 1
  for (var in all_vars) {
    
    mis_idx <- mask_missing[, var]
    true_vals <- df_complete[[var]][mis_idx]
    imp_vals  <- imputed_df[[var]][mis_idx]
    
    if (var %in% vars_cont) {
      sd_complete <- sd(df_complete[[var]], na.rm = TRUE)
      rmse <- sqrt(mean((as.numeric(imp_vals) - as.numeric(true_vals))^2))
      results[[k]] <- data.frame(
        variable = var,
        nrmse = rmse / sd_complete,
        pfc = NA_real_
      )
    } else {
      pfc <- mean(as.character(imp_vals) != as.character(true_vals))
      results[[k]] <- data.frame(
        variable = var,
        nrmse = NA_real_,
        pfc = pfc
      )
    }
    k <- k + 1
  }
  dplyr::bind_rows(results)
}


# 2. Calculate NRMSE/PFC for each run 
long_nrmse_pfc <- purrr::map_dfr(
  names(list_of_imputation_results),
  function(method_name) {
    
    imputed_dfs_for_method <- list_of_imputation_results[[method_name]]
    
    purrr::map2_dfr(
      imputed_dfs_for_method,
      seq_along(imputed_dfs_for_method),
      function(df_imputed_current, run_idx) {
        
        df_complete_current <- all_complete_datasets[[run_idx]]
        df_mar_current <- all_mar_datasets[[run_idx]]
        
        mask_missing_current <- is.na(df_mar_current)
        
        vars_present <- intersect(
          c(vars_cont, vars_dicho, vars_cat),
          names(df_mar_current)
        )
        
        df_metrics <- eval_run_imputation(
          imputed_df = df_imputed_current,
          df_complete = df_complete_current,
          mask_missing = mask_missing_current,
          vars_cont  = intersect(vars_cont, vars_present),
          vars_dicho = intersect(vars_dicho, vars_present),
          vars_cat   = intersect(vars_cat, vars_present)
        )
        
        df_metrics$Method <- method_name
        df_metrics$Run <- run_idx
        df_metrics
      }
    )
  }
)


# 3. Average out
summary_nrmse_pfc <- long_nrmse_pfc %>%
  group_by(Method, variable) %>%
  summarise(
    Mean_nrmse = mean(nrmse, na.rm = TRUE),
    SD_nrmse = sd(nrmse, na.rm = TRUE),
    Mean_pfc = mean(pfc, na.rm = TRUE),
    SD_pfc = sd(pfc, na.rm = TRUE),
    N_runs = n_distinct(Run),
    .groups = "drop"
  )




# ---------------------------------------------
# 2. ACCURACY OF REGRESSION COEFFICIENTS; BIAS
# ---------------------------------------------

# 1. final
bias_coef_final <- coefs_final %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  group_by(term) %>%
  summarise(
    true_coefs = first(true_coefs),
    mean_estimate = mean(estimate),
    bias = mean_estimate - first(true_coefs),
    se_empirical = sd(estimate),
    mean_se = mean(std.error),
    n_runs = n(),
    .groups = "drop"
  )


# 2. quickpred
bias_coef_quickpred <- coefs_quickpred %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  group_by(term) %>%
  summarise(
    true_coefs = first(true_coefs),
    mean_estimate = mean(estimate),
    bias = mean_estimate - first(true_coefs),
    se_empirical = sd(estimate),
    mean_se = mean(std.error),
    n_runs = n(),
    .groups = "drop"
  )

# 3. Lasso
bias_coef_lasso <- coefs_lasso %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  group_by(term) %>%
  summarise(
    true_coefs = first(true_coefs),
    mean_estimate = mean(estimate),
    bias = mean_estimate - first(true_coefs),
    se_empirical = sd(estimate),
    mean_se = mean(std.error),
    n_runs = n(),
    .groups = "drop"
  )

# 4. include all
bias_coef_includeall <- coefs_includeall %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  group_by(term) %>%
  summarise(
    true_coefs = first(true_coefs),
    mean_estimate = mean(estimate),
    bias = mean_estimate - first(true_coefs),
    se_empirical = sd(estimate),
    mean_se = mean(std.error),
    n_runs = n(),
    .groups = "drop"
  )

# 5. rf
bias_coef_rf <- coefs_rf %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  group_by(term) %>%
  summarise(
    true_coefs = first(true_coefs),
    mean_estimate = mean(estimate),
    bias = mean_estimate - first(true_coefs),
    se_empirical = sd(estimate),
    mean_se = mean(std.error),
    n_runs = n(),
    .groups = "drop"
  )


# 6. missForest
bias_coef_missForest <- coefs_missForest %>%
  map(~ .x %>% as.data.frame() %>% rownames_to_column("term")) %>%
  bind_rows() %>%
  mutate(term = as.character(term)) %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  group_by(term) %>%
  summarise(
    true_coefs    = first(true_coefs),
    mean_estimate = mean(Estimate, na.rm = TRUE),
    bias          = mean_estimate - first(true_coefs),
    se_empirical  = sd(Estimate, na.rm = TRUE),
    mean_se       = mean(`Std. Error`, na.rm = TRUE),
    n_runs        = n(),
    .groups = "drop"
  ) %>%
  dplyr::select(
    term, 
    true_coefs, 
    mean_estimate, 
    bias, 
    abs_bias, 
    se_empirical, 
    mean_se, 
    n_runs
  )


# 7. knn
bias_coef_knn <- coefs_knn %>%
  map(~ .x %>% as.data.frame() %>% rownames_to_column("term")) %>%
  bind_rows() %>%
  mutate(term = as.character(term)) %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  group_by(term) %>%
  summarise(
    true_coefs    = first(true_coefs),
    mean_estimate = mean(Estimate, na.rm = TRUE),
    bias          = mean_estimate - first(true_coefs),
    se_empirical  = sd(Estimate, na.rm = TRUE),
    mean_se       = mean(`Std. Error`, na.rm = TRUE),
    n_runs        = n(),
    .groups = "drop"
  ) %>%
  dplyr::select(
    term, 
    true_coefs, 
    mean_estimate, 
    bias, 
    abs_bias, 
    se_empirical, 
    mean_se, 
    n_runs
  )

# 8. SVM
bias_coef_svm <- coefs_svm %>%
  map(~ .x %>% as.data.frame() %>% rownames_to_column("term")) %>%
  bind_rows() %>%
  mutate(term = as.character(term)) %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  group_by(term) %>%
  summarise(
    true_coefs    = first(true_coefs),
    mean_estimate = mean(Estimate, na.rm = TRUE),
    bias          = mean_estimate - first(true_coefs),
    se_empirical  = sd(Estimate, na.rm = TRUE),
    mean_se       = mean(`Std. Error`, na.rm = TRUE),
    n_runs        = n(),
    .groups = "drop"
  ) %>%
  dplyr::select(
    term, 
    true_coefs, 
    mean_estimate, 
    bias, 
    abs_bias, 
    se_empirical, 
    mean_se, 
    n_runs
  )


all_bias_results <- bind_rows(
  "final" = bias_coef_final,
  "quickpred" = bias_coef_quickpred,
  "lasso" = bias_coef_lasso,
  "includeall" = bias_coef_includeall,
  "rf" = bias_coef_rf,
  "missForest" = bias_coef_missForest,
  "knn" = bias_coef_knn,
  "svm" = bias_coef_svm,
  .id = "methode" 
) %>%
  dplyr::select(methode, term, true_coefs, mean_estimate, bias, abs_bias, se_empirical, mean_se, n_runs)


# ---------------------------------------------
# 3. ACCURACY OF REGRESSION COEFFICIENTS; COVERAGE
# ---------------------------------------------

# 1. final
coverage_final <- coefs_final %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower_ci = estimate - 1.96 * std.error,
    upper_ci = estimate + 1.96 * std.error,
    is_covered = (true_coefs >= lower_ci) & (true_coefs <= upper_ci)
  ) %>%
  group_by(term) %>%
  summarise(
    coverage_percentage = mean(is_covered, na.rm = TRUE) * 100,
    total_runs = n(),
    .groups = "drop"
  ) %>%
  mutate(methode = "final", .before = term)

# 2. quickpred
coverage_quickpred <- coefs_quickpred %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower_ci = estimate - 1.96 * std.error,
    upper_ci = estimate + 1.96 * std.error,
    is_covered = (true_coefs >= lower_ci) & (true_coefs <= upper_ci)
  ) %>%
  group_by(term) %>%
  summarise(
    coverage_percentage = mean(is_covered, na.rm = TRUE) * 100,
    total_runs = n(),
    .groups = "drop"
  ) %>%
  mutate(methode = "quickpred", .before = term)

# 3. lasso
coverage_lasso <- coefs_lasso %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower_ci = estimate - 1.96 * std.error,
    upper_ci = estimate + 1.96 * std.error,
    is_covered = (true_coefs >= lower_ci) & (true_coefs <= upper_ci)
  ) %>%
  group_by(term) %>%
  summarise(
    coverage_percentage = mean(is_covered, na.rm = TRUE) * 100,
    total_runs = n(),
    .groups = "drop"
  ) %>%
  mutate(methode = "lasso", .before = term)

# 4. include all
coverage_includeall <- coefs_includeall %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower_ci = estimate - 1.96 * std.error,
    upper_ci = estimate + 1.96 * std.error,
    is_covered = (true_coefs >= lower_ci) & (true_coefs <= upper_ci)
  ) %>%
  group_by(term) %>%
  summarise(
    coverage_percentage = mean(is_covered, na.rm = TRUE) * 100,
    total_runs = n(),
    .groups = "drop"
  ) %>%
  mutate(methode = "includeall", .before = term)

# 4. rf
coverage_rf <- coefs_rf %>%
  bind_rows() %>%
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower_ci = estimate - 1.96 * std.error,
    upper_ci = estimate + 1.96 * std.error,
    is_covered = (true_coefs >= lower_ci) & (true_coefs <= upper_ci)
  ) %>%
  group_by(term) %>%
  summarise(
    coverage_percentage = mean(is_covered, na.rm = TRUE) * 100,
    total_runs = n(),
    .groups = "drop"
  ) %>%
  mutate(methode = "rf", .before = term)


# 6. missForest
coverage_missForest <- coefs_missForest %>%
  map(~ .x %>% as.data.frame() %>% rownames_to_column("term")) %>%
  bind_rows() %>%
  rename(estimate = Estimate, std.error = `Std. Error`) %>% 
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower_ci = estimate - 1.96 * std.error,
    upper_ci = estimate + 1.96 * std.error,
    is_covered = (true_coefs >= lower_ci) & (true_coefs <= upper_ci)
  ) %>%
  group_by(term) %>%
  summarise(
    coverage_percentage = mean(is_covered, na.rm = TRUE) * 100,
    total_runs = n(),
    .groups = "drop"
  ) %>%
  mutate(methode = "missForest", .before = term)


# 7. knn
coverage_knn <- coefs_knn %>%
  map(~ .x %>% as.data.frame() %>% rownames_to_column("term")) %>%
  bind_rows() %>%
  rename(estimate = Estimate, std.error = `Std. Error`) %>% 
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower_ci = estimate - 1.96 * std.error,
    upper_ci = estimate + 1.96 * std.error,
    is_covered = (true_coefs >= lower_ci) & (true_coefs <= upper_ci)
  ) %>%
  group_by(term) %>%
  summarise(
    coverage_percentage = mean(is_covered, na.rm = TRUE) * 100,
    total_runs = n(),
    .groups = "drop"
  ) %>%
  mutate(methode = "knn", .before = term)

# 8. svm
coverage_svm <- coefs_svm %>%
  map(~ .x %>% as.data.frame() %>% rownames_to_column("term")) %>%
  bind_rows() %>%
  rename(estimate = Estimate, std.error = `Std. Error`) %>% 
  left_join(true_values, by = "term") %>%
  filter(term != "(Intercept)") %>%
  mutate(
    lower_ci = estimate - 1.96 * std.error,
    upper_ci = estimate + 1.96 * std.error,
    is_covered = (true_coefs >= lower_ci) & (true_coefs <= upper_ci)
  ) %>%
  group_by(term) %>%
  summarise(
    coverage_percentage = mean(is_covered, na.rm = TRUE) * 100,
    total_runs = n(),
    .groups = "drop"
  ) %>%
  mutate(methode = "svm", .before = term)


all_coverage_results <- bind_rows(
  "final" = coverage_final,
  "quickpred" = coverage_quickpred,
  "lasso" = coverage_lasso,
  "includeall" = coverage_includeall,
  "rf" = coverage_rf,
  "missForest" = coverage_missForest,
  "knn" = coverage_knn,
  "svm" = coverage_svm,
  .id = "methode" 
) %>%
  select(methode, everything())



# ---------------------------------------------
# 3. ACCURACY OF PREDICTION: AUC
# ---------------------------------------------
auc_df <- as.data.frame(results_auc)
auc_long <- auc_df %>%
  set_names(c("final", "quickpred", "lasso", "includeall", "rf", "missForest", "knn", "svm")) %>%
  mutate(run = row_number()) %>%
  pivot_longer(
    cols = -run, # Alle kolommen behalve 'run'
    names_to = "method",
    values_to = "auc"
  )

auc_summary <- auc_long %>%
  group_by(method) %>%
  summarise(
    mean_auc = mean(auc, na.rm = TRUE),
    sd_runs = sd(auc, na.rm = TRUE),
    n_runs = n(),
    se_mean_auc = sd_runs / sqrt(n_runs),
    lower_ci_95 = mean_auc - (1.96 * se_mean_auc),
    upper_ci_95 = mean_auc + (1.96 * se_mean_auc),
    
    .groups = "drop"
  ) %>%
  
  # Kolomvolgorde aanpassen
  select(
    method,
    mean_auc,
    lower_ci_95,
    upper_ci_95,
    se_mean_auc,
    sd_runs,
    n_runs
  )


# ---------------------------------------------
# 4. SELECTED PREDICTORS
# ---------------------------------------------
extract_predictor_counts <- function(matrix_list, target_vars) {
  var_names <- colnames(matrix_list[[1]])
  num_iter <- length(matrix_list)
  num_targets <- length(target_vars)
  num_predictors <- length(var_names)

  count_array <- array(0, dim = c(num_iter, num_targets, num_predictors),
                       dimnames = list(NULL, target_vars, var_names))
  
  for (i in 1:num_iter) {
    pred_matrix <- matrix_list[[i]]
    binary_rows <- (pred_matrix[target_vars, ] == 1)
    count_array[i, , ] <- binary_rows
  }
  return(count_array)
}

target_vars <- paste0("x", 1:6)
counts_quickpred <- extract_predictor_counts(pred_matrices_quickpred_results, target_vars)
counts_lasso <- extract_predictor_counts(pred_matrices_lasso_results, target_vars)

num_predictors_quickpred <- apply(counts_quickpred, c(1, 2), sum)
num_predictors_lasso <- apply(counts_lasso, c(1, 2), sum)

summary_quickpred_selection <- t(apply(num_predictors_quickpred, 2, function(x) {
  c(mean = mean(x), 
    median = median(x), 
    min = min(x), 
    max = max(x))
}))

summary_lasso_selection <- t(apply(num_predictors_lasso, 2, function(x) {
  c(mean = mean(x), 
    median = median(x), 
    min = min(x), 
    max = max(x))
}))

