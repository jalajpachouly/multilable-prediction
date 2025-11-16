# multilable-prediction Configuration Guide

This document explains every configuration variable in the JSON config files for the multilable-prediction project. Each setting is described in detail, including its impact on model performance and experiment behavior.

---

## Top-Level

### `experiment_name`
- **Description:** Name of the experiment. Used for logging and output folder naming.
- **Impact:** No direct effect on performance, but helps organize results.

### `output_directory`
- **Description:** Path where all output files (results, visualizations) are saved.
- **Impact:** No effect on performance, but keeps outputs organized.

---

## Data Section (`data`)

- **`dataset_path`**: Path to the CSV dataset file. Must be correct for successful data loading.
- **`random_state`**: Seed for reproducibility. Ensures consistent splits and results.
- **`test_size`**: Fraction of data used for testing (e.g., 0.2 = 20%). Larger test size means less training data, possibly lower accuracy.
- **`sample_size`**: Number of samples to use. Useful for quick tests. Smaller samples may reduce performance and stability.
- **`sample_random_state`**: Seed for sampling. Ensures reproducibility when sampling.
- **`run_balanced`**: If true, runs experiments on a balanced dataset (equal samples per class). Can improve performance for imbalanced data.
- **`run_unbalanced`**: If true, runs experiments on the original (possibly imbalanced) dataset.
- **`balanced_target_count`**: Number of samples per class in balanced mode. Higher values may improve learning but require more data.

---

## Feature Engineering (`feature_engineering`)

- **`use_feature_selection`**: If true, selects top features using Chi-Square. Reduces dimensionality, can improve speed and generalization.
- **`top_k`**: Number of top features to select. Too low may miss important features; too high may add noise.
- **`top_k_plot`**: Number of top features to plot. Visualization only.
- **`use_wordcloud_vocabulary`**: If true, restricts TF-IDF vocabulary to words from word clouds. Can focus on relevant terms, but may miss others.
- **`max_words_per_label`**: Number of top words per label for word clouds. Affects vocabulary size and focus.
- **`tfidf`**:
  - **`ngram_range`**: Range of n-grams (e.g., [1,2] for unigrams and bigrams). Higher n-grams capture more context but increase dimensionality.
  - **`min_df`**: Minimum document frequency for terms. Filters rare words; higher values reduce noise but may remove useful terms.
  - **`use_idf`**: If true, uses inverse document frequency weighting. Usually improves text feature quality.

---

## Visualizations (`visualizations`)

- **`enabled`**: Master switch for all visualizations.
- **`description_length`**: Plots distribution of report lengths. No direct impact, but helps understand data.
- **`class_distribution`**: Plots class label distribution. Reveals imbalance issues.
- **`correlation_matrix`**: Plots label correlations. Can inform feature engineering.
- **`label_frequency`**: Plots label frequencies. Similar to class distribution.
- **`word_clouds`**: Generates word clouds for each label. No direct impact, but helps interpret features.
- **`top_features`**: Plots top features by Chi-Square. Visualization only.
- **`f1_scores`**: Plots F1 score distribution for models. No direct impact, but helps compare models.
- **`all_metrics_boxplot`**: Plots all metrics in a boxplot. Visualization only.
- **`nb_metrics`**: Plots metrics for Naive Bayes. Visualization only.

---

## Models (`models`)

### Traditional ML (`traditional_ml`)
- **`enabled`**: Master switch for traditional ML models.
- **`run_cross_validation`**: If true, runs cross-validation for ML models. Improves reliability of results.
- **`cv_n_splits`**: Number of CV folds. More folds = more stable estimates, but slower.
- **`multinomial_nb`**:
  - **`enabled`**: If true, runs Multinomial Naive Bayes.
  - **`use_classifier_chain`**: If true, wraps NB in a classifier chain for multi-label prediction.
- **`logistic_regression`**:
  - **`enabled`**: If true, runs Logistic Regression.
  - **`use_classifier_chain`**: If true, wraps LR in a classifier chain.
  - **`max_iter`**: Maximum iterations for solver. Higher values may be needed for convergence.
- **`random_forest`**:
  - **`enabled`**: If true, runs Random Forest.
  - **`use_classifier_chain`**: If true, wraps RF in a classifier chain.
  - **`n_estimators`**: Number of trees. More trees can improve accuracy but slow down training.
  - **`random_state`**: Seed for reproducibility.

### Deep Learning (`deep_learning`)
- **`enabled`**: Master switch for deep learning models.
- **`mlp`**:
  - **`enabled`**: If true, runs MLP model.
  - **`run_cross_validation`**: If true, runs CV for MLP.
  - **`cv_n_splits`**: Number of CV folds.
  - **`cv_epochs`**: Epochs for CV runs.
  - **`cv_batch_size`**: Batch size for CV runs.
  - **`epochs`**: Training epochs for final model. More epochs can improve learning but risk overfitting.
  - **`batch_size`**: Batch size for training. Larger batches are faster but may reduce generalization.
  - **`validation_split`**: Fraction of training data for validation. Used for early stopping.
  - **`early_stopping_patience`**: Number of epochs with no improvement before stopping. Prevents overfitting.
  - **`architecture`**:
    - **`layer1_units`**: Units in first hidden layer. More units = more capacity, but risk overfitting.
    - **`layer1_dropout`**: Dropout rate for first layer. Higher dropout reduces overfitting.
    - **`layer2_units`**: Units in second hidden layer.
    - **`layer2_dropout`**: Dropout rate for second layer.
- **`cnn`**:
  - **`enabled`**: If true, runs CNN model.
  - **`run_cross_validation`**: If true, runs CV for CNN.
  - **`cv_n_splits`**: Number of CV folds.
  - **`cv_epochs`**: Epochs for CV runs.
  - **`cv_batch_size`**: Batch size for CV runs.
  - **`epochs`**: Training epochs for final model.
  - **`batch_size`**: Batch size for training.
  - **`validation_split`**: Fraction of training data for validation.
  - **`early_stopping_patience`**: Early stopping patience.
  - **`max_words`**: Max vocabulary size for tokenizer. Larger vocab = more features, but slower and risk of overfitting.
  - **`max_len`**: Max sequence length for padding. Longer sequences capture more context but increase computation.
  - **`embedding_dim`**: Size of word embeddings. Higher values capture more semantic info but increase model size.
  - **`conv_filters`**: Number of convolutional filters. More filters = more capacity.
  - **`conv_kernel_size`**: Size of convolutional kernel. Larger kernels capture broader context.
  - **`dense_units`**: Units in dense layer after convolution.
  - **`dropout`**: Dropout rate for dense layer.

---

## Output (`output`)
- **`save_results_csv`**: If true, saves main results to CSV.
- **`save_cv_results_csv`**: If true, saves cross-validation results to CSV.
- **`results_filename`**: Filename for main results.
- **`cv_results_filename`**: Filename for CV results.

---

## How Settings Impact Model Performance

- **Data settings** affect how much and what kind of data is used. More data and balanced classes usually improve performance.
- **Feature engineering** controls which features are used. Good feature selection and TF-IDF settings can boost accuracy and reduce overfitting.
- **Model hyperparameters** (e.g., number of trees, layers, units, dropout) directly affect learning capacity, speed, and risk of overfitting.
- **Cross-validation** provides more reliable estimates of model performance, especially for small datasets.
- **Deep learning architecture** (layers, units, dropout) must be tuned for your data size and complexity.
- **Output settings** do not affect performance, but help organize and save results for analysis.

---

## Best Practices
- Start with default settings, then tune one parameter at a time.
- Use cross-validation for reliable performance estimates.
- Monitor overfitting by adjusting dropout, early stopping, and validation split.
- Use balanced data if your classes are imbalanced.
- Save and review all results and visualizations for insights.

---

For further details, see the code comments and experiment logs.
