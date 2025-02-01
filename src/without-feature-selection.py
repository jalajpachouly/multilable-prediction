"""
Multi-Label Classification for Bug Reports

This script performs multi-label classification on bug reports using several machine learning models,
including traditional classifiers and deep learning models like MLP and CNN. It includes functions
for data loading, preprocessing, visualization, model training, evaluation, and result visualization.

Author: Your Name
Date: October 2023
"""

# ====================================
# Import Libraries
# ====================================

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP and text processing
import nltk

# Machine learning and evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, hamming_loss
from sklearn.feature_selection import chi2

# Multi-label stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Optimization
from scipy.optimize import nnls

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# Initialize NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Define global labels
LABELS = ["type_blocker", "type_bug", "type_documentation", "type_enhancement"]


# ====================================
# Data Loading and Preparation
# ====================================

def build_conditional_prob_matrix(df, labels):
    """
    Build a conditional probability matrix for label co-occurrence.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the labels.
    - labels (list of str): List of label column names.

    Returns:
    - cooc_norm (np.ndarray): Normalized co-occurrence matrix.
    """
    cooc = df[labels].values.T.dot(df[labels].values)
    cooc_norm = cooc.copy().astype(np.float32)
    for i in range(cooc_norm.shape[0]):
        cooc_norm[:, i] /= cooc[i, i]
    return cooc_norm


def nnls_sample(df, labels, target_count, cond_prob):
    """
    Perform stratified sampling to balance the dataset based on label co-occurrence.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data and labels.
    - labels (list of str): List of label column names.
    - target_count (int): Desired number of samples per label.
    - cond_prob (np.ndarray): Conditional probability matrix from build_conditional_prob_matrix.

    Returns:
    - sampled_df (pd.DataFrame): The resampled DataFrame.
    """
    target_counts = np.array([target_count for _ in labels])
    optimal_samples, residuals = nnls(cond_prob, target_counts)
    optimal_samples = np.ceil(optimal_samples).astype(np.int32)
    df_subs = []
    for i, label in enumerate(labels):
        sub_df = df[df[label] == 1]
        df_subs.append(sub_df.sample(optimal_samples[i],
                                     replace=len(sub_df) < optimal_samples[i]))
    sampled_df = pd.concat(df_subs)
    return sampled_df


def load_data(csv_path: str):
    """
    Load the dataset from a CSV file and split it into training and testing sets using stratified splitting.

    Parameters:
    - csv_path (str): Path to the CSV file containing the dataset.

    Returns:
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Testing set features.
    - y_train (pd.DataFrame): Training set labels.
    - y_test (pd.DataFrame): Testing set labels.
    """
    # Load the dataset from a CSV file
    data = pd.read_csv(csv_path)

    # Check if required columns exist
    required_columns = ['report', 'type_blocker', 'type_bug', 'type_documentation', 'type_enhancement']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

    # Feature column (text data)
    X = data[['report']]

    # Label columns (multi-label targets)
    y = data[['type_blocker', 'type_bug', 'type_documentation', 'type_enhancement']]

    # Initialize the stratified shuffle split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Perform the split
    for train_index, test_index in msss.split(X, y):
        X_train = X.iloc[train_index].reset_index(drop=True)
        X_test = X.iloc[test_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)

    # Print label counts to check for class imbalance
    print("Label counts in y_train:")
    print(y_train.sum())
    print("\nLabel counts in y_test:")
    print(y_test.sum())
    return X_train, X_test, y_train, y_test


def load_data_balanced(csv_path: str):
    """
    Load the dataset from a CSV file, balance it, and split into training and testing sets.

    Parameters:
    - csv_path (str): Path to the CSV file containing the dataset.

    Returns:
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Testing set features.
    - y_train (pd.DataFrame): Training set labels.
    - y_test (pd.DataFrame): Testing set labels.
    """
    # Load the dataset from a CSV file
    data = pd.read_csv(csv_path)

    # Build conditional probability matrix and perform NNLS sampling
    cooc_norm = build_conditional_prob_matrix(data, LABELS)
    resampled_df = nnls_sample(data, LABELS, 600, cooc_norm)
    data = resampled_df

    # Check if required columns exist
    required_columns = ['report', 'type_blocker', 'type_bug', 'type_documentation', 'type_enhancement']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

    # Feature column (text data)
    X = data[['report']]

    # Label columns (multi-label targets)
    y = data[['type_blocker', 'type_bug', 'type_documentation', 'type_enhancement']]

    # Initialize the stratified shuffle split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Perform the split
    for train_index, test_index in msss.split(X, y):
        X_train = X.iloc[train_index].reset_index(drop=True)
        X_test = X.iloc[test_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)

    # Print label counts to check for class imbalance
    print("Label counts in y_train:")
    print(y_train.sum())
    print("\nLabel counts in y_test:")
    print(y_test.sum())
    return X_train, X_test, y_train, y_test


def prepare_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, top_k=1000, vocabulary=None):
    """
    Convert text data to TF-IDF features and perform feature selection using the Chi-Square test.

    Parameters:
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Testing set features.
    - y_train (pd.DataFrame): Training set labels.
    - top_k (int): Number of top features to select based on Chi-Square scores.
    - vocabulary (list or None): Predefined vocabulary to use for TF-IDF vectorizer.

    Returns:
    - X_train_selected (sparse matrix): Selected training set features.
    - X_test_selected (sparse matrix): Selected testing set features.
    - selected_features (np.array): Names of the selected features.
    - chi2_scores_max (np.array): Maximum Chi-Square scores for each feature.
    - vector (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    # Initialize the TF-IDF vectorizer
    vector = TfidfVectorizer(
        ngram_range=(1, 4),
        analyzer='word',
        stop_words='english',
        strip_accents='unicode',
        use_idf=True,
        min_df=2
    )

    # Apply TF-IDF on the 'report' column
    X_train_tfidf = vector.fit_transform(X_train['report'])
    X_test_tfidf = vector.transform(X_test['report'])

    # Feature selection using Chi-Square test
    chi2_scores = []
    for i in range(y_train.shape[1]):
        chi2_score_values, p_value = chi2(X_train_tfidf, y_train.iloc[:, i])
        chi2_scores.append(chi2_score_values)

    # Aggregate Chi-Square scores across labels by taking the maximum score for each feature
    chi2_scores_max = np.mean(np.array(chi2_scores), axis=0)

    # Select top K features based on Chi-Square scores
    if top_k > len(chi2_scores_max):
        top_k = len(chi2_scores_max)
    selected_indices = np.argsort(chi2_scores_max)[::-1][:top_k]
    selected_indices = selected_indices.astype(int)
    X_train_selected = X_train_tfidf[:, selected_indices]
    X_test_selected = X_test_tfidf[:, selected_indices]

    # Retrieve selected feature names
    selected_features = np.array(vector.get_feature_names_out())[selected_indices]
    print(f"\nSelected top {top_k} features based on Chi-Square scores:")
    print(selected_features[:20])
    X_train_tfidf = vector.fit_transform(X_train['report'])
    X_test_tfidf = vector.transform(X_test['report'])
    return X_train_tfidf, X_test_tfidf, selected_features, chi2_scores_max, vector

    #return X_train_selected, X_test_selected, selected_features, chi2_scores_max, vector


def prepare_data_for_deep_learning(X_train_texts, X_test_texts, max_words=5000, max_len=100):
    """
    Tokenize and pad text data for deep learning models.

    Parameters:
    - X_train_texts: Training text data.
    - X_test_texts: Testing text data.
    - max_words: Maximum number of words to consider in the vocabulary.
    - max_len: Maximum length of sequences after padding.

    Returns:
    - X_train_pad: Padded training sequences.
    - X_test_pad: Padded testing sequences.
    - tokenizer: Fitted Keras tokenizer.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token='')
    tokenizer.fit_on_texts(X_train_texts)

    X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
    X_test_seq = tokenizer.texts_to_sequences(X_test_texts)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    return X_train_pad, X_test_pad, tokenizer


def prepare_data_for_transformer(X_texts, tokenizer, max_len=100):
    """
    Tokenize text data for the Transformer model.

    Parameters:
    - X_texts: List or Series of text data.
    - tokenizer: Pretrained tokenizer from Hugging Face.
    - max_len: Maximum sequence length.

    Returns:
    - input_ids: Token IDs for each text.
    - attention_masks: Attention masks for each text.
    """
    encodings = tokenizer.batch_encode_plus(
        X_texts.tolist(),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )

    return encodings['input_ids'], encodings['attention_mask']


# ====================================
# Visualization Functions
# ====================================

def visualize_description_length(X_train: pd.DataFrame, data_type: str):
    """
    Visualize the distribution of description lengths in the given DataFrame.

    Parameters:
    - X_train (pd.DataFrame): DataFrame containing the 'report' column.
    - data_type (str): String indicating the type of data (e.g., 'Balanced').
    """
    sns.set(style="darkgrid")
    X_train['report'] = X_train['report'].astype(str)
    description_len = X_train['report'].str.len()
    plt.figure(figsize=(10, 6))
    sns.histplot(description_len, kde=False, bins=20, color="steelblue")
    plt.xlabel('Description Length')
    plt.ylabel('Frequency')
    plt.title('Description Length Distribution')
    plt.tight_layout()
    plt.savefig(f'description_length_distribution_{data_type}.png')
    plt.close()
    print(f"Description length distribution plot saved as 'description_length_distribution_{data_type}.png'.")


def visualize_class_distribution(y_train: pd.DataFrame, y_test: pd.DataFrame, data_type: str,
                                 save_path='class_distribution.png'):
    """
    Visualize the distribution of classes within each label for both training and test datasets.

    Parameters:
    - y_train (pd.DataFrame): Training set labels.
    - y_test (pd.DataFrame): Testing set labels.
    - data_type (str): String indicating the type of data (e.g., 'Balanced').
    - save_path (str): Filename for saving the plot.
    """
    labels = y_train.columns.tolist()
    bar_width = 0.2
    bars1 = [sum(y_train[label] == 1) for label in labels]
    bars2 = [sum(y_train[label] == 0) for label in labels]
    bars3 = [sum(y_test[label] == 1) for label in labels]
    bars4 = [sum(y_test[label] == 0) for label in labels]

    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    plt.figure(figsize=(12, 8))
    plt.bar(r1, bars1, color='steelblue', width=bar_width, label='Train Labeled = 1')
    plt.bar(r2, bars2, color='lightsteelblue', width=bar_width, label='Train Labeled = 0')
    plt.bar(r3, bars3, color='darkorange', width=bar_width, label='Test Labeled = 1')
    plt.bar(r4, bars4, color='navajowhite', width=bar_width, label='Test Labeled = 0')

    plt.xlabel('Labels', fontweight='bold')
    plt.xticks([r + bar_width * 1.5 for r in range(len(bars1))], labels, rotation=45)
    plt.legend()
    plt.title('Distribution of Classes within Each Label')
    plt.tight_layout()
    plt.savefig(f'{data_type}_{save_path}')
    plt.close()
    print(f"Class distribution plot saved as '{data_type}_{save_path}'.")


def visualize_word_cloud(X_train: pd.DataFrame, y_train: pd.DataFrame, token: str, max_words=500):
    """
    Visualize the most common words contributing to the token and return top words.

    Parameters:
    - X_train (pd.DataFrame): Training set features.
    - y_train (pd.DataFrame): Training set labels.
    - token (str): The label to visualize word cloud for.
    - max_words (int): Number of top words to return.

    Returns:
    - top_words (list): List of top words based on frequency.
    """
    description_context = X_train.join(y_train)
    description_context = description_context[description_context[token] == 1]
    if description_context.empty:
        print(f"No instances for label '{token}'; skipping word cloud.")
        return []

    description_text = description_context['report']
    combined_text = ' '.join(description_text)
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color='white').generate(combined_text)
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud.recolor(colormap="Blues"), interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Most Common Words Associated with '{token}' Defects", size=20)
    plt.tight_layout()
    plt.savefig(f'wordcloud_{token}.png')
    plt.close()
    print(f"Word cloud for '{token}' saved as 'wordcloud_{token}.png'.")

    # Extract word frequencies from the word cloud
    word_freq = wordcloud.words_
    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    # Get top N words
    top_words = [word for word, freq in sorted_words[:max_words]]
    return top_words


def visualize_f1_scores(methods: pd.DataFrame, data_type: str):
    """
    Visualize F1 score results through a box plot with jittered points.

    Parameters:
    - methods (pd.DataFrame): DataFrame containing evaluation metrics.
    """
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Model', y='F1', data=methods, palette="Blues")
    sns.stripplot(x='Model', y='F1', data=methods, size=8, jitter=True,
                  edgecolor="gray", linewidth=2, palette="Blues")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    plt.title('F1 Score Distribution by Model')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(f'f1_score_distribution_{data_type}.png')
    plt.close()
    print("F1 score distribution plot saved as 'f1_score_distribution.png'.")


def visualize_all_metrics_boxplot(methods: pd.DataFrame, data_type: str):
    """
    Create a box plot comparing Recall, F1-score, and Hamming Loss across all models.

    Parameters:
    - methods (pd.DataFrame): DataFrame containing evaluation metrics.
    - data_type (str): String indicating the type of data (e.g., 'Balanced').
    """
    # Melt the DataFrame to have Metrics in a single column
    metrics_melted = methods.melt(id_vars=['Model', 'Label'], value_vars=['Recall', 'F1', 'Hamming Loss'],
                                  var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Metric', y='Score', hue='Model', data=metrics_melted, palette="Set2")
    sns.stripplot(x='Metric', y='Score', hue='Model', data=metrics_melted, dodge=True,
                  color='gray', alpha=0.6, size=5, jitter=True)
    plt.title('Comparison of Evaluation Metrics Across Models')
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    # Handle legends to avoid duplication
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'all_metrics_comparison_boxplot_{data_type}.png')
    plt.close()
    print(f"All metrics comparison box plot saved as 'all_metrics_comparison_boxplot_{data_type}.png'.")


def visualize_nb_metrics(methods: pd.DataFrame, data_type: str):
    """
    Create a bar graph of F1 and Recall across each label for Multinomial Naive Bayes.

    Parameters:
    - methods (pd.DataFrame): DataFrame containing evaluation metrics.
    - data_type (str): String indicating the type of data (e.g., 'Balanced').
    """
    print("Plot for Multinomial Naive Bayes regression")
    m2 = methods[methods.Model == 'MultinomialNB'].copy()
    if m2.empty:
        print("No data available for MultinomialNB metrics; skipping plot.")
        return
    m2.set_index(["Label"], inplace=True)
    ax = m2[['Recall', 'F1']].plot(figsize=(16, 8), kind='bar', title='Multinomial Naive Bayes Metrics by Label',
                                   rot=60, ylim=(0.0, 1), colormap='tab10')
    plt.ylabel('Score')
    plt.xlabel('Labels')
    plt.tight_layout()
    plt.savefig(f'mnb_metrics_per_label_{data_type}.png')
    plt.close()
    print(f"Multinomial Naive Bayes metrics per label plot saved as 'mnb_metrics_per_label_{data_type}.png'.")


def visualize_correlation_matrix(y_train: pd.DataFrame, data_type: str):
    """
    Visualize the cross-correlation matrix across labels in the given DataFrame.

    Parameters:
    - y_train (pd.DataFrame): Training set labels.
    - data_type (str): String indicating the type of data (e.g., 'Balanced').
    """
    label_columns = y_train.columns.tolist()
    if not label_columns:
        print("No label columns found in the DataFrame for correlation matrix; skipping plot.")
        return

    train_corr = y_train.copy()
    if train_corr.empty:
        print("No data available for correlation matrix; skipping plot.")
        return

    corr = train_corr.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    plt.title('Correlation Matrix of Labels')
    plt.tight_layout()
    plt.savefig(f'label_correlation_matrix_{data_type}.png')
    plt.close()
    print(f"Label correlation matrix plot saved as 'label_correlation_matrix_{data_type}.png'.")


def visualize_label_frequency(y_train: pd.DataFrame, data_type: str):
    """
    Visualize the frequency of specified labels in the given DataFrame.

    Parameters:
    - y_train (pd.DataFrame): Training set labels.
    - data_type (str): String indicating the type of data (e.g., 'Balanced').
    """
    labels = ["type_bug", "type_documentation", "type_enhancement"]
    # Filter labels that exist in the DataFrame
    present_labels = [label for label in labels if label in y_train.columns]
    if not present_labels:
        print("No specified labels found in the DataFrame; skipping label frequency plot.")
        return

    label_count = y_train[present_labels].sum()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_count.index, y=label_count.values, color="steelblue")
    plt.title('Labels Frequency')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'label_frequency_{data_type}.png')
    plt.close()
    print(f"Label frequency plot saved as 'label_frequency_{data_type}.png'.")


def plot_top_features(selected_features, chi2_scores_max, data_type: str, top_k_plot=20):
    """
    Plot the top features based on Chi-Square scores.

    Parameters:
    - selected_features (np.array): Array of selected feature names.
    - chi2_scores_max (np.array): Maximum Chi-Square scores for each feature.
    - data_type (str): String indicating the type of data (e.g., 'Balanced').
    - top_k_plot (int): Number of top features to plot.
    """
    top_features = selected_features[:top_k_plot]
    selected_indices = np.argsort(chi2_scores_max)[::-1][:top_k_plot]
    selected_indices = selected_indices.astype(int)
    top_scores = chi2_scores_max[selected_indices[:top_k_plot]]
    # Plot the top features
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_scores, y=top_features)
    plt.title(f'Top {top_k_plot} Features Based on Chi-Square Scores')
    plt.xlabel('Chi-Square Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'chi2_features_{data_type}.png')
    plt.close()

    print(f"\nTop {top_k_plot} features have been plotted and saved as 'chi2_features_{data_type}.png'.")


# ====================================
# Model Training and Evaluation
# ====================================

def cross_validation_score_multilabel(classifier, X, y, n_splits=10):
    """
    Perform cross-validation and compute average Recall and F1-score.

    Parameters:
    - classifier: The classifier to evaluate.
    - X (sparse matrix or ndarray): Feature matrix.
    - y (ndarray): Label matrix.
    - n_splits (int): Number of cross-validation splits.

    Returns:
    - dict: Dictionary containing average Recall and F1-score.
    """
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall_scores = []
    f1_scores = []

    for fold, (train_index, test_index) in enumerate(mskf.split(X, y), 1):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        classifier.fit(X_train_cv, y_train_cv)
        y_pred_cv = classifier.predict(X_test_cv)

        # Compute recall and F1 score
        recall = recall_score(y_test_cv, y_pred_cv, average='macro', zero_division=0)
        f1 = f1_score(y_test_cv, y_pred_cv, average='macro', zero_division=0)

        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Fold {fold}: Recall = {recall:.4f}, F1-Score = {f1:.4f}")

    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    return {'Recall': avg_recall, 'F1': avg_f1}


def cross_validation_score_deep_learning(model_builder, X, y, n_splits=10, epochs=10, batch_size=32):
    """
    Perform cross-validation for a Deep Learning model and compute average Recall and F1-score.

    Parameters:
    - model_builder: Function to build the model.
    - X (ndarray): Feature matrix.
    - y (ndarray): Label matrix.
    - n_splits (int): Number of cross-validation splits.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.

    Returns:
    - dict: Dictionary containing average Recall and F1-score.
    """
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall_scores = []
    f1_scores = []

    for fold, (train_index, test_index) in enumerate(mskf.split(X, y), 1):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        model = model_builder()

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(
            X_train_cv,
            y_train_cv,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_cv, y_test_cv),
            callbacks=[early_stop],
            verbose=0
        )

        y_pred_cv_prob = model.predict(X_test_cv)
        y_pred_cv = (y_pred_cv_prob >= 0.5).astype(int)

        # Compute recall and F1 score
        recall = recall_score(y_test_cv, y_pred_cv, average='macro', zero_division=0)
        f1 = f1_score(y_test_cv, y_pred_cv, average='macro', zero_division=0)

        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Fold {fold}: {model_builder.__name__} Recall = {recall:.4f}, F1-Score = {f1:.4f}")

    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    return {'Recall': avg_recall, 'F1': avg_f1}


def evaluate_classifier(clf, clf_name, X_train, y_train, X_test, y_test, label_names):
    """
    Train the classifier, make predictions, and evaluate performance.

    Parameters:
    - clf: The classifier to evaluate.
    - clf_name (str): Name of the classifier.
    - X_train (sparse matrix or ndarray): Training feature matrix.
    - y_train (ndarray): Training labels.
    - X_test (sparse matrix or ndarray): Testing feature matrix.
    - y_test (ndarray): Testing labels.
    - label_names (list): List of label names.

    Returns:
    - list of dicts: List containing evaluation metrics for each label.
    """
    print(f"\n===== Evaluating {clf_name} =====")

    # Fit the model
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Calculate Hamming Loss
    hamming_loss_value = hamming_loss(y_test, predictions)

    # Calculate metrics for each label
    metrics = []
    n_labels = y_test.shape[1]

    for label_idx in range(n_labels):
        y_true_label = y_test[:, label_idx]
        y_pred_label = predictions[:, label_idx]

        recall = recall_score(y_true_label, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)

        metrics.append({
            'Model': clf_name,
            'Label': label_names[label_idx],
            'Recall': recall,
            'F1': f1,
            'Hamming Loss': hamming_loss_value
        })

    # Print Hamming Loss
    print(f"Hamming Loss for {clf_name}: {hamming_loss_value}")

    return metrics


def evaluate_deep_learning_model(model, X_test, y_test, model_name, label_names):
    """
    Evaluate the Deep Learning model on the test set.

    Parameters:
    - model: The trained Keras model.
    - X_test (sparse matrix or ndarray): Testing feature matrix.
    - y_test (ndarray): Testing labels.
    - model_name (str): Name of the model for reporting.
    - label_names (list): List of label names.

    Returns:
    - list of dicts: List containing evaluation metrics for each label.
    """
    print(f"\n===== Evaluating {model_name} Model =====")

    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Calculate Hamming Loss
    hamming_loss_value = hamming_loss(y_test, y_pred)

    # Calculate metrics for each label
    metrics = []
    n_labels = y_test.shape[1]

    for label_idx in range(n_labels):
        y_true_label = y_test[:, label_idx]
        y_pred_label = y_pred[:, label_idx]

        recall = recall_score(y_true_label, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)

        metrics.append({
            'Model': model_name,
            'Label': label_names[label_idx],
            'Recall': recall,
            'F1': f1,
            'Hamming Loss': hamming_loss_value
        })

    # Print Hamming Loss
    print(f"Hamming Loss for {model_name} Model: {hamming_loss_value}")

    return metrics


def build_deep_learning_model(input_dim, output_dim):
    """
    Build and compile a Multilayer Perceptron (MLP) model for multi-label classification.

    Parameters:
    - input_dim (int): Number of input features.
    - output_dim (int): Number of output labels.

    Returns:
    - model (Sequential): Compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid'))  # Sigmoid for multi-label classification

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_cnn_model():
    """
    Build and compile a CNN model for text classification.

    Returns:
    - model: Compiled Keras model.
    """
    global vocab_size, embedding_dim, max_len, output_dim
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid'))  # Sigmoid activation for multi-label classification

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# ====================================
# Main Execution
# ====================================

def main(data_type='Unbalanced'):
    """
    Main function to execute data processing, model training, evaluation, and visualization.

    Parameters:
    - data_type (str): Type of data to process ('Unbalanced' or 'Balanced').
    """
    # ----------------------------
    # Load Data
    # ----------------------------
    csv_path = "dataset.csv"
    try:
        if data_type == 'Balanced':
            X_train_df, X_test_df, y_train_df, y_test_df = load_data_balanced(csv_path)
        else:
            X_train_df, X_test_df, y_train_df, y_test_df = load_data(csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    '''
    # ----------------------------
    # Visualizations
    # ----------------------------
    visualize_description_length(X_train_df, data_type)
    visualize_class_distribution(y_train_df, y_test_df, data_type)
    visualize_correlation_matrix(y_train_df, data_type)
    visualize_label_frequency(y_train_df, data_type)
    '''
    # ----------------------------
    # Word Clouds and Vocabulary Collection
    # ----------------------------
    vocab_set = set()  # Initialize an empty set to collect unique words

    for label in LABELS:
        top_words = visualize_word_cloud(X_train_df, y_train_df, label,1000)
        vocab_set.update(top_words)

    wordcloud_vocab = list(vocab_set)
    print(f"\nTotal unique words collected from word clouds: {len(wordcloud_vocab)}")

    # ----------------------------
    # Prepare Data with Vocabulary
    # ----------------------------
    top_k = 1000  # Number of top features to select based on Chi-Square scores
    try:
        X_train_tfidf, X_test_tfidf, selected_features, chi2_scores_max, vector = prepare_data(
            X_train_df, X_test_df, y_train_df, top_k=top_k,vocabulary=wordcloud_vocab
        )
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return

    # ----------------------------
    # Check for Selected Features
    # ----------------------------
    if X_train_tfidf.shape[1] == 0:
        raise ValueError(
            "No features were selected. Consider reducing the 'top_k' parameter or using alternative feature selection methods.")

    # ----------------------------
    # Convert Labels to NumPy Arrays
    # ----------------------------
    y_train_np = y_train_df.to_numpy()
    y_test_np = y_test_df.to_numpy()

    # Get label names
    label_names = y_test_df.columns.tolist()

    # Plot Top Features
    # plot_top_features(selected_features, chi2_scores_max, data_type, top_k_plot=20)

    # ----------------------------
    # Define Classifiers
    # ----------------------------
    clf1 = ClassifierChain(MultinomialNB())
    clf2 = ClassifierChain(LogisticRegression(max_iter=10000))
    clf3 = ClassifierChain(RandomForestClassifier(n_estimators=100, random_state=42))

    # ----------------------------
    # Cross-Validation for Traditional Models
    # ----------------------------
    meth_cv = []
    for clf, model_name in zip([clf1, clf2, clf3], ['MultinomialNB', 'LogisticRegression', 'RandomForest']):
        print(f"\n===== Cross-Validating {model_name} =====")
        cv_scores = cross_validation_score_multilabel(clf, X_train_tfidf, y_train_np)
        meth_cv.append({'Model': model_name, 'Recall': cv_scores['Recall'], 'F1': cv_scores['F1']})
    meth_cv = pd.DataFrame(meth_cv)
    print("\nCross-validation results:")
    print(meth_cv[['Model', 'Recall', 'F1']])

    # ----------------------------
    # Evaluate Classifiers on Test Set
    # ----------------------------
    results_nb = evaluate_classifier(clf1, 'MultinomialNB', X_train_tfidf, y_train_np, X_test_tfidf, y_test_np,
                                     label_names)
    results_lr = evaluate_classifier(clf2, 'LogisticRegression', X_train_tfidf, y_train_np, X_test_tfidf, y_test_np,
                                     label_names)
    results_rf = evaluate_classifier(clf3, 'RandomForest', X_train_tfidf, y_train_np, X_test_tfidf, y_test_np,
                                     label_names)

    # ----------------------------
    # Cross-Validation for Deep Learning Model
    # ----------------------------
    print("\n===== Training and Evaluating Deep Learning Model via Cross-Validation =====")
    deep_learning_cv_scores = cross_validation_score_deep_learning(
        lambda: build_deep_learning_model(X_train_tfidf.shape[1], y_train_np.shape[1]),
        X_train_tfidf.toarray(), y_train_np, n_splits=10, epochs=100, batch_size=16
    )
    print("\nDeep Learning Cross-validation results:")
    print(f"Recall: {deep_learning_cv_scores['Recall']:.4f}")
    print(f"F1-score: {deep_learning_cv_scores['F1']:.4f}")

    # ----------------------------
    # Train Deep Learning Model on Entire Training Set
    # ----------------------------
    print("\n===== Training Deep Learning Model on Entire Training Set =====")
    deep_learning_model = build_deep_learning_model(input_dim=X_train_tfidf.shape[1], output_dim=y_train_np.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    deep_learning_model.fit(
        X_train_tfidf.toarray(),
        y_train_np,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    # ----------------------------
    # Evaluate Deep Learning Model on Test Set
    # ----------------------------
    results_dl = evaluate_deep_learning_model(deep_learning_model, X_test_tfidf, y_test_np, 'MLP', label_names)
    '''
    # ----------------------------
    # Prepare Data for CNN
    # ----------------------------
    X_train_dl, X_test_dl, tokenizer = prepare_data_for_deep_learning(
        X_train_df['report'],
        X_test_df['report'],
        max_words=5000,
        max_len=100
    )

    # Parameters for CNN
    global vocab_size, embedding_dim, max_len, output_dim
    vocab_size = min(len(tokenizer.word_index) + 1, 5000)
    embedding_dim = 100
    max_len = X_train_dl.shape[1]
    output_dim = y_train_np.shape[1]

    # ----------------------------
    # Cross-Validation for CNN Model
    # ----------------------------
    print("\n===== Training and Evaluating CNN Model via Cross-Validation =====")
    cnn_cv_scores = cross_validation_score_deep_learning(
        build_cnn_model, X_train_dl, y_train_np, n_splits=10, epochs=10, batch_size=32
    )
    print("\nCNN Cross-validation results:")
    print(f"Recall: {cnn_cv_scores['Recall']:.4f}")
    print(f"F1-score: {cnn_cv_scores['F1']:.4f}")

    # ----------------------------
    # Train CNN Model on Entire Training Set
    # ----------------------------
    print("\n===== Training CNN Model on Entire Training Set =====")

    cnn_model = build_cnn_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    cnn_model.fit(
        X_train_dl,
        y_train_np,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # ----------------------------
    # Evaluate CNN Model on Test Set
    # ----------------------------
    results_cnn = evaluate_deep_learning_model(cnn_model, X_test_dl, y_test_np, 'CNN', label_names)

    # ----------------------------
    # Combine Results
    # ----------------------------
    '''
    combined_results = results_nb + results_lr + results_rf + results_dl
    df_results = pd.DataFrame(combined_results)

    # Convert 'Hamming Loss' to numeric
    df_results['Hamming Loss'] = pd.to_numeric(df_results['Hamming Loss'], errors='coerce')

    # ----------------------------
    # Visualization of Results
    # ----------------------------
    sns.set(style="whitegrid")
    '''
    # Box plot for F1 Score Distribution
    visualize_f1_scores(df_results,data_type)

    # Box plot comparing Recall, F1-score, and Hamming Loss across all models
    visualize_all_metrics_boxplot(df_results, data_type)

    # Bar plots for Multinomial Naive Bayes metrics
    visualize_nb_metrics(df_results, data_type)
    '''
    print("\nAll visualization processes completed successfully. Plots have been saved.")


# ====================================
# Execute Scripts
# ====================================

if __name__ == "__main__":
    print("\nProcessing with Unbalanced Data.")
    main(data_type='Unbalanced')
    print("\n---------------------------------------------------------")
    print("\nProcessing with Balanced Data.")
    main(data_type='Balanced')