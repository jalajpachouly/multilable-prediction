"""Feature engineering functions."""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, Optional
from scipy.sparse import csr_matrix


def prepare_data(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.DataFrame, 
    top_k: int = 50, 
    vocabulary: Optional[list] = None,
    ngram_range: tuple = (1, 1),
    min_df: int = 1,
    use_idf: bool = True
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, TfidfVectorizer]:
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
        ngram_range=ngram_range,
        analyzer='word',
        stop_words='english',
        strip_accents='unicode',
        use_idf=use_idf,
        min_df=min_df,
        vocabulary=vocabulary
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
    chi2_scores_max = np.max(np.array(chi2_scores), axis=0)

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

    return X_train_selected, X_test_selected, selected_features, chi2_scores_max, vector


def prepare_data_no_selection(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.DataFrame
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Convert text data to TF-IDF features without feature selection.

    Parameters:
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Testing set features.
    - y_train (pd.DataFrame): Training set labels.

    Returns:
    - X_train_tfidf (sparse matrix): Full TF-IDF training features.
    - X_test_tfidf (sparse matrix): Full TF-IDF testing features.
    - selected_features (np.array): All feature names.
    - chi2_scores_max (np.array): Chi-Square scores for all features.
    - vector (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    # Initialize the TF-IDF vectorizer without vocabulary restrictions
    vector = TfidfVectorizer(
        ngram_range=(1, 1),
        analyzer='word',
        stop_words='english',
        strip_accents='unicode',
        use_idf=True,
        min_df=1
    )

    # Apply TF-IDF on the 'report' column
    X_train_tfidf = vector.fit_transform(X_train['report'])
    X_test_tfidf = vector.transform(X_test['report'])
    
    # Calculate Chi-Square scores for reference
    chi2_scores = []
    for i in range(y_train.shape[1]):
        chi2_score_values, p_value = chi2(X_train_tfidf, y_train.iloc[:, i])
        chi2_scores.append(chi2_score_values)

    chi2_scores_max = np.max(np.array(chi2_scores), axis=0)
    selected_features = np.array(vector.get_feature_names_out())
    
    return X_train_tfidf, X_test_tfidf, selected_features, chi2_scores_max, vector


def prepare_data_for_deep_learning(
    X_train_texts: pd.Series, 
    X_test_texts: pd.Series, 
    max_words: int = 5000, 
    max_len: int = 100
) -> Tuple[np.ndarray, np.ndarray, Tokenizer]:
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
