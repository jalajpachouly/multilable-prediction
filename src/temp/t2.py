# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import recall_score, f1_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelBinarizer

import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from wordcloud import WordCloud

# Initialize NLTK resources
nltk.download('wordnet')

# Suppress warnings
warnings.filterwarnings('ignore')


# ====================================
# Data Loading and Preparation
# ====================================

def load_data(csv_path: str):
    """
    Loads the dataset from a CSV file and splits it into training and testing sets using stratified splitting.

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
    required_columns = ['additional_info', 'type_blocker', 'type_bug', 'type_documentation',
                        'type_enhancement', 'type_regression']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

    # Feature column (text data)
    X = data[['additional_info']]

    # Label columns (multi-label targets)
    y = data[['type_blocker', 'type_bug', 'type_documentation', 'type_enhancement', 'type_regression']]

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


def prepare_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, top_k=50, vocabulary=None):
    """
    Converts text data to TF-IDF features and performs feature selection using the Chi-Square test.

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
    # Initialize the TF-IDF vectorizer with the predefined vocabulary if provided
    vector = TfidfVectorizer(
        ngram_range=(1, 1),
        analyzer='word',
        stop_words='english',
        strip_accents='unicode',
        use_idf=True,
        min_df=1,  # Capture all features appearing at least once
        vocabulary=vocabulary  # <-- Updated to accept predefined vocabulary
    )

    # Apply TF-IDF on the 'additional_info' column
    X_train_tfidf = vector.fit_transform(X_train['additional_info'])
    X_test_tfidf = vector.transform(X_test['additional_info'])

    # Feature selection using Chi-Square test
    chi2_scores = []
    for i in range(y_train.shape[1]):
        chi2_score, p_value = chi2(X_train_tfidf, y_train.iloc[:, i])
        chi2_scores.append(chi2_score)

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


# ====================================
# Visualization Functions
# ====================================

def visualize_description_length(train: pd.DataFrame):
    """Visualizes the distribution of description lengths in the given DataFrame."""
    sns.set(style="darkgrid")
    train['additional_info'] = train['additional_info'].astype(str)
    description_len = train['additional_info'].str.len()
    plt.figure(figsize=(10, 6))
    sns.histplot(description_len, kde=False, bins=20, color="steelblue")
    plt.xlabel('Description Length')
    plt.ylabel('Frequency')
    plt.title('Description Length Distribution')
    plt.tight_layout()
    plt.savefig('description_length_distribution.png')  # Save the plot
    plt.close()
    print("Description length distribution plot saved as 'description_length_distribution.png'.")


def visualize_class_distribution(y_train: pd.DataFrame, y_test: pd.DataFrame):
    """Visualizes the distribution of classes within each label in the given DataFrame."""
    labels = y_train.columns.tolist()

    barWidth = 0.25
    bars1 = [sum(y_train[label] == 1) for label in labels]
    bars2 = [sum(y_train[label] == 0) for label in labels]

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(12, 8))
    plt.bar(r1, bars1, color='steelblue', width=barWidth, label='Labeled = 1')
    plt.bar(r2, bars2, color='lightsteelblue', width=barWidth, label='Labeled = 0')

    plt.xlabel('Labels', fontweight='bold')
    plt.xticks([r + barWidth / 2 for r in range(len(bars1))], labels, rotation=45)
    plt.legend()
    plt.title('Distribution of Classes within Each Label')
    plt.tight_layout()
    plt.savefig('class_distribution.png')  # Save the plot
    plt.close()
    print("Class distribution plot saved as 'class_distribution.png'.")


def W_Cloud(train: pd.DataFrame, token: str, max_words=50):
    """
    Visualize the most common words contributing to the token and return top words.

    Parameters:
    - train (pd.DataFrame): Training set DataFrame including labels.
    - token (str): The label to visualize word cloud for.
    - max_words (int): Number of top words to return.

    Returns:
    - top_words (list): List of top words based on frequency.
    """
    description_context = train[train[token] == 1]
    if description_context.empty:
        print(f"No instances for label '{token}'; skipping word cloud.")
        return []

    description_text = description_context['additional_info']
    combined_text = ' '.join(description_text)
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color='white').generate(combined_text)
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud.recolor(colormap="Blues"), interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Most Common Words Associated with '{token}' Defects", size=20)
    plt.tight_layout()
    plt.savefig(f'wordcloud_{token}.png')  # Save the plot
    plt.close()
    print(f"Word cloud for '{token}' saved as 'wordcloud_{token}.png'.")

    # Extract word frequencies from the word cloud
    word_freq = wordcloud.words_
    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    # Get top N words
    top_words = [word for word, freq in sorted_words[:max_words]]
    return top_words  # Return the top words


def visualize_f1_scores(methods: pd.DataFrame):
    """Visualizes F1 score results through a box plot with jittered points."""
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Model', y='F1', data=methods, palette="Blues")
    sns.stripplot(x='Model', y='F1', data=methods, size=8, jitter=True,
                  edgecolor="gray", linewidth=2, palette="Blues")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    plt.title('F1 Score Distribution by Model')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig('f1_score_distribution.png')  # Save the plot
    plt.close()
    print("F1 score distribution plot saved as 'f1_score_distribution.png'.")


def visualize_all_metrics_boxplot(methods: pd.DataFrame):
    """Creates a box plot comparing Recall, F1-score, and Hamming Loss across all models."""
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
    plt.savefig('all_metrics_comparison_boxplot.png')  # Save the plot
    plt.close()
    print("All metrics comparison box plot saved as 'all_metrics_comparison_boxplot.png'.")


def visualize_nb_metrics(methods: pd.DataFrame):
    """Creates a bar graph of F1 and Recall across each label for Multinomial Naive Bayes."""
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
    plt.savefig('mnb_metrics_per_label.png')  # Save the plot
    plt.close()
    print("Multinomial Naive Bayes metrics per label plot saved as 'mnb_metrics_per_label.png'.")


def visualize_correlation_matrix(y_train: pd.DataFrame):
    """Visualizes the cross-correlation matrix across labels in the given DataFrame."""
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
    plt.savefig('label_correlation_matrix.png')  # Save the plot
    plt.close()
    print("Label correlation matrix plot saved as 'label_correlation_matrix.png'.")


def visualize_label_frequency(y_train: pd.DataFrame):
    """Visualizes the frequency of specified labels in the given DataFrame."""
    labels = ["type_blocker", "type_bug", "type_documentation", "type_enhancement", "type_regression"]
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
    plt.savefig('label_frequency.png')  # Save the plot
    plt.close()
    print("Label frequency plot saved as 'label_frequency.png'.")


def plot_top_features(selected_features, chi2_scores_max, top_k_plot=20):
    """
    Plots the top features based on Chi-Square scores.

    Parameters:
    - selected_features (np.array): Array of selected feature names.
    - chi2_scores_max (np.array): Maximum Chi-Square scores for each feature.
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
    plt.savefig('chi2_features.png')
    plt.close()

    print(f"\nTop {top_k_plot} features have been plotted and saved as 'chi2_features.png'.")


# ====================================
# Model Training and Evaluation
# ====================================

def cross_validation_score_multilabel(classifier, X, y, n_splits=5):
    """
    Performs cross-validation and computes average Recall and F1-score.

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


def cross_validation_score_deep_learning(X, y, n_splits=5, epochs=100, batch_size=16):
    """
    Performs cross-validation for a Deep Learning model and computes average Recall and F1-score.

    Parameters:
    - X (sparse matrix or ndarray): Feature matrix.
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
        X_train_cv, X_test_cv = X[train_index].toarray(), X[test_index].toarray()
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        model = build_deep_learning_model(input_dim=X_train_cv.shape[1], output_dim=y_train_cv.shape[1])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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

        print(f"Fold {fold}: Deep Learning Recall = {recall:.4f}, F1-Score = {f1:.4f}")

    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    return {'Recall': avg_recall, 'F1': avg_f1}


def calculate_cv_scores(clf1, clf2, clf3, X_train, y_train_bin):
    """
    Calculates cross-validation scores for multiple classifiers.

    Parameters:
    - clf1, clf2, clf3: The classifiers to evaluate.
    - X_train (sparse matrix or ndarray): Training feature matrix.
    - y_train_bin (ndarray): Binarized training labels.

    Returns:
    - pd.DataFrame: DataFrame containing cross-validation results.
    """
    methods_cv = []

    for clf, model_name in zip([clf1, clf2, clf3], ['MultinomialNB', 'LogisticRegression', 'RandomForest']):
        print(f"\n===== Cross-Validating {model_name} =====")
        cv_scores = cross_validation_score_multilabel(clf, X_train, y_train_bin)
        methods_cv.append({
            'Model': model_name,
            'Recall': cv_scores['Recall'],
            'F1': cv_scores['F1']
        })

    return pd.DataFrame(methods_cv)


def evaluate_classifier(clf, clf_name, X_train, y_train, X_test, y_test):
    """
    Trains the classifier, makes predictions, and evaluates performance.

    Parameters:
    - clf: The classifier to evaluate.
    - clf_name (str): Name of the classifier.
    - X_train (sparse matrix or ndarray): Training feature matrix.
    - y_train (ndarray): Training labels.
    - X_test (sparse matrix or ndarray): Testing feature matrix.
    - y_test (ndarray): Testing labels.

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
    label_names = [f'Label_{idx}' for idx in range(n_labels)]

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


def evaluate_deep_learning_model(model, X_test, y_test):
    """
    Evaluates the Deep Learning model on the test set.

    Parameters:
    - model: The trained Keras model.
    - X_test (sparse matrix or ndarray): Testing feature matrix.
    - y_test (ndarray): Testing labels.

    Returns:
    - list of dicts: List containing evaluation metrics for each label.
    """
    print(f"\n===== Evaluating Deep Learning Model =====")

    # Convert X_test to dense if it's a sparse matrix
    if isinstance(X_test, (np.matrix, np.ndarray, tf.Tensor)):
        X_test_dense = X_test
    else:
        X_test_dense = X_test.toarray()

    # Make predictions
    y_pred_prob = model.predict(X_test_dense)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Calculate Hamming Loss
    hamming_loss_value = hamming_loss(y_test, y_pred)

    # Calculate metrics for each label
    metrics = []
    n_labels = y_test.shape[1]
    label_names = [f'Label_{idx}' for idx in range(n_labels)]

    for label_idx in range(n_labels):
        y_true_label = y_test[:, label_idx]
        y_pred_label = y_pred[:, label_idx]

        recall = recall_score(y_true_label, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)

        metrics.append({
            'Model': 'DeepLearning',
            'Label': label_names[label_idx],
            'Recall': recall,
            'F1': f1,
            'Hamming Loss': hamming_loss_value
        })

    # Print Hamming Loss
    print(f"Hamming Loss for Deep Learning Model: {hamming_loss_value}")

    return metrics


def build_deep_learning_model(input_dim, output_dim):
    """
    Builds and compiles a Multilayer Perceptron (MLP) model for multi-label classification.

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


# ====================================
# Main Execution
# ====================================

def main():
    """Main function to execute the data processing, model training, evaluation, and visualization."""

    # ----------------------------
    # Load Data
    # ----------------------------
    csv_path = "D:\\Phd-Jalaj\\dataset\\final\\training.csv"  # Update this path if necessary
    try:
        X_train_df, X_test_df, y_train_df, y_test_df = load_data(csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # ----------------------------
    # Visualize Description Length
    # ----------------------------
    visualize_description_length(X_train_df)

    # ----------------------------
    # Visualize Class Distribution
    # ----------------------------
    visualize_class_distribution(y_train_df, y_test_df)

    # ----------------------------
    # Visualize Word Clouds for Each Label and Collect Vocabulary
    # ----------------------------
    labels = ["type_blocker", "type_bug", "type_documentation", "type_enhancement", "type_regression"]
    vocab_set = set()  # Initialize an empty set to collect unique words

    for label in labels:
        top_words = W_Cloud(X_train_df.join(y_train_df), label)  # Ensure 'additional_info' and labels are present
        vocab_set.update(top_words)  # Add the top words to the vocabulary set

    wordcloud_vocab = list(vocab_set)  # Convert the set to a list
    print(f"\nTotal unique words collected from word clouds: {len(wordcloud_vocab)}")

    # ----------------------------
    # Prepare Data with Vocabulary from Word Clouds
    # ----------------------------
    top_k = 50  # Number of top features to select based on Chi-Square scores
    try:
        X_train, X_test, selected_features, chi2_scores_max, vector = prepare_data(
            X_train_df, X_test_df, y_train_df, top_k=top_k, vocabulary=wordcloud_vocab  # <-- Pass the vocabulary here
        )
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return

    # ----------------------------
    # Check for Selected Features
    # ----------------------------
    if X_train.shape[1] == 0:
        raise ValueError(
            "No features were selected. Consider reducing the 'top_k' parameter or using alternative feature selection methods.")

    # ----------------------------
    # Convert Labels to NumPy Arrays
    # ----------------------------
    y_train_np = y_train_df.to_numpy()
    y_test_np = y_test_df.to_numpy()

    # ----------------------------
    # Define Classifiers
    # ----------------------------
    clf1 = ClassifierChain(MultinomialNB())
    clf2 = ClassifierChain(LogisticRegression(max_iter=10000))
    clf3 = ClassifierChain(RandomForestClassifier(n_estimators=100, random_state=42))

    # ----------------------------
    # Cross-Validation for Traditional Models
    # ----------------------------
    meth_cv = calculate_cv_scores(clf1, clf2, clf3, X_train, y_train_np)
    print("\nCross-validation results:")
    print(meth_cv[['Model', 'Recall', 'F1']])

    # ----------------------------
    # Evaluate Classifiers on Test Set
    # ----------------------------
    results_nb = evaluate_classifier(clf1, 'MultinomialNB', X_train, y_train_np, X_test, y_test_np)
    results_lr = evaluate_classifier(clf2, 'LogisticRegression', X_train, y_train_np, X_test, y_test_np)
    results_rf = evaluate_classifier(clf3, 'RandomForest', X_train, y_train_np, X_test, y_test_np)

    # ----------------------------
    # Cross-Validation for Deep Learning Model
    # ----------------------------
    print("\n===== Training and Evaluating Deep Learning Model via Cross-Validation =====")
    deep_learning_cv_scores = cross_validation_score_deep_learning(
        X_train, y_train_np, n_splits=5, epochs=100, batch_size=16
    )
    print("\nDeep Learning Cross-validation results:")
    print(f"Recall: {deep_learning_cv_scores['Recall']:.4f}")
    print(f"F1-score: {deep_learning_cv_scores['F1']:.4f}")

    # ----------------------------
    # Train Deep Learning Model on Entire Training Set
    # ----------------------------
    print("\n===== Training Deep Learning Model on Entire Training Set =====")
    deep_learning_model = build_deep_learning_model(input_dim=X_train.shape[1], output_dim=y_train_np.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    deep_learning_model.fit(
        X_train.toarray(),
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
    results_dl = evaluate_deep_learning_model(deep_learning_model, X_test, y_test_np)

    # ----------------------------
    # Combine Results
    # ----------------------------
    combined_results = results_nb + results_lr + results_rf + results_dl
    df_results = pd.DataFrame(combined_results)

    # Convert 'Hamming Loss' to numeric
    df_results['Hamming Loss'] = pd.to_numeric(df_results['Hamming Loss'], errors='coerce')

    # ----------------------------
    # Visualization of Results
    # ----------------------------
    sns.set(style="whitegrid")

    # Box plot for F1 Score Distribution
    visualize_f1_scores(df_results)

    # Box plot comparing Recall, F1-score, and Hamming Loss across all models
    visualize_all_metrics_boxplot(df_results)

    # Bar plots for each classifier's metrics
    visualize_nb_metrics(df_results)

    # Correlation Matrix of Labels
    visualize_correlation_matrix(y_train_df)

    # Label Frequency
    visualize_label_frequency(y_train_df)

    # Plot Top Features
    plot_top_features(selected_features, chi2_scores_max,  top_k_plot=20)

    print("\nAll visualization processes completed successfully. Plots have been saved.")


if __name__ == "__main__":
    main()