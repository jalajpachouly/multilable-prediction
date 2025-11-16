"""
Configurable Multi-Label Classification for Bug Reports

This file takes the EXACT main() function logic from main.py and makes it configurable via JSON.
100% based on the original working code - same imports, same flow, same results.
"""

# Standard library imports
import json
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Third-party imports  
import pandas as pd
import numpy as np
import nltk
import seaborn as sns

# Sklearn imports
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Keras imports
from tensorflow.keras.callbacks import EarlyStopping

# Local imports (EXACTLY as in main.py)
from utils.config import LABELS, DATASET_PATH, TrainingConfig
from utils.data_loading import load_data, load_data_balanced
from utils.feature_engineering import prepare_data, prepare_data_for_deep_learning
from utils.models import build_mlp_model, build_cnn_model
from utils.evaluation import (
    cross_validation_score_multilabel,
    cross_validation_score_deep_learning,
    evaluate_classifier,
    evaluate_deep_learning_model
)
from utils.visualization import (
    visualize_word_cloud,
    visualize_description_length,
    visualize_class_distribution,
    visualize_correlation_matrix,
    visualize_f1_scores,
    plot_top_features
)

# Initialize NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


def load_config(config_path='main_config.json'):
    """Load configuration from JSON file"""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"[OK] Loaded configuration: {config.get('experiment_name', 'Unnamed')}\n")
    return config


def main(data_type='Unbalanced', config=None):
    output_dir = config['output_directory'] if config and 'output_directory' in config else 'output'
    # Feature engineering config
    fe_conf = config['feature_engineering'] if config else {}
    tfidf_conf = fe_conf.get('tfidf', {})
    tfidf_ngram_range = tuple(tfidf_conf.get('ngram_range', [1, 1]))
    tfidf_min_df = tfidf_conf.get('min_df', 1)
    tfidf_use_idf = tfidf_conf.get('use_idf', True)
    use_wordcloud_vocab = fe_conf.get('use_wordcloud_vocabulary', False)
    max_words_per_label = fe_conf.get('max_words_per_label', 50)
    top_k_plot = fe_conf.get('top_k_plot', 20)
    """
    Main function to execute data processing, model training, evaluation, and visualization.
    This is the EXACT main() function from main.py lines 51-205, now made configurable.

    Parameters:
    - data_type (str): Type of data to process ('Unbalanced' or 'Balanced').
    - config (dict): Configuration dictionary loaded from JSON. If None, uses hardcoded defaults.
    """
    # ====================================
    # Configuration Setup
    # ====================================
    training_config = TrainingConfig()
    
    if config is None:
        # Use hardcoded defaults (original main.py behavior)
        csv_path = str(DATASET_PATH)
        sample_size = None
        top_k = 50
        run_visualizations = True
        run_ml = {'MultinomialNB': True, 'LogisticRegression': True, 'RandomForest': True}
        run_dl_mlp = True
        run_dl_cnn = True
        run_cv = True
    else:
        # Use JSON config
        project_root = Path(__file__).parent.parent
        csv_path = str(project_root / config['data']['dataset_path'])
        sample_size = config['data'].get('sample_size')
        top_k = config['feature_engineering'].get('top_k', 50)
        run_visualizations = config['visualizations']['enabled']
        
        # Extract enabled models
        run_ml = {}
        trad_ml = config['models']['traditional_ml']
        run_ml['MultinomialNB'] = trad_ml.get('multinomial_nb', {}).get('enabled', False)
        run_ml['LogisticRegression'] = trad_ml.get('logistic_regression', {}).get('enabled', False)
        run_ml['RandomForest'] = trad_ml.get('random_forest', {}).get('enabled', False)
        
        run_dl_mlp = config['models']['deep_learning']['mlp']['enabled']
        run_dl_cnn = config['models']['deep_learning']['cnn']['enabled']
        run_cv = trad_ml.get('run_cross_validation', False)
        
        # Override training config with JSON values if MLP enabled
        if run_dl_mlp:
            mlp_config = config['models']['deep_learning']['mlp']
            training_config.epochs = mlp_config['epochs']
            training_config.batch_size = mlp_config['batch_size']
            training_config.early_stopping_patience = mlp_config['early_stopping_patience']
            training_config.validation_split = mlp_config['validation_split']
            training_config.n_cv_splits = mlp_config.get('cv_splits', mlp_config.get('cv_n_splits', 10))
    
    # ====================================
    # Load Data (EXACT logic from main.py lines 62-69)
    # ====================================
    try:
        if data_type == 'Balanced':
            X_train_df, X_test_df, y_train_df, y_test_df = load_data_balanced(csv_path, LABELS, sample_size=sample_size)
        else:
            X_train_df, X_test_df, y_train_df, y_test_df = load_data(csv_path, LABELS, sample_size=sample_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # ====================================
    # Visualizations (EXACT logic from main.py lines 71-74)
    # ====================================
    if run_visualizations:
        visualize_description_length(X_train_df, data_type, output_dir)
        visualize_class_distribution(y_train_df, y_test_df, data_type, output_dir)
        visualize_correlation_matrix(y_train_df, data_type, output_dir)

    # ====================================
    # Word Clouds and Vocabulary Collection (EXACT logic from main.py lines 76-82)
    # ====================================
    vocab_set = set()
    wordcloud_vocab = None
    if run_visualizations and (config is None or config['visualizations'].get('word_clouds', True)):
        for label in LABELS:
            top_words = visualize_word_cloud(X_train_df, y_train_df, label, max_words=max_words_per_label, output_dir=output_dir)
            vocab_set.update(top_words)
        if use_wordcloud_vocab:
            wordcloud_vocab = list(vocab_set) if vocab_set else None
        if wordcloud_vocab:
            print(f"\nTotal unique words collected from word clouds: {len(wordcloud_vocab)}")

    # ====================================
    # Prepare Data with Vocabulary (EXACT logic from main.py lines 84-92)
    # ====================================
    try:
        X_train_tfidf, X_test_tfidf, selected_features, chi2_scores_max, vector = prepare_data(
            X_train_df, X_test_df, y_train_df,
            top_k=top_k,
            vocabulary=wordcloud_vocab,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df,
            use_idf=tfidf_use_idf
        )
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return

    if X_train_tfidf.shape[1] == 0:
        raise ValueError("No features were selected.")

    # ====================================
    # Convert Labels to NumPy Arrays (EXACT logic from main.py lines 94-97)
    # ====================================
    y_train_np = y_train_df.to_numpy()
    y_test_np = y_test_df.to_numpy()
    label_names = y_test_df.columns.tolist()

    # ====================================
    # Plot Top Features (EXACT logic from main.py line 100)
    # ====================================
    if run_visualizations:
        plot_top_features(selected_features, chi2_scores_max, data_type, top_k_plot=top_k_plot, output_dir=output_dir)

    # ====================================
    # Define Classifiers (EXACT logic from main.py lines 102-105)
    # Now conditional based on config
    # ====================================
    classifiers = []
    model_names = []
    
    if run_ml.get('MultinomialNB', False):
        nb_conf = trad_ml.get('multinomial_nb', {})
        clf = MultinomialNB()
        if nb_conf.get('use_classifier_chain', True):
            clf = ClassifierChain(clf)
        classifiers.append(clf)
        model_names.append('MultinomialNB')

    if run_ml.get('LogisticRegression', False):
        lr_conf = trad_ml.get('logistic_regression', {})
        clf = LogisticRegression(max_iter=lr_conf.get('max_iter', 10000))
        if lr_conf.get('use_classifier_chain', True):
            clf = ClassifierChain(clf)
        classifiers.append(clf)
        model_names.append('LogisticRegression')

    if run_ml.get('RandomForest', False):
        rf_conf = trad_ml.get('random_forest', {})
        clf = RandomForestClassifier(n_estimators=rf_conf.get('n_estimators', 100), random_state=rf_conf.get('random_state', 42))
        if rf_conf.get('use_classifier_chain', True):
            clf = ClassifierChain(clf)
        classifiers.append(clf)
        model_names.append('RandomForest')

    # ====================================
    # Cross-Validation for Traditional Models (EXACT logic from main.py lines 107-115)
    # ====================================
    meth_cv = []
    if run_cv and classifiers:
        for clf, model_name in zip(classifiers, model_names):
            print(f"\n===== Cross-Validating {model_name} =====")
            cv_scores = cross_validation_score_multilabel(clf, X_train_tfidf, y_train_np)
            meth_cv.append({'Model': model_name, 'Recall': cv_scores['Recall'], 'F1': cv_scores['F1']})
        meth_cv = pd.DataFrame(meth_cv)
        print("\nCross-validation results:")
        print(meth_cv[['Model', 'Recall', 'F1']])

    # ====================================
    # Evaluate Classifiers on Test Set (EXACT logic from main.py lines 117-120)
    # ====================================
    all_results = []
    for clf, model_name in zip(classifiers, model_names):
        results = evaluate_classifier(clf, model_name, X_train_tfidf, y_train_np, X_test_tfidf, y_test_np, label_names)
        all_results.extend(results)

    # ====================================
    # Deep Learning Model - MLP (EXACT logic from main.py lines 122-152)
    # ====================================
    if run_dl_mlp:
        mlp_conf = config['models']['deep_learning']['mlp']
        arch_conf = mlp_conf.get('architecture', {})
        from utils.config import MLPConfig
        mlp_config_obj = MLPConfig(
            hidden_layer_1=arch_conf.get('layer1_units', 256),
            hidden_layer_2=arch_conf.get('layer2_units', 128),
            dropout_rate=arch_conf.get('layer1_dropout', 0.5),
            output_dim=y_train_np.shape[1]
        )
        print("\n===== Training and Evaluating Deep Learning Model via Cross-Validation =====")
        deep_learning_cv_scores = cross_validation_score_deep_learning(
            lambda: build_mlp_model(X_train_tfidf.shape[1], y_train_np.shape[1], config=mlp_config_obj),
            X_train_tfidf.toarray(), y_train_np, 
            n_splits=mlp_conf.get('cv_n_splits', 10), 
            epochs=mlp_conf.get('cv_epochs', mlp_conf.get('epochs', 100)), 
            batch_size=mlp_conf.get('cv_batch_size', mlp_conf.get('batch_size', 16)),
            patience=mlp_conf.get('early_stopping_patience', 5)
        )
        print(f"\nDeep Learning Cross-validation results:")
        print(f"Recall: {deep_learning_cv_scores['Recall']:.4f}")
        print(f"F1-score: {deep_learning_cv_scores['F1']:.4f}")

        # Train Deep Learning Model on Entire Training Set
        print("\n===== Training Deep Learning Model on Entire Training Set =====")
        deep_learning_model = build_mlp_model(input_dim=X_train_tfidf.shape[1], output_dim=y_train_np.shape[1], config=mlp_config_obj)
        early_stop = EarlyStopping(monitor='val_loss', patience=mlp_conf.get('early_stopping_patience', 5), restore_best_weights=True)

        deep_learning_model.fit(
            X_train_tfidf.toarray(),
            y_train_np,
            epochs=mlp_conf.get('epochs', 100),
            batch_size=mlp_conf.get('batch_size', 16),
            validation_split=mlp_conf.get('validation_split', 0.2),
            callbacks=[early_stop],
            verbose=0
        )

        # Evaluate Deep Learning Model on Test Set
        results_dl = evaluate_deep_learning_model(deep_learning_model, X_test_tfidf, y_test_np, 'MLP', label_names)
        all_results.extend(results_dl)

    # ====================================
    # Prepare Data for CNN (EXACT logic from main.py lines 154-160)
    # ====================================
    if run_dl_cnn:
        cnn_conf = config['models']['deep_learning']['cnn']
        X_train_dl, X_test_dl, tokenizer = prepare_data_for_deep_learning(
            X_train_df['report'], X_test_df['report'],
            max_words=cnn_conf.get('max_words', 5000),
            max_len=cnn_conf.get('max_len', 100)
        )

        # Parameters for CNN
        vocab_size = min(len(tokenizer.word_index) + 1, cnn_conf.get('max_words', 5000))
        embedding_dim = cnn_conf.get('embedding_dim', 100)
        max_len = X_train_dl.shape[1]
        output_dim = y_train_np.shape[1]
        conv_filters = cnn_conf.get('conv_filters', 128)
        conv_kernel_size = cnn_conf.get('conv_kernel_size', 5)
        dense_units = cnn_conf.get('dense_units', 128)
        dropout = cnn_conf.get('dropout', 0.5)

        from utils.models import build_cnn_model
        def cnn_builder():
            return build_cnn_model(
                vocab_size, embedding_dim, max_len, output_dim,
                conv_filters=conv_filters,
                conv_kernel_size=conv_kernel_size,
                dense_units=dense_units,
                dropout=dropout
            )

        # Cross-Validation for CNN Model
        print("\n===== Training and Evaluating CNN Model via Cross-Validation =====")
        cnn_cv_scores = cross_validation_score_deep_learning(
            cnn_builder,
            X_train_dl, y_train_np,
            n_splits=cnn_conf.get('cv_n_splits', 10),
            epochs=cnn_conf.get('cv_epochs', cnn_conf.get('epochs', 20)),
            batch_size=cnn_conf.get('cv_batch_size', cnn_conf.get('batch_size', 32)),
            patience=cnn_conf.get('early_stopping_patience', 5)
        )
        print(f"\nCNN Cross-validation results:")
        print(f"Recall: {cnn_cv_scores['Recall']:.4f}")
        print(f"F1-score: {cnn_cv_scores['F1']:.4f}")

        # Train CNN Model on Entire Training Set
        print("\n===== Training CNN Model on Entire Training Set =====")
        cnn_model = cnn_builder()
        early_stop = EarlyStopping(monitor='val_loss', patience=cnn_conf.get('early_stopping_patience', 5), restore_best_weights=True)

        cnn_model.fit(
            X_train_dl, y_train_np,
            epochs=cnn_conf.get('epochs', 20),
            batch_size=cnn_conf.get('batch_size', 32),
            validation_split=cnn_conf.get('validation_split', 0.2),
            callbacks=[early_stop],
            verbose=1
        )

        # Evaluate CNN Model on Test Set
        results_cnn = evaluate_deep_learning_model(cnn_model, X_test_dl, y_test_np, 'CNN', label_names)
        all_results.extend(results_cnn)

    # ====================================
    # Combine Results and Visualize (EXACT logic from main.py lines 184-194)
    # ====================================
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results['Hamming Loss'] = pd.to_numeric(df_results['Hamming Loss'], errors='coerce')

        # Visualization of Results
        if run_visualizations:
            sns.set(style="whitegrid")
            visualize_f1_scores(df_results, data_type, output_dir)

        print("\nAll processes completed successfully.")
        return df_results
    else:
        print("\n[WARNING] No models were run. Check your configuration.")
        return None


if __name__ == "__main__":
    """
    Entry point - reads config and runs experiments.
    This replaces the hardcoded execution from original main.py lines 197-201.
    """
    # Get config file from command line or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'main_config.json'

    # Get config name (without extension and path)
    config_name = Path(config_file).stem
    output_base = Path('output') / config_name
    output_base.mkdir(parents=True, exist_ok=True)

    # Load configuration
    json_config = load_config(config_file)

    # Override output_directory in config to always use output/<config_name>
    json_config['output_directory'] = str(output_base)

    # Determine which data types to run
    data_types = []
    if json_config['data']['run_unbalanced']:
        data_types.append('Unbalanced')
    if json_config['data']['run_balanced']:
        data_types.append('Balanced')

    if not data_types:
        print("[ERROR] No data types enabled in config. Enable run_balanced or run_unbalanced.")
        sys.exit(1)

    # Run experiments (EXACT same structure as original main.py)
    for data_type in data_types:
        print(f"\n{'='*70}")
        print(f"Processing with {data_type} Data")
        print(f"{'='*70}\n")
        results = main(data_type=data_type, config=json_config)

        # Save results if configured
        if results is not None and json_config['output'].get('save_csv', False):
            output_file = output_base / f"results_{data_type.lower()}.csv"
            results.to_csv(output_file, index=False)
            print(f"\n[OK] Results saved to: {output_file}")
