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
import matplotlib.pyplot as plt
from scipy import stats

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
    plot_top_features,
    generate_performance_charts
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


def write_run_metadata(run_folder, run_name, timestamp, data_types, models, config_file=None, hyperparameters=None, charts=None):
    """Persist high level information about a completed run."""
    metadata = {
        "run_name": run_name,
        "timestamp": timestamp,
        "data_types": sorted(set(data_types)),
        "models": sorted(set(models)),
        "config_file": str(config_file) if config_file else None,
        "status": "complete",
    }
    if hyperparameters is not None:
        metadata["hyperparameters"] = hyperparameters
    if charts:
        metadata["charts"] = charts
    meta_path = Path(run_folder) / 'metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as meta_file:
        json.dump(metadata, meta_file, indent=2)


def collect_model_hyperparameters(config, run_ml_flags, run_dl_mlp, run_dl_cnn):
    """Build a list of hyperparameters for the models that are enabled for this run."""
    params = []
    models_conf = config.get('models', {})
    trad_ml = models_conf.get('traditional_ml', {})

    if run_ml_flags.get('RandomForest'):
        rf = trad_ml.get('random_forest', {})
        params.append({
            "model": "RandomForest",
            "family": "Traditional ML",
            "parameters": {
                "n_estimators": rf.get('n_estimators'),
                "random_state": rf.get('random_state'),
                "classifier_chain": rf.get('use_classifier_chain')
            }
        })

    if run_ml_flags.get('LogisticRegression'):
        lr = trad_ml.get('logistic_regression', {})
        params.append({
            "model": "LogisticRegression",
            "family": "Traditional ML",
            "parameters": {
                "max_iter": lr.get('max_iter'),
                "classifier_chain": lr.get('use_classifier_chain')
            }
        })

    if run_ml_flags.get('MultinomialNB'):
        nb = trad_ml.get('multinomial_nb', {})
        params.append({
            "model": "MultinomialNB",
            "family": "Traditional ML",
            "parameters": {
                "classifier_chain": nb.get('use_classifier_chain')
            }
        })

    deep_learning = models_conf.get('deep_learning', {})
    if run_dl_mlp:
        mlp = deep_learning.get('mlp', {})
        arch = mlp.get('architecture', {})
        params.append({
            "model": "MLP",
            "family": "Deep Learning",
            "parameters": {
                "epochs": mlp.get('epochs'),
                "batch_size": mlp.get('batch_size'),
                "validation_split": mlp.get('validation_split'),
                "early_stopping_patience": mlp.get('early_stopping_patience'),
                "layer1_units": arch.get('layer1_units'),
                "layer2_units": arch.get('layer2_units'),
                "layer1_dropout": arch.get('layer1_dropout'),
                "layer2_dropout": arch.get('layer2_dropout')
            }
        })

    if run_dl_cnn:
        cnn = deep_learning.get('cnn', {})
        params.append({
            "model": "CNN",
            "family": "Deep Learning",
            "parameters": {
                "epochs": cnn.get('epochs'),
                "batch_size": cnn.get('batch_size'),
                "validation_split": cnn.get('validation_split'),
                "max_words": cnn.get('max_words'),
                "max_len": cnn.get('max_len'),
                "embedding_dim": cnn.get('embedding_dim'),
                "conv_filters": cnn.get('conv_filters'),
                "conv_kernel_size": cnn.get('conv_kernel_size'),
                "dense_units": cnn.get('dense_units'),
                "dropout": cnn.get('dropout')
            }
        })

    return params


def compute_significance_from_folds(fold_metrics):
    """
    Given a mapping of model -> fold-wise F1 scores, compute paired tests against the best model.
    Returns a dict with best model and comparison rows, or None if not enough data.
    """
    if not fold_metrics:
        return None
    cleaned = {
        model: [float(v) for v in (scores or []) if v is not None and not np.isnan(v)]
        for model, scores in fold_metrics.items()
    }
    cleaned = {m: s for m, s in cleaned.items() if len(s) >= 2}
    if len(cleaned) < 2:
        return None

    best_model, best_scores = max(cleaned.items(), key=lambda kv: np.mean(kv[1]))
    comparisons = []
    for model, scores in cleaned.items():
        if model == best_model:
            continue
        paired_len = min(len(best_scores), len(scores))
        if paired_len < 2:
            continue
        a = np.array(best_scores[:paired_len])
        b = np.array(scores[:paired_len])
        test_used = "Wilcoxon"
        try:
            stat, p_val = stats.wilcoxon(a, b)
        except Exception:
            # Fall back to paired t-test if Wilcoxon cannot run (e.g., constant differences)
            test_used = "Paired t-test"
            _, p_val = stats.ttest_rel(a, b)
        comparisons.append({
            "baseline": model,
            "test": test_used,
            "p_value": float(p_val) if p_val is not None else np.nan,
            "n_pairs": paired_len
        })
    if not comparisons:
        return None
    return {"best_model": best_model, "comparisons": comparisons}


def plot_model_summary_bar(df_results, data_type, output_dir):
    if df_results is None or df_results.empty:
        return None
    summary = df_results.groupby('Model')['F1'].mean().reset_index()
    if summary.empty:
        return None
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x='Model', y='F1', data=summary, palette="Blues_d")
    ax.set_ylim(0, 1)
    ax.set_title(f'Average F1 by Model ({data_type})')
    ax.set_ylabel('Average F1')
    ax.set_xlabel('Model')
    plt.xticks(rotation=20)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chart_path = Path(output_dir) / f"model_f1_summary_{data_type.lower()}.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def plot_combined_model_comparison(all_results, data_types, run_folder):
    frames = []
    for data_type in data_types:
        df = all_results.get(data_type)
        if df is None or df.empty:
            continue
        summary = df.groupby('Model')['F1'].mean().reset_index()
        summary['DataType'] = data_type
        frames.append(summary)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=combined, x='Model', y='F1', hue='DataType')
    ax.set_ylim(0, 1)
    ax.set_title('Average F1 by Model (Balanced vs Unbalanced)')
    plt.xticks(rotation=20)
    plt.ylabel('Average F1')
    plt.xlabel('Model')
    plt.legend(title='Data Type')
    chart_path = run_folder / 'model_f1_comparison.png'
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def plot_metric_comparisons(all_results, data_types, run_folder):
    """Create cross-dataset comparisons for key metrics."""
    metrics = ['F1', 'Recall', 'Hamming Loss']
    chart_paths = {}
    for metric in metrics:
        frames = []
        for data_type in data_types:
            df = all_results.get(data_type)
            if df is None or df.empty or metric not in df.columns:
                continue
            summary = df.groupby('Model')[metric].mean().reset_index()
            summary['DataType'] = data_type
            frames.append(summary)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        plt.figure(figsize=(9, 5))
        ax = sns.barplot(data=combined, x='Model', y=metric, hue='DataType')
        if metric != 'Hamming Loss':
            ax.set_ylim(0, 1)
        ax.set_title(f'Average {metric} by Model (Balanced vs Unbalanced)')
        plt.xticks(rotation=20)
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.legend(title='Data Type')
        chart_path = run_folder / f"{metric.lower().replace(' ', '_')}_by_model_comparison.png"
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        chart_paths[metric] = chart_path
    return chart_paths


def categorize_diagnostic_image(image_name):
    if not image_name:
        return None
    lower = image_name.lower()
    if 'model_f1' in lower or 'f1_comparison' in lower:
        return None
    if 'class_distribution' in lower:
        return "Class Distribution"
    if 'description_length' in lower:
        return "Description Length"
    if 'label_correlation' in lower:
        return "Correlation Matrix"
    if 'wordcloud' in lower:
        return "Word Clouds"
    if 'chi2_features' in lower or 'chi2' in lower:
        return "Feature Importance"
    return None


def main(
    data_type='Unbalanced',
    config=None,
    run_folder=None,
    run_name=None,
    timestamp=None,
    config_path=None,
    generate_report=True,
):
    import shutil
    import datetime
    import glob
    project_root = Path(__file__).parent.parent

    resolved_run_name = run_name or (config.get('experiment_name', 'run') if config else 'run')
    resolved_timestamp = timestamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_folder is None:
        run_folder = project_root / 'output' / 'reports' / f"{resolved_run_name}_{resolved_timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    run_name = resolved_run_name
    timestamp = resolved_timestamp
    output_dir = str(run_folder)
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
    model_hyperparams = []
    performance_charts = []
    fold_metrics = {}
    
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
        model_hyperparams = collect_model_hyperparameters(config, run_ml, run_dl_mlp, run_dl_cnn)
        
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
            if cv_scores.get('fold_f1'):
                fold_metrics[model_name] = cv_scores['fold_f1']
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
        if deep_learning_cv_scores.get('fold_f1'):
            fold_metrics['MLP'] = deep_learning_cv_scores['fold_f1']

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
    # Save metrics/results CSV
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results['Hamming Loss'] = pd.to_numeric(df_results['Hamming Loss'], errors='coerce')
        results_csv = run_folder / f"results_{data_type.lower()}.csv"
        df_results.to_csv(results_csv, index=False)
        # Visualization of Results
        if run_visualizations:
            sns.set(style="whitegrid")
            visualize_f1_scores(df_results, data_type, output_dir)
            performance_charts = generate_performance_charts(df_results, data_type, output_dir)
        print("\nAll processes completed successfully.")
    else:
        print("\n[WARNING] No models were run. Check your configuration.")

    models_run = sorted({r['Model'] for r in all_results}) if all_results else []
    summary_chart_path = plot_model_summary_bar(df_results, data_type, output_dir) if all_results else None
    # Gather images and logs for HTML report or later aggregation
    images = list(run_folder.glob('*.png'))
    # Also include wordclouds and other images from configured output directory
    output_images = glob.glob(str(project_root / config.get('output_directory', 'output/quick_test') / '*.png')) if config else []
    for img_path in output_images:
        img_name = Path(img_path).name
        dest_path = run_folder / img_name
        if not dest_path.exists():
            shutil.copy(img_path, dest_path)
            images.append(dest_path)
    # Find CSVs and logs
    csvs = list(run_folder.glob('*.csv'))
    logs = list(run_folder.glob('*.log'))

    if generate_report:
        # Generate HTML report
        html_path = run_folder / 'report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"<html><head><title>Run Report: {run_name} ({timestamp})</title></head><body>")
            f.write(f"<h1>Run Report: {run_name}</h1>")
            f.write(f"<h3>Timestamp: {timestamp}</h3>")

            f.write("<nav><ol>")
            toc = [
                ("overview", "1. Overview"),
                ("setup", "2. Experiment Setup"),
                ("performance", "3. Model Performance"),
                ("data", "4. Data Diagnostics"),
                ("artifacts", "5. Artifacts")
            ]
            for anchor, label in toc:
                f.write(f"<li><a href='#{anchor}'>{label}</a></li>")
            f.write("</ol></nav>")

            f.write("<section id='overview'>")
            f.write("<h2>1. Overview</h2>")
            if models_run:
                f.write("<p><strong>Models executed:</strong> " + ", ".join(models_run) + "</p>")
            else:
                f.write("<p>No models executed for this configuration.</p>")
            f.write("</section>")

            f.write("<section id='setup'>")
            f.write("<h2>2. Experiment Setup</h2>")
            if model_hyperparams:
                f.write("<table border='1'><tr><th>Model</th><th>Family</th><th>Parameters</th></tr>")
                for entry in model_hyperparams:
                    param_text = '<br>'.join(f"{k}: {v}" for k, v in entry.get('parameters', {}).items())
                    f.write(f"<tr><td>{entry.get('model')}</td><td>{entry.get('family')}</td><td>{param_text}</td></tr>")
                f.write("</table>")
            else:
                f.write("<p>Hyperparameter details were not captured.</p>")
            f.write("</section>")

            def render_img(img, caption):
                f.write(f"<figure><img src='{img}' style='max-width:720px;'><figcaption>{caption}</figcaption></figure><br>")

            f.write("<section id='performance'>")
            f.write("<h2>3. Model Performance</h2>")
            if all_results:
                f.write("<h3>3.1 Metric Table (All Models, Both Data Types)</h3>")
                f.write("<table border='1'><tr><th>Data Type</th><th>Model</th><th>Label</th><th>Recall</th><th>F1</th><th>Hamming Loss</th></tr>")
                for data_type in data_types:
                    results = all_results.get(data_type)
                    if results is None:
                        continue
                    for _, r in results.iterrows():
                        f.write(f"<tr><td>{data_type}</td><td>{r['Model']}</td><td>{r['Label']}</td><td>{r['Recall']:.4f}</td><td>{r['F1']:.4f}</td><td>{r['Hamming Loss']:.4f}</td></tr>")
                f.write("</table>")
                links = []
                for dt in data_types:
                    csv_name = f"results_{dt.lower()}.csv"
                    if (run_folder / csv_name).exists():
                        links.append(f"<a href='{csv_name}' target='_blank'>Download {dt} metrics</a>")
                if links:
                    f.write("<p>" + " | ".join(links) + "</p>")
            else:
                f.write("<p>No metrics available.</p>")

            f.write("<h3>3.2 Statistical Significance Testing</h3>")
            significance = compute_significance_from_folds(fold_metrics)
            if significance:
                f.write("<p>Paired tests on fold-wise macro-F1 (Wilcoxon signed-rank; falls back to paired t-test if ties prevent Wilcoxon).</p>")
                f.write("<table border='1'><tr><th>Best Model</th><th>Baseline</th><th>Test</th><th>p-value</th></tr>")
                significant_baselines = []
                for comp in significance["comparisons"]:
                    p_val = comp.get("p_value")
                    p_text = f"{p_val:.4f}" if p_val is not None and not np.isnan(p_val) else "N/A"
                    if p_val is not None and not np.isnan(p_val) and p_val < 0.05:
                        significant_baselines.append(comp["baseline"])
                    f.write(f"<tr><td>{significance['best_model']}</td><td>{comp['baseline']}</td><td>{comp['test']}</td><td>{p_text}</td></tr>")
                if significant_baselines:
                    f.write(f"<p>{significance['best_model']} outperformed {', '.join(significant_baselines)} (p &lt; 0.05).</p>")
                else:
                    f.write(f"<p>No baseline differences versus {significance['best_model']} were significant at p &lt; 0.05.</p>")
            else:
                f.write("<p>Significance testing skipped (needs cross-validation fold metrics for at least two models).</p>")

            # Cross-dataset model comparison
            if comparison_chart_path or cross_metric_charts:
                f.write("<h3>3.3 Model Comparison: Balanced vs Unbalanced</h3>")
                if comparison_chart_path:
                    render_img(Path(comparison_chart_path).name, "Average F1 by Model (Balanced vs Unbalanced)")
                for metric, chart_path in (cross_metric_charts or {}).items():
                    render_img(Path(chart_path).name, f"Average {metric} by Model (Balanced vs Unbalanced)")

            # Label trends
            f.write("<h3>3.4 Label-Level Trends</h3>")
            plt.figure(figsize=(8,5))
            plotted = False
            for data_type in data_types:
                results = all_results.get(data_type)
                if results is not None and not results.empty:
                    plt.plot(results['Label'], results['F1'], marker='o', label=data_type)
                    plotted = True
            if plotted:
                plt.legend()
                plt.title('F1 Score by Label (Balanced vs Unbalanced)')
                plt.xlabel('Label')
                plt.ylabel('F1 Score')
                img_path = run_folder / 'f1_comparison.png'
                plt.savefig(img_path)
                plt.close()
                render_img('f1_comparison.png', 'F1 Score by Label (Balanced vs Unbalanced)')
                chart_names_for_metadata.append('f1_comparison.png')

            # Per-data-type summaries
            f.write("<h3>3.5 Per Data-Type Model Summaries</h3>")
            for data_type in data_types:
                chart_name = summary_charts.get(data_type)
                if chart_name:
                    render_img(chart_name, f"Average F1 by Model ({data_type})")
                    chart_names_for_metadata.append(chart_name)

            # Per-data-type detailed charts
            for data_type in data_types:
                charts = performance_charts_by_type.get(data_type) or []
                if charts:
                    f.write(f"<h3>3.6 Detailed Performance Views ({data_type})</h3>")
                    caption_map = {
                        "recall_by_model_label": "Recall by Model and Label",
                        "f1_by_model_label": "F1 by Model and Label",
                        "f1_heatmap": "F1 Heatmap",
                        "recall_heatmap": "Recall Heatmap",
                        "hamming_loss_by_model": "Hamming Loss by Model",
                        "f1_trend": "F1 Trend across Labels",
                        "recall_vs_f1": "Recall vs F1 (bubble size ~ 1 - Hamming Loss)"
                    }
                    for chart in charts:
                        chart_names_for_metadata.append(chart)
                        base = Path(chart).stem
                        caption = next((v for k, v in caption_map.items() if k in base), base)
                        render_img(chart, f"{caption} ({data_type})")
            if cross_metric_charts:
                chart_names_for_metadata.extend(Path(p).name for p in cross_metric_charts.values())
            if comparison_chart_path:
                chart_names_for_metadata.append(Path(comparison_chart_path).name)
            f.write("</section>")

            f.write("<section id='data'>")
            f.write("<h2>4. Data Diagnostics</h2>")
            diagnostics = {}
            for img in images:
                category = categorize_diagnostic_image(img.name)
                if category:
                    diagnostics.setdefault(category, []).append(img.name)
            if diagnostics:
                for category, imgs in diagnostics.items():
                    f.write(f"<h4>{category}</h4>")
                    for name in imgs:
                        ds = "Balanced" if "balanced" in name.lower() else ("Unbalanced" if "unbalanced" in name.lower() else None)
                        caption = f"{category}" + (f" ({ds})" if ds else "")
                        render_img(name, caption)
            else:
                f.write("<p>No diagnostic visuals available.</p>")
            f.write("</section>")

            f.write("<section id='artifacts'>")
            f.write("<h2>5. Artifacts</h2>")
            f.write("<h4>Logs</h4>")
            if logs:
                for log in logs:
                    f.write(f"<a href='{log.name}' target='_blank'>{log.name}</a><br>")
            else:
                f.write("<p>No log files generated.</p>")
            f.write("<h4>CSV Outputs</h4>")
            if csvs:
                for csv_file in csvs:
                    f.write(f"<a href='{csv_file.name}' target='_blank'>{csv_file.name}</a><br>")
            else:
                f.write("<p>No CSV outputs created.</p>")
            f.write("</section>")

            f.write("<hr><p>Run complete.</p>")
            f.write("</body></html>")

        charts = []
        if summary_chart_path:
            charts.append(Path(summary_chart_path).name)
        if performance_charts:
            charts.extend([Path(p).name for p in performance_charts])
        charts.extend(chart_names_for_metadata)
        write_run_metadata(run_folder, run_name, timestamp, [data_type], models_run, config_path, hyperparameters={data_type: model_hyperparams}, charts=charts)
        # Mark run as complete
        with open(run_folder / 'COMPLETE.flag', 'w') as flagf:
            flagf.write('complete')

    return (df_results if all_results else None, model_hyperparams, summary_chart_path, performance_charts, fold_metrics)


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

    import io
    import datetime
    import contextlib
    import shutil
    # Prepare run folder
    run_name = json_config.get('experiment_name', 'run')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    project_root = Path(__file__).parent.parent
    run_folder = project_root / 'output' / 'reports' / f"{run_name}_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)

    # Capture stdout/stderr to log file
    log_path = run_folder / 'run.log'
    log_stream = io.StringIO()
    with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
        all_results = {}
        all_feature_k = {}
        all_images = {}
        csv_outputs = {}
        log_outputs = {}
        models_by_type = {}
        combined_models = set()
        hyperparams_by_type = {}
        summary_charts = {}
        performance_charts_by_type = {}
        fold_metrics_by_type = {}
        cross_metric_charts = {}
        for data_type in data_types:
            print(f"\n{'='*70}")
            print(f"Processing with {data_type} Data")
            print(f"{'='*70}\n")
            existing_png = set(run_folder.glob('*.png'))
            existing_csv = set(run_folder.glob('*.csv'))
            existing_logs = set(run_folder.glob('*.log'))

            results_df, hyperparams, summary_chart, perf_charts, fold_metrics = main(
                data_type=data_type,
                config=json_config,
                run_folder=run_folder,
                run_name=run_name,
                timestamp=timestamp,
                config_path=config_file,
                generate_report=False,
            )

            all_results[data_type] = results_df
            hyperparams_by_type[data_type] = hyperparams or []
            fold_metrics_by_type[data_type] = fold_metrics or {}
            summary_charts[data_type] = Path(summary_chart).name if summary_chart else None
            performance_charts_by_type[data_type] = [Path(p).name for p in (perf_charts or [])]
            # Feature selection K data
            if results_df is not None and 'Feature' in results_df.columns and 'Chi2' in results_df.columns:
                all_feature_k[data_type] = results_df[['Feature', 'Chi2']].head(json_config['feature_engineering'].get('top_k', 20))
            # Collect images
            new_images = [p for p in run_folder.glob('*.png') if p not in existing_png]
            all_images[data_type] = new_images
            # Collect csv/log outputs for reference
            new_csvs = [p for p in run_folder.glob('*.csv') if p not in existing_csv]
            new_logs = [p for p in run_folder.glob('*.log') if p not in existing_logs]
            csv_outputs[data_type] = new_csvs
            log_outputs[data_type] = new_logs
            if results_df is not None and 'Model' in results_df.columns:
                models = sorted(results_df['Model'].dropna().unique().tolist())
                models_by_type[data_type] = models
                combined_models.update(models)
            else:
                models_by_type[data_type] = []
        cross_metric_charts = {}
    # Write log file
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(log_stream.getvalue())

    comparison_chart_path = plot_combined_model_comparison(all_results, data_types, run_folder)
    cross_metric_charts = plot_metric_comparisons(all_results, data_types, run_folder)
    chart_names_for_metadata = [Path(p).name for p in cross_metric_charts.values()]
    if comparison_chart_path:
        chart_names_for_metadata.append(Path(comparison_chart_path).name)

    # Generate comparison table and visual
    html_path = run_folder / 'report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"<html><head><title>Run Report: {run_name} ({timestamp})</title></head><body>")
        f.write(f"<h1>Run Report: {run_name}</h1>")
        f.write(f"<h3>Timestamp: {timestamp}</h3>")

        toc = [
            ("overview", "1. Overview"),
            ("setup", "2. Experiment Setup"),
            ("performance", "3. Model Performance"),
            ("data", "4. Data Diagnostics"),
            ("artifacts", "5. Artifacts")
        ]
        f.write("<nav><ol>")
        for anchor, label in toc:
            f.write(f"<li><a href='#{anchor}'>{label}</a></li>")
        f.write("</ol></nav>")

        f.write("<section id='overview'>")
        f.write("<h2>1. Overview</h2>")
        f.write("<ul>")
        for data_type in data_types:
            models = ', '.join(models_by_type.get(data_type, [])) or 'No models executed'
            f.write(f"<li><strong>{data_type}</strong>: {models}</li>")
        f.write("</ul>")
        f.write("</section>")

        f.write("<section id='setup'>")
        f.write("<h2>2. Experiment Setup</h2>")
        for data_type in data_types:
            entries = hyperparams_by_type.get(data_type, [])
            f.write(f"<h4>{data_type}</h4>")
            if entries:
                f.write("<table border='1'><tr><th>Model</th><th>Family</th><th>Parameters</th></tr>")
                for entry in entries:
                    param_text = '<br>'.join(f"{k}: {v}" for k, v in entry.get('parameters', {}).items())
                    f.write(f"<tr><td>{entry.get('model')}</td><td>{entry.get('family')}</td><td>{param_text}</td></tr>")
                f.write("</table>")
            else:
                f.write("<p>No models executed for this scenario.</p>")
        f.write("</section>")

        f.write("<section id='performance'>")
        f.write("<h2>3. Model Performance</h2>")
        f.write("<h3>3.1 Metric Table</h3>")
        f.write("<table border='1'><tr><th>Data Type</th><th>Model</th><th>Label</th><th>Recall</th><th>F1</th><th>Hamming Loss</th></tr>")
        for data_type in data_types:
            results = all_results.get(data_type)
            if results is not None:
                for _, r in results.iterrows():
                    f.write(f"<tr><td>{data_type}</td><td>{r['Model']}</td><td>{r['Label']}</td><td>{r['Recall']:.4f}</td><td>{r['F1']:.4f}</td><td>{r['Hamming Loss']:.4f}</td></tr>")
        f.write("</table>")
        chart_names_for_metadata = []

        f.write("<h3>3.2 Statistical Significance Testing</h3>")
        any_sig = False
        for data_type in data_types:
            sig = compute_significance_from_folds(fold_metrics_by_type.get(data_type))
            if not sig:
                continue
            any_sig = True
            f.write(f"<h4>{data_type}</h4>")
            f.write("<p>Paired tests on fold-wise macro-F1 (Wilcoxon signed-rank; fallback to paired t-test for tied folds).</p>")
            f.write("<table border='1'><tr><th>Best Model</th><th>Baseline</th><th>Test</th><th>p-value</th></tr>")
            significant_baselines = []
            for comp in sig["comparisons"]:
                p_val = comp.get("p_value")
                p_text = f"{p_val:.4f}" if p_val is not None and not np.isnan(p_val) else "N/A"
                if p_val is not None and not np.isnan(p_val) and p_val < 0.05:
                    significant_baselines.append(comp["baseline"])
                f.write(f"<tr><td>{sig['best_model']}</td><td>{comp['baseline']}</td><td>{comp['test']}</td><td>{p_text}</td></tr>")
            if significant_baselines:
                f.write(f"<p>{sig['best_model']} outperformed {', '.join(significant_baselines)} (p &lt; 0.05).</p>")
            else:
                f.write(f"<p>No baseline differences versus {sig['best_model']} were significant at p &lt; 0.05.</p>")
        if not any_sig:
            f.write("<p>Significance testing skipped (requires cross-validation fold metrics for at least two models per data type).</p>")

        f.write("<h3>3.3 F1 Score Trends</h3>")
        plt.figure(figsize=(8,5))
        for data_type in data_types:
            results = all_results.get(data_type)
            if results is not None:
                plt.plot(results['Label'], results['F1'], marker='o', label=data_type)
        plt.legend()
        plt.title('F1 Score Comparison')
        plt.xlabel('Label')
        plt.ylabel('F1 Score')
        img_path = run_folder / 'f1_comparison.png'
        plt.savefig(img_path)
        plt.close()
        chart_names_for_metadata.append('f1_comparison.png')
        f.write(f"<div><img src='f1_comparison.png' style='max-width:600px;'><br>f1_comparison.png</div><br>")
        if comparison_chart_path:
            chart_names_for_metadata.append(Path(comparison_chart_path).name)
            f.write("<h3>3.4 Average F1 by Model</h3>")
            f.write(f"<div><img src='{Path(comparison_chart_path).name}' style='max-width:600px;'><br>{Path(comparison_chart_path).name}</div><br>")
        f.write("<h3>3.5 Per Data-Type Summaries</h3>")
        for data_type in data_types:
            chart_name = summary_charts.get(data_type)
            if chart_name:
                f.write(f"<p><strong>{data_type}</strong></p>")
                f.write(f"<div><img src='{chart_name}' style='max-width:600px;'><br>{chart_name}</div>")
                chart_names_for_metadata.append(chart_name)
        for data_type in data_types:
            charts = performance_charts_by_type.get(data_type) or []
            if charts:
                f.write(f"<h3>3.6 Additional Performance Charts ({data_type})</h3>")
                for chart in charts:
                    chart_names_for_metadata.append(chart)
                    f.write(f"<div><img src='{chart}' style='max-width:600px;'><br>{chart}</div><br>")
        f.write("</section>")

        f.write("<section id='data'>")
        f.write("<h2>4. Data Diagnostics</h2>")
        for data_type in data_types:
            f.write(f"<h3>{data_type}</h3>")
            feature_df = all_feature_k.get(data_type)
            if feature_df is not None and not feature_df.empty:
                f.write("<h4>Top K Features (Chi2)</h4>")
                f.write("<table border='1'><tr><th>Feature</th><th>Chi2 Score</th></tr>")
                for _, row in feature_df.iterrows():
                    f.write(f"<tr><td>{row['Feature']}</td><td>{row['Chi2']:.4f}</td></tr>")
                f.write("</table>")
            diagnostics = {}
            for img in all_images.get(data_type, []):
                category = categorize_diagnostic_image(img.name)
                if category:
                    diagnostics.setdefault(category, []).append(img.name)
            if diagnostics:
                for category, images in diagnostics.items():
                    f.write(f"<h4>{category}</h4>")
                    for name in images:
                        f.write(f"<div><img src='{name}' style='max-width:600px;'><br>{name}</div><br>")
            else:
                f.write("<p>No diagnostic visuals captured.</p>")
        f.write("</section>")

        f.write("<section id='artifacts'>")
        f.write("<h2>5. Artifacts</h2>")
        f.write("<h3>Pipeline Log</h3>")
        f.write("<a href='run.log' target='_blank'>Download/View Log</a><br>")
        f.write("<h3>Results CSVs</h3>")
        for data_type, csv_list in csv_outputs.items():
            if not csv_list:
                f.write(f"<p>No CSV files produced for {data_type}.</p>")
            else:
                f.write(f"<p>{data_type}:</p>")
                for csv_file in csv_list:
                    f.write(f"<a href='{csv_file.name}' target='_blank'>{csv_file.name}</a><br>")
        f.write("<h3>Step Logs</h3>")
        for data_type, logs in log_outputs.items():
            if logs:
                f.write(f"<p>{data_type}:</p>")
                for log in logs:
                    f.write(f"<a href='{log.name}' target='_blank'>{log.name}</a><br>")
        f.write("</section>")

        f.write("<hr><p>Run complete.</p>")
        f.write("</body></html>")

    write_run_metadata(run_folder, run_name, timestamp, data_types, combined_models, config_file, hyperparameters=hyperparams_by_type, charts=chart_names_for_metadata)
    # Mark run as complete
    with open(run_folder / 'COMPLETE.flag', 'w') as flagf:
        flagf.write('complete')
