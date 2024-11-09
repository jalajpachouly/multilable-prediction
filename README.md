# Multi-Label Classification for Bug Reports

This project performs multi-label classification on bug reports using several machine learning models, including traditional classifiers and deep learning models like MLP and CNN. It aims to classify bug reports into multiple categories based on textual descriptions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Introduction

Bug tracking systems often contain reports that need to be categorized into multiple types (e.g., bug, enhancement, documentation). This project explores various machine learning approaches to perform multi-label classification on such reports, comparing traditional models with deep learning techniques.

## Dataset

The dataset should be a CSV file named `data_ml_1.csv` located in the `dataset` directory. It must contain the following columns:

- `report`: The textual description of the bug report.
- `type_blocker`, `type_bug`, `type_documentation`, `type_enhancement`: Binary columns indicating the presence of each label.

## Prerequisites

- Python 3.x
- Required libraries: See `requirements.txt`
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - nltk
    - scikit-learn
    - tensorflow
    - transformers
    - wordcloud
    - iterstrat

## Usage

1. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK resources:**

   The script automatically downloads the necessary NLTK resources (`wordnet` and `stopwords`).

3. **Prepare the Dataset:**

   - Place your dataset CSV file in the `dataset` directory.
   - Ensure it is named `data_ml_1.csv` and contains the required columns.

4. **Run the Script:**

   ```bash
   python main.py
   ```

   Replace `script_name.py` with the actual name of the script.

5. **Output:**

   - The script will process both unbalanced and balanced data.
   - Evaluation metrics and visualizations will be saved in the `output` directory.

## Project Structure

- `main.py`: The main script containing data processing, model training, and evaluation.
- `dataset/`: Directory containing the dataset CSV file.
- `output/`: Directory where all plots and result files will be saved.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

## Results

The script evaluates multiple models:

- **Traditional Machine Learning Models:**
  - Multinomial Naive Bayes
  - Logistic Regression
  - Random Forest

- **Deep Learning Models:**
  - Multilayer Perceptron (MLP)
  - Convolutional Neural Network (CNN)

Evaluation metrics include:

- Recall
- F1-Score
- Hamming Loss

Cross-validation is performed to ensure robust evaluation.

## Visualization

The script generates several plots saved in the `output/` directory:

- **Description Length Distribution**
- **Class Distribution**
- **Correlation Matrix of Labels**
- **Label Frequency**
- **Word Clouds for Each Label**
- **Top Features Based on Chi-Square Scores**
- **F1 Score Distribution by Model**
- **Comparison of Evaluation Metrics Across Models**
- **Metrics Per Label for Multinomial Naive Bayes**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

