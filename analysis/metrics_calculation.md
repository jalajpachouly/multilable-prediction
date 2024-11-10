# Calculation of Model Performance Metrics Based on ML Execution Output

This document provides a detailed calculation of the performance metrics for various models on both balanced and unbalanced datasets, based on the provided machine learning execution outputs. The metrics calculated are **Mean Recall**, **Mean F1-Score**, and **Hamming Loss**, which are presented in Table 3 and Table 4 of the paper.

---

## Table to Check

### Table 3. Model Performance on Balanced Dataset

| Model                               | Mean Recall | Mean F1-Score | Hamming Loss |
|-------------------------------------|-------------|---------------|--------------|
| **Convolutional Neural Network (CNN)** | 0.9559      | 0.9505        | 0.1494       |
| **Random Forest**                   | 0.8803      | 0.8717        | 0.1015       |
| **Multilayer Perceptron (MLP)**     | 0.8533      | 0.8396        | 0.1944       |
| **Logistic Regression**             | 0.8336      | 0.8096        | 0.1925       |
| **Multinomial Naive Bayes**         | 0.7920      | 0.7944        | 0.1772       |

### Table 4. Model Performance on Unbalanced Dataset

| Model                               | Mean Recall | Mean F1-Score | Hamming Loss |
|-------------------------------------|-------------|---------------|--------------|
| **Convolutional Neural Network (CNN)** | 0.7827      | 0.7708        | 0.1864       |
| **Random Forest**                   | 0.7320      | 0.7478        | 0.2016       |
| **Multilayer Perceptron (MLP)**     | 0.7473      | 0.7518        | 0.2231       |
| **Logistic Regression**             | 0.6995      | 0.7155        | 0.2213       |
| **Multinomial Naive Bayes**         | 0.6229      | 0.6095        | 0.2267       |

---

## Calculations Based on ML Execution Output

### Balanced Dataset

#### **Multinomial Naive Bayes**

From the cross-validation results:

- **Fold Recalls**: 0.7718, 0.7913, 0.7933, 0.7729, 0.8572, 0.6962, 0.8424, 0.8305, 0.7650, 0.7992
- **Fold F1-Scores**: 0.7827, 0.7880, 0.7757, 0.7823, 0.8592, 0.7032, 0.8427, 0.8258, 0.7847, 0.8000

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \frac{0.7718 + 0.7913 + \dots + 0.7992}{10} = \mathbf{0.7920}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \frac{0.7827 + 0.7880 + \dots + 0.8000}{10} = \mathbf{0.7944}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.1772**

---

#### **Logistic Regression**

From the cross-validation results:

- **Fold Recalls**: 0.8427, 0.8515, 0.8306, 0.8201, 0.8732, 0.7965, 0.8530, 0.8618, 0.8022, 0.8040
- **Fold F1-Scores**: 0.8092, 0.8244, 0.7884, 0.8001, 0.8599, 0.7665, 0.8331, 0.8384, 0.7891, 0.7864

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.8336}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.8096}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.1925**

---

#### **Random Forest**

From the cross-validation results:

- **Fold Recalls**: 0.8568, 0.8872, 0.8944, 0.8657, 0.8997, 0.8605, 0.8927, 0.8979, 0.8813, 0.8664
- **Fold F1-Scores**: 0.8477, 0.8775, 0.8814, 0.8589, 0.8897, 0.8445, 0.8851, 0.8906, 0.8808, 0.8610

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.8803}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.8717}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.1015**

---

#### **Multilayer Perceptron (MLP)**

From the deep learning cross-validation results:

- **Recalls**: 0.8353, 0.8230, 0.8456, 0.8157, 0.8774, 0.8131, 0.8890, 0.9097, 0.8982, 0.8259
- **F1-Scores**: 0.8202, 0.8150, 0.8165, 0.8211, 0.8649, 0.7906, 0.8909, 0.8780, 0.8823, 0.8168

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.8533}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.8396}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.1944**

---

#### **Convolutional Neural Network (CNN)**

From the CNN cross-validation results:

- **Recalls**: 0.9462, 0.9565, 0.9726, 0.9476, 0.9451, 0.9360, 0.9842, 0.9566, 0.9563, 0.9577
- **F1-Scores**: 0.9258, 0.9524, 0.9542, 0.9528, 0.9434, 0.9266, 0.9846, 0.9519, 0.9522, 0.9609

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.9559}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.9505}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.1494**

---

### Unbalanced Dataset

#### **Multinomial Naive Bayes**

From the cross-validation results:

- **Fold Recalls**: 0.6414, 0.6099, 0.6113, 0.6042, 0.6158, 0.5911, 0.5882, 0.6977, 0.6451, 0.6246
- **Fold F1-Scores**: 0.6323, 0.5886, 0.5831, 0.5758, 0.5897, 0.5753, 0.5656, 0.7093, 0.6528, 0.6230

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.6229}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.6095}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.2267**

---

#### **Logistic Regression**

From the cross-validation results:

- **Fold Recalls**: 0.7486, 0.6686, 0.6600, 0.6725, 0.6929, 0.7153, 0.6276, 0.7885, 0.6997, 0.7216
- **Fold F1-Scores**: 0.7631, 0.6888, 0.6659, 0.6729, 0.7057, 0.7457, 0.6545, 0.8178, 0.7137, 0.7267

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.6995}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.7155}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.2213**

---

#### **Random Forest**

From the cross-validation results:

- **Fold Recalls**: 0.7357, 0.7634, 0.7598, 0.6976, 0.6953, 0.7697, 0.7239, 0.7715, 0.6798, 0.7235
- **Fold F1-Scores**: 0.7474, 0.7657, 0.7652, 0.7261, 0.7180, 0.7670, 0.7468, 0.7973, 0.7053, 0.7390

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.7320}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.7478}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.2016**

---

#### **Multilayer Perceptron (MLP)**

From the deep learning cross-validation results:

- **Recalls**: 0.8357, 0.7631, 0.6986, 0.6995, 0.7120, 0.7621, 0.7083, 0.8192, 0.7192, 0.7554
- **F1-Scores**: 0.8295, 0.7685, 0.7127, 0.7166, 0.7295, 0.7472, 0.7125, 0.8242, 0.7134, 0.7638

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.7473}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.7518}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.2231**

---

#### **Convolutional Neural Network (CNN)**

From the CNN cross-validation results:

- **Recalls**: 0.7587, 0.8604, 0.7243, 0.7501, 0.8100, 0.8586, 0.7636, 0.7259, 0.7842, 0.7916
- **F1-Scores**: 0.7677, 0.8321, 0.7025, 0.7486, 0.7881, 0.8162, 0.7611, 0.7392, 0.7498, 0.8025

Calculating **Mean Recall**:

\[
\text{Mean Recall} = \frac{1}{10} \sum_{i=1}^{10} \text{Recall}_i = \mathbf{0.7827}
\]

Calculating **Mean F1-Score**:

\[
\text{Mean F1-Score} = \frac{1}{10} \sum_{i=1}^{10} \text{F1}_i = \mathbf{0.7708}
\]

**Hamming Loss**:

- Reported Hamming Loss: **0.1864**

---

## Conclusion

The calculated **Mean Recall**, **Mean F1-Score**, and **Hamming Loss** for each model on both balanced and unbalanced datasets are presented in Table 3 and Table 4. These values are derived directly from the cross-validation and evaluation outputs provided, demonstrating the models' performance in the multi-label classification task.

---