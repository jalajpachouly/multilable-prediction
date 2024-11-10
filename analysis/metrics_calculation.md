# Metrics Calculations from Output Data

In this document, we'll provide step-by-step calculations to derive the **Mean Recall**, **Mean F1-Score**, and **Hamming Loss** for each model using the provided output data. This process will help understand how the values in Tables 3 and 4 are obtained.


---

## Unbalanced Dataset Calculations

### 1. Multinomial Naive Bayes (MNB)

#### Per-Fold Metrics:

```
Fold 1: Recall = 0.6426, F1-Score = 0.6405
Fold 2: Recall = 0.6433, F1-Score = 0.6197
Fold 3: Recall = 0.6642, F1-Score = 0.6623
Fold 4: Recall = 0.6354, F1-Score = 0.6344
Fold 5: Recall = 0.6357, F1-Score = 0.6446
Fold 6: Recall = 0.6727, F1-Score = 0.6849
Fold 7: Recall = 0.6307, F1-Score = 0.6333
Fold 8: Recall = 0.6134, F1-Score = 0.5797
Fold 9: Recall = 0.5694, F1-Score = 0.5497
Fold 10: Recall = 0.5917, F1-Score = 0.5709
```

#### Calculating Mean Recall:

\[
\text{Mean Recall} = \frac{\sum_{i=1}^{10} \text{Recall}_i}{10}
\]

\[
\text{Mean Recall} = \frac{0.6426 + 0.6433 + 0.6642 + 0.6354 + 0.6357 + 0.6727 + 0.6307 + 0.6134 + 0.5694 + 0.5917}{10}
\]

\[
\text{Mean Recall} = \frac{6.2991}{10} = 0.6299
\]

#### Calculating Mean F1-Score:

\[
\text{Mean F1-Score} = \frac{\sum_{i=1}^{10} \text{F1-Score}_i}{10}
\]

\[
\text{Mean F1-Score} = \frac{0.6405 + 0.6197 + 0.6623 + 0.6344 + 0.6446 + 0.6849 + 0.6333 + 0.5797 + 0.5497 + 0.5709}{10}
\]

\[
\text{Mean F1-Score} = \frac{6.2200}{10} = 0.6220
\]

#### Hamming Loss:

From the output:
```
Hamming Loss for MultinomialNB: 0.21762589928057555
```

**Summary:**

- **Mean Recall:** 0.6299
- **Mean F1-Score:** 0.6220
- **Hamming Loss:** 0.2176

---

### 2. Logistic Regression (LR)

#### Per-Fold Metrics:

```
Fold 1: Recall = 0.7561, F1-Score = 0.7772
Fold 2: Recall = 0.6756, F1-Score = 0.6941
Fold 3: Recall = 0.7457, F1-Score = 0.7677
Fold 4: Recall = 0.6621, F1-Score = 0.6621
Fold 5: Recall = 0.6758, F1-Score = 0.6872
Fold 6: Recall = 0.7137, F1-Score = 0.7245
Fold 7: Recall = 0.6640, F1-Score = 0.6883
Fold 8: Recall = 0.6378, F1-Score = 0.6406
Fold 9: Recall = 0.6806, F1-Score = 0.6992
Fold 10: Recall = 0.7081, F1-Score = 0.7354
```

#### Calculating Mean Recall:

\[
\text{Mean Recall} = \frac{\sum_{i=1}^{10} \text{Recall}_i}{10} = \frac{6.9195}{10} = 0.6919
\]

#### Calculating Mean F1-Score:

\[
\text{Mean F1-Score} = \frac{\sum_{i=1}^{10} \text{F1-Score}_i}{10} = \frac{7.0763}{10} = 0.7076
\]

#### Hamming Loss:

From the output:
```
Hamming Loss for LogisticRegression: 0.19424460431654678
```

**Summary:**

- **Mean Recall:** 0.6919
- **Mean F1-Score:** 0.7076
- **Hamming Loss:** 0.1942

---

### 3. Random Forest (RF)

#### Per-Fold Metrics:

```
Fold 1: Recall = 0.7557, F1-Score = 0.7778
Fold 2: Recall = 0.6865, F1-Score = 0.7135
Fold 3: Recall = 0.7890, F1-Score = 0.8077
Fold 4: Recall = 0.8029, F1-Score = 0.7724
Fold 5: Recall = 0.6887, F1-Score = 0.7059
Fold 6: Recall = 0.7465, F1-Score = 0.7367
Fold 7: Recall = 0.7450, F1-Score = 0.7759
Fold 8: Recall = 0.6343, F1-Score = 0.6434
Fold 9: Recall = 0.7251, F1-Score = 0.7226
Fold 10: Recall = 0.6763, F1-Score = 0.6920
```

#### Calculating Mean Recall:

\[
\text{Mean Recall} = \frac{\sum_{i=1}^{10} \text{Recall}_i}{10} = \frac{7.2500}{10} = 0.7250
\]

#### Calculating Mean F1-Score:

\[
\text{Mean F1-Score} = \frac{\sum_{i=1}^{10} \text{F1-Score}_i}{10} = \frac{7.3479}{10} = 0.7348
\]

#### Hamming Loss:

From the output:
```
Hamming Loss for RandomForest: 0.1960431654676259
```

**Summary:**

- **Mean Recall:** 0.7250
- **Mean F1-Score:** 0.7348
- **Hamming Loss:** 0.1960

---

### 4. Multilayer Perceptron (MLP)

#### Cross-Validation Results:

```
Recall: 0.7333
F1-score: 0.7395
```

#### Hamming Loss:

From the output:
```
Hamming Loss for MLP Model: 0.18165467625899281
```

**Summary:**

- **Mean Recall:** 0.7333
- **Mean F1-Score:** 0.7395
- **Hamming Loss:** 0.1817

---

### 5. Convolutional Neural Network (CNN)

#### Per-Fold Metrics:

```
Fold 1: Recall = 0.7824, F1-Score = 0.7873
Fold 2: Recall = 0.7881, F1-Score = 0.7745
Fold 3: Recall = 0.8843, F1-Score = 0.8409
Fold 4: Recall = 0.7803, F1-Score = 0.7387
Fold 5: Recall = 0.7260, F1-Score = 0.7430
Fold 6: Recall = 0.8614, F1-Score = 0.8154
Fold 7: Recall = 0.5735, F1-Score = 0.5627
Fold 8: Recall = 0.7977, F1-Score = 0.7765
Fold 9: Recall = 0.7862, F1-Score = 0.7631
Fold 10: Recall = 0.7216, F1-Score = 0.7251
```

#### Calculating Mean Recall:

\[
\text{Mean Recall} = \frac{7.7015}{10} = 0.7702
\]

#### Calculating Mean F1-Score:

\[
\text{Mean F1-Score} = \frac{7.2272}{10} = 0.7227
\]

However, the output reports:

```
CNN Cross-validation results:
Recall: 0.7702
F1-score: 0.7527
```

We'll use the reported **Mean F1-Score** from the output: 0.7527.

#### Hamming Loss:

From the output:

```
Hamming Loss for CNN Model: 0.22032374100719423
```

**Summary:**

- **Mean Recall:** 0.7702
- **Mean F1-Score:** 0.7527
- **Hamming Loss:** 0.2203

---

## Updated Table for Unbalanced Dataset

**Table: Model Performance on Unbalanced Dataset**

| Model                            | Mean Recall | Mean F1-Score | Hamming Loss |
|----------------------------------|-------------|---------------|--------------|
| Convolutional Neural Network (CNN) | 0.7702      | 0.7527        | 0.2203       |
| Random Forest                      | 0.7250      | 0.7348        | 0.1960       |
| Multilayer Perceptron (MLP)        | 0.7333      | 0.7395        | 0.1817       |
| Logistic Regression                | 0.6919      | 0.7076        | 0.1942       |
| Multinomial Naive Bayes            | 0.6299      | 0.6220        | 0.2176       |

---

## Balanced Dataset Calculations

### 1. Multinomial Naive Bayes (MNB)

#### Per-Fold Metrics:

```
Fold 1: Recall = 0.7581, F1-Score = 0.7854
Fold 2: Recall = 0.8041, F1-Score = 0.8128
Fold 3: Recall = 0.7142, F1-Score = 0.7271
Fold 4: Recall = 0.7712, F1-Score = 0.7780
Fold 5: Recall = 0.7463, F1-Score = 0.7534
Fold 6: Recall = 0.7667, F1-Score = 0.7909
Fold 7: Recall = 0.7958, F1-Score = 0.7979
Fold 8: Recall = 0.7644, F1-Score = 0.7732
Fold 9: Recall = 0.8012, F1-Score = 0.8078
Fold 10: Recall = 0.7694, F1-Score = 0.7783
```

#### Calculating Mean Recall:

\[
\text{Mean Recall} = \frac{7.6915}{10} = 0.7691
\]

#### Calculating Mean F1-Score:

\[
\text{Mean F1-Score} = \frac{7.8049}{10} = 0.7805
\]

#### Hamming Loss:

From the output:

```
Hamming Loss for MultinomialNB: 0.191793893129771
```

**Summary:**

- **Mean Recall:** 0.7691
- **Mean F1-Score:** 0.7805
- **Hamming Loss:** 0.1918

---

### 2. Logistic Regression (LR)

#### Per-Fold Metrics:

```
Fold 1: Recall = 0.8121, F1-Score = 0.8057
Fold 2: Recall = 0.8299, F1-Score = 0.8073
Fold 3: Recall = 0.7739, F1-Score = 0.7498
Fold 4: Recall = 0.8237, F1-Score = 0.8095
Fold 5: Recall = 0.7842, F1-Score = 0.7729
Fold 6: Recall = 0.8091, F1-Score = 0.7955
Fold 7: Recall = 0.8327, F1-Score = 0.7916
Fold 8: Recall = 0.8343, F1-Score = 0.8074
Fold 9: Recall = 0.8343, F1-Score = 0.8089
Fold 10: Recall = 0.8498, F1-Score = 0.8184
```

#### Calculating Mean Recall:

\[
\text{Mean Recall} = \frac{8.1841}{10} = 0.8184
\]

#### Calculating Mean F1-Score:

\[
\text{Mean F1-Score} = \frac{7.9669}{10} = 0.7967
\]

#### Hamming Loss:

From the output:

```
Hamming Loss for LogisticRegression: 0.19274809160305342
```

**Summary:**

- **Mean Recall:** 0.8184
- **Mean F1-Score:** 0.7967
- **Hamming Loss:** 0.1927

---

### 3. Random Forest (RF)

#### Per-Fold Metrics:

```
Fold 1: Recall = 0.8584, F1-Score = 0.8596
Fold 2: Recall = 0.8607, F1-Score = 0.8512
Fold 3: Recall = 0.8714, F1-Score = 0.8665
Fold 4: Recall = 0.8659, F1-Score = 0.8650
Fold 5: Recall = 0.8780, F1-Score = 0.8731
Fold 6: Recall = 0.8775, F1-Score = 0.8724
Fold 7: Recall = 0.9300, F1-Score = 0.9176
Fold 8: Recall = 0.8551, F1-Score = 0.8504
Fold 9: Recall = 0.8549, F1-Score = 0.8459
Fold 10: Recall = 0.8925, F1-Score = 0.8807
```

#### Calculating Mean Recall:

\[
\text{Mean Recall} = \frac{8.7444}{10} = 0.8744
\]

#### Calculating Mean F1-Score:

\[
\text{Mean F1-Score} = \frac{8.6824}{10} = 0.8682
\]

#### Hamming Loss:

From the output:

```
Hamming Loss for RandomForest: 0.11450381679389313
```

**Summary:**

- **Mean Recall:** 0.8744
- **Mean F1-Score:** 0.8682
- **Hamming Loss:** 0.1145

---

### 4. Multilayer Perceptron (MLP)

#### Cross-Validation Results:

```
Recall: 0.8334
F1-score: 0.8223
```

#### Hamming Loss:

From the output:

```
Hamming Loss for MLP Model: 0.19370229007633588
```

**Summary:**

- **Mean Recall:** 0.8334
- **Mean F1-Score:** 0.8223
- **Hamming Loss:** 0.1937

---

### 5. Convolutional Neural Network (CNN)

#### Per-Fold Metrics:

```
Fold 1: Recall = 0.9042, F1-Score = 0.9209
Fold 2: Recall = 0.9572, F1-Score = 0.9470
Fold 3: Recall = 0.9183, F1-Score = 0.9203
Fold 4: Recall = 0.9410, F1-Score = 0.9314
Fold 5: Recall = 0.9249, F1-Score = 0.9322
Fold 6: Recall = 0.9319, F1-Score = 0.9465
Fold 7: Recall = 0.9354, F1-Score = 0.9359
Fold 8: Recall = 0.9458, F1-Score = 0.9387
Fold 9: Recall = 0.9565, F1-Score = 0.9394
Fold 10: Recall = 0.9455, F1-Score = 0.9390
```

#### Calculating Mean Recall:

\[
\text{Mean Recall} = \frac{9.3610}{10} = 0.9361
\]

#### Calculating Mean F1-Score:

\[
\text{Mean F1-Score} = \frac{9.3508}{10} = 0.9351
\]

#### Hamming Loss:

From the output:

```
Hamming Loss for CNN Model: 0.10400763358778627
```

**Summary:**

- **Mean Recall:** 0.9361
- **Mean F1-Score:** 0.9351
- **Hamming Loss:** 0.1040

---

## Updated Table for Balanced Dataset

**Table: Model Performance on Balanced Dataset**

| Model                            | Mean Recall | Mean F1-Score | Hamming Loss |
|----------------------------------|-------------|---------------|--------------|
| **Convolutional Neural Network (CNN)** | **0.9361**  | **0.9351**    | 0.1040       |
| Random Forest                      | 0.8744      | 0.8682        | 0.1145       |
| Multilayer Perceptron (MLP)        | 0.8334      | 0.8223        | 0.1937       |
| Logistic Regression                | 0.8184      | 0.7967        | 0.1927       |
| Multinomial Naive Bayes            | 0.7691      | 0.7805        | 0.1918       |

---

## Conclusion

The calculations above provide a step-by-step derivation of the Mean Recall, Mean F1-Score, and Hamming Loss for each model using the output data from your 10-fold cross-validation experiments. These updated values should replace any previous metrics to accurately reflect the performance of each model on both unbalanced and balanced datasets.

