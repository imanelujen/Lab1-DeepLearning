# Deep Learning Lab — PyTorch

## Overview

This project explores deep learning techniques using PyTorch for two supervised learning tasks:

- **Part 1 — Regression:** Predicting the next closing price of a stock using historical stock market data.
- **Part 2 — Multiclass Classification:** Predicting machine failure categories using machine operating conditions.

The project covers data preprocessing, exploratory data analysis (EDA), neural network implementation, hyperparameter tuning, regularization, model evaluation, and visualization of training performance.

## Technologies Used

- Python
- PyTorch
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Imbalanced-learn
- skorch
- Kaggle Notebooks

---

# Part 1 — Stock Price Regression

## 1. Objective

The objective of this part is to build a Deep Neural Network (DNN) capable of predicting the next closing price of a stock based on historical stock market observations.

This is a **regression problem** because the target variable is a continuous numerical value.

## 2. Dataset

**Dataset:** [NYSE — New York Stock Exchange](https://www.kaggle.com/datasets/dgawlik/nyse)

The dataset contains historical stock prices and company-related information.

The file used for the experiment was `prices-split-adjusted.csv`.

### Features

The input features include:

- Opening price
- Closing price
- Lowest price
- Highest price
- Trading volume
- Stock symbol and date for organizing observations

The target variable is `target_next_close`, representing the next recorded closing price for the same stock.

The target was created by shifting the closing price by one observation within each stock symbol.

## 3. Data Preprocessing

The following preprocessing operations were performed:

1. Loaded the dataset and inspected its structure.
2. Sorted observations by stock symbol and date.
3. Created the target variable using the next closing price for each stock.
4. Removed observations for which the next closing price was unavailable.
5. Checked for missing values and examined the feature distributions.
6. Split the dataset into training, validation, and test sets.
7. Applied feature scaling using `StandardScaler`, fitted only on the training data.

The resulting dataset contained approximately 850,000 observations before the final target and feature preparation.

### Exploratory Data Analysis

The EDA examined:

- Stock price distributions
- Summary statistics
- Trading volume and potential outliers
- Temporal trends
- Relationships between input features and the target

The median next closing price was approximately 48.49.

## 4. Model Architecture

A feedforward Deep Neural Network was implemented using PyTorch.

The network consists of:

- An input layer corresponding to the numerical features
- Fully connected hidden layers with ReLU activation
- An output layer producing a single continuous prediction

The model was trained using a regression loss function and an optimizer. Training and validation losses were monitored across epochs.

## 5. Hyperparameter Tuning

Scikit-learn's `GridSearchCV` was used to investigate different configurations.

The search considered:

- Learning rate: `0.001`, `0.01`
- Optimizer: Adam, SGD
- Hidden-layer architecture: `(32,)`, `(64, 32)`
- Number of epochs: `5`

The objective was to identify a configuration that minimized validation error.

## 6. Regularization Experiments

The experiments compared the baseline model with regularized configurations, including L2 regularization and Dropout.

L2 regularization penalizes large weights, while Dropout randomly disables selected activations during training.

The purpose of these experiments was to investigate how regularization influences generalization and overfitting.

## 7. Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)
- Training and validation loss curves

### Reported Results

| Experiment | Best Epoch | Best Validation MSE | MAE | RMSE | R² |
|---|---:|---:|---:|---:|---:|
| Baseline | 5 | 0.0013 | 0.9999 | 2.2208 | 0.9995 |
| L2 | 4 | 0.0014 | 1.0437 | 2.2632 | 0.9995 |
| Dropout | 2 | 0.0017 | — | — | — |

The reported baseline evaluation in an earlier run also produced MAE = 1.1037, RMSE = 2.3980, and R² = 0.9995. The exact evaluation run should be specified in the final report because these results differ from the comparison table.

The results suggest that the baseline performed well on the regression task. The regularization experiments produced slightly higher reported errors, indicating that regularization did not improve every metric in these experiments.

## 8. Conclusion

This part demonstrated how a PyTorch DNN can be applied to stock price regression. Feature scaling, chronological organization, hyperparameter tuning, and evaluation metrics were important parts of the workflow.

The results were strong according to R² and the reported prediction errors. However, predicting stock prices accurately on historical data does not guarantee reliable predictions of future market movements.

---

# Part 2 — Machine Predictive Maintenance Classification

## 1. Objective

The objective of this part is to develop a Deep Neural Network that classifies machine operating conditions into different failure categories.

This is a **multiclass classification problem** involving six classes.

## 2. Dataset

**Dataset:** [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

The dataset contains 10,000 observations and 10 original columns describing machine operating conditions and failure information.

### Input Features

The model uses:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Encoded machine type

The identifier columns `UDI` and `Product ID` were removed.

The binary `Target` column was also excluded from the input features to prevent target leakage when predicting `Failure Type`.

### Target Classes

The six classes are:

1. Heat Dissipation Failure
2. No Failure
3. Overstrain Failure
4. Power Failure
5. Random Failures
6. Tool Wear Failure

## 3. Data Preprocessing

The preprocessing workflow included:

1. Inspecting the dataset structure and data types.
2. Checking missing values and duplicate rows.
3. Removing identifier columns.
4. Encoding the categorical machine type using one-hot encoding.
5. Encoding the target classes using `LabelEncoder`.
6. Creating stratified training, validation, and test sets.
7. Scaling features using `StandardScaler`, fitted only on the training set.
8. Applying random oversampling to the training set to mitigate class imbalance.

### Dataset Split

| Split | Observations |
|---|---:|
| Training | 7,000 |
| Validation | 1,500 |
| Test | 1,500 |

The training set was oversampled from 7,000 to 40,536 observations, giving each of the six classes the same number of training examples.

The validation and test sets retained their original class distributions.

### Class Imbalance

The original dataset is highly imbalanced:

- No Failure: 96.52%
- Heat Dissipation Failure: 1.12%
- Power Failure: 0.95%
- Overstrain Failure: 0.78%
- Tool Wear Failure: 0.45%
- Random Failures: 0.18%

This imbalance makes it difficult for a model to learn rare failure categories.

## 4. DNN Architecture

The baseline model was implemented using PyTorch.

Its architecture consists of:

- Input layer: 8 features
- First hidden layer: 64 neurons with ReLU
- Second hidden layer: 32 neurons with ReLU
- Output layer: 6 neurons, one per class

The model outputs class scores, and `CrossEntropyLoss` is used as the classification loss function.

Adam was used as the optimizer with a learning rate of `0.001` for the baseline and regularization experiments.

## 5. Hyperparameter Tuning

`GridSearchCV` was combined with `skorch` to search neural network configurations.

The tested hyperparameters included:

| Hyperparameter | Values |
|---|---|
| First hidden layer | 32, 64 neurons |
| Second hidden layer | 16, 32 neurons |
| Learning rate | 0.001, 0.01 |
| Optimizer | Adam, SGD |
| Epochs per candidate | 8 |
| Cross-validation folds | 3 |

Macro F1-score was used as the selection metric because it gives equal importance to each class.

### Selected Configuration

The best configuration returned by GridSearchCV was:

- First hidden layer: 32 neurons
- Second hidden layer: 32 neurons
- Optimizer: SGD
- Learning rate: 0.01
- Best cross-validation macro F1-score: approximately 0.1788

The selected configuration did not outperform the baseline in the subsequent test evaluation.

An important limitation is that oversampling was performed before cross-validation. This may allow duplicate observations to appear in separate folds and make the cross-validation score less reliable. A more rigorous workflow would place oversampling inside each training fold.

## 6. Regularization Experiments

Three configurations were compared using the same training, validation, and test splits.

| Experiment | Dropout Rate | Weight Decay |
|---|---:|---:|
| Baseline | 0.0 | 0 |
| L2 Regularization | 0.0 | 0.001 |
| Dropout | 0.3 | 0 |

All three configurations used the same baseline architecture and Adam optimizer with a learning rate of `0.001`.

The model checkpoint with the lowest validation loss was retained for each experiment.

## 7. Evaluation Metrics

The following metrics were used:

- **Accuracy:** proportion of correctly classified observations.
- **Balanced accuracy:** average recall across classes.
- **Precision:** proportion of predicted examples of a class that belong to that class.
- **Recall:** proportion of actual examples of a class correctly identified.
- **Macro F1-score:** average F1-score across classes.
- **Weighted F1-score:** F1-score averaged according to class support.

Macro F1 and balanced accuracy are particularly useful for this dataset because of the severe class imbalance.

## 8. Results

### Regularization Comparison

| Metric | Baseline | L2 | Dropout |
|---|---:|---:|---:|
| Best Epoch | 14 | 28 | 28 |
| Best Validation Loss | 0.2693 | 0.2478 | 0.3732 |
| Test Accuracy | 93.47% | 91.47% | 87.60% |
| Balanced Accuracy | 74.15% | 71.44% | 70.77% |
| Macro F1-score | 54.97% | 49.83% | 45.40% |
| Weighted F1-score | 95.35% | 94.01% | 91.69% |

### Interpretation

The baseline model achieved the highest test accuracy, balanced accuracy, macro F1-score, and weighted F1-score among the three configurations.

L2 regularization achieved the lowest validation loss, but this did not translate into the highest test classification metrics.

Dropout reduced training accuracy and produced lower test performance in this experiment. The dropout rate of 0.3 did not improve the selected metrics.

The results illustrate that regularization does not necessarily improve every performance measure. Its effectiveness depends on the dataset, architecture, and training configuration.

### Minority-Class Performance

All three models failed to correctly identify the two test examples belonging to the Random Failures class.

The baseline also achieved a relatively low F1-score for Tool Wear Failure, despite high overall accuracy.

Consequently, the results should not be interpreted using accuracy alone. Per-class recall, balanced accuracy, and macro F1 provide a more complete picture of the model's limitations.

## 9. Visualizations

The experiments include plots of:

- Feature distributions and exploratory data analysis
- Training and validation loss across epochs
- Training and validation accuracy across epochs
- Model comparison metrics
- Classification performance by failure category

These visualizations help identify learning patterns, compare configurations, and assess potential overfitting.

## 10. Conclusion

This part demonstrated how a PyTorch DNN can be applied to multiclass predictive maintenance.

The preprocessing workflow included categorical encoding, feature scaling, stratified splitting, and training-set oversampling. GridSearchCV was used to explore hyperparameters, and L2 regularization and Dropout were compared with the baseline.

The baseline achieved the strongest overall test metrics in the reported experiments. However, the models struggled to identify some rare failure categories because of the limited number of minority-class examples.

Further improvements could include cross-validation with oversampling performed inside each fold, more extensive hyperparameter tuning, class-weighted loss functions, and additional evaluation on rare failure categories.

---

# General Conclusion

The two parts demonstrate the application of deep learning to different supervised learning tasks.

- **Regression:** predicts a continuous stock price using a neural network.
- **Classification:** predicts one of six machine failure categories using a neural network.

Across both tasks, the experiments highlight the importance of preprocessing, suitable evaluation metrics, hyperparameter tuning, regularization, and careful interpretation of model performance.

## Project Structure

An example repository structure is:

```text
deep-learning-lab/
├── README.md
├── lab-1part1.ipynb
└── lab1-part-2-mcc-predictive-maintenance.ipynb
```

## How to Run

1. Clone or download this repository.
2. Open the notebooks in Kaggle or Jupyter.
3. Install the required Python libraries if they are not already available.
4. Download the datasets from the linked Kaggle pages.
5. Update the dataset paths in the notebooks.
6. Execute the notebook cells in order to reproduce the experiments.

## Author

**Name:** Imane Chikh

**Course:** Deep Learning

**Academic Year:** 2026–2027
