# 🔢 Handwritten Digit Recognition Using Machine Learning

This project focuses on recognizing handwritten digits (**0–9**) from **8x8 grayscale images** using **Machine Learning algorithms**.  
Automatic digit recognition is widely applied in:  
- 📬 Postal automation  
- 🏦 Bank check processing  
- 🔐 Security systems (e.g., CAPTCHA)  

The task is framed as a **multi-class classification problem** where the model predicts one of ten digit classes.

---

## 📌 Table of Contents
- [Dataset Overview](#-dataset-overview)
- [Data Exploration](#-data-exploration)
- [Preprocessing](#-preprocessing)
- [Algorithms Used](#-algorithms-used)
- [Model Training Setup](#-model-training-setup)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Confusion Matrix Insights](#-confusion-matrix-insights)
- [Challenges](#-challenges)
- [Conclusion](#-conclusion)
- [Future Improvements](#-future-improvements)

---

## 📂 Dataset Overview
- **Samples**: 1,797 handwritten digits  
- **Features**: 64 (pixel values of 8x8 images)  
- **Labels**: Digits `0` to `9`  
- **Pixel Values**: Range from `0` (white) to `16` (black)  
- **Balance**: Dataset is evenly distributed across digit classes  
- **Clean Data**: No missing values  

---

## 🔍 Data Exploration
- Pixel intensity distributions visualized for clarity.  
- Dataset is **balanced** and suitable for supervised learning.  
- Strong **visual patterns** exist, allowing ML algorithms to differentiate digits.  

---

## ⚙️ Preprocessing
Steps applied before training:  
- ✨ **Feature Scaling** using `StandardScaler` to normalize pixel values.  
- ✂️ **Train-Test Split** → 80% training, 20% testing for generalization.  

---

## 🤖 Algorithms Used
- **Logistic Regression** → baseline linear classifier  
- **K-Nearest Neighbors (KNN)** → similarity-based classification  
- **Naive Bayes** → probabilistic approach with independence assumption  
- **Decision Tree** → splits based on feature thresholds  
- **Support Vector Machine (SVM)** → linear kernel for class separation  

---

## 🛠 Model Training Setup
- **Logistic Regression** → `max_iter=1000` for convergence  
- **KNN** → 5 nearest neighbors  
- **SVM** → linear kernel (simple, effective)  
- **Decision Tree / Naive Bayes** → default parameters  

All models were trained on **80% training data** and evaluated on **20% unseen test data**.

---

## 📊 Evaluation Metrics
- **Accuracy** → percentage of correct predictions  
- **Precision** → correctness of predicted labels  
- **Recall** → ability to capture actual digits  
- **F1 Score** → balance of precision & recall  

---

## 📈 Results

| Model                | Accuracy | Precision | Recall | F1 Score |
|-----------------------|----------|-----------|--------|----------|
| **KNN**              | **0.975** | **0.976** | **0.977** | **0.976** |
| **SVM (Linear)**     | **0.975** | 0.975     | 0.975  | 0.975    |
| Logistic Regression  | 0.972    | 0.974     | 0.974  | 0.974    |
| Decision Tree        | 0.842    | 0.846     | 0.837  | 0.840    |
| Naive Bayes          | 0.767    | 0.817     | 0.770  | 0.760    |

🏆 **Top Performers** → **KNN** & **SVM** (both ~97.5% accuracy)  
⚡ Logistic Regression is a strong, fast baseline  
⚠️ Decision Tree and Naive Bayes underperform  

---

## 🔢 Confusion Matrix Insights
- **Naive Bayes** → frequently misclassifies `5` as `3` or `8`  
- **Decision Tree** → random misclassifications due to overfitting  

---

## ⚠️ Challenges
- 📉 **Small dataset size** (1,797 samples) → risk of overfitting  
- 👀 **Visually similar digits** (e.g., `3`, `5`, `8`) cause confusion across models  

---

## ✅ Conclusion
- **KNN & SVM** tied as best models (~97.5% accuracy)  
- **Logistic Regression** → efficient, high-performing baseline  
- **Naive Bayes** struggles due to unrealistic independence assumption  
- **Decision Tree** suffers from **generalization issues**  

---

## 🔮 Future Improvements
- 🔧 Use **SVM with RBF kernel** for non-linear separation  
- 📊 Apply **k-fold cross-validation** for robust generalization  
- 📉 Implement **PCA** for dimensionality reduction  
- 🌲 Explore **ensemble models** (Random Forest, Voting Classifier)  
