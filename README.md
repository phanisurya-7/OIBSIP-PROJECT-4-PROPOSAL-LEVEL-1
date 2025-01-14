# OIBSIP_Project-4-Proposal-Level_1

# **Sentiment Analysis Project**

## **Project Overview**
This project focuses on building a **Sentiment Analysis** model to classify text data as positive, negative, or neutral. By analyzing datasets from social media and app reviews, the project provides insights into public opinion, customer feedback, and trends in social media.

---

## **Features**
- **Natural Language Processing (NLP)**: Preprocessing and analyzing text data for sentiment classification.
- **Machine Learning Models**: Implemented algorithms like Naive Bayes and Support Vector Machines (SVM).
- **Data Visualization**: Generated charts and graphs to visualize sentiment distribution.
- **Datasets Used**:
  - [Twitter Sentiment Dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset)
  - [Play Store Dataset](https://www.kaggle.com/datasets/mmmarchetti/play-store-dataset)

---

## **How It Works**
1. **Data Preprocessing**:
   - Cleaned the text by removing stop words and converting text to lowercase.
   - Applied tokenization and vectorization for feature extraction.

2. **Model Training**:
   - Trained a **Multinomial Naive Bayes** model on the Twitter dataset.
   - Split the dataset into training and testing sets to evaluate performance.

3. **Evaluation**:
   - Measured accuracy and other metrics like precision, recall, and F1-score.
   - Visualized sentiment distribution using bar plots and pie charts.

---

## **Installation and Usage**
### **Requirements**
- Python 3.8 or later
- Libraries: `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`

### **Steps to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/phanisurya-7/OIBSIP_Project-4-Proposal-Level_1.git
   ```
2. Navigate to the project folder:
   ```bash
   cd OIBSIP_Project-4-Proposal-Level_1
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python main.py
   ```

---

## **Output**
The project provides:
- Sentiment predictions for text data.
- Accuracy metrics and classification reports.
- Visualized sentiment trends and patterns.

---

## **Contributions**
Feel free to fork the repository, make changes, and submit pull requests!

---
