# Book Popularity Classification using Decision Trees

This project aims to classify books into **three popularity classes** — `popular`, `average`, and `unpopular` — using a **Decision Tree Classifier**. The classification is based on metadata from a book dataset, including features like average rating, number of pages, and author information.

---

## Objective
To build a classification model that predicts the popularity of books using cleaned and feature-engineered metadata. This can help publishers or users understand which types of books are more likely to be popular.

---

## Data Cleaning
- Removed rows with null values and duplicates (`title`, `authors`)
- Dropped irrelevant columns: `bookID`, `isbn`, `isbn13`, `publisher` etc. 
- Filtered out:
  - Books with fewer than 30 pages
  - Entries with `NOT A BOOK` as author
  - Books with 0 ratings or reviews
  - Books with an average rating below 2.0 and etc.

---

## Feature Engineering
- `book_age`: Derived from publication year (`2025 - year`)
- `is_box_set`: Binary flag for books with more than 1000 pages
- `num_authors`: Counted authors by splitting `authors` column
- `author_type`: Categorized into `solo`, `collab`, and `group`
- `log_` features: Applied log transformation to reduce skewness
  - `log_ratings_count`
  - `log_text_review_count`
  - `log_num_pages`
- `popularity_class`: Defined using `log_ratings_count` as:
  - `< 5.5`: Unpopular
  - `5.5 - 7.5`: Average
  - `> 7.5`: Popular

---

## Modeling Approach
- **Model**: `DecisionTreeClassifier` from scikit-learn
- **Validation**: 10-fold Stratified Cross-Validation
- **Encoding**:
  - Label Encoding for `popularity_class`
  - One-hot encoding for `author_type` (`collab`, `group`)
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix for per-class performance

---

## Results
- **Average Accuracy**: ~83.6%
- **Best predicted class**: Average
- **Most confused class**: Unpopular
- **Insight**: Class 2 (`popular`) is predicted well with low misclassification.
  
---

## Dataset 
- `[books.csv](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks)`: Original raw dataset
---
## Lessons Learned: Handling Data Leakage

This project was very special to me because it helped me discover and understand a critical issue in machine learning — **data leakage**.

### What Happened?
In an earlier version of this project, I made a mistake by:
- **Creating the target label** (`popularity_class`) based on `log_ratings_count`
- But then also using `log_ratings_count` as an input feature for training

This led to **artificially high performance** (accuracy, precision, recall, F1 all near 1.0), because the model was effectively being given the answer during training — a classic case of **data leakage**.

### What I Learned
- Always check for **correlation between your features and labels**, especially if the target was derived from a feature.
- If a feature was used to define the label, it should be **excluded** from the feature set during modeling.
- Realistic performance metrics are critical for building trustworthy models.

This mistake became a powerful learning moment that improved my understanding of feature engineering and model validation.

---
## Conclusion
This project demonstrates the use of basic Decision Tree classifiers on real-world, messy book data with the help of thoughtful cleaning and feature engineering. The model performs well, especially considering the interpretability of Decision Trees.

---
