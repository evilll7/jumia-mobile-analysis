# Jumia Mobile Price Analysis Project

# Project Overview
This project is a complete data science pipeline that scrapes mobile phone data from Jumia Egypt website, processes and cleans the data, performs exploratory data analysis, engineers new features, and applies machine learning models to predict mobile price categories.

---

# Objective
The goal of this project is to:
- Extract real-world mobile phone data using web scraping
- Clean and preprocess the dataset
- Perform data analysis and visualization
- Build classification models to predict price category

---

# Dataset Source
The data was collected using web scraping from:
- https://www.jumia.com.eg/mobile-phones/

The dataset includes:
- Product name
- Price

---

# Technologies Used
- Python 
- BeautifulSoup (Web Scraping)
- Pandas & NumPy (Data Processing)
- Matplotlib (Visualization)
- Scikit-learn (Machine Learning)
- MySQL (Database Storage)

---

# Data Cleaning
- Removed currency symbols (EGP)
- Converted price to numeric values
- Handled missing values
- Removed outliers using IQR method

---

# Feature Engineering
New features were created such as:
- Brand extraction
- Word count of product name
- Name length
- Detection of 5G, RAM, Storage (8GB, 128GB, etc.)
- Apple / Samsung detection
- Price per word

---

# Data Visualization
- Price distribution histogram
- Product categories distribution bar chart

---

# Machine Learning Models
The following models were trained:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

# Model Evaluation
Models were evaluated using:
- Accuracy Score
- Cross Validation

---

# Database Integration
Data was stored in MySQL database:
Table: products

Columns:
- id
- product
- price
- category
- label

---

# Results Summary
- Multiple ML models were compared
- Random Forest showed the best performance

---

# How to Run Project

```bash
pip install -r requirements.txt
python src/main.py
