import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np  
all_data = []

for page in range(1, 5):

    url = f"https://www.jumia.com.eg/mobile-phones/?page={page}"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    products = soup.find_all("article", class_="prd _fb col c-prd")

    for product in products:

        name = product.find("h3", class_="name")
        price = product.find("div", class_="prc")

        if name and price:
            all_data.append([name.text, price.text])

df = pd.DataFrame(all_data, columns=["Product", "Price"])
print(df.head())
#CLEANING
df["Price"] = df["Price"].str.replace("EGP", "")
df["Price"] = df["Price"].str.replace(",", "")
df["Price"] = df["Price"].astype(float)

#OUTLIERS
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df["Price"] < lower) | (df["Price"] > upper)]
print(outliers)
df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]

#SORTING 
print(df.head())
df.sort_values(by="Price", ascending=False).head(10)
df.sort_values(by="Price").head(10)
df["Price"].mean()
print("Max Price:", df["Price"].max())
print("Min Price:", df["Price"].min())
#انا عارف ان ملوش لازمه اوي بس عملته علشان بعدين اقارنه بقيم ال database quires 


#Category 
def price_category(price):
    if price < 8000:
        return 0
    elif price < 20000:
        return 1
    else:
        return 2

df["Label"] = df["Price"].apply(price_category)
def price_category(price):
    if price < 7000:
        return "Low"
    elif price < 15000:
        return "Medium"
    else:
        return "High"

df["Category"] = df["Price"].apply(price_category)
print(df.head())

#VISULISATION
import matplotlib.pyplot as plt
plt.hist(df["Price"], bins=10)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Products")
plt.show()
df["Category"].value_counts().plot(kind="bar")
plt.title("Products by Price Category")
plt.show()


#Classification Problem
median_price = df["Price"].median()
df["Label"] = (df["Price"] > df["Price"].quantile(0.7)).astype(int)

#models traning 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Data Cleaning
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna()
df["Price_log"] = np.log1p(df["Price"])

#Feature 
df["Brand"] = df["Product"].str.split().str[0]
df["NameLength"] = df["Product"].apply(len)
df["Has5G"] = df["Product"].str.contains("5G").astype(int)
df["Has8GB"] = df["Product"].str.contains("8GB").astype(int)
df["Has12GB"] = df["Product"].str.contains("12GB").astype(int)
df["Has128GB"] = df["Product"].str.contains("128GB").astype(int)
df["Has256GB"] = df["Product"].str.contains("256GB").astype(int)
df["Brand"] = df["Product"].str.split().str[0]
df["NameLength"] = df["Product"].apply(len)
df["WordCount"] = df["Product"].apply(lambda x: len(x.split()))
df["IsApple"] = df["Product"].str.contains("Apple", case=False).astype(int)
df["IsSamsung"] = df["Product"].str.contains("Samsung", case=False).astype(int)
df["HasGB"] = df["Product"].str.contains("GB", case=False).astype(int)
df["Price_per_word"] = df["Price"] / df["WordCount"]
df["HasDualSIM"] = df["Product"].str.contains("Dual").astype(int)
df["IsSamsung"] = df["Product"].str.contains("Samsung", case=False).astype(int)
df["IsApple"] = df["Product"].str.contains("Apple", case=False).astype(int)
df["TextLen"] = df["Product"].apply(len)
df["DigitCount"] = df["Product"].str.count(r"\d")

#DATABASE

import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="evil",
        database="jumia_db"
    )
    if conn.is_connected():
        print("Connected to MySQL Successfully")
    cursor = conn.cursor()
except Exception as e:
    print("Connection Error:", e)
    exit()
####################################
cursor.execute("""
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product TEXT,
    price FLOAT,
    category TEXT,
    label INT
)
""")
df_reset = df.reset_index(drop=True)

for _, row in df_reset.iterrows():

    cursor.execute("""
        INSERT INTO products (product, price, category, label)
        VALUES (%s, %s, %s, %s)
    """, (
        row["Product"],
        row["Price"],
        row["Category"],
        row["Label"]
    ))
conn.commit()
product_id = cursor.lastrowid
conn.close()
################################################################
# Encoding
df = pd.get_dummies(df, columns=["Brand"], drop_first=True)
np.random.seed(42)
df["WordCount_noise"] = df["WordCount"] + np.random.normal(0, 0.5, len(df))
feature_cols = [
    "NameLength",
    "WordCount",
    #"Has5G",
    #"Has8GB",
    #"Has12GB",
    #"Has128GB",
    #"Has256GB",
    "IsApple",
    "IsSamsung",
    "HasGB",
    "WordCount_noise"
]
brand_cols = [col for col in df.columns if col.startswith("Brand_")]
X = df[feature_cols + brand_cols]
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model 1: Decision Tree
model_dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model_dt.fit(X_train, y_train)
pred_dt = model_dt.predict(X_test)

#Model 2: Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)

#Model 3: Random Forest
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

# Model Comparison
print("Logistic Regression:", accuracy_score(y_test, pred_lr))
print("Decision Tree:", accuracy_score(y_test, pred_dt))
print("Random Forest:", accuracy_score(y_test, pred_rf))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_rf, X, y, cv=5)
print("Cross Validation Accuracy:", scores.mean())