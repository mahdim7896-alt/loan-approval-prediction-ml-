
"""
Project Title: Bank Loan Status Prediction
Description:
    This project aims to predict bank loan approval status (Approved/Rejected) 
    based on customer profile data (e.g., income, education, marital status). 
    It implements an end-to-end Machine Learning pipeline using Logistic Regression.

Pipeline Steps:
    1. Import Libraries & Setup
    2. Data Loading & Initial Exploration (EDA)
    3. Data Preprocessing & Feature Scaling (Imputation, Encoding, Standardization)
    4. Model Training & Prediction (Logistic Regression)
    5. Model Evaluation & Performance Metrics (Confusion Matrix visualization)

Author: Mahdi
Tech Stack: Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
"""

#=========================================================
#----- Import Libraries & Dependencies --------
#=========================================================
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#=========================================================
#----- Data Loading & Initial Exploration --------
#=========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
path_csv = os.path.join(PROJECT_DIR, 'data', 'loan.csv')
df=pd.read_csv(path_csv)
print(df.head())
print(df.dtypes)
df.drop(['Loan_ID'],axis=1,inplace=True)

plt.style.use("seaborn-v0_8")
target = "Loan_Status"
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
str_cols = df.select_dtypes(include=[str]).columns
print(num_cols)
print(str_cols)
print("\nMissing values:")
print(df.isnull().sum())



# plt.figure(figsize=(6, 4))
# sns.countplot(x=target, data=df)
# plt.xlabel("Loan Status")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()

# for col in str_cols:
#     plt.figure(figsize=(8, 4))
#     order = df[col].value_counts().index
#     sns.countplot(x=col, data=df, order=order)
#     plt.title(f"Count Plot of {col}")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# for col in num_cols:
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=target, y=col, data=df)
#     plt.title(f"{col} vs {target}")
#     plt.tight_layout()
#     plt.show()




#===================================================
#----Data Cleaning and Transformation------
#===================================================

for col in str_cols :
   df[col] = df[col].fillna(df[col].mode()[0])


for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

le = LabelEncoder()
for col in str_cols :
    df[col] = le.fit_transform(df[col].astype(str))
# print(df.head())
x=df.drop(['Loan_Status'],axis=1)
scaler = StandardScaler()
x[num_cols] = scaler.fit_transform(x[num_cols])
X_scaled_df = x.copy()

print(X_scaled_df.head())



#===========================================
#----Model Training & Prediction ----
#===========================================
X = X_scaled_df
y=df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)



#===========================================
#----- Model Evaluation ------------
#===========================================

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#=================================================
# --- Model Evaluation & Visualization -----
#=================================================

cm = confusion_matrix(y_test, y_pred)

#---- Plotting a Heatmap -----
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Rejected (0)', 'Approved (1)'], 
            yticklabels=['Rejected (0)', 'Approved (1)'])

plt.title('Confusion Matrix - Loan Prediction')
plt.ylabel('Actual Label (real)')
plt.xlabel('Predicted Label (predicted)')
plt.tight_layout()
 
plt.show()