import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set Streamlit page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# App Title
st.title('💳 Credit Card Fraud Detection')

# Description
st.markdown("""
This application uses **Machine Learning** to detect **fraudulent credit card transactions**. 
Upload your transaction dataset (CSV format) and get insights, predictions, and fraud alerts.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your credit card transactions dataset", type=["csv"])

if uploaded_file:
    # Load and show dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Dataset info
    st.subheader("📈 Dataset Info")
    st.markdown(f"- **Shape**: `{df.shape}`")
    class_counts = df['Class'].value_counts()
    st.markdown(f"- **Class Distribution**:\n- Legit: {class_counts[0]}\n- Fraud: {class_counts[1]}")

    # Preprocessing
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("✅ Model Accuracy")
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {accuracy * 100:.2f}%")

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Dynamic figure size based on dataset size
    def dynamic_figsize(num_rows):
        base_height = 4
        base_width = 6
        scale_factor = min(max(num_rows // 1000, 1), 5)
        return (base_width * scale_factor, base_height * scale_factor)

    # Confusion matrix
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=dynamic_figsize(len(df)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Fraud data
    fraud = df[df['Class'] == 1]
    if not fraud.empty:
        st.subheader(f"🚨 Fraudulent Transactions Detected: {len(fraud)}")
        st.dataframe(fraud.head())
        st.warning(f"{len(fraud)} fraudulent transactions found!")

    # Transaction amount distribution
    st.subheader("💰 Distribution of Transaction Amounts (Fraud vs Legit)")
    fig_dist, ax_dist = plt.subplots(figsize=dynamic_figsize(len(df)))
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='green', label='Legit', kde=True, ax=ax_dist)
    sns.histplot(fraud['Amount'], bins=50, color='red', label='Fraud', kde=True, ax=ax_dist)
    ax_dist.set_title('Transaction Amount Distribution')
    ax_dist.legend()
    st.pyplot(fig_dist)

else:
    st.info("Please upload a CSV file to begin.")
