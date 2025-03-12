import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import ttest_ind

# Streamlit UI for File Upload
st.title("Customer Segmentation and Analysis")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Data Information
    st.write("### Data Information:")
    st.text(data.info())

    # Data Preprocessing
    st.write("### Data Preprocessing:")
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    # Handle missing values
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # Standardizing the numerical features
    numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Clustering (K-Means)
    st.write("### K-Means Clustering")
    kmeans = KMeans(n_clusters=3, random_state=42)
    data["KMeans_Labels"] = kmeans.fit_predict(data[numerical_features])

    # Visualization
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['KMeans_Labels'], cmap='viridis')
    ax.set_title("K-Means Clustering")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    st.pyplot(fig)

    # Classification (Decision Tree)
    st.write("### Decision Tree Classification")
    target = 'Gender'
    X = data[numerical_features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(dt, filled=True, feature_names=numerical_features, class_names=['Male', 'Female'], ax=ax)
    st.pyplot(fig)

    # Classification Report
    y_pred = dt.predict(X_test)
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Statistical Testing (T-test)
    st.write("### Hypothesis Testing")
    group1 = data[data['KMeans_Labels'] == 0]['Annual Income (k$)']
    group2 = data[data['KMeans_Labels'] == 1]['Annual Income (k$)']
    t_stat, p_value = ttest_ind(group1, group2)

    st.write(f"T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        st.write("Statistically significant difference detected.")
    else:
        st.write("No statistically significant difference detected.")

    st.success("Lab Completed Successfully!")

