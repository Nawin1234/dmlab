from google.colab import drive
drive.mount('/content/drive')

# Path to your file in Google Drive
file_path = '/content/drive/MyDrive/Mall_Customers.csv'

# Load the CSV file into a DataFrame
import pandas as pd
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Import necessary libraries
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


print("Data Information:")
print(data.info())
print("\nData Description:")
print(data.describe())


# --- Section 1: Data Preprocessing ---
# Handle missing values by filling them with the mean (if any)
# Encode categorical features
print("\nEncoding Categorical Features...")
label_encoder = LabelEncoder()  # Initialize LabelEncoder
data['Gender'] = label_encoder.fit_transform(data['Gender'])

print("\nHandling Missing Values...")
numerical_columns = data.select_dtypes(include=[np.number]).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Encode categorical features
print("\nEncoding Categorical Features...")
label_encoder = LabelEncoder()  # Initialize LabelEncoder
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Select numerical features for scaling
print("\nScaling Numerical Features...")
numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
# Encode categorical features
print("\nEncoding Categorical Features...")
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

# Select numerical features for scaling
print("\nScaling Numerical Features...")
numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])


# Calculate Euclidean distance between two rows
print("\nEuclidean Distance Example:")
distance = euclidean_distances(data[numerical_features].iloc[:2])
print(distance)

# Calculate Cosine Similarity
print("\nCosine Similarity Example:")
similarity = cosine_similarity(data[numerical_features].iloc[:2])
print(similarity)


# --- Section 3: Clustering ---
# K-means clustering
print("\nK-means Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data[numerical_features])
data['KMeans_Labels'] = kmeans_labels

# Visualize clusters
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=kmeans_labels, cmap='viridis')
plt.title("K-means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# DBSCAN clustering
print("\nDBSCAN Clustering...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data[numerical_features])
data['DBSCAN_Labels'] = dbscan_labels

# Visualize DBSCAN results
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=dbscan_labels, cmap='viridis')
plt.title("DBSCAN Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# --- Section 4: Classification ---
# Decision Tree Classifier
print("\nDecision Tree Classification...")
# Using "Gender" as the target variable for demonstration
target = 'Gender'
X = data[numerical_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Plot decision tree
plt.figure(figsize=(10, 8))
plot_tree(dt, filled=True, feature_names=numerical_features, class_names=['Male', 'Female'])
plt.show()

# Evaluate the model
y_pred = dt.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- Section 5: Statistical Testing ---
# Hypothesis Testing using t-test
print("\nHypothesis Testing...")
group1 = data[data['KMeans_Labels'] == 0]['Annual Income (k$)']
group2 = data[data['KMeans_Labels'] == 1]['Annual Income (k$)']
t_stat, p_value = ttest_ind(group1, group2)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Conclusion based on p-value
if p_value < 0.05:
    print("Statistically significant difference detected.")
else:
    print("No statistically significant difference detected.")


print("\nLab Completed Successfully.")
