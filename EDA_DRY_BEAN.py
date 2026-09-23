import numpy as np
import pandas as pd
from data_profiling import ProfileReport 
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# 2. Loaded with correct subfolder path
input_data = pd.read_excel('Dry_Bean_Dataset.xlsx')
df = pd.DataFrame(input_data)
#df.info()
pd.set_option('display.max_columns', None)
# 3. OPTIMIZED: Added minimal=True so it takes seconds instead of minutes
#profile = ProfileReport(df, title="Dry Bean Dataset Profiling Report", minimal=True, explorative=True)

# 4. Save the report to an HTML file
#profile.to_file("dry_bean_report.html")
#print("Report generated successfully!")
#print(df.head())
#print(df.describe())
#print(df.isnull().sum())
#print(df.shape)


# 1. Create a folder to save the graphs if it doesn't exist
# ==========================================
# STEP 1: Setup Directory Infrastructure
# ==========================================
"""
base_folder = "eda_visualizations"
univariate_folder = os.path.join(base_folder, "univariate_normality")
bivariate_folder = os.path.join(base_folder, "bivariate_class_analysis")

os.makedirs(univariate_folder, exist_ok=True)
os.makedirs(bivariate_folder, exist_ok=True)

# ==========================================
# STEP 2: Extract Numeric Target Columns
# ==========================================
# Filters down to only columns containing numbers (excludes categorical columns)
numeric_cols = df.select_dtypes(include=["number"]).columns
print(f"Starting EDA Engine. Found {len(numeric_cols)} numeric columns to process...")

# ==========================================
# STEP 3: Automated Processing Engine Loop
# ==========================================
for i, col in enumerate(numeric_cols, 1):
    # Clean the column names for OS filename safety
    clean_col_name = col.replace("/", "_").replace(" ", "_")
    
    # --------------------------------------
    # A. EXECUTE UNIVARIATE ANALYSIS (Normality Verification)
    # --------------------------------------
    # Creates side-by-side subplot canvas
    fig_uni, axes_uni = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left subplot: Histogram + KDE Distribution Curve
    sns.histplot(df[col], kde=True, ax=axes_uni[0], color="royalblue")
    axes_uni[0].set_title(f"Histogram & KDE for {col}", fontsize=11, fontweight="bold")
    axes_uni[0].set_xlabel(col)
    axes_uni[0].set_ylabel("Frequency")
    
    # Right subplot: Quantile-Quantile (Q-Q) Probability Plot
    stats.probplot(df[col], dist="norm", plot=axes_uni[1])
    axes_uni[1].get_lines()[0].set_color("royalblue")     # Data scatter points
    axes_uni[1].get_lines()[1].set_color("firebrick")     # Normal reference baseline
    axes_uni[1].set_title(f"Q-Q Normal Distribution Plot for {col}", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    uni_path = os.path.join(univariate_folder, f"{i:02d}_{clean_col_name}_normality.png")
    plt.savefig(uni_path, dpi=150)
    plt.close(fig_uni)  # Purge figure from memory
    
    # --------------------------------------
    # B. EXECUTE BIVARIATE ANALYSIS (Class Separability)
    # --------------------------------------
    fig_bi = plt.figure(figsize=(12, 6))
    
    # Render boxplot mapping the feature range against the target labels
    sns.boxplot(
        x="Class", 
        y=col, 
        data=df, 
        palette="Set2", 
        fliersize=0  # Disables duplicate outlier dots
    )
    
    # Superimpose high-transparency jitter plot to view genuine point density
    sns.stripplot(
        x="Class", 
        y=col, 
        data=df, 
        color="black", 
        alpha=0.1, 
        jitter=0.2
    )
    
    # Aesthetic adjustments and labeling
    plt.title(f"Bivariate Analysis: Feature '{col}' Stratified by Class", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Target Bean Class", fontsize=11)
    plt.ylabel(col, fontsize=11)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    bi_path = os.path.join(bivariate_folder, f"{i:02d}_{clean_col_name}_vs_Class.png")
    plt.savefig(bi_path, dpi=150)
    plt.close(fig_bi)  # Purge figure from memory
    
    print(f"Processed [{i:02d}/{len(numeric_cols)}]: Visualizations generated for {col}")

print(f"\nSuccess! Visual Audit Complete.")
print(f" -> Check '{univariate_folder}' for normality checks.")
print(f" -> Check '{bivariate_folder}' for separation metrics.")
"""
"""
corr_matrix = df.drop(columns=['Class']).corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix to Identify Redundant Features")
plt.show()

"""
feature_to_drop = ['Perimeter','EquivDiameter','ConvexArea','Eccentricity', 'ShapeFactor3']
df_clean = df.drop(columns=feature_to_drop)
X = df_clean.drop(columns=['Class'])
y = df_clean['Class']
print(X.columns.to_list())
"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', ax=ax)
plt.title("Random Forest Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""
print(X.min())
print(X.max())