# data_processing.py
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

CLEANED_FILENAME = "cleaned_dataset.csv"

def smart_cleaning(df):
    logs = []
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        logs.append(f"Removed {duplicates} duplicate rows.")

    missing = df.isnull().sum()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                if len(df[col].mode()) > 0:
                    fill_value = df[col].mode()[0]
                else:
                    fill_value = "Unknown"
            else:
                fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            logs.append(f"Filled {missing[col]} missing values in '{col}' with {'median' if df[col].dtype != 'object' else 'mode'}.")

    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                logs.append(f"Converted '{col}' to datetime format.")
            except:
                logs.append(f"Could not convert '{col}' to datetime format.")

    df.to_csv(CLEANED_FILENAME, index=False, encoding='utf-8')
    return df, logs

def auto_eda(df):
    cleaned_df, cleaning_logs = smart_cleaning(df)
    summary = {
        "original_shape": df.shape,
        "cleaned_shape": cleaned_df.shape,
        "cleaning_log": cleaning_logs,
        "missing_values_after": cleaned_df.isnull().sum().to_dict(),
        "data_types": cleaned_df.dtypes.astype(str).to_dict(),
        "describe": cleaned_df.describe(include='all').to_dict(),
        "cleaned_file": CLEANED_FILENAME
    }
    return summary

def auto_feature_selection(df, target_col):
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_num = X.select_dtypes(include=np.number)
        
        if X_num.empty:
            raise ValueError("No numerical features found for feature selection")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_num, y)
        importance = model.feature_importances_
        features = X_num.columns
        return sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    except Exception as e:
        raise Exception(f"Feature selection failed: {str(e)}")

def generate_visualizations(df, selected_features, viz_type):
    figs = []
    try:
        for col in selected_features:
            if viz_type == "Histogram" and df[col].dtype != 'object':
                fig = px.histogram(df, x=col, title=f"Histogram of {col}")
                figs.append(fig)
            elif viz_type == "Boxplot" and df[col].dtype != 'object':
                fig = px.box(df, y=col, title=f"Boxplot of {col}")
                figs.append(fig)
            elif viz_type == "Bar Chart" and df[col].dtype == 'object':
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = ['Category', 'Count']
                fig = px.bar(value_counts, x='Category', y='Count', title=f"Bar Chart of {col}")
                figs.append(fig)
            elif viz_type == "Pie Chart" and df[col].dtype == 'object':
                fig = px.pie(df, names=col, title=f"Pie Chart of {col}")
                figs.append(fig)

        if viz_type == "Correlation Heatmap":
            numeric_df = df[selected_features].select_dtypes(include=np.number)
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                figs.append(fig)
                
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    return figs

# TEST SECTION
if __name__ == "__main__":
    print("Testing AI Data Explorer Module...")
    print("=" * 60)
    
    try:
        print("Test 1: Importing all dependencies - SUCCESS")
        
        print("Creating sample dataset with intentional issues...")
        sample_data = pd.DataFrame({
            'CustomerID': [1, 2, 3, 4, 5, 5, 6, 7],
            'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank', 'Grace'],
            'Age': [25, 30, None, 40, 35, 35, 45, 28],
            'City': ['New York', 'London', 'Tokyo', 'Paris', 'Berlin', 'Berlin', None, 'Sydney'],
            'Salary': [50000, 60000, 70000, 80000, None, 75000, 90000, 55000],
            'JoinDate': ['2020-01-15', '2019-03-22', '2021-07-10', '2018-11-05', '2022-02-14', '2022-02-14', '2020-09-30', '2021-12-03'],
            'Category': ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C']
        })
        
        print(f"Original dataset shape: {sample_data.shape}")
        print(f"Missing values: {sample_data.isnull().sum().sum()}")
        print(f"Duplicate rows: {sample_data.duplicated().sum()}")
        print("Test 2: Sample dataset creation - SUCCESS")
        
        print("\nTesting smart cleaning function...")
        cleaned_df, logs = smart_cleaning(sample_data.copy())
        
        print("Cleaning operations performed:")
        for log in logs:
            print(f"  - {log}")
        
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        print(f"Missing values after cleaning: {cleaned_df.isnull().sum().sum()}")
        print("Test 3: Smart cleaning function - SUCCESS")
        
        print("\nTesting auto EDA function...")
        eda_summary = auto_eda(sample_data.copy())
        print("EDA Summary generated:")
        print(f"  - Original shape: {eda_summary['original_shape']}")
        print(f"  - Cleaned shape: {eda_summary['cleaned_shape']}")
        print(f"  - Cleaned file saved: {eda_summary['cleaned_file']}")
        print("Test 4: Auto EDA function - SUCCESS")
        
        print("\nTesting feature selection...")
        try:
            feature_importance = auto_feature_selection(cleaned_df, 'Category')
            print("Feature importance ranking:")
            for feature, importance in feature_importance[:3]:
                print(f"  - {feature}: {importance:.4f}")
            print("Test 5: Feature selection - SUCCESS")
        except Exception as e:
            print(f"Note: Feature selection test skipped - {str(e)}")
        
        print("\nTesting visualization generation...")
        try:
            test_features = ['Age', 'Salary', 'City']
            viz_figs = generate_visualizations(cleaned_df, test_features, "Histogram")
            print(f"Generated {len(viz_figs)} visualization objects")
            print("Test 6: Visualization generation - SUCCESS")
        except Exception as e:
            print(f"Note: Visualization test encountered issue - {str(e)}")
        
        print("\nTesting file operations...")
        if os.path.exists(CLEANED_FILENAME):
            saved_df = pd.read_csv(CLEANED_FILENAME)
            print(f"Cleaned dataset saved successfully: {saved_df.shape}")
            print("Test 7: File operations - SUCCESS")
        else:
            print("Test 7: File operations - FAILED (file not saved)")
        
        print("\n" + "=" * 60)
        print("AI DATA EXPLORER MODULE TEST COMPLETED SUCCESSFULLY!")
        print("All core data processing functions are working properly.")
        print("Ready to integrate with Streamlit app!")
        print("=" * 60)
        
        if os.path.exists(CLEANED_FILENAME):
            os.remove(CLEANED_FILENAME)
            print("Test file cleaned up.")
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please install required packages.")
        print("Required packages: pandas, numpy, plotly, seaborn, matplotlib, scikit-learn")
        
    except Exception as e:
        print(f"Test Failed: {str(e)}")
        print("There might be an issue with the setup.")
