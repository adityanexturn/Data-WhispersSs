import streamlit as st
import pandas as pd
import numpy as np
from data_processing import smart_cleaning, auto_eda, auto_feature_selection, generate_visualizations
from rag_module import RAGModule
import os

# Page configuration
st.set_page_config(
    page_title="Data WhisPersSs",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center; 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #1f77b4; 
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Question box styling */
    .question-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #6495ED;
    }
    
    /* Answer box styling */
    .answer-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #50C878;
    }
    
    /* Label styling within boxes */
    .box-label {
        font-weight: bold;
        color: #333333;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }
    
    /* Text styling within boxes */
    .box-text {
        color: #424242;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 1px solid #cccccc;
        padding: 10px 15px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'cleaning_logs' not in st.session_state:
    st.session_state.cleaning_logs = []
if 'rag_module' not in st.session_state:
    st.session_state.rag_module = None
if 'rag_ready' not in st.session_state:
    st.session_state.rag_ready = False
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

# Centered title
st.markdown("<h1 class='main-title'>Data WhisPersSs</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
tabs = ["Upload & Preview", "Data Cleaning", "Feature Selection & Visualization", "AI Assistant"]
selected_tab = st.sidebar.radio("Choose a section:", tabs)

# Exit button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Exit Application", type="secondary"):
    st.sidebar.success("Application closed. You can close this browser tab.")
    st.stop()

# Groq API Key input in sidebar
st.sidebar.subheader("Configuration")

try:
    default_api_key = st.secrets.get("GROQ_API_KEY", "")
except (AttributeError, FileNotFoundError):
    default_api_key = ""

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key:", 
    value=default_api_key,
    type="password",
    help="API key can be loaded from Streamlit secrets."
)

if groq_api_key and st.session_state.rag_module is None:
    try:
        with st.spinner("Initializing AI module..."):
            st.session_state.rag_module = RAGModule(groq_api_key=groq_api_key)
        st.sidebar.success("AI module configured successfully!")
    except Exception as e:
        st.sidebar.error(f"Error initializing AI module: {str(e)}")

# Helper function to read CSV
def read_csv_with_encoding(uploaded_file):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue
    return None

# Tab 1: Upload & Preview
if selected_tab == "Upload & Preview":
    st.markdown("<h3>Upload & Preview Dataset</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt'],
        help="Upload a CSV or TXT file for analysis"
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_type == 'csv':
                df = read_csv_with_encoding(uploaded_file)
                if df is not None:
                    st.session_state.uploaded_data = df
                    st.session_state.file_type = "csv"
                    st.success(f"Successfully uploaded CSV file with {df.shape[0]} rows and {df.shape[1]} columns")
                    
                    st.subheader("Dataset Preview")
                    st.dataframe(df.head(10))
                    
                    st.subheader("Dataset Info")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", df.shape[0])
                    with col2:
                        st.metric("Total Columns", df.shape[1])
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    
                    st.subheader("Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes,
                        'Missing Values': df.isnull().sum(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(col_info)
                else:
                    st.error("Could not read the CSV file. Please check its format and encoding.")
            
            elif file_type == 'txt':
                text_content = uploaded_file.read().decode('utf-8', errors='ignore')
                st.session_state.uploaded_data = text_content
                st.session_state.file_type = "txt"
                st.success(f"Successfully uploaded text file with {len(text_content)} characters")
                
                st.subheader("Text Preview")
                st.text_area("File Content (first 1000 characters):", text_content[:1000], height=200)
                
                st.subheader("Text Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Characters", len(text_content))
                with col2:
                    st.metric("Total Words", len(text_content.split()))
                with col3:
                    st.metric("Total Lines", len(text_content.split('\n')))

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Tab 2: Data Cleaning
elif selected_tab == "Data Cleaning":
    st.header("Data Cleaning")
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload a dataset first in the 'Upload & Preview' tab")
    elif st.session_state.file_type != "csv":
        st.info("Data cleaning is available only for CSV files. Text files will be processed directly.")
    else:
        df = st.session_state.uploaded_data
        
        st.subheader("Original Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Rows", df.shape[0])
        with col2:
            st.metric("Original Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        if st.button("Clean Data", type="primary"):
            with st.spinner("Cleaning data..."):
                try:
                    cleaned_df, logs = smart_cleaning(df.copy())
                    st.session_state.cleaned_data = cleaned_df
                    st.session_state.cleaning_logs = logs
                    st.success("Data cleaning completed!")
                except Exception as e:
                    st.error(f"Error during data cleaning: {str(e)}")
        
        if st.session_state.cleaned_data is not None:
            cleaned_df = st.session_state.cleaned_data
            
            st.subheader("Cleaning Operations Performed")
            for log in st.session_state.cleaning_logs:
                st.write(f"- {log}")
            
            st.subheader("Cleaned Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cleaned Rows", cleaned_df.shape[0])
            with col2:
                st.metric("Cleaned Columns", cleaned_df.shape[1])
            with col3:
                st.metric("Missing Values", cleaned_df.isnull().sum().sum())
            
            st.subheader("Cleaned Dataset Preview")
            st.dataframe(cleaned_df.head(10))
            
            try:
                csv = cleaned_df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="Download Cleaned Dataset",
                    data=csv,
                    file_name='cleaned_dataset.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Error preparing download: {str(e)}")

# Tab 3: Feature Selection & Visualization
elif selected_tab == "Feature Selection & Visualization":
    st.header("Feature Selection & Visualization")
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload a dataset first")
    elif st.session_state.file_type != "csv":
        st.info("Feature selection and visualization are available only for CSV files")
    elif st.session_state.cleaned_data is None:
        st.warning("Please clean your data first in the 'Data Cleaning' tab")
    else:
        cleaned_df = st.session_state.cleaned_data
        
        st.subheader("Feature Selection")
        
        selected_features = st.multiselect(
            "Select features for visualization:",
            options=cleaned_df.columns.tolist(),
            default=cleaned_df.columns.tolist()[:3] if len(cleaned_df.columns) >= 3 else cleaned_df.columns.tolist()
        )
        
        st.write("**Automatic Feature Selection (Optional):**")
        target_options = cleaned_df.columns.tolist()
        target_col = st.selectbox(
            "Select target column for feature importance:",
            options=["None"] + target_options,
            help="Choose a target column to automatically rank features by importance"
        )
        
        if target_col != "None" and st.button("Calculate Feature Importance"):
            try:
                with st.spinner("Calculating feature importance..."):
                    feature_importance = auto_feature_selection(cleaned_df, target_col)
                
                st.subheader("Feature Importance Ranking")
                importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
                st.dataframe(importance_df)
                
                import plotly.express as px
                fig = px.bar(importance_df.head(10), x='Importance', y='Feature', 
                           title='Top 10 Feature Importance', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating feature importance: {str(e)}")
        
        if selected_features:
            st.subheader("Data Visualization")
            
            viz_type = st.selectbox(
                "Choose visualization type:",
                ["Histogram", "Boxplot", "Bar Chart", "Pie Chart", "Correlation Heatmap"]
            )
            
            if st.button("Generate Visualizations"):
                with st.spinner("Generating visualizations..."):
                    try:
                        figs = generate_visualizations(cleaned_df, selected_features, viz_type)
                        
                        if figs:
                            for fig in figs:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No visualizations could be generated for the selected features and visualization type")
                            
                    except Exception as e:
                        st.error(f"Error generating visualizations: {str(e)}")

# Tab 4: AI Assistant
elif selected_tab == "AI Assistant":
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload a dataset first")
    elif not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar")
    elif st.session_state.rag_module is None:
        st.error("AI module not initialized. Please check your Groq API key.")
    else:
        if not st.session_state.rag_ready:
            st.subheader("Prepare AI Assistant")
            st.write("Click the button below to process your data and prepare the AI assistant")
            
            if st.button("Prepare AI Assistant", type="primary"):
                with st.spinner("Processing data for AI assistant..."):
                    try:
                        if st.session_state.file_type == "csv":
                            data_to_process = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data
                        else:
                            data_to_process = st.session_state.uploaded_data
                        
                        result = st.session_state.rag_module.process_uploaded_data(
                            data_to_process, 
                            st.session_state.file_type
                        )
                        
                        if result["success"]:
                            st.session_state.rag_ready = True
                            st.success(result["message"])
                            st.info("You can now ask questions about your data below!")
                        else:
                            st.error(result["message"])
                            
                    except Exception as e:
                        st.error(f"Error preparing AI assistant: {str(e)}")
        
        if st.session_state.rag_ready:
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history with improved styling
            if st.session_state.chat_history:
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    # Question container
                    st.markdown(f"""
                    <div class='question-box'>
                        <div class='box-label'>You:</div>
                        <div class='box-text'>{question}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Answer container
                    st.markdown(f"""
                    <div class='answer-box'>
                        <div class='box-label'>Assistant:</div>
                        <div class='box-text'>{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Function to handle question submission
            def submit_question():
                user_question = st.session_state.get("user_input", "").strip()
                if user_question:
                    try:
                        with st.spinner("Getting answer from AI..."):
                            answer = st.session_state.rag_module.ask_rag(user_question)
                            st.session_state.chat_history.append((user_question, answer))
                            st.session_state.user_input = ""  # Clear input
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")

            # Chat input using on_change for Enter key submission
            st.text_input(
                "Chat Input",
                key="user_input",
                on_change=submit_question,
                placeholder="Enter your question here...",
                label_visibility="collapsed"
            )
            
            if st.session_state.chat_history:
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
