import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Web Traffic Prediction Dashboard", layout="wide")

# Add a title and description
st.title("Web Traffic Prediction Dashboard")
st.markdown("""
This dashboard allows you to analyze web traffic data and predict future traffic patterns
based on metrics like session duration, bounce rate, traffic source, and more.
""")

# Create sample data based on the image
def load_sample_data():
    data = {
        'Page Views': [5, 4],
        'Session Duration': [11.051381, 3.429316],
        'Bounce Rate': [0.230652, 0.391001],
        'Traffic Source': ['Organic', 'Social'],
        'Time on Page': [3.890460, 8.478174],
        'Previous Visits': [3, 0],
        'Conversion Rate': [1.0, 1.0]
    }
    return pd.DataFrame(data)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = load_sample_data()

# Function to make predictions
def predict_traffic(data, features, target, days_to_predict=7):
    # Process categorical features
    cat_features = data.select_dtypes(include=['object']).columns
    num_features = [col for col in features if col not in cat_features]
    
    # Handle categorical data
    if len(cat_features) > 0 and any(col in features for col in cat_features):
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        cat_cols = [col for col in cat_features if col in features]
        if cat_cols:
            cat_encoded = encoder.fit_transform(data[cat_cols])
            cat_encoded_df = pd.DataFrame(
                cat_encoded, 
                columns=encoder.get_feature_names_out(cat_cols)
            )
            
            # Combine with numerical features
            X = pd.concat([data[num_features].reset_index(drop=True), 
                          cat_encoded_df.reset_index(drop=True)], axis=1)
        else:
            X = data[num_features]
    else:
        X = data[num_features]
    
    y = data[target]
    
    # Train a simple model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions for future days
    last_row = X.iloc[-1:].copy()
    predictions = []
    
    for _ in range(days_to_predict):
        next_prediction = model.predict(last_row)[0]
        predictions.append(next_prediction)
        
        # Update numerical features for next prediction (simple incremental approach)
        for col in num_features:
            if col != target:  # Don't update the target column
                # Apply a small random change
                last_row[col] = last_row[col] * (1 + np.random.normal(0, 0.05))
    
    return predictions

# Sidebar for data input and settings
st.sidebar.header("Settings")

# Option to upload data or use sample data
data_option = st.sidebar.radio("Choose data source", ["Use sample data", "Upload your own CSV"])

if data_option == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.sidebar.success("Data successfully loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")

# Manual data entry option
if st.sidebar.checkbox("Add new data point"):
    st.sidebar.subheader("Enter new data point")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        page_views = st.number_input("Page Views", min_value=0, value=5)
        session_duration = st.number_input("Session Duration", min_value=0.0, value=5.0)
        bounce_rate = st.number_input("Bounce Rate", min_value=0.0, max_value=1.0, value=0.3)
        traffic_source = st.selectbox("Traffic Source", ["Organic", "Social", "Direct", "Referral", "Email"])
    
    with col2:
        time_on_page = st.number_input("Time on Page", min_value=0.0, value=4.0)
        previous_visits = st.number_input("Previous Visits", min_value=0, value=1)
        conversion_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.8)
    
    if st.sidebar.button("Add Data Point"):
        new_data = pd.DataFrame({
            'Page Views': [page_views],
            'Session Duration': [session_duration],
            'Bounce Rate': [bounce_rate],
            'Traffic Source': [traffic_source],
            'Time on Page': [time_on_page],
            'Previous Visits': [previous_visits],
            'Conversion Rate': [conversion_rate]
        })
        st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
        st.sidebar.success("New data point added!")

# Prediction settings
st.sidebar.subheader("Prediction Settings")
target_col = st.sidebar.selectbox("Select target column to predict", 
                                 options=st.session_state.data.columns.tolist(),
                                 index=0)  # Default to Page Views

available_features = [col for col in st.session_state.data.columns if col != target_col]
selected_features = st.sidebar.multiselect(
    "Select features for prediction",
    options=available_features,
    default=available_features[:3]  # Default to first 3 features
)

days_to_predict = st.sidebar.slider("Days to predict", min_value=1, max_value=30, value=7)

# Main content area - Organized in tabs
tab1, tab2, tab3 = st.tabs(["Data Explorer", "Predictions", "Insights"])

with tab1:
    st.header("Data Explorer")
    st.dataframe(st.session_state.data, use_container_width=True)
    
    # Show basic statistics
    if st.checkbox("Show statistics"):
        st.subheader("Data Statistics")
        st.write(st.session_state.data.describe())
    
    # Data visualization
    st.subheader("Data Visualization")
    
    viz_type = st.selectbox(
        "Select visualization type",
        ["Correlation Heatmap", "Time Series", "Scatter Plot", "Distribution"]
    )
    
    if viz_type == "Correlation Heatmap":
        numeric_data = st.session_state.data.select_dtypes(include=['number'])
        corr = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    elif viz_type == "Time Series":
        if len(st.session_state.data) > 1:
            col_to_plot = st.selectbox("Select column to plot", options=st.session_state.data.columns)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(st.session_state.data.index, st.session_state.data[col_to_plot], marker='o')
            ax.set_title(f"{col_to_plot} Over Time")
            ax.set_xlabel("Data Point Index")
            ax.set_ylabel(col_to_plot)
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("Need more data points for time series visualization")
    
    elif viz_type == "Scatter Plot":
        if len(available_features) >= 2:
            x_col = st.selectbox("Select X-axis", options=available_features, index=0)
            y_col = st.selectbox("Select Y-axis", options=[c for c in available_features if c != x_col], index=0)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(st.session_state.data[x_col], st.session_state.data[y_col])
            ax.set_title(f"{y_col} vs {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("Need at least two features for scatter plot")
    
    elif viz_type == "Distribution":
        col_to_plot = st.selectbox("Select column", options=st.session_state.data.columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if st.session_state.data[col_to_plot].dtype == 'object':
            st.session_state.data[col_to_plot].value_counts().plot(kind='bar', ax=ax)
        else:
            sns.histplot(st.session_state.data[col_to_plot], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col_to_plot}")
        ax.grid(True)
        st.pyplot(fig)

with tab2:
    st.header("Traffic Predictions")
    
    # Check if we have enough data and features
    if len(st.session_state.data) > 1 and len(selected_features) > 0:
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    predictions = predict_traffic(
                        st.session_state.data, 
                        selected_features, 
                        target_col, 
                        days_to_predict
                    )
                    
                    # Show predictions
                    st.subheader(f"Predicted {target_col} for the next {days_to_predict} days")
                    
                    # Create a DataFrame for the predictions
                    pred_df = pd.DataFrame({
                        'Day': range(1, days_to_predict + 1),
                        f'Predicted {target_col}': predictions
                    })
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Plot predictions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(pred_df['Day'], pred_df[f'Predicted {target_col}'], marker='o', linestyle='-')
                    ax.set_title(f"Predicted {target_col} for Next {days_to_predict} Days")
                    ax.set_xlabel("Day")
                    ax.set_ylabel(target_col)
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Display additional statistics
                    st.subheader("Prediction Statistics")
                    st.markdown(f"**Average predicted {target_col}:** {np.mean(predictions):.2f}")
                    st.markdown(f"**Maximum predicted {target_col}:** {np.max(predictions):.2f}")
                    st.markdown(f"**Minimum predicted {target_col}:** {np.min(predictions):.2f}")
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {e}")
    else:
        st.warning("Need more data points and at least one feature selected for prediction")

with tab3:
    st.header("Traffic Insights")
    
    # Only show insights if we have enough data
    if len(st.session_state.data) > 1:
        # Traffic source analysis
        st.subheader("Traffic Source Analysis")
        if 'Traffic Source' in st.session_state.data.columns:
            traffic_source_counts = st.session_state.data['Traffic Source'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            traffic_source_counts.plot(kind='bar', ax=ax)
            ax.set_title("Traffic by Source")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
            # Performance by traffic source
            st.subheader("Performance by Traffic Source")
            source_perf = st.session_state.data.groupby('Traffic Source').agg({
                'Page Views': 'mean',
                'Session Duration': 'mean',
                'Bounce Rate': 'mean',
                'Conversion Rate': 'mean'
            }).reset_index()
            
            st.dataframe(source_perf, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Key Correlation Insights")
        numeric_data = st.session_state.data.select_dtypes(include=['number'])
        if len(numeric_data.columns) > 1:
            correlations = numeric_data.corr()[target_col].drop(target_col)
            
            # Sort correlations and show top influences
            sorted_corr = correlations.abs().sort_values(ascending=False)
            
            st.markdown(f"**Top factors influencing {target_col}:**")
            for col, corr_val in sorted_corr.items():
                direction = "positive" if correlations[col] > 0 else "negative"
                st.markdown(f"- **{col}**: {corr_val:.2f} ({direction} correlation)")
            
            # Plot key relationships
            if len(sorted_corr) > 0:
                top_factor = sorted_corr.index[0]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(st.session_state.data[top_factor], st.session_state.data[target_col])
                ax.set_title(f"Relationship: {target_col} vs {top_factor}")
                ax.set_xlabel(top_factor)
                ax.set_ylabel(target_col)
                ax.grid(True)
                
                # Add trend line
                z = np.polyfit(st.session_state.data[top_factor], st.session_state.data[target_col], 1)
                p = np.poly1d(z)
                ax.plot(st.session_state.data[top_factor], p(st.session_state.data[top_factor]), 
                        "r--", alpha=0.8)
                
                st.pyplot(fig)
        
        # Recommendations section
        st.subheader("Recommendations")
        st.markdown("""
        Based on the data analysis, here are some recommendations to improve web traffic:
        
        1. **Monitor key metrics** that have strong correlations with your target variable
        2. **Analyze traffic sources** to identify which channels are performing best
        3. **Optimize for conversion** by focusing on pages with higher conversion rates
        4. **Reduce bounce rates** on key pages to improve overall engagement
        5. **Increase session duration** by providing more engaging content
        """)
    else:
        st.info("Add more data points to see insights")

st.markdown("---")
st.caption("Web Traffic Prediction Dashboard | Created with Streamlit")