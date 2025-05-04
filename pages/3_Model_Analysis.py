"""
Model Analysis page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import plot_feature_importance, plot_metrics_comparison
from utils.data_loader import load_models
from config import MODELS_DIR

# Configure page
st.set_page_config(
    page_title="Model Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

# Create session state for data persistence if not present
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Check if we have data and models
def ensure_models():
    # Check if we have data
    if st.session_state.df is None:
        st.error("No data loaded. Please load data first from the Home page.")
        return False
    
    # Check if we have models
    if st.session_state.models is None:
        with st.spinner("Loading models..."):
            models, feature_importances, metrics = load_models()
            
            if models:
                st.session_state.models = models
                st.session_state.feature_importances = feature_importances
                st.session_state.metrics = metrics
            else:
                st.error("No trained models available. Please train models first.")
                return False
    
    return True

def display_model_performance(metrics, nn_metrics=None):
    """Display model performance metrics"""
    st.subheader("Model Performance Metrics")
    
    # Add model type selector
    model_type = st.radio(
        "Select Model Type",
        ["Random Forest", "Neural Network", "Ensemble Comparison"],
        horizontal=True,
        help="Choose which model type to analyze"
    )
    
    if model_type == "Random Forest":
        # Display Random Forest metrics
        metrics_records = []
        for work_type, metric in metrics.items():
            metrics_records.append({
                'Work Type': work_type,
                'MAE': metric.get('MAE', np.nan),
                'RMSE': metric.get('RMSE', np.nan),
                'R²': metric.get('R²', np.nan),
                'MAPE (%)': metric.get('MAPE', np.nan)
            })
        
        metrics_df = pd.DataFrame(metrics_records)
        
        # Sort by MAE by default
        metrics_df = metrics_df.sort_values('MAE')
        
        st.dataframe(
            metrics_df,
            use_container_width=True,
            column_config={
                'MAE': st.column_config.NumberColumn('MAE', format="%.4f"),
                'RMSE': st.column_config.NumberColumn('RMSE', format="%.4f"),
                'R²': st.column_config.NumberColumn('R²', format="%.4f"),
                'MAPE (%)': st.column_config.NumberColumn('MAPE (%)', format="%.2f")
            }
        )
        
        # Simple bar chart for MAE
        st.subheader("Random Forest: MAE by Work Type")
        
        # Sort by MAE for the chart
        chart_df = metrics_df.sort_values('MAE').head(15)  # Limit to top 15 for readability
        
        fig = px.bar(
            chart_df,
            x='Work Type',
            y='MAE',
            title='Random Forest: Mean Absolute Error by Work Type',
            color='MAE',
            color_continuous_scale='RdYlGn_r',  # Red for high error, green for low
        )
        
        fig.update_layout(
            xaxis_title='Work Type',
            yaxis_title='Mean Absolute Error (MAE)',
            xaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif model_type == "Neural Network":
        # Check if Neural Network metrics are available
        if nn_metrics is None or len(nn_metrics) == 0:
            nn_path = os.path.join(MODELS_DIR, 'work_utilization_nn_metrics.pkl')
            if os.path.exists(nn_path):
                try:
                    import pickle
                    with open(nn_path, 'rb') as f:
                        nn_metrics = pickle.load(f)
                except Exception as e:
                    st.error(f"Could not load Neural Network metrics: {str(e)}")
                    st.info("Please train Neural Network models first.")
                    return
            else:
                st.warning("Neural Network metrics are not available. Please train Neural Network models first.")
                return
        
        # Display Neural Network metrics
        nn_metrics_records = []
        for work_type, metric in nn_metrics.items():
            nn_metrics_records.append({
                'Work Type': work_type,
                'MAE': metric.get('MAE', np.nan),
                'RMSE': metric.get('RMSE', np.nan),
                'R²': metric.get('R²', np.nan),
                'MAPE (%)': metric.get('MAPE', np.nan)
            })
        
        nn_metrics_df = pd.DataFrame(nn_metrics_records)
        
        # Sort by MAE by default
        nn_metrics_df = nn_metrics_df.sort_values('MAE')
        
        st.dataframe(
            nn_metrics_df,
            use_container_width=True,
            column_config={
                'MAE': st.column_config.NumberColumn('MAE', format="%.4f"),
                'RMSE': st.column_config.NumberColumn('RMSE', format="%.4f"),
                'R²': st.column_config.NumberColumn('R²', format="%.4f"),
                'MAPE (%)': st.column_config.NumberColumn('MAPE (%)', format="%.2f")
            }
        )
        
        # Simple bar chart for MAE
        st.subheader("Neural Network: MAE by Work Type")
        
        # Sort by MAE for the chart
        nn_chart_df = nn_metrics_df.sort_values('MAE').head(15)  # Limit to top 15 for readability
        
        fig = px.bar(
            nn_chart_df,
            x='Work Type',
            y='MAE',
            title='Neural Network: Mean Absolute Error by Work Type',
            color='MAE',
            color_continuous_scale='RdYlGn_r',  # Red for high error, green for low
        )
        
        fig.update_layout(
            xaxis_title='Work Type',
            yaxis_title='Mean Absolute Error (MAE)',
            xaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Ensemble Comparison
        # Check if Neural Network metrics are available
        if nn_metrics is None or len(nn_metrics) == 0:
            nn_path = os.path.join(MODELS_DIR, 'work_utilization_nn_metrics.pkl')
            if os.path.exists(nn_path):
                try:
                    import pickle
                    with open(nn_path, 'rb') as f:
                        nn_metrics = pickle.load(f)
                except Exception as e:
                    st.error(f"Could not load Neural Network metrics: {str(e)}")
                    st.info("Please train Neural Network models first.")
                    return
            else:
                st.warning("Neural Network models are not available. Cannot compare ensemble performance.")
                return
        
        # Create a comparison of RF vs NN for common work types
        common_work_types = set(metrics.keys()) & set(nn_metrics.keys())
        
        if not common_work_types:
            st.warning("No common work types found between Random Forest and Neural Network models.")
            return
        
        # Create comparison dataframe
        comparison_records = []
        
        for wt in common_work_types:
            rf_metrics = metrics.get(wt, {})
            nn_metrics_wt = nn_metrics.get(wt, {})
            
            # For each metric, determine the better model
            for metric_name in ['MAE', 'RMSE', 'R²']:
                rf_value = rf_metrics.get(metric_name, np.nan)
                nn_value = nn_metrics_wt.get(metric_name, np.nan)
                
                # For R², higher is better. For MAE and RMSE, lower is better
                if metric_name == 'R²':
                    better_model = 'Neural Network' if nn_value > rf_value else 'Random Forest'
                else:
                    better_model = 'Neural Network' if nn_value < rf_value else 'Random Forest'
                
                comparison_records.append({
                    'Work Type': wt,
                    'Metric': metric_name,
                    'Random Forest': rf_value,
                    'Neural Network': nn_value,
                    'Better Model': better_model
                })
        
        comparison_df = pd.DataFrame(comparison_records)
        
        # Display the comparison
        st.subheader("Model Comparison: Random Forest vs Neural Network")
        
        # Filter for specific metric
        metric_to_compare = st.selectbox(
            "Select Metric to Compare",
            options=['MAE', 'RMSE', 'R²'],
            index=0
        )
        
        filtered_comparison = comparison_df[comparison_df['Metric'] == metric_to_compare]
        
        # Sort by RF metric by default
        filtered_comparison = filtered_comparison.sort_values('Random Forest')
        
        # Create a comparison chart
        st.subheader(f"Model Comparison: {metric_to_compare} by Work Type")
        
        # Limit to top 10 for readability
        chart_comparison = filtered_comparison.head(10)
        
        # Create a grouped bar chart
        fig = go.Figure()
        
        # Add Random Forest bars
        fig.add_trace(go.Bar(
            x=chart_comparison['Work Type'],
            y=chart_comparison['Random Forest'],
            name='Random Forest',
            marker_color='blue'
        ))
        
        # Add Neural Network bars
        fig.add_trace(go.Bar(
            x=chart_comparison['Work Type'],
            y=chart_comparison['Neural Network'],
            name='Neural Network',
            marker_color='red'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Random Forest vs Neural Network: {metric_to_compare} Comparison',
            xaxis_title='Work Type',
            yaxis_title=metric_to_compare,
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary table
        st.subheader("Model Performance Summary")
        
        # Count wins by model
        model_wins = filtered_comparison['Better Model'].value_counts().reset_index()
        model_wins.columns = ['Model', 'Number of Work Types']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(model_wins, use_container_width=True)
        
        with col2:
            # Create a pie chart of model wins
            fig = px.pie(
                model_wins, 
                values='Number of Work Types', 
                names='Model',
                title=f'Better Model by {metric_to_compare}',
                color='Model',
                color_discrete_map={'Random Forest': 'blue', 'Neural Network': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_feature_importance(feature_importances):
    """Display feature importance analysis"""
    st.subheader("Feature Importance Analysis")
    
    # Work type selector
    selected_work_type = st.selectbox(
        "Select Work Type",
        options=sorted(list(feature_importances.keys())),
        index=0 if feature_importances else None
    )
    
    if selected_work_type:
        # Get feature importance for the selected work type
        importance_dict = feature_importances[selected_work_type]
        importance_df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        
        # Sort by importance and take top 15
        importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
        
        # Create a simple bar chart
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top 15 Features for {selected_work_type}',
            color='Importance',
            color_continuous_scale='Blues',
        )
        
        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='Feature',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature explanation
        st.subheader("Feature Explanations")
        
        feature_explanations = {
            'DayOfWeek_feat': 'Day of the week (0=Monday, 6=Sunday)',
            'Month_feat': 'Month of the year (1-12)',
            'Year_feat': 'Year',
            'IsWeekend_feat': 'Whether the day is a weekend (1) or weekday (0)',
            'NoOfMan_lag_1': 'Number of workers from 1 day ago',
            'NoOfMan_lag_7': 'Number of workers from 7 days ago (same day last week)',
            'NoOfMan_rolling_mean_7': 'Average number of workers over the past 7 days',
            'NoOfMan_7day_trend': 'Trend over 7 days (difference between today and 7 days ago)',
            'NoOfMan_1day_trend': 'Short-term trend (difference between today and yesterday)'
        }
        
        # Show explanations for the top features in a compact format
        st.write("Key feature meanings:")
        
        # Create 2-column layout for explanations
        col1, col2 = st.columns(2)
        
        # Get top features
        top_features = importance_df['Feature'].tolist()
        
        # Display explanations in columns
        for i, feature in enumerate(top_features):
            explanation = feature_explanations.get(feature, "Time-related feature")
            
            # Display in alternating columns
            if i % 2 == 0:
                with col1:
                    st.markdown(f"**{feature}**: {explanation}")
            else:
                with col2:
                    st.markdown(f"**{feature}**: {explanation}")

def main():
    st.header("Model Analysis")
    
    # Check if models are loaded
    if not ensure_models():
        return
    
    # Get models and metrics
    models = st.session_state.models
    metrics = st.session_state.metrics
    feature_importances = st.session_state.feature_importances
    
    # Check for neural network metrics
    nn_metrics = None
    nn_path = os.path.join(MODELS_DIR, 'work_utilization_nn_metrics.pkl')
    if os.path.exists(nn_path):
        try:
            import pickle
            with open(nn_path, 'rb') as f:
                nn_metrics = pickle.load(f)
        except Exception as e:
            logger.error(f"Could not load Neural Network metrics: {str(e)}")
    
    # Model overview
    st.subheader("Model Overview")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of RF Models", f"{len(models)}")
    
    with col2:
        # Number of NN models
        nn_model_count = len(nn_metrics) if nn_metrics else 0
        st.metric("Number of NN Models", f"{nn_model_count}")
    
    with col3:
        # Average MAE across all RF models
        if metrics:
            avg_mae = np.mean([m.get('MAE', 0) for m in metrics.values()])
            st.metric("RF Average MAE", f"{avg_mae:.4f}")
    
    with col4:
        # Average MAE across all NN models
        if nn_metrics:
            nn_avg_mae = np.mean([m.get('MAE', 0) for m in nn_metrics.values()])
            st.metric("NN Average MAE", f"{nn_avg_mae:.4f}")
    
    # Tabs for different analyses
    tab1, tab2 = st.tabs(["Model Performance", "Feature Importance"])
    
    with tab1:
        display_model_performance(metrics, nn_metrics)
    
    with tab2:
        display_feature_importance(feature_importances)

# Run the main function
if __name__ == "__main__":
    main()