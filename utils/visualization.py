"""
Visualization utilities for work utilization data and predictions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import logging
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback

# Configure logger
logger = logging.getLogger(__name__)

def plot_worktype_distribution(df, top_n=10):
    """
    Plot the distribution of NoOfMan across different WorkTypes
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    top_n : int
        Number of top WorkTypes to show
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    try:
        # Group by WorkType and calculate mean NoOfMan
        worktype_stats = df.groupby('WorkType')['NoOfMan'].agg(['mean', 'count']).reset_index()
        
        # Sort by mean and get top N
        top_worktypes = worktype_stats.sort_values('mean', ascending=False).head(top_n)
        
        # Create figure
        fig = px.bar(
            top_worktypes,
            x='WorkType',
            y='mean',
            text='mean',
            color='mean',
            labels={
                'WorkType': 'Work Type',
                'mean': 'Average Number of Workers'
            },
            title=f'Top {top_n} Work Types by Average Workers Required',
            height=500
        )
        
        # Add count as hover data
        fig.update_traces(
            hovertemplate='<b>Work Type:</b> %{x}<br><b>Avg Workers:</b> %{y:.2f}<br><b>Count:</b> %{customdata}',
            customdata=top_worktypes['count']
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Work Type',
            yaxis_title='Average Number of Workers',
            xaxis={'categoryorder': 'total descending'},
            plot_bgcolor='white',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting WorkType distribution: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error generating WorkType distribution plot: {str(e)}")
        return None

def plot_time_series(df, work_type=None, resample='D', start_date=None, end_date=None):
    """
    Plot time series of NoOfMan
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    work_type : str or list, optional
        WorkType(s) to filter for, or None for all
    resample : str, optional
        Time period for resampling ('D' for daily, 'W' for weekly, 'M' for monthly)
    start_date : datetime, optional
        Start date for filtering
    end_date : datetime, optional
        End date for filtering
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    try:
        # Make a copy of the dataframe
        plot_df = df.copy()
        
        # Apply date filters if provided
        if start_date is not None:
            plot_df = plot_df[plot_df['Date'] >= start_date]
        
        if end_date is not None:
            plot_df = plot_df[plot_df['Date'] <= end_date]
        
        # Apply WorkType filter if provided
        if work_type is not None:
            if isinstance(work_type, list):
                plot_df = plot_df[plot_df['WorkType'].isin(work_type)]
            else:
                plot_df = plot_df[plot_df['WorkType'] == work_type]
        
        # Group by date and WorkType
        time_series_data = plot_df.groupby(['Date', 'WorkType'])['NoOfMan'].sum().reset_index()
        
        # Create figure
        fig = px.line(
            time_series_data, 
            x='Date', 
            y='NoOfMan', 
            color='WorkType',
            labels={
                'Date': 'Date',
                'NoOfMan': 'Number of Workers',
                'WorkType': 'Work Type'
            },
            title='Worker Requirements Over Time',
            height=500
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Workers',
            plot_bgcolor='white',
            hovermode='x unified',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting time series: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error generating time series plot: {str(e)}")
        return None

def plot_predictions(predictions, actual=None, work_types=None):
    """
    Plot predictions for each WorkType using bar charts
    
    Parameters:
    -----------
    predictions : dict
        Dictionary of predictions with dates as keys and WorkType predictions as values
    actual : pd.DataFrame, optional
        DataFrame with actual values for comparison
    work_types : list, optional
        List of WorkTypes to include, or None for all
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    try:
        # Convert predictions to DataFrame
        pred_records = []
        for date, work_type_preds in predictions.items():
            for work_type, value in work_type_preds.items():
                if work_types is None or work_type in work_types:
                    pred_records.append({
                        'Date': date,
                        'WorkType': work_type,
                        'NoOfMan': value,
                        'Type': 'Predicted',
                        'Day': date.strftime('%a')  # Short day name
                    })
        
        if not pred_records:
            return None
            
        pred_df = pd.DataFrame(pred_records)
        
        # Create figure for bar chart
        fig = go.Figure()
        
        # Add bars for each date with the work types
        for date in sorted(pred_df['Date'].unique()):
            date_df = pred_df[pred_df['Date'] == date]
            day_name = date.strftime('%a')
            date_str = date.strftime('%Y-%m-%d')
            
            fig.add_trace(go.Bar(
                x=date_df['WorkType'],
                y=date_df['NoOfMan'],
                name=f"{day_name} ({date_str})",
                hovertemplate='<b>%{x}</b><br>Workers: %{y}<br>Date: ' + date_str + '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='Predicted Worker Requirements by Work Type and Date',
            xaxis_title='Work Type',
            yaxis_title='Number of Workers',
            barmode='group',
            plot_bgcolor='white',
            margin=dict(t=50, b=50, l=50, r=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting predictions: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error generating predictions plot: {str(e)}")
        return None

def create_grouped_bar_chart(predictions, work_types=None, date_format='%m-%d'):
    """
    Create a grouped bar chart for multi-day predictions showing bars by work type
    
    Parameters:
    -----------
    predictions : dict
        Dictionary of predictions by date and work type
    work_types : list, optional
        List of work types to include in the plot
    date_format : str, optional
        Format for date labels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    try:
        # Create a list of records from the predictions
        records = []
        
        for date, preds in predictions.items():
            for work_type, value in preds.items():
                if work_types is None or work_type in work_types:
                    records.append({
                        'Date': date,
                        'WorkType': work_type,
                        'NoOfMan': value,
                        'Day': date.strftime('%a')  # Short day name
                    })
        
        if not records:
            return None
        
        # Create a dataframe
        df = pd.DataFrame(records)
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Create figure
        fig = go.Figure()
        
        # Add a trace for each work type
        for wt in df['WorkType'].unique():
            wt_data = df[df['WorkType'] == wt]
            
            fig.add_trace(go.Bar(
                x=wt_data['Date'].dt.strftime(date_format),
                y=wt_data['NoOfMan'],
                name=wt,
                hovertemplate='<b>%{x}</b><br>Workers: %{y}<br>Work Type: ' + wt + '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='Predicted Workers by Work Type and Date',
            xaxis_title='Date',
            yaxis_title='Predicted Number of Workers',
            barmode='group',
            legend_title='Work Type',
            height=600,
            plot_bgcolor='white',
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis=dict(
                tickmode='array',
                tickvals=df['Date'].dt.strftime(date_format).unique(),
                ticktext=[f"{d.strftime(date_format)} ({d.strftime('%a')})" for d in df['Date'].unique()]
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating grouped bar chart: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error generating grouped bar chart: {str(e)}")
        return None

def plot_feature_importance(feature_importances, work_type, top_n=15):
    """
    Plot feature importance for a specific WorkType
    
    Parameters:
    -----------
    feature_importances : dict
        Dictionary of feature importances
    work_type : str
        WorkType to plot feature importance for
    top_n : int, optional
        Number of top features to show
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    try:
        if work_type not in feature_importances:
            st.warning(f"No feature importance data available for WorkType: {work_type}")
            return None
        
        # Get feature importance for the specified WorkType
        importance_dict = feature_importances[work_type]
        
        # Convert to DataFrame
        imp_df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        
        # Sort by importance and get top N
        imp_df = imp_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create figure
        fig = px.bar(
            imp_df,
            y='Feature',
            x='Importance',
            orientation='h',
            color='Importance',
            labels={
                'Feature': 'Feature',
                'Importance': 'Importance Score'
            },
            title=f'Top {top_n} Features for WorkType: {work_type}',
            height=600
        )
        
        # Update layout
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='white',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error generating feature importance plot: {str(e)}")
        return None

def plot_metrics_comparison(metrics, metric_name='MAE'):
    """
    Compare a specific metric across WorkTypes
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics for each WorkType
    metric_name : str
        Name of the metric to compare
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    try:
        # Extract the specified metric for each WorkType
        metric_values = []
        for work_type, work_metrics in metrics.items():
            if metric_name in work_metrics:
                metric_values.append({
                    'WorkType': work_type,
                    'Value': work_metrics[metric_name]
                })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metric_values)
        
        # Sort by metric value
        metrics_df = metrics_df.sort_values('Value')
        
        # Create figure
        fig = px.bar(
            metrics_df,
            x='WorkType',
            y='Value',
            color='Value',
            labels={
                'WorkType': 'Work Type',
                'Value': metric_name
            },
            title=f'{metric_name} by Work Type',
            height=500
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Work Type',
            yaxis_title=metric_name,
            plot_bgcolor='white',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting metrics comparison: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error generating metrics comparison plot: {str(e)}")
        return None
    
def create_actual_vs_predicted_chart(results_df, work_type=None):
    """
    Create a line chart comparing actual vs predicted values
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with prediction results
    work_type : str, optional
        Work type to filter for, or None for all
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    
    # Filter for specific work type if provided
    if work_type is not None:
        df = results_df[results_df['Work Type'] == work_type].copy()
    else:
        df = results_df.copy()
    
    # Make sure we have data
    if len(df) == 0:
        return None
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Create figure
    fig = go.Figure()
    
    # Add actual line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Actual'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add predicted line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Predicted'],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Update layout
    title = "Actual vs Predicted Values"
    if work_type:
        title += f" for {work_type}"
        
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Workers',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark",
        hovermode="x unified",
        height=500
    )
    
    # Add day of week annotations to x-axis
    date_ticks = []
    for date in df['Date']:
        day_name = date.strftime('%a')
        date_str = date.strftime('%Y-%m-%d')
        date_ticks.append(f"{day_name}<br>{date_str}")
    
    fig.update_xaxes(
        ticktext=date_ticks,
        tickvals=df['Date']
    )
    
    return fig