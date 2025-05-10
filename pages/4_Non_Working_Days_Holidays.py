"""
Non-Working Days page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import calendar

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.holiday_utils import get_swedish_holidays, is_non_working_day
from config import MODELS_DIR, DATA_DIR

# Configure page
st.set_page_config(
    page_title="Non-Working Days",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

def main():
    st.header("📅 Non-Working Days Calendar")
    
    st.info("""
    This page provides information about non-working days for workforce planning.
    
    **At this company:**
    - Swedish holidays are non-working days
    - Saturdays are non-working days
    - Sundays are working days
    
    The prediction system automatically accounts for this and sets workforce requirements to zero on non-working days.
    """)
    
    # Year selection
    current_year = datetime.now().year
    selected_year = st.selectbox(
        "Select Year",
        options=list(range(current_year-2, current_year+3)),
        index=2  # Default to current year
    )
    
    # Get Swedish holidays for the selected year
    with st.spinner("Loading Swedish holidays..."):
        holidays = get_swedish_holidays(selected_year)
        
        if not holidays:
            st.error("Could not load holiday information. Please check your internet connection or install the 'holidays' package.")
            st.info("You may need to run: `pip install holidays` to install the required package.")
            return
        
        # Convert holidays dictionary to DataFrame for display
        holiday_records = []
        for date, name in holidays.items():
            holiday_records.append({
                'Date': date,
                'Reason': f"Swedish Holiday: {name}",
                'Type': 'Holiday',
                'Day of Week': date.strftime('%A'),
                'Month': date.strftime('%B')
            })
        
        # Add all Saturdays for the year
        for month in range(1, 13):
            num_days = calendar.monthrange(selected_year, month)[1]
            for day in range(1, num_days + 1):
                date = datetime(selected_year, month, day).date()
                if date.weekday() == 5:  # 5 = Saturday
                    holiday_records.append({
                        'Date': date,
                        'Reason': 'Saturday (Weekend)',
                        'Type': 'Weekend',
                        'Day of Week': 'Saturday',
                        'Month': date.strftime('%B')
                    })
        
        non_working_days_df = pd.DataFrame(holiday_records)
        non_working_days_df = non_working_days_df.sort_values('Date')
    
    # Display holidays in a table
    st.subheader(f"Non-Working Days for {selected_year}")
    
    st.dataframe(
        non_working_days_df,
        use_container_width=True,
        column_config={
            'Date': st.column_config.DateColumn(
                'Date',
                format="YYYY-MM-DD"
            ),
            'Reason': st.column_config.TextColumn(
                'Reason'
            ),
            'Type': st.column_config.TextColumn(
                'Type',
                help="Type of non-working day"
            ),
            'Day of Week': st.column_config.TextColumn(
                'Day of Week'
            ),
            'Month': st.column_config.TextColumn(
                'Month'
            )
        }
    )
    
    # Visualize holidays distribution
    st.subheader("Distribution of Non-Working Days by Month")
    
    # Count holidays by month
    days_by_month = non_working_days_df.groupby(['Month', 'Type']).size().reset_index(name='Count')
    
    # Ensure months are in correct order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
    days_by_month['Month'] = pd.Categorical(days_by_month['Month'], categories=month_order, ordered=True)
    days_by_month = days_by_month.sort_values(['Month', 'Type'])
    
    # Create a bar chart
    fig = px.bar(
        days_by_month,
        x='Month',
        y='Count',
        color='Type',
        labels={
            'Month': 'Month',
            'Count': 'Number of Non-Working Days',
            'Type': 'Type'
        },
        title=f'Non-Working Days by Month in {selected_year}',
        color_discrete_map={'Holiday': '#ff7f0e', 'Weekend': '#1f77b4'}
    )
    
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Days',
        xaxis={'categoryorder': 'array', 'categoryarray': month_order},
        barmode='stack'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create calendar view
    st.subheader("Calendar View")
    
    # Create month tabs
    month_tabs = st.tabs(month_order)
    
    # Get holiday dates for faster checking
    holiday_dates = [d for d in holidays.keys()]
    
    # Display calendar for each month
    for i, month_tab in enumerate(month_tabs):
        with month_tab:
            # Get the number of days in the month
            month_num = i + 1
            num_days = calendar.monthrange(selected_year, month_num)[1]
            
            # Create a calendar grid
            # Days of the week
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Get the first day of the month
            first_day = datetime(selected_year, month_num, 1)
            first_weekday = first_day.weekday()  # 0 = Monday, 6 = Sunday
            
            # Initialize calendar data
            calendar_data = []
            day_num = 1
            
            # Create weeks
            for week in range(6):  # Maximum 6 weeks in a month
                week_data = {'Week': week + 1}
                
                for day_idx, day in enumerate(days):
                    if week == 0 and day_idx < first_weekday:
                        # Empty cells before the first day
                        week_data[day] = ""
                    elif day_num <= num_days:
                        # Regular day
                        current_date = datetime(selected_year, month_num, day_num).date()
                        is_nonworking, reason = is_non_working_day(current_date)
                        
                        if is_nonworking:
                            if day_idx == 5:  # Saturday
                                week_data[day] = f"{day_num} 🌙 Weekend"
                            else:
                                week_data[day] = f"{day_num} 🌟 {reason}"
                        else:
                            if day_idx == 6:  # Sunday but a working day
                                week_data[day] = f"{day_num} 💼"
                            else:
                                week_data[day] = str(day_num)
                            
                        day_num += 1
                    else:
                        # Empty cells after the last day
                        week_data[day] = ""
                
                calendar_data.append(week_data)
                
                # Stop if we've reached the end of the month
                if day_num > num_days:
                    break
            
            # Convert to DataFrame
            calendar_df = pd.DataFrame(calendar_data)
            
            # Display calendar
            st.write(f"### {month_order[i]} {selected_year}")
            st.write("Legend: 🌟 = Holiday, 🌙 = Saturday (Non-working), 💼 = Sunday (Working Day)")
            st.dataframe(
                calendar_df.set_index('Week'),
                use_container_width=True,
                height=35 + len(calendar_data) * 35
            )
            
            # Get holidays for this month
            month_holidays = non_working_days_df[non_working_days_df['Month'] == month_order[i]]
            
            if not month_holidays.empty:
                st.write("#### Non-Working Days this month:")
                for _, row in month_holidays.iterrows():
                    st.write(f"• **{row['Date'].strftime('%B %d')}** ({row['Day of Week']}): {row['Reason']}")
            else:
                st.write("No non-working days this month.")
    
    # Add exporting functionality
    st.subheader("Export Non-Working Days Calendar")
    
    export_format = st.radio(
        "Export Format",
        ["CSV", "Excel"],
        horizontal=True
    )
    
    if st.button("Download Calendar", type="primary"):
        if export_format == "CSV":
            csv_data = non_working_days_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"non_working_days_{selected_year}.csv",
                mime="text/csv"
            )
        else:
            # Excel export
            import io
            buffer = io.BytesIO()
            non_working_days_df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            
            st.download_button(
                label="Download Excel",
                data=buffer,
                file_name=f"non_working_days_{selected_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()