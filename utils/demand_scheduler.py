"""
Demand scheduling utilities for handling non-working days and demand shifting.
Enterprise-grade demand management for workforce planning.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
from typing import Dict, List, Tuple, Optional

from utils.holiday_utils import is_non_working_day

# Configure logger
logger = logging.getLogger(__name__)

class DemandScheduler:
    """
    Enterprise-grade demand scheduler that handles non-working days
    and automatically shifts demand to working days.
    """
    
    def __init__(self, max_shift_days: int = 7):
        """
        Initialize the demand scheduler
        
        Parameters:
        -----------
        max_shift_days : int
            Maximum number of days to look ahead when shifting demand
        """
        self.max_shift_days = max_shift_days
        logger.info(f"DemandScheduler initialized with max_shift_days={max_shift_days}")
    
    def find_next_working_day(self, start_date: date, max_days: int = None) -> Optional[date]:
        """
        Find the next working day after the given date
        
        Parameters:
        -----------
        start_date : date
            Starting date to search from
        max_days : int, optional
            Maximum days to search ahead (defaults to self.max_shift_days)
            
        Returns:
        --------
        date or None
            Next working day, or None if no working day found within max_days
        """
        if max_days is None:
            max_days = self.max_shift_days
            
        current_date = start_date
        
        for i in range(max_days):
            current_date = start_date + timedelta(days=i + 1)
            is_non_working, reason = is_non_working_day(current_date)
            
            if not is_non_working:
                logger.info(f"Found next working day: {current_date} (after {start_date})")
                return current_date
                
        logger.warning(f"No working day found within {max_days} days after {start_date}")
        return None
    
    def find_previous_working_day(self, start_date: date, max_days: int = None) -> Optional[date]:
        """
        Find the previous working day before the given date
        
        Parameters:
        -----------
        start_date : date
            Starting date to search from
        max_days : int, optional
            Maximum days to search back (defaults to self.max_shift_days)
            
        Returns:
        --------
        date or None
            Previous working day, or None if no working day found within max_days
        """
        if max_days is None:
            max_days = self.max_shift_days
            
        current_date = start_date
        
        for i in range(max_days):
            current_date = start_date - timedelta(days=i + 1)
            is_non_working, reason = is_non_working_day(current_date)
            
            if not is_non_working:
                logger.info(f"Found previous working day: {current_date} (before {start_date})")
                return current_date
                
        logger.warning(f"No working day found within {max_days} days before {start_date}")
        return None
    
    def shift_demand_for_non_working_day(self, 
                                    demand_df: pd.DataFrame, 
                                    punch_code_values: List[str],
                                    shift_direction: str = "forward",
                                    actual_cells: set = None,
                                    modified_cells: set = None) -> Tuple[pd.DataFrame, Dict, set, set]:
        """
        Shift demand from non-working days to working days
        
        Parameters:
        -----------
        demand_df : pd.DataFrame
            DataFrame with demand data (Date column + punch code columns)
        punch_code_values : List[str]
            List of punch code column names
        shift_direction : str
            Direction to shift: "forward" (to next working day) or "backward" (to previous working day)
        actual_cells : set, optional
            Set of cell IDs that contain actual quantities (for visual indicators)
        modified_cells : set, optional
            Set of cell IDs that contain user modifications (for visual indicators)
            
        Returns:
        --------
        tuple
            (updated_dataframe, shift_log, updated_actual_cells, updated_modified_cells)
        """
        logger.info(f"Starting demand shift process with direction: {shift_direction}")
        
        # Create copies to avoid modifying the originals
        df = demand_df.copy()
        updated_actual_cells = actual_cells.copy() if actual_cells else set()
        updated_modified_cells = modified_cells.copy() if modified_cells else set()
        shift_log = {}
        
        # Process each row (date)
        for idx, row in df.iterrows():
            date_str = row['Date']
            
            # Extract date from the format "YYYY-MM-DD (Day)"
            try:
                actual_date_str = date_str.split(' (')[0]
                date_obj = datetime.strptime(actual_date_str, '%Y-%m-%d').date()
            except ValueError as e:
                logger.error(f"Error parsing date '{date_str}': {e}")
                continue
            
            # Check if this date is a non-working day
            is_non_working, reason = is_non_working_day(date_obj)
            
            if is_non_working:
                logger.info(f"Processing non-working day: {date_obj} ({reason})")
                
                # Find target working day
                if shift_direction == "forward":
                    target_date = self.find_next_working_day(date_obj)
                else:
                    target_date = self.find_previous_working_day(date_obj)
                
                if target_date is None:
                    logger.warning(f"Could not find target working day for {date_obj}")
                    continue
                
                # Find target row in dataframe
                target_date_str = None
                target_idx = None
                
                for target_idx_candidate, target_row in df.iterrows():
                    target_row_date_str = target_row['Date'].split(' (')[0]
                    try:
                        target_row_date = datetime.strptime(target_row_date_str, '%Y-%m-%d').date()
                        if target_row_date == target_date:
                            target_date_str = target_row['Date']
                            target_idx = target_idx_candidate
                            break
                    except ValueError:
                        continue
                
                if target_idx is None:
                    logger.warning(f"Target date {target_date} not found in dataframe")
                    continue
                
                # Shift demand from non-working day to working day
                shifted_quantities = {}
                
                for punch_code in punch_code_values:
                    current_quantity = row[punch_code]
                    
                    # Extract numeric value if it's formatted
                    if isinstance(current_quantity, str):
                        # Remove formatting symbols
                        clean_value = current_quantity.replace('✓', '').replace('●', '').replace('**', '').strip()
                        try:
                            current_quantity = int(clean_value)
                        except ValueError:
                            current_quantity = 0
                    else:
                        current_quantity = int(current_quantity) if pd.notnull(current_quantity) else 0
                    
                    if current_quantity > 0:
                        # Get target quantity (also clean if formatted)
                        target_quantity = df.at[target_idx, punch_code]
                        if isinstance(target_quantity, str):
                            clean_target = target_quantity.replace('✓', '').replace('●', '').replace('**', '').strip()
                            try:
                                target_quantity = int(clean_target)
                            except ValueError:
                                target_quantity = 0
                        else:
                            target_quantity = int(target_quantity) if pd.notnull(target_quantity) else 0
                        
                        # Add to target day
                        new_target_quantity = target_quantity + current_quantity
                        df.at[target_idx, punch_code] = new_target_quantity
                        
                        # Clear from non-working day
                        df.at[idx, punch_code] = 0
                        
                        # IMPORTANT: Move the visual indicators too!
                        source_cell_id = f"{date_str}_{punch_code}"
                        target_cell_id = f"{target_date_str}_{punch_code}"
                        
                        # If the source cell was marked as actual or modified, move that marking to target
                        if source_cell_id in updated_actual_cells:
                            updated_actual_cells.remove(source_cell_id)
                            updated_actual_cells.add(target_cell_id)
                            logger.info(f"Moved actual cell indicator from {source_cell_id} to {target_cell_id}")
                        
                        if source_cell_id in updated_modified_cells:
                            updated_modified_cells.remove(source_cell_id)
                            updated_modified_cells.add(target_cell_id)
                            logger.info(f"Moved modified cell indicator from {source_cell_id} to {target_cell_id}")
                        
                        shifted_quantities[punch_code] = current_quantity
                        
                        logger.info(f"Shifted {current_quantity} units of {punch_code} from {date_obj} to {target_date}")
                
                # Log the shift
                if shifted_quantities:
                    shift_log[date_obj] = {
                        'reason': reason,
                        'target_date': target_date,
                        'shifted_quantities': shifted_quantities,
                        'direction': shift_direction
                    }
        
        logger.info(f"Demand shift completed. Processed {len(shift_log)} non-working days")
        return df, shift_log, updated_actual_cells, updated_modified_cells
    
    def generate_shift_summary_report(self, shift_log: Dict) -> str:
        """
        Generate a human-readable summary of demand shifts
        
        Parameters:
        -----------
        shift_log : Dict
            Log of shifts from shift_demand_for_non_working_day
            
        Returns:
        --------
        str
            Formatted summary report
        """
        if not shift_log:
            return "No demand shifts were required."
        
        report_lines = [
            "📊 DEMAND SHIFT SUMMARY REPORT",
            "=" * 50,
            ""
        ]
        
        total_shifts = 0
        total_quantity = 0
        
        for non_working_date, shift_info in shift_log.items():
            report_lines.append(f"🚫 Non-Working Day: {non_working_date.strftime('%Y-%m-%d')} ({non_working_date.strftime('%A')})")
            report_lines.append(f"   Reason: {shift_info['reason']}")
            report_lines.append(f"   ➡️  Shifted to: {shift_info['target_date'].strftime('%Y-%m-%d')} ({shift_info['target_date'].strftime('%A')})")
            report_lines.append("   Quantities moved:")
            
            for punch_code, quantity in shift_info['shifted_quantities'].items():
                report_lines.append(f"     • Punch Code {punch_code}: {quantity:,} units")
                total_quantity += quantity
                total_shifts += 1
            
            report_lines.append("")
        
        report_lines.extend([
            "📈 SUMMARY:",
            f"   • Total non-working days processed: {len(shift_log)}",
            f"   • Total punch code shifts: {total_shifts}",
            f"   • Total quantity shifted: {total_quantity:,} units",
            ""
        ])
        
        return "\n".join(report_lines)

# Convenience functions for backward compatibility
def shift_demand_forward(demand_df: pd.DataFrame, punch_code_values: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to shift demand forward to next working days
    """
    scheduler = DemandScheduler()
    return scheduler.shift_demand_for_non_working_day(demand_df, punch_code_values, "forward")

def shift_demand_backward(demand_df: pd.DataFrame, punch_code_values: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to shift demand backward to previous working days
    """
    scheduler = DemandScheduler()
    return scheduler.shift_demand_for_non_working_day(demand_df, punch_code_values, "backward")

def get_next_working_day(start_date: date) -> Optional[date]:
    """
    Convenience function to get the next working day
    """
    scheduler = DemandScheduler()
    return scheduler.find_next_working_day(start_date)