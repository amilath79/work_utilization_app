"""
Implements robust business logic with fallbacks and validation.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import traceback

from config import WORKFORCE_PREDICTION_CONFIG, PREDICTION_VALIDATION
from utils.sql_data_connector import extract_sql_data
from config import SQL_SERVER, SQL_DATABASE, SQL_DATABASE_LIVE, SQL_TRUSTED_CONNECTION

logger = logging.getLogger(__name__)

class WorkforceCalculator:
    """
    Enterprise-grade workforce calculator that handles both demand-based 
    and ML-based predictions with robust error handling and business rules.
    """
    
    def __init__(self):
        self.config = WORKFORCE_PREDICTION_CONFIG
        self.validation = PREDICTION_VALIDATION
        
    def get_crew_limits(self, punch_code: str) -> Tuple[int, int]:
        """Get minimum and maximum crew size for a punch code"""
        try:
            punch_code_int = int(punch_code)
            return self.config['CREW_LIMITS'].get(punch_code_int, (1, 20))
        except (ValueError, TypeError):
            logger.warning(f"Invalid punch code format: {punch_code}")
            return (1, 20)  # Default limits
    
    def get_historical_productivity(self, punch_code: str) -> float:
        """Get historical productivity (items per worker per hour) for fallback"""
        try:
            punch_code_int = int(punch_code)
            return self.config['HISTORICAL_PRODUCTIVITY'].get(
                punch_code_int, 
                self.config['DEFAULT_KPI_FALLBACK']
            )
        except (ValueError, TypeError):
            logger.warning(f"Invalid punch code for historical productivity: {punch_code}")
            return self.config['DEFAULT_KPI_FALLBACK']
    
    def validate_kpi_value(self, kpi_value: float, punch_code: str) -> bool:
        """Validate if KPI value is within reasonable bounds"""
        if pd.isna(kpi_value) or kpi_value is None:
            return False
            
        return (self.validation['MIN_KPI_THRESHOLD'] <= kpi_value <= 
                self.validation['MAX_KPI_THRESHOLD'])
    
    def calculate_demand_based_workforce(self, quantity: float, kpi_value: float, 
                                       punch_code: str) -> Dict:
        """
        Calculate workforce requirement based on demand and KPI.
        
        Business Logic:
        1. If KPI valid: Workers = Quantity / (KPI × Hours_per_Day)
        2. If KPI invalid: Use historical productivity as fallback
        3. Apply crew size limits and rounding
        4. Return detailed calculation breakdown
        """
        try:
            calculation_details = {
                'punch_code': punch_code,
                'quantity': quantity,
                'kpi_value': kpi_value,
                'method': 'demand_based',
                'fallback_used': False,
                'warnings': []
            }
            
            hours_per_day = self.config['DEFAULT_HOURS_PER_DAY']
            min_crew, max_crew = self.get_crew_limits(punch_code)
            
            # Handle zero quantity case
            if quantity <= 0:
                if kpi_value > 0:
                    # Use next day KPI as specified
                    required_workers = min_crew  # Minimum crew to maintain KPI
                    calculation_details['warnings'].append("Zero quantity - using minimum crew for KPI maintenance")
                else:
                    required_workers = 0
                    calculation_details['warnings'].append("Zero quantity and no valid KPI - no workers required")
            else:
                # Normal calculation
                if self.validate_kpi_value(kpi_value, punch_code):
                    # Primary calculation: Quantity / (KPI × Hours)
                    required_workers = quantity / (kpi_value * hours_per_day)
                    logger.info(f"Demand calculation for {punch_code}: {quantity} / ({kpi_value} × {hours_per_day}) = {required_workers}")
                else:
                    # Fallback to historical productivity
                    historical_kpi = self.get_historical_productivity(punch_code)
                    required_workers = quantity / (historical_kpi * hours_per_day)
                    calculation_details['fallback_used'] = True
                    calculation_details['historical_kpi_used'] = historical_kpi
                    calculation_details['warnings'].append(f"Invalid KPI ({kpi_value}) - used historical productivity ({historical_kpi})")
                    logger.warning(f"Using historical KPI for {punch_code}: {historical_kpi}")
            
            # Apply business rules and rounding
            raw_workers = required_workers
            rounded_workers = round(required_workers)
            constrained_workers = max(min_crew, min(max_crew, rounded_workers))
            
            # Detect and flag outliers
            if constrained_workers > self.validation['MAX_DAILY_WORKERS_PER_CODE']:
                calculation_details['warnings'].append(f"High workforce requirement ({constrained_workers}) - please review")
            
            # Update calculation details
            calculation_details.update({
                'raw_calculation': raw_workers,
                'rounded_workers': rounded_workers,
                'final_workers': constrained_workers,
                'min_crew_limit': min_crew,
                'max_crew_limit': max_crew,
                'hours_per_day': hours_per_day
            })
            
            return calculation_details
            
        except Exception as e:
            logger.error(f"Error in demand-based calculation for {punch_code}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return safe fallback
            min_crew, _ = self.get_crew_limits(punch_code)
            return {
                'punch_code': punch_code,
                'final_workers': min_crew,
                'method': 'error_fallback',
                'error': str(e),
                'warnings': ['Calculation error - using minimum crew size']
            }
    
    def calculate_hours_from_workers(self, workers: int, punch_code: str, 
                                   base_hours_per_worker: float = 8.0) -> float:
        """Calculate hours requirement from worker count"""
        try:
            # You can implement more sophisticated logic here
            # For now, simple multiplication
            return workers * base_hours_per_worker
        except Exception as e:
            logger.error(f"Error calculating hours for {punch_code}: {str(e)}")
            return workers * 8.0  # Safe fallback

# Global instance for use across the application
workforce_calculator = WorkforceCalculator()