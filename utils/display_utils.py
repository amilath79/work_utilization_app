"""
Display utilities for punch code names and formatting
"""
from config import PUNCH_CODE_NAMES

def get_display_name(punch_code):
    """Get friendly display name for punch code"""
    return PUNCH_CODE_NAMES.get(str(punch_code), str(punch_code))

def get_punch_code_tooltip(punch_code):
    """Get tooltip text combining code and name"""
    name = get_display_name(punch_code)
    if name != str(punch_code):
        return f"Code: {punch_code} - {name}"
    return f"Code: {punch_code}"