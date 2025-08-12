"""
Enhanced Training Script (train_models3.py)
Next-generation time-series workforce prediction training
Based on proven methodology achieving R² > 0.99 for punch code 217
"""
import argparse
import logging
import os
import sys
import traceback
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.time_series_trainer import train_punch_code_with_new_pipeline
from config import MODELS_DIR, ENHANCED_WORK_TYPES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "time_series_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("time_series_train")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def print_banner():
    """Print training banner"""
    print("=" * 70)
    print("🚀 ENHANCED TIME-SERIES WORKFORCE PREDICTION TRAINING")
    print("   Based on proven methodology: R² > 0.99, MAE < 1.0")
    print("=" * 70)

def train_specific_punch_code(punch_code: str, data_source: str = None) -> bool:
    """Train a specific punch code using time-series methodology"""
    
    logger.info(f"🎯 Starting time-series training for punch code: {punch_code}")
    
    # Use the universal training function
    return train_punch_code_with_new_pipeline(punch_code, data_source)

def validate_punch_code(punch_code: str) -> bool:
    """Validate that punch code is supported"""
    
    # Now support ALL enhanced punch codes
    if punch_code not in ENHANCED_WORK_TYPES:
        logger.error(f"❌ Punch code {punch_code} not in enhanced work types")
        logger.info(f"📋 Supported codes: {ENHANCED_WORK_TYPES}")
        return False
    
    return True

def main():
    """Main training function with enhanced argument parsing"""
    
    print_banner()
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Enhanced Time-Series Workforce Prediction Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models3.py --punch-code 217                    # Train punch code 217
  python train_models3.py -p 217 --data-file data.xlsx       # Train with custom data file
  python train_models3.py --list                              # List available punch codes
        """
    )
    
    parser.add_argument(
        '--punch-code', '-p', 
        type=str, 
        help='Train specific punch code (e.g., 217)'
    )
    
    parser.add_argument(
        '--data-file', '-d',
        type=str,
        help='Optional: Custom data file path (.xlsx or .pkl)'
    )
    
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List available punch codes for training'
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        print("📋 AVAILABLE PUNCH CODES FOR TIME-SERIES TRAINING:")
        print("   ✅ ALL ENHANCED PUNCH CODES NOW SUPPORTED:")
        for i, code in enumerate(ENHANCED_WORK_TYPES, 1):
            print(f"   {i:2d}. {code}")
        print("\n🚀 USAGE EXAMPLES:")
        print("   python train_models3.py --punch-code 217")
        print("   python train_models3.py --punch-code 206") 
        print("   python train_models3.py --punch-code 202")
        return
    
    # Validate arguments
    if not args.punch_code:
        logger.error("❌ Please specify a punch code to train")
        logger.info("💡 Use: python train_models3.py --punch-code 217")
        logger.info("💡 Or: python train_models3.py --list (to see available codes)")
        return
    
    # Validate punch code
    if not validate_punch_code(args.punch_code):
        return
    
    # Validate data file if provided
    if args.data_file and not os.path.exists(args.data_file):
        logger.error(f"❌ Data file not found: {args.data_file}")
        return
    
    # Start training
    logger.info(f"🚀 Starting enhanced training for punch code: {args.punch_code}")
    if args.data_file:
        logger.info(f"📁 Using custom data file: {args.data_file}")
    else:
        logger.info("📊 Loading data from SQL database")
    
    # Train the model
    start_time = datetime.now()
    
    success = train_specific_punch_code(args.punch_code, args.data_file)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print results
    print("\n" + "=" * 70)
    if success:
        logger.info(f"🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Training duration: {duration:.1f} seconds")
        logger.info(f"📁 Model saved to: {MODELS_DIR}")
        logger.info(f"🔄 Model ready for use in prediction interface")
        print("=" * 70)
        
        print("\n🚀 NEXT STEPS:")
        print("1. Test the new model in your prediction interface")
        print("2. Compare performance with previous 217 model")
        print("3. If satisfied, we can implement other punch codes")
        
    else:
        logger.error(f"❌ TRAINING FAILED for punch code {args.punch_code}")
        logger.info(f"⏱️  Duration before failure: {duration:.1f} seconds")
        logger.info("💡 Check the logs above for specific error details")
        print("=" * 70)

if __name__ == "__main__":
    main()