#!/usr/bin/env python
"""
Initialize src/ data into the database
This script loads all CSV and shapefile data from the src/ folder into the database
for improved performance and centralized data management.
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DatabaseManager
from config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_src_directories():
    """Check if all required src/ directories exist"""
    required_dirs = [
        'src/polygons/tapsifood_marketing_areas',
        'src/polygons/tehran_districts',
        'src/vendor',
        'src/targets'
    ]
    
    missing_dirs = []
    existing_files = {}
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            # List files in the directory
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            existing_files[dir_path] = files
            logger.info(f"✓ Found directory: {dir_path} ({len(files)} files)")
        else:
            missing_dirs.append(dir_path)
            logger.warning(f"✗ Missing directory: {dir_path}")
    
    return missing_dirs, existing_files

def initialize_database_with_src_data(force_reload=False):
    """Initialize the database with all src/ data"""
    logger.info("=" * 60)
    logger.info("Initializing Database with src/ Data")
    logger.info("=" * 60)
    
    # Check directories
    missing_dirs, existing_files = check_src_directories()
    
    if missing_dirs:
        logger.error(f"Missing required directories: {missing_dirs}")
        logger.error("Please ensure all src/ directories are present")
        return False
    
    # Initialize database manager
    config = get_config()
    db_manager = DatabaseManager(config.DATABASE_PATH)
    
    logger.info(f"Using database: {config.DATABASE_PATH}")
    
    # Get initial stats
    initial_stats = db_manager.get_database_stats()
    logger.info(f"Initial database size: {initial_stats.get('database_size_bytes', 0) / 1024 / 1024:.2f} MB")
    
    # Initialize src/ data
    logger.info("\nStep 1: Loading spatial data...")
    results = db_manager.initialize_src_data(force_reload=force_reload)
    
    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("Data Loading Results:")
    logger.info("=" * 60)
    
    success_count = 0
    for data_type, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{data_type}: {status}")
        if success:
            success_count += 1
    
    # Get final stats
    final_stats = db_manager.get_database_stats()
    final_size_mb = final_stats.get('database_size_bytes', 0) / 1024 / 1024
    
    logger.info("\n" + "=" * 60)
    logger.info("Database Statistics:")
    logger.info("=" * 60)
    logger.info(f"Final database size: {final_size_mb:.2f} MB")
    
    # Show counts for each table
    table_counts = {
        'Marketing Areas': final_stats.get('marketing_areas_count', 0),
        'District Boundaries': final_stats.get('district_boundaries_count', 0),
        'Coverage Targets': final_stats.get('coverage_targets_count', 0),
        'Vendors': final_stats.get('vendors_count', 0),
        'Orders': final_stats.get('orders_count', 0)
    }
    
    for table_name, count in table_counts.items():
        logger.info(f"{table_name}: {count:,} records")
    
    # Overall success
    logger.info("\n" + "=" * 60)
    if success_count == len(results):
        logger.info("🎉 ALL DATA LOADED SUCCESSFULLY!")
        logger.info("The database is now ready for optimized operations.")
        
        # Test spatial data access
        logger.info("\nTesting spatial data access...")
        try:
            # Test marketing areas
            ma_tehran = db_manager.get_marketing_areas('tehran')
            logger.info(f"✓ Tehran marketing areas: {len(ma_tehran)} loaded")
            
            # Test district boundaries
            districts = db_manager.get_district_boundaries('tehran', 'region')
            logger.info(f"✓ Tehran region districts: {len(districts)} loaded")
            
            # Test coverage targets
            targets = db_manager.get_coverage_targets()
            logger.info(f"✓ Coverage targets: {len(targets)} loaded")
            
            # Test target lookup
            target_lookup = db_manager.get_target_lookup_dict('tehran')
            logger.info(f"✓ Target lookup dictionary: {len(target_lookup)} entries")
            
            logger.info("✅ All spatial data access tests passed!")
            
        except Exception as e:
            logger.error(f"❌ Error testing spatial data access: {e}")
            return False
            
    else:
        failed_count = len(results) - success_count
        logger.warning(f"⚠️ {failed_count} data types failed to load")
        logger.warning("Some features may not work properly")
    
    logger.info("=" * 60)
    return success_count == len(results)

def show_database_contents():
    """Show what's currently in the database"""
    logger.info("=" * 60)
    logger.info("Current Database Contents:")
    logger.info("=" * 60)
    
    config = get_config()
    db_manager = DatabaseManager(config.DATABASE_PATH)
    
    stats = db_manager.get_database_stats()
    
    # Show all tables
    tables_info = [
        ("Orders", stats.get('orders_count', 0)),
        ("Vendors", stats.get('vendors_count', 0)),
        ("Marketing Areas", stats.get('marketing_areas_count', 0)),
        ("District Boundaries", stats.get('district_boundaries_count', 0)),
        ("Coverage Targets", stats.get('coverage_targets_count', 0)),
        ("Coverage Grid Cache", stats.get('coverage_grid_cache_count', 0)),
        ("Heatmap Cache", stats.get('heatmap_cache_count', 0))
    ]
    
    for table_name, count in tables_info:
        logger.info(f"{table_name}: {count:,} records")
    
    # Show last update times
    logger.info("\nLast Updates:")
    update_keys = [
        'vendors_last_update', 'orders_last_update',
        'marketing_areas_loaded', 'district_boundaries_loaded', 'coverage_targets_loaded'
    ]
    
    for key in update_keys:
        value = stats.get(key, 'Never')
        logger.info(f"{key}: {value}")
    
    # Database size
    size_mb = stats.get('database_size_bytes', 0) / 1024 / 1024
    logger.info(f"\nDatabase size: {size_mb:.2f} MB")
    
    logger.info("=" * 60)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize src/ data into database")
    parser.add_argument('--force-reload', action='store_true', 
                       help='Force reload all data even if already loaded')
    parser.add_argument('--show-contents', action='store_true',
                       help='Show current database contents and exit')
    
    args = parser.parse_args()
    
    if args.show_contents:
        show_database_contents()
        return
    
    try:
        success = initialize_database_with_src_data(force_reload=args.force_reload)
        
        if success:
            logger.info("\n🚀 You can now run the optimized application:")
            logger.info("python run_production_optimized.py")
            sys.exit(0)
        else:
            logger.error("\n❌ Initialization failed. Please check the logs above.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
