# scheduler_fixed.py - Fixed version of scheduler with data type handling
import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
import pandas as pd
import traceback
from typing import Optional
import os

import requests
from models import DatabaseManager
from config import Config

logger = logging.getLogger(__name__)

def fetch_question_data(question_id: int, metabase_url: str, username: str, password: str, team: str = None, workers: int = None, page_size: int = None) -> pd.DataFrame:
    """Fetch data from Metabase question"""
    try:
        # Login to Metabase
        login_url = f"{metabase_url}/api/session"
        login_data = {"username": username, "password": password}
        response = requests.post(login_url, json=login_data)
        response.raise_for_status()
        session_id = response.json()["id"]
        
        # Fetch question data
        headers = {"X-Metabase-Session": session_id}
        data_url = f"{metabase_url}/api/card/{question_id}/query/csv"
        response = requests.post(data_url, headers=headers)
        response.raise_for_status()
        
        # Convert to DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        logger.info(f"Fetched {len(df)} rows from Metabase question {question_id}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from Metabase question {question_id}: {e}")
        return pd.DataFrame()

def fix_timestamp_columns(df):
    """Convert timestamp columns to proper string format for SQLite"""
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['time', 'date', 'created_at', 'updated_at']):
            if col in df.columns and not df[col].empty:
                try:
                    # Convert to datetime first, then to string
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # Replace NaT with None
                    df[col] = df[col].replace('NaT', None)
                    df[col] = df[col].where(pd.notna(df[col]), None)
                except Exception as e:
                    logger.warning(f"Could not convert timestamp column {col}: {e}")
                    # If conversion fails, convert to string anyway
                    df[col] = df[col].astype(str).replace('nan', None)
    return df

def fix_vendor_grades(df_vendors):
    """Fix vendor grade handling to avoid categorical issues"""
    # Handle grades file loading with better error handling
    grades_file = os.path.join('src', 'vendor', 'graded.csv')
    
    if os.path.exists(grades_file):
        try:
            df_graded = pd.read_csv(grades_file, dtype={'vendor_code': 'str'})
            
            # Ensure vendor_code is string in both dataframes
            if 'vendor_code' in df_vendors.columns:
                df_vendors['vendor_code'] = df_vendors['vendor_code'].astype(str)
                df_graded['vendor_code'] = df_graded['vendor_code'].astype(str)
                
                # Remove existing grade column if it exists
                if 'grade' in df_vendors.columns:
                    df_vendors = df_vendors.drop('grade', axis=1)
                
                # Merge grades
                df_vendors = pd.merge(
                    df_vendors, 
                    df_graded[['vendor_code', 'grade']], 
                    on='vendor_code', 
                    how='left'
                )
                
                logger.info(f"Merged grades for {df_vendors['grade'].notna().sum()} vendors")
        except Exception as e:
            logger.warning(f"Could not load or merge grades file: {e}")
    
    # Ensure grade column exists and handle missing values
    if 'grade' not in df_vendors.columns:
        df_vendors['grade'] = 'Ungraded'
    else:
        df_vendors['grade'] = df_vendors['grade'].fillna('Ungraded').astype(str)
    
    return df_vendors

class DataScheduler:
    """Handles scheduled data fetching and updates with fixed data type handling"""
    
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.running = False
        self.scheduler_thread = None
        
        # City mapping
        self.city_id_map = {1: "mashhad", 2: "tehran", 5: "shiraz"}
        self.city_name_to_id_map = {v: k for k, v in self.city_id_map.items()}
        
    def start(self):
        """Start the background scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
            
        self.running = True
        
        # Schedule jobs
        schedule.every(10).minutes.do(self._fetch_vendors_job)
        schedule.every().day.at("09:00").do(self._fetch_orders_job)
        
        # Also schedule cache cleanup
        schedule.every().day.at("02:00").do(self._cleanup_cache_job)
        
        # Run initial data check if database is empty
        self._initial_data_check()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Data scheduler started successfully")
        
    def stop(self):
        """Stop the background scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Data scheduler stopped")
        
    def _run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
                
    def _initial_data_check(self):
        """Check if we need to fetch initial data"""
        try:
            # Check if we have recent data
            vendors_last_update = self.db_manager.get_metadata("vendors_last_update")
            orders_last_update = self.db_manager.get_metadata("orders_last_update")
            
            current_time = datetime.now()
            
            # If no vendors data or older than 10 minutes, fetch vendors
            if not vendors_last_update:
                logger.info("No vendors data found, fetching initial vendors data...")
                self._fetch_vendors_job()
            else:
                last_update = datetime.fromisoformat(vendors_last_update)
                if current_time - last_update > timedelta(minutes=10):
                    logger.info("Vendors data is stale, fetching fresh data...")
                    self._fetch_vendors_job()
                    
            # If no orders data or older than 1 day, fetch orders
            if not orders_last_update:
                logger.info("No orders data found, fetching initial orders data...")
                self._fetch_orders_job()
            else:
                last_update = datetime.fromisoformat(orders_last_update)
                if current_time - last_update > timedelta(days=1):
                    logger.info("Orders data is stale, fetching fresh data...")
                    self._fetch_orders_job()
                    
        except Exception as e:
            logger.error(f"Error in initial data check: {e}")
            
    def _fetch_vendors_job(self):
        """Scheduled job to fetch vendors data every 10 minutes"""
        try:
            logger.info("🚀 Starting scheduled vendors data fetch...")
            start_time = time.time()
            
            # Fetch vendors data from Metabase
            df_vendors_raw = fetch_question_data(
                question_id=self.config.VENDOR_DATA_QUESTION_ID,
                metabase_url=self.config.METABASE_URL,
                username=self.config.METABASE_USERNAME,
                password=self.config.METABASE_PASSWORD,
                team="growth",
                workers=self.config.WORKER_COUNT,
                page_size=self.config.PAGE_SIZE
            )
            
            if df_vendors_raw is None or df_vendors_raw.empty:
                logger.error("Failed to fetch vendors data or received empty dataset")
                return
                
            # Process and clean vendors data with fixes
            df_vendors = self._process_vendors_data(df_vendors_raw)
            
            # Store in database
            inserted_count = self.db_manager.upsert_vendors(df_vendors)
            
            # Update metadata
            self.db_manager.set_metadata("vendors_last_update", datetime.now().isoformat())
            self.db_manager.set_metadata("vendors_count", str(len(df_vendors)))
            
            duration = time.time() - start_time
            logger.info(f"✅ Vendors data updated successfully: {inserted_count} records in {duration:.2f}s")
            
            # Clear related caches since vendor data changed
            self._invalidate_vendor_related_caches()
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch vendors data: {e}")
            logger.error(traceback.format_exc())
            
    def _fetch_orders_job(self):
        """Scheduled job to fetch orders data daily at 9am"""
        try:
            logger.info("🚀 Starting scheduled orders data fetch...")
            start_time = time.time()
            
            # Fetch orders data from Metabase
            df_orders_raw = fetch_question_data(
                question_id=self.config.ORDER_DATA_QUESTION_ID,
                metabase_url=self.config.METABASE_URL,
                username=self.config.METABASE_USERNAME,
                password=self.config.METABASE_PASSWORD,
                team="growth",
                workers=self.config.WORKER_COUNT,
                page_size=self.config.PAGE_SIZE
            )
            
            if df_orders_raw is None or df_orders_raw.empty:
                logger.error("Failed to fetch orders data or received empty dataset")
                return
                
            # Process and clean orders data with fixes
            df_orders = self._process_orders_data(df_orders_raw)
            
            # Store in database (this might take a while for large datasets)
            logger.info(f"Inserting {len(df_orders)} orders into database...")
            inserted_count = self.db_manager.upsert_orders(df_orders)
            
            # Update metadata
            self.db_manager.set_metadata("orders_last_update", datetime.now().isoformat())
            self.db_manager.set_metadata("orders_count", str(len(df_orders)))
            
            duration = time.time() - start_time
            logger.info(f"✅ Orders data updated successfully: {inserted_count} records in {duration:.2f}s")
            
            # Clear related caches since order data changed
            self._invalidate_order_related_caches()
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch orders data: {e}")
            logger.error(traceback.format_exc())
            
    def _process_vendors_data(self, df_vendors_raw: pd.DataFrame) -> pd.DataFrame:
        """Process and clean vendors data with improved error handling"""
        df_vendors = df_vendors_raw.copy()
        
        # Fix timestamp columns first
        df_vendors = fix_timestamp_columns(df_vendors)
        
        # Add city name mapping
        if 'city_id' in df_vendors.columns:
            df_vendors['city_name'] = df_vendors['city_id'].map(self.city_id_map)
            # Convert to string instead of category to avoid issues
            df_vendors['city_name'] = df_vendors['city_name'].astype(str)
        
        # Handle grades with improved error handling
        df_vendors = fix_vendor_grades(df_vendors)
            
        # Ensure required columns exist
        required_columns = ['latitude', 'longitude', 'vendor_name', 'radius', 'status_id', 'visible', 'open', 'vendor_code']
        for col in required_columns:
            if col not in df_vendors.columns:
                df_vendors[col] = None
                
        # Clean data types with better error handling
        for col, convert_func in [('visible', pd.to_numeric), ('open', pd.to_numeric), ('status_id', pd.to_numeric)]:
            if col in df_vendors.columns:
                try:
                    df_vendors[col] = convert_func(df_vendors[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
                    
        if 'vendor_code' in df_vendors.columns:
            df_vendors['vendor_code'] = df_vendors['vendor_code'].astype(str)
            
        # Store original radius for reset functionality
        if 'radius' in df_vendors.columns:
            df_vendors['original_radius'] = df_vendors['radius'].copy()
            
        # business_line is now included directly in vendor data from Metabase
        # Ensure business_line column exists and handle any missing values
        if 'business_line' not in df_vendors.columns:
            df_vendors['business_line'] = None
        else:
            # Convert to string and handle NaN values
            df_vendors['business_line'] = df_vendors['business_line'].astype(str).replace('nan', None)
            bl_count = df_vendors['business_line'].notna().sum()
            logger.info(f"Found business_line data for {bl_count}/{len(df_vendors)} vendors")
            
        return df_vendors
        
    def _process_orders_data(self, df_orders_raw: pd.DataFrame) -> pd.DataFrame:
        """Process and clean orders data with improved error handling"""
        df_orders = df_orders_raw.copy()
        
        # Fix timestamp columns first
        df_orders = fix_timestamp_columns(df_orders)
        
        # Add city name mapping
        if 'city_id' in df_orders.columns:
            df_orders['city_name'] = df_orders['city_id'].map(self.city_id_map)
            # Fill NaN values with proper mapping, then convert to string
            df_orders['city_name'] = df_orders['city_name'].fillna('unknown').astype(str)
            
            # Double-check: if we still have 'nan' strings, fix them
            df_orders.loc[df_orders['city_name'] == 'nan', 'city_name'] = 'unknown'
            
        # Handle organic column
        if 'organic' not in df_orders.columns:
            # Create synthetic organic data for demo (70% non-organic, 30% organic)
            import numpy as np
            df_orders['organic'] = np.random.choice([0, 1], size=len(df_orders), p=[0.7, 0.3])
        
        # Convert to standard types instead of categories to avoid SQLite issues
        if 'organic' in df_orders.columns:
            df_orders['organic'] = pd.to_numeric(df_orders['organic'], errors='coerce').fillna(0).astype(int)
            
        # Set string types for text columns to avoid categorical issues
        text_columns = ['business_line', 'marketing_area', 'city_name', 'vendor_code', 'user_id']
        for col in text_columns:
            if col in df_orders.columns:
                df_orders[col] = df_orders[col].astype(str)
                
        return df_orders
        
    def _cleanup_cache_job(self):
        """Scheduled job to clean up old cache entries"""
        try:
            logger.info("🧹 Starting cache cleanup...")
            self.db_manager.cleanup_old_cache(days_old=7)
            logger.info("✅ Cache cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")
            
    def _invalidate_vendor_related_caches(self):
        """Invalidate caches that depend on vendor data"""
        try:
            with self.db_manager.get_connection() as conn:
                # Clear coverage grid cache as it depends on vendor locations
                conn.execute("DELETE FROM coverage_grid_cache")
                conn.commit()
                logger.info("Invalidated vendor-related caches")
        except Exception as e:
            logger.error(f"Failed to invalidate vendor-related caches: {e}")
            
    def _invalidate_order_related_caches(self):
        """Invalidate caches that depend on order data"""
        try:
            with self.db_manager.get_connection() as conn:
                # Clear heatmap cache as it depends on order data
                conn.execute("DELETE FROM heatmap_cache")
                conn.commit()
                logger.info("Invalidated order-related caches")
        except Exception as e:
            logger.error(f"Failed to invalidate order-related caches: {e}")
            
    def force_vendors_update(self):
        """Manually trigger vendors data update"""
        logger.info("🔄 Force updating vendors data...")
        self._fetch_vendors_job()
        
    def force_orders_update(self):
        """Manually trigger orders data update"""
        logger.info("🔄 Force updating orders data...")
        self._fetch_orders_job()
        
    def get_status(self) -> dict:
        """Get scheduler status and statistics"""
        stats = self.db_manager.get_database_stats()
        
        return {
            "running": self.running,
            "database_stats": stats,
            "last_vendors_update": self.db_manager.get_metadata("vendors_last_update"),
            "last_orders_update": self.db_manager.get_metadata("orders_last_update"),
            "next_scheduled_runs": [str(job) for job in schedule.jobs]
        }

# Singleton instance for easy access
_scheduler_instance = None

def get_scheduler() -> Optional[DataScheduler]:
    """Get the global scheduler instance"""
    return _scheduler_instance

def init_scheduler(config: Config, db_manager: DatabaseManager) -> DataScheduler:
    """Initialize the global scheduler instance"""
    global _scheduler_instance
    _scheduler_instance = DataScheduler(config, db_manager)
    return _scheduler_instance