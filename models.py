# models.py - Enhanced version with src/ data integration
import sqlite3
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import json
import logging
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import hashlib
from shapely import wkt

logger = logging.getLogger(__name__)

def convert_timestamps_to_string(df, timestamp_columns=None):
    """Convert timestamp columns to string format for SQLite compatibility"""
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # If specific columns are provided, use those
    if timestamp_columns:
        cols_to_convert = timestamp_columns
    else:
        # Auto-detect timestamp columns
        cols_to_convert = []
        for col in df_copy.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'created_at', 'updated_at']):
                cols_to_convert.append(col)
            # Also check for actual datetime/timestamp types
            elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                cols_to_convert.append(col)
    
    # Convert each timestamp column
    for col in cols_to_convert:
        if col in df_copy.columns:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                
                # Convert to string format
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Handle NaT values
                df_copy[col] = df_copy[col].replace('NaT', None)
                df_copy[col] = df_copy[col].where(pd.notna(df_copy[col]), None)
                
                logger.debug(f"Converted timestamp column '{col}' to string format")
                
            except Exception as e:
                logger.warning(f"Could not convert timestamp column '{col}': {e}")
                # Fallback: convert to string anyway
                df_copy[col] = df_copy[col].astype(str).replace('nan', None)
    
    return df_copy

class DatabaseManager:
    """Enhanced database manager with src/ data integration"""
    
    def __init__(self, db_path: str = "tapsi_food_data.db"):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with proper cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Better for concurrent access
            conn.execute("PRAGMA synchronous = NORMAL")  # Balance between safety and speed
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database with all required tables and indexes"""
        with self.get_connection() as conn:
            # Metadata table for tracking updates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            
            # Orders table with proper indexing
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    vendor_code TEXT,
                    city_id INTEGER,
                    city_name TEXT,
                    business_line TEXT,
                    marketing_area TEXT,
                    customer_latitude REAL,
                    customer_longitude REAL,
                    user_id TEXT,
                    organic INTEGER,
                    created_at TEXT,
                    imported_at TEXT,
                    UNIQUE(order_id, vendor_code, created_at)
                )
            """)
            
            # Vendors table with spatial capabilities
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vendors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vendor_code TEXT UNIQUE,
                    vendor_name TEXT,
                    city_id INTEGER,
                    city_name TEXT,
                    business_line TEXT,
                    latitude REAL,
                    longitude REAL,
                    radius REAL,
                    original_radius REAL,
                    status_id REAL,
                    visible REAL,
                    open REAL,
                    grade TEXT,
                    updated_at TEXT
                )
            """)
            
            # NEW: Marketing areas table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS marketing_areas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    area_id TEXT UNIQUE,
                    city_name TEXT,
                    name TEXT,
                    wkt_geometry TEXT,
                    created_at TEXT
                )
            """)
            
            # NEW: District boundaries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS district_boundaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    district_id TEXT,
                    city_name TEXT,
                    district_type TEXT,
                    name TEXT,
                    wkt_geometry TEXT,
                    pop INTEGER,
                    pop_density REAL,
                    created_at TEXT
                )
            """)
            
            # NEW: Coverage targets table  
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    area_id TEXT,
                    marketing_area_name TEXT,
                    business_line TEXT,
                    target_value INTEGER,
                    created_at TEXT
                )
            """)
            
            # Coverage grid cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_grid_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE,
                    city_name TEXT,
                    business_line TEXT,
                    vendor_filters TEXT,  -- JSON string of filters
                    grid_data TEXT,       -- JSON string of grid results
                    point_count INTEGER,
                    created_at TEXT,
                    last_accessed TEXT
                )
            """)
            
            # Heatmap cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS heatmap_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE,
                    heatmap_type TEXT,
                    city_name TEXT,
                    date_range TEXT,
                    business_line TEXT,
                    zoom_level INTEGER,
                    heatmap_data TEXT,    -- JSON string of heatmap points
                    created_at TEXT
                )
            """)
            
            # Create indexes for performance
            self._create_indexes(conn)
            conn.commit()
            
    def _create_indexes(self, conn):
        """Create all necessary indexes for optimal query performance"""
        indexes = [
            # Orders indexes
            "CREATE INDEX IF NOT EXISTS idx_orders_city_name ON orders(city_name)",
            "CREATE INDEX IF NOT EXISTS idx_orders_business_line ON orders(business_line)",
            "CREATE INDEX IF NOT EXISTS idx_orders_vendor_code ON orders(vendor_code)",
            "CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_orders_location ON orders(customer_latitude, customer_longitude)",
            "CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_marketing_area ON orders(marketing_area)",
            
            # Vendors indexes
            "CREATE INDEX IF NOT EXISTS idx_vendors_city_name ON vendors(city_name)",
            "CREATE INDEX IF NOT EXISTS idx_vendors_business_line ON vendors(business_line)",
            "CREATE INDEX IF NOT EXISTS idx_vendors_location ON vendors(latitude, longitude)",
            "CREATE INDEX IF NOT EXISTS idx_vendors_status_id ON vendors(status_id)",
            "CREATE INDEX IF NOT EXISTS idx_vendors_grade ON vendors(grade)",
            "CREATE INDEX IF NOT EXISTS idx_vendors_visible ON vendors(visible)",
            "CREATE INDEX IF NOT EXISTS idx_vendors_open ON vendors(open)",
            
            # NEW: Spatial data indexes
            "CREATE INDEX IF NOT EXISTS idx_marketing_areas_city ON marketing_areas(city_name)",
            "CREATE INDEX IF NOT EXISTS idx_marketing_areas_name ON marketing_areas(name)",
            "CREATE INDEX IF NOT EXISTS idx_districts_city_type ON district_boundaries(city_name, district_type)",
            "CREATE INDEX IF NOT EXISTS idx_districts_name ON district_boundaries(name)",
            "CREATE INDEX IF NOT EXISTS idx_targets_area_bl ON coverage_targets(area_id, business_line)",
            
            # Cache indexes
            "CREATE INDEX IF NOT EXISTS idx_coverage_cache_key ON coverage_grid_cache(cache_key)",
            "CREATE INDEX IF NOT EXISTS idx_coverage_city_bl ON coverage_grid_cache(city_name, business_line)",
            "CREATE INDEX IF NOT EXISTS idx_coverage_accessed ON coverage_grid_cache(last_accessed)",
            "CREATE INDEX IF NOT EXISTS idx_heatmap_cache_key ON heatmap_cache(cache_key)",
            "CREATE INDEX IF NOT EXISTS idx_heatmap_type_city ON heatmap_cache(heatmap_type, city_name)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.Error as e:
                logger.warning(f"Could not create index: {e}")

    # NEW: Methods for loading src/ data into database
    def load_marketing_areas_from_csv(self, force_reload: bool = False) -> bool:
        """Load marketing areas from CSV files into database"""
        if not force_reload:
            # Check if already loaded
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM marketing_areas")
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.info(f"Marketing areas already loaded ({count} records)")
                    return True
        
        marketing_areas_base = os.path.join('src', 'polygons', 'tapsifood_marketing_areas')
        city_files = ['mashhad_polygons.csv', 'tehran_polygons.csv', 'shiraz_polygons.csv']
        
        total_loaded = 0
        
        with self.get_connection() as conn:
            # Clear existing data if force reload
            if force_reload:
                conn.execute("DELETE FROM marketing_areas")
            
            for city_file in city_files:
                city_name = city_file.split('_')[0]
                file_path = os.path.join(marketing_areas_base, city_file)
                
                if not os.path.exists(file_path):
                    logger.warning(f"Marketing areas file not found: {file_path}")
                    continue
                
                try:
                    df_poly = pd.read_csv(file_path, encoding='utf-8')
                    
                    if 'WKT' not in df_poly.columns:
                        logger.warning(f"No WKT column in {city_file}")
                        continue
                    
                    # Create area IDs and clean names
                    df_poly['area_id'] = f"{city_name}_" + df_poly.index.astype(str)
                    
                    if 'name' not in df_poly.columns:
                        df_poly['name'] = [f"{city_name}_area_{i+1}" for i in range(len(df_poly))]
                    else:
                        df_poly['name'] = df_poly['name'].astype(str).str.strip()
                    
                    # Insert into database
                    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    for _, row in df_poly.iterrows():
                        conn.execute("""
                            INSERT OR REPLACE INTO marketing_areas 
                            (area_id, city_name, name, wkt_geometry, created_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            row['area_id'], city_name, row['name'], 
                            row['WKT'], created_at
                        ))
                    
                    total_loaded += len(df_poly)
                    logger.info(f"Loaded {len(df_poly)} marketing areas for {city_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading marketing areas from {city_file}: {e}")
            
            conn.commit()
        
        logger.info(f"Total marketing areas loaded: {total_loaded}")
        self.set_metadata("marketing_areas_loaded", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return total_loaded > 0
    
    def load_district_boundaries_from_shp(self, force_reload: bool = False) -> bool:
        """Load district boundaries from shapefiles into database"""
        if not force_reload:
            # Check if already loaded
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM district_boundaries")
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.info(f"District boundaries already loaded ({count} records)")
                    return True
        
        districts_base = os.path.join('src', 'polygons', 'tehran_districts')
        district_files = [
            ('RegionTehran_WGS1984.shp', 'region'),
            ('Tehran_WGS1984.shp', 'main')
        ]
        
        total_loaded = 0
        
        with self.get_connection() as conn:
            # Clear existing data if force reload
            if force_reload:
                conn.execute("DELETE FROM district_boundaries")
            
            for file_name, district_type in district_files:
                file_path = os.path.join(districts_base, file_name)
                
                if not os.path.exists(file_path):
                    logger.warning(f"District file not found: {file_path}")
                    continue
                
                try:
                    # Load shapefile with encoding detection
                    gdf = self._load_shapefile_with_encoding(file_path)
                    
                    if gdf.empty:
                        logger.warning(f"No data loaded from {file_name}")
                        continue
                    
                    # Ensure WGS84 projection
                    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")
                    
                    # Determine name column
                    name_col = 'Name' if 'Name' in gdf.columns else 'NAME_MAHAL'
                    if name_col not in gdf.columns:
                        logger.warning(f"No name column found in {file_name}")
                        continue
                    
                    # Insert into database
                    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    for idx, row in gdf.iterrows():
                        district_id = f"tehran_{district_type}_{idx}"
                        name = str(row[name_col]).strip()
                        wkt_geom = row['geometry'].wkt
                        pop = row.get('Pop', None)
                        pop_density = row.get('PopDensity', None)
                        
                        # Convert population values to appropriate types
                        if pop is not None:
                            try:
                                pop = int(float(pop))
                            except (ValueError, TypeError):
                                pop = None
                        
                        if pop_density is not None:
                            try:
                                pop_density = float(pop_density)
                            except (ValueError, TypeError):
                                pop_density = None
                        
                        conn.execute("""
                            INSERT OR REPLACE INTO district_boundaries 
                            (district_id, city_name, district_type, name, wkt_geometry, pop, pop_density, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            district_id, 'tehran', district_type, name, 
                            wkt_geom, pop, pop_density, created_at
                        ))
                    
                    total_loaded += len(gdf)
                    logger.info(f"Loaded {len(gdf)} {district_type} districts")
                    
                except Exception as e:
                    logger.error(f"Error loading districts from {file_name}: {e}")
            
            conn.commit()
        
        logger.info(f"Total district boundaries loaded: {total_loaded}")
        self.set_metadata("district_boundaries_loaded", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return total_loaded > 0
    
    def load_coverage_targets_from_csv(self, force_reload: bool = False) -> bool:
        """Load coverage targets from CSV into database"""
        if not force_reload:
            # Check if already loaded
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM coverage_targets")
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.info(f"Coverage targets already loaded ({count} records)")
                    return True
        
        targets_file = os.path.join('src', 'targets', 'tehran_coverage.csv')
        
        if not os.path.exists(targets_file):
            logger.warning(f"Coverage targets file not found: {targets_file}")
            return False
        
        try:
            # Load marketing areas mapping for area_id lookup
            area_name_to_id = {}
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, area_id FROM marketing_areas WHERE city_name = 'tehran'")
                area_name_to_id = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Load targets CSV
            df_targets = pd.read_csv(targets_file, encoding='utf-8')
            
            if 'marketing_area' not in df_targets.columns:
                logger.error("No marketing_area column in targets file")
                return False
            
            df_targets['marketing_area'] = df_targets['marketing_area'].str.strip()
            
            # Map area names to IDs
            df_targets['area_id'] = df_targets['marketing_area'].map(area_name_to_id)
            
            # Remove unmapped areas
            df_targets = df_targets.dropna(subset=['area_id'])
            
            # Melt to long format
            df_melted = df_targets.melt(
                id_vars=['area_id', 'marketing_area'],
                var_name='business_line',
                value_name='target_value'
            )
            
            # Remove null targets
            df_melted = df_melted.dropna(subset=['target_value'])
            
            # Insert into database
            with self.get_connection() as conn:
                if force_reload:
                    conn.execute("DELETE FROM coverage_targets")
                
                created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                for _, row in df_melted.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO coverage_targets 
                        (area_id, marketing_area_name, business_line, target_value, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        row['area_id'], row['marketing_area'], 
                        row['business_line'], int(row['target_value']), created_at
                    ))
                
                conn.commit()
            
            logger.info(f"Loaded {len(df_melted)} coverage targets")
            self.set_metadata("coverage_targets_loaded", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            return True
            
        except Exception as e:
            logger.error(f"Error loading coverage targets: {e}")
            return False
    
    def _load_shapefile_with_encoding(self, file_path: str) -> gpd.GeoDataFrame:
        """Load shapefile with automatic encoding detection"""
        tried_encodings = [None, 'cp1256', 'utf-8', 'latin1']
        
        for enc in tried_encodings:
            try:
                gdf = gpd.read_file(file_path, encoding=enc)
                logger.debug(f"Loaded {file_path} using encoding='{enc or 'default'}'")
                return gdf
            except Exception as e:
                logger.debug(f"Failed to load {file_path} with encoding {enc}: {e}")
        
        logger.error(f"Could not load {file_path} with any encoding")
        return gpd.GeoDataFrame()
    
    def initialize_src_data(self, force_reload: bool = False) -> Dict[str, bool]:
        """Initialize all src/ data into database"""
        logger.info("Initializing src/ data into database...")
        
        results = {}
        
        # Load marketing areas first (needed for targets)
        results['marketing_areas'] = self.load_marketing_areas_from_csv(force_reload)
        
        # Load district boundaries
        results['district_boundaries'] = self.load_district_boundaries_from_shp(force_reload)
        
        # Load coverage targets (depends on marketing areas)
        results['coverage_targets'] = self.load_coverage_targets_from_csv(force_reload)
        
        logger.info(f"Src data initialization results: {results}")
        return results

    # NEW: Enhanced getter methods for spatial data
    def get_marketing_areas(self, city_name: str = None) -> gpd.GeoDataFrame:
        """Get marketing areas as GeoDataFrame"""
        where_clause = "WHERE city_name = ?" if city_name else ""
        params = [city_name] if city_name else []
        
        with self.get_connection() as conn:
            query = f"SELECT * FROM marketing_areas {where_clause}"
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return gpd.GeoDataFrame()
            
            # Convert WKT to geometry
            df['geometry'] = df['wkt_geometry'].apply(wkt.loads)
            return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    def get_district_boundaries(self, city_name: str = 'tehran', district_type: str = None) -> gpd.GeoDataFrame:
        """Get district boundaries as GeoDataFrame"""
        where_conditions = ["city_name = ?"]
        params = [city_name]
        
        if district_type:
            where_conditions.append("district_type = ?")
            params.append(district_type)
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        with self.get_connection() as conn:
            query = f"SELECT * FROM district_boundaries {where_clause}"
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return gpd.GeoDataFrame()
            
            # Convert WKT to geometry
            df['geometry'] = df['wkt_geometry'].apply(wkt.loads)
            return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    def get_coverage_targets(self, area_id: str = None, business_line: str = None) -> pd.DataFrame:
        """Get coverage targets"""
        where_conditions = []
        params = []
        
        if area_id:
            where_conditions.append("area_id = ?")
            params.append(area_id)
        
        if business_line:
            where_conditions.append("business_line = ?")
            params.append(business_line)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        with self.get_connection() as conn:
            query = f"SELECT * FROM coverage_targets {where_clause}"
            return pd.read_sql_query(query, conn, params=params)
    
    def get_target_lookup_dict(self, city_name: str = 'tehran') -> Dict[tuple[str, str], int]:
        """Get target lookup dictionary for fast access"""
        with self.get_connection() as conn:
            query = """
                SELECT ct.area_id, ct.business_line, ct.target_value 
                FROM coverage_targets ct
                JOIN marketing_areas ma ON ct.area_id = ma.area_id
                WHERE ma.city_name = ?
            """
            cursor = conn.cursor()
            cursor.execute(query, (city_name,))
            
            return {(row[0], row[1]): row[2] for row in cursor.fetchall()}

    # Existing methods (unchanged but included for completeness)
    def upsert_orders(self, df_orders: pd.DataFrame) -> int:
        """Insert or update orders data with conflict resolution"""
        if df_orders.empty:
            return 0
            
        # Prepare data and handle timestamps
        df_clean = df_orders.copy()
        df_clean['imported_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # String format directly
        
        # Convert any timestamp columns to string format
        df_clean = convert_timestamps_to_string(df_clean)
        
        # Convert to records for insertion
        records = df_clean.to_dict('records')
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Use INSERT OR REPLACE for upsert behavior
            insert_sql = """
                INSERT OR REPLACE INTO orders (
                    order_id, vendor_code, city_id, city_name, business_line,
                    marketing_area, customer_latitude, customer_longitude,
                    user_id, organic, created_at, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Batch insert for performance
            batch_size = 1000
            inserted_count = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_data = [
                    (
                        r.get('order_id'), r.get('vendor_code'), r.get('city_id'),
                        r.get('city_name'), r.get('business_line'), r.get('marketing_area'),
                        r.get('customer_latitude'), r.get('customer_longitude'),
                        r.get('user_id'), r.get('organic'), r.get('created_at'),
                        r.get('imported_at')
                    ) for r in batch
                ]
                
                cursor.executemany(insert_sql, batch_data)
                inserted_count += len(batch)
                
                if i % 10000 == 0:  # Progress logging
                    logger.info(f"Inserted {inserted_count}/{len(records)} orders...")
            
            conn.commit()
            logger.info(f"Successfully upserted {inserted_count} orders")
            return inserted_count

    def upsert_vendors(self, df_vendors: pd.DataFrame) -> int:
        """Insert or update vendors data"""
        if df_vendors.empty:
            return 0
            
        df_clean = df_vendors.copy()
        df_clean['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # String format directly
        
        # Convert any timestamp columns to string format
        df_clean = convert_timestamps_to_string(df_clean)
        
        records = df_clean.to_dict('records')
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            insert_sql = """
                INSERT OR REPLACE INTO vendors (
                    vendor_code, vendor_name, city_id, city_name, business_line,
                    latitude, longitude, radius, original_radius, status_id,
                    visible, open, grade, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            batch_data = [
                (
                    r.get('vendor_code'), r.get('vendor_name'), r.get('city_id'),
                    r.get('city_name'), r.get('business_line'), r.get('latitude'),
                    r.get('longitude'), r.get('radius'), r.get('original_radius'),
                    r.get('status_id'), r.get('visible'), r.get('open'),
                    r.get('grade'), r.get('updated_at')
                ) for r in records
            ]
            
            cursor.executemany(insert_sql, batch_data)
            conn.commit()
            
            logger.info(f"Successfully upserted {len(records)} vendors")
            return len(records)

    def get_orders(self, 
                   city_name: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   business_lines: Optional[List[str]] = None,
                   vendor_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """Retrieve filtered orders data"""
        
        where_conditions = []
        params = []
        
        if city_name and city_name != "all":
            where_conditions.append("city_name = ?")
            params.append(city_name)
            
        if start_date:
            where_conditions.append("created_at >= ?")
            params.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
            
        if end_date:
            where_conditions.append("created_at <= ?")
            params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))
            
        if business_lines:
            placeholders = ','.join(['?' for _ in business_lines])
            where_conditions.append(f"business_line IN ({placeholders})")
            params.extend(business_lines)
            
        if vendor_codes:
            placeholders = ','.join(['?' for _ in vendor_codes])
            where_conditions.append(f"vendor_code IN ({placeholders})")
            params.extend(vendor_codes)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        sql = f"""
            SELECT * FROM orders 
            {where_clause}
            ORDER BY created_at DESC
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
            # Convert string timestamps back to datetime for processing
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            return df

    def get_vendors(self, 
                    city_name: Optional[str] = None,
                    business_lines: Optional[List[str]] = None,
                    vendor_codes: Optional[List[str]] = None,
                    status_ids: Optional[List[int]] = None,
                    grades: Optional[List[str]] = None,
                    visible: Optional[int] = None,
                    is_open: Optional[int] = None) -> pd.DataFrame:
        """Retrieve filtered vendors data"""
        
        where_conditions = []
        params = []
        
        if city_name and city_name != "all":
            where_conditions.append("city_name = ?")
            params.append(city_name)
            
        if business_lines:
            placeholders = ','.join(['?' for _ in business_lines])
            where_conditions.append(f"business_line IN ({placeholders})")
            params.extend(business_lines)
            
        if vendor_codes:
            placeholders = ','.join(['?' for _ in vendor_codes])
            where_conditions.append(f"vendor_code IN ({placeholders})")
            params.extend(vendor_codes)
            
        if status_ids:
            placeholders = ','.join(['?' for _ in status_ids])
            where_conditions.append(f"status_id IN ({placeholders})")
            params.extend(status_ids)
            
        if grades:
            placeholders = ','.join(['?' for _ in grades])
            where_conditions.append(f"grade IN ({placeholders})")
            params.extend(grades)
            
        if visible is not None:
            where_conditions.append("visible = ?")
            params.append(visible)
            
        if is_open is not None:
            where_conditions.append("open = ?")
            params.append(is_open)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        sql = f"SELECT * FROM vendors {where_clause}"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def cache_coverage_grid(self, cache_key: str, city_name: str, 
                           business_line: str, vendor_filters: Dict[str, Any],
                           grid_data: List[Dict]) -> bool:
        """Cache coverage grid results"""
        try:
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO coverage_grid_cache 
                    (cache_key, city_name, business_line, vendor_filters, grid_data, point_count, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key, city_name, business_line,
                    json.dumps(vendor_filters), json.dumps(grid_data), len(grid_data),
                    created_at, created_at
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to cache coverage grid: {e}")
            return False

    def get_cached_coverage_grid(self, cache_key: str) -> Optional[List[Dict]]:
        """Retrieve cached coverage grid results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT grid_data FROM coverage_grid_cache 
                    WHERE cache_key = ?
                """, (cache_key,))
                
                result = cursor.fetchone()
                if result:
                    # Update last accessed timestamp
                    last_accessed = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    conn.execute("""
                        UPDATE coverage_grid_cache 
                        SET last_accessed = ? 
                        WHERE cache_key = ?
                    """, (last_accessed, cache_key))
                    conn.commit()
                    
                    return json.loads(result[0])
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached coverage grid: {e}")
            return None

    def cache_heatmap(self, cache_key: str, heatmap_type: str, city_name: str,
                     date_range: str, business_line: str, zoom_level: int,
                     heatmap_data: List[Dict]) -> bool:
        """Cache heatmap results"""
        try:
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO heatmap_cache 
                    (cache_key, heatmap_type, city_name, date_range, business_line, zoom_level, heatmap_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key, heatmap_type, city_name, date_range,
                    business_line, zoom_level, json.dumps(heatmap_data), created_at
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to cache heatmap: {e}")
            return False

    def get_cached_heatmap(self, cache_key: str) -> Optional[List[Dict]]:
        """Retrieve cached heatmap results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT heatmap_data FROM heatmap_cache 
                    WHERE cache_key = ?
                """, (cache_key,))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached heatmap: {e}")
            return None

    def cleanup_old_cache(self, days_old: int = 7):
        """Clean up old cache entries to prevent database bloat"""
        cutoff_date = (datetime.now() - timedelta(days=days_old)).strftime('%Y-%m-%d %H:%M:%S')
        
        with self.get_connection() as conn:
            # Clean old coverage grid cache
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM coverage_grid_cache 
                WHERE last_accessed < ?
            """, (cutoff_date,))
            
            coverage_deleted = cursor.rowcount
            
            # Clean old heatmap cache
            cursor.execute("""
                DELETE FROM heatmap_cache 
                WHERE created_at < ?
            """, (cutoff_date,))
            
            heatmap_deleted = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleaned up {coverage_deleted} coverage cache entries and {heatmap_deleted} heatmap cache entries")

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else None

    def set_metadata(self, key: str, value: str):
        """Set metadata value"""
        updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, updated_at))
            conn.commit()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records in each table
            tables = ['orders', 'vendors', 'coverage_grid_cache', 'heatmap_cache', 
                     'marketing_areas', 'district_boundaries', 'coverage_targets']
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                except:
                    stats[f'{table}_count'] = 0
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            stats['database_size_bytes'] = cursor.fetchone()[0]
            
            # Last update times
            cursor.execute("SELECT key, updated_at FROM metadata WHERE key LIKE '%_last_update' OR key LIKE '%_loaded'")
            for key, updated_at in cursor.fetchall():
                stats[key] = updated_at
            
            return stats

def generate_cache_key(city_name: str, business_lines: List[str], 
                      vendor_filters: Dict[str, Any], 
                      additional_params: Dict[str, Any] = None) -> str:
    """Generate a consistent cache key for given parameters"""
    key_data = {
        'city': city_name,
        'business_lines': sorted(business_lines) if business_lines else [],
        'vendor_filters': vendor_filters,
        'additional': additional_params or {}
    }
    
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()