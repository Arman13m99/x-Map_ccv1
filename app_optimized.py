# app_optimized.py - Optimized Flask application with coverage grid caching
import os
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import webbrowser

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import Point, Polygon
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from scipy import stats
import random
import hashlib

# Import our new modules
from config import Config, get_config
from models import DatabaseManager, generate_cache_key
from scheduler import DataScheduler, init_scheduler
from cache_manager import CoverageGridCacheManager, init_cache_manager, get_cache_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize configuration
config = get_config()

# Initialize Flask app
app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app)

# Configure Flask
app.config['JSON_SORT_KEYS'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600

# Global instances (initialized in create_app)
db_manager: Optional[DatabaseManager] = None
scheduler: Optional[DataScheduler] = None
coverage_cache: Optional[CoverageGridCacheManager] = None

# Polygon data (loaded once at startup)
gdf_marketing_areas = {}
gdf_tehran_region = None
gdf_tehran_main_districts = None
df_coverage_targets = None
target_lookup_dict = {}

# Constants
CITY_ID_MAP = {1: "mashhad", 2: "tehran", 5: "shiraz"}
CITY_NAME_TO_ID_MAP = {v: k for k, v in CITY_ID_MAP.items()}

# --- Helper Functions (optimized versions) ---

def safe_tolist(series):
    """Safely convert pandas series to list"""
    if series.empty:
        return []
    cleaned = series.dropna().unique()
    if pd.api.types.is_numeric_dtype(cleaned.dtype):
        return [item.item() if hasattr(item, 'item') else item for item in cleaned]
    return cleaned.tolist()

def load_tehran_shapefile(filename: str) -> gpd.GeoDataFrame:
    """Load Tehran district shapefiles with encoding detection"""
    shp_path = os.path.join('src', 'polygons', 'tehran_districts', filename)
    
    if not os.path.exists(shp_path):
        logger.warning(f"Shapefile not found: {shp_path}")
        return gpd.GeoDataFrame()
    
    tried_encodings = [None, 'cp1256', 'utf-8']
    
    for enc in tried_encodings:
        try:
            gdf_temp = gpd.read_file(shp_path, encoding=enc)
            logger.info(f"Loaded {filename} using encoding='{enc or 'default'}'")
            
            # Reproject to WGS84 if needed
            if gdf_temp.crs and gdf_temp.crs.to_string() != "EPSG:4326":
                gdf_temp = gdf_temp.to_crs("EPSG:4326")
                logger.info(f"Reprojected {filename} to EPSG:4326")
            
            return gdf_temp
            
        except Exception as e:
            logger.debug(f"Failed to load {filename} with encoding {enc}: {e}")
    
    logger.error(f"Could not load {filename} with any encoding")
    return gpd.GeoDataFrame()

def get_district_names_from_gdf(gdf: gpd.GeoDataFrame, default_prefix: str = "District") -> List[str]:
    """Extract district names from GeoDataFrame"""
    if gdf is None or gdf.empty:
        return []
    
    name_cols = ['Name', 'name', 'NAME', 'Region', 'REGION_N', 'NAME_MAHAL', 'NAME_1', 'NAME_2', 'district']
    
    for col in name_cols:
        if col in gdf.columns and gdf[col].dtype == 'object':
            return sorted(safe_tolist(gdf[col].astype(str)))
    
    # Fallback to any object column
    for col in gdf.columns:
        if col != 'geometry' and gdf[col].dtype == 'object':
            return sorted(safe_tolist(gdf[col].astype(str)))
    
    # Last resort - generate names
    return [f"{default_prefix} {i+1}" for i in range(len(gdf))]

def generate_coverage_grid(city_name: str, grid_size_meters: int = 200) -> List[Dict[str, float]]:
    """Generate a grid of points for coverage analysis"""
    if city_name not in config.CITY_BOUNDARIES:
        return []
    
    bounds = config.CITY_BOUNDARIES[city_name]
    
    # Convert grid size from meters to approximate degrees
    grid_size_deg = grid_size_meters / 111000.0
    
    grid_points = []
    lat = bounds["min_lat"]
    
    while lat <= bounds["max_lat"]:
        lng = bounds["min_lng"]
        while lng <= bounds["max_lng"]:
            grid_points.append({"lat": lat, "lng": lng})
            lng += grid_size_deg
        lat += grid_size_deg
    
    return grid_points

def apply_radius_modifications_to_vendors(vendors_df: pd.DataFrame, radius_modifier: float, radius_mode: str, radius_fixed: float) -> pd.DataFrame:
    """Apply radius modifications to vendors DataFrame"""
    if vendors_df.empty or 'radius' not in vendors_df.columns:
        return vendors_df
        
    vendors_modified = vendors_df.copy()
    
    if radius_mode == 'percentage':
        # Use original_radius if available, otherwise fall back to radius
        base_radius = vendors_df.get('original_radius', vendors_df['radius'])
        vendors_modified['radius'] = base_radius * radius_modifier
        
        if radius_modifier != 1.0:
            logger.info(f"Applied {radius_modifier*100:.0f}% radius modifier to {len(vendors_modified)} vendors")
            
    elif radius_mode == 'fixed':
        # Set all vendors to fixed radius
        vendors_modified['radius'] = radius_fixed
        logger.info(f"Set fixed radius {radius_fixed}km for {len(vendors_modified)} vendors")
        
    return vendors_modified

def calculate_coverage_for_grid_vectorized(grid_points: List[Dict], 
                                         df_vendors_filtered: pd.DataFrame, 
                                         city_name: str) -> List[Dict]:
    """Calculate vendor coverage for all grid points using vectorized operations"""
    if df_vendors_filtered.empty or not grid_points:
        return []
    
    # Filter vendors with valid data
    valid_vendors = df_vendors_filtered.dropna(subset=['latitude', 'longitude', 'radius'])
    if valid_vendors.empty:
        return []
    
    # Convert to numpy arrays for faster computation
    grid_lats = np.array([p['lat'] for p in grid_points])
    grid_lngs = np.array([p['lng'] for p in grid_points])
    
    vendor_lats = valid_vendors['latitude'].values
    vendor_lngs = valid_vendors['longitude'].values
    vendor_radii = valid_vendors['radius'].values * 1000  # Convert km to meters
    
    # Pre-extract vendor attributes with proper handling of categorical data
    vendor_business_lines = None
    if 'business_line' in valid_vendors.columns:
        if pd.api.types.is_categorical_dtype(valid_vendors['business_line']):
            vendor_business_lines = valid_vendors['business_line'].cat.add_categories(['Unknown']).fillna('Unknown').values
        else:
            vendor_business_lines = valid_vendors['business_line'].fillna('Unknown').values
    
    vendor_grades = None
    if 'grade' in valid_vendors.columns:
        if pd.api.types.is_categorical_dtype(valid_vendors['grade']):
            vendor_grades = valid_vendors['grade'].cat.add_categories(['Unknown']).fillna('Unknown').values
        else:
            vendor_grades = valid_vendors['grade'].fillna('Unknown').values
    
    coverage_results = []
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(grid_points), batch_size):
        batch_end = min(i + batch_size, len(grid_points))
        batch_lats = grid_lats[i:batch_end]
        batch_lngs = grid_lngs[i:batch_end]
        
        # Vectorized distance calculation
        lat_diff = batch_lats[:, np.newaxis] - vendor_lats[np.newaxis, :]
        lng_diff = batch_lngs[:, np.newaxis] - vendor_lngs[np.newaxis, :]
        
        # Approximate distance in meters
        distances_meters = np.sqrt(
            (lat_diff * 111000)**2 + 
            (lng_diff * 111000 * np.cos(np.radians(batch_lats[:, np.newaxis])))**2
        )
        
        # Check which vendors cover each point
        coverage_matrix = distances_meters <= vendor_radii[np.newaxis, :]
        
        # Process results for each point in batch
        for j, point_idx in enumerate(range(i, batch_end)):
            covering_vendors = np.where(coverage_matrix[j])[0]
            
            coverage_data = {
                "lat": grid_points[point_idx]['lat'],
                "lng": grid_points[point_idx]['lng'],
                "total_vendors": len(covering_vendors),
                "by_business_line": {},
                "by_grade": {}
            }
            
            if len(covering_vendors) > 0:
                # Count by business line
                if vendor_business_lines is not None:
                    bl_counts = {}
                    for vendor_idx in covering_vendors:
                        bl = str(vendor_business_lines[vendor_idx])
                        bl_counts[bl] = bl_counts.get(bl, 0) + 1
                    coverage_data["by_business_line"] = bl_counts
                
                # Count by grade
                if vendor_grades is not None:
                    grade_counts = {}
                    for vendor_idx in covering_vendors:
                        grade = str(vendor_grades[vendor_idx])
                        grade_counts[grade] = grade_counts.get(grade, 0) + 1
                    coverage_data["by_grade"] = grade_counts
            
            coverage_results.append(coverage_data)
    
    return coverage_results

def find_marketing_areas_for_points(points: List[Dict], city_name: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """Find which marketing area each point belongs to using spatial indexing"""
    if city_name not in gdf_marketing_areas or gdf_marketing_areas[city_name].empty:
        return [(None, None)] * len(points)
    
    gdf_areas = gdf_marketing_areas[city_name]
    results = []
    
    try:
        from shapely.strtree import STRtree
        
        area_geoms = gdf_areas.geometry.values
        area_ids = gdf_areas['area_id'].values if 'area_id' in gdf_areas else [None] * len(gdf_areas)
        area_names = gdf_areas['name'].values if 'name' in gdf_areas else [None] * len(gdf_areas)
        
        # Create spatial index
        tree = STRtree(area_geoms)
        
        for point in points:
            point_geom = Point(point['lng'], point['lat'])
            candidate_indices = tree.query(point_geom)
            found_area = False
            
            for idx in candidate_indices:
                if area_geoms[idx].contains(point_geom):
                    results.append((area_ids[idx], area_names[idx]))
                    found_area = True
                    break
            
            if not found_area:
                results.append((None, None))
                
    except ImportError:
        logger.warning("shapely.strtree not available, using slower point-in-polygon check")
        # Fallback method
        for point in points:
            point_geom = Point(point['lng'], point['lat'])
            found = False
            for idx, area in gdf_areas.iterrows():
                if area.geometry.contains(point_geom):
                    area_id = area.get('area_id')
                    area_name = area.get('name')
                    results.append((area_id, area_name))
                    found = True
                    break
            if not found:
                results.append((None, None))
    
    return results

def calculate_coverage_grid_direct(city_name: str, 
                                 business_lines: List[str],
                                 vendor_filters: Dict[str, Any],
                                 radius_modifier: float = 1.0,
                                 radius_mode: str = 'percentage',
                                 radius_fixed: float = 3.0) -> List[Dict]:
    """Calculate coverage grid directly without caching"""
    try:
        logger.info(f"Calculating coverage grid directly for: {city_name}, BL: {business_lines}, radius_mode: {radius_mode}, modifier: {radius_modifier}")
        
        # Get filtered vendors from database
        bl_filter = None if business_lines == ["All"] else business_lines
        
        vendors_df = db_manager.get_vendors(
            city_name=city_name,
            business_lines=bl_filter,
            status_ids=vendor_filters.get('status_ids'),
            grades=vendor_filters.get('grades'),
            visible=vendor_filters.get('visible'),
            is_open=vendor_filters.get('is_open')
        )
        
        if vendors_df.empty:
            logger.warning(f"No vendors found for city: {city_name}, BL: {business_lines}, filters: {vendor_filters}")
            return []
        
        logger.info(f"Found {len(vendors_df)} vendors for coverage calculation")
        
        # Generate grid points
        grid_points = generate_coverage_grid(city_name, config.GRID_SIZE_METERS)
        if not grid_points:
            logger.warning(f"No grid points generated for city: {city_name}")
            return []
        
        logger.info(f"Generated coverage grid with {len(grid_points)} points (200m spacing)")
        
        # Apply radius modifications to vendors BEFORE calculating coverage
        vendors_df = apply_radius_modifications_to_vendors(vendors_df, radius_modifier, radius_mode, radius_fixed)
        
        # Find marketing areas for points
        point_area_info = find_marketing_areas_for_points(grid_points, city_name)
        
        # Calculate coverage
        coverage_results = calculate_coverage_for_grid_vectorized(grid_points, vendors_df, city_name)
        
        # Process results with target-based logic if applicable
        processed_grid_data = process_coverage_results_with_targets(
            coverage_results, point_area_info, business_lines, city_name
        )
        
        logger.info(f"Calculated coverage grid: {len(processed_grid_data)} points with coverage")
        return processed_grid_data
        
    except Exception as e:
        logger.error(f"Error calculating coverage grid: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def process_coverage_results_with_targets(coverage_results: List[Dict],
                                        point_area_info: List[Tuple],
                                        business_lines: List[str],
                                        city_name: str) -> List[Dict]:
    """Process coverage results with target-based analysis"""
    processed_data = []
    
    # Load target lookup if available
    target_lookup = {}
    if city_name == "tehran" and business_lines and len(business_lines) == 1:
        try:
            target_lookup = db_manager.get_target_lookup_dict('tehran')
            logger.info(f"Loaded target lookup for {city_name}: {len(target_lookup)} targets")
        except Exception as e:
            logger.error(f"Failed to load target lookup: {e}")
            target_lookup = {}
    
    for i, coverage in enumerate(coverage_results):
        if coverage['total_vendors'] > 0:
            area_id, area_name = point_area_info[i] if i < len(point_area_info) else (None, None)
            
            point_data = {
                'lat': coverage['lat'],
                'lng': coverage['lng'],
                'coverage': coverage,
                'marketing_area': area_name
            }
            
            # Add target-based analysis if applicable
            if target_lookup and area_id and len(business_lines) == 1:
                target_key = (area_id, business_lines[0])
                if target_key in target_lookup:
                    target_value = target_lookup[target_key]
                    actual_value = coverage['by_business_line'].get(business_lines[0], 0)
                    
                    point_data.update({
                        'target_business_line': business_lines[0],
                        'target_value': target_value,
                        'actual_value': actual_value,
                        'performance_ratio': (actual_value / target_value) if target_value > 0 else 2.0
                    })
            
            processed_data.append(point_data)
    
    return processed_data

def generate_improved_heatmap_data(heatmap_type_req: str, 
                                 df_orders_filtered: pd.DataFrame, 
                                 zoom_level: int = 11) -> List[Dict]:
    """Generate optimized heatmap data with caching"""
    
    # Create cache key for heatmap
    cache_params = {
        'type': heatmap_type_req,
        'order_count': len(df_orders_filtered),
        'zoom': zoom_level,
        'date_range': f"{df_orders_filtered['created_at'].min()}_{df_orders_filtered['created_at'].max()}" if not df_orders_filtered.empty and 'created_at' in df_orders_filtered.columns else "all"
    }
    
    cache_key = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
    
    # Check cache first
    if db_manager:
        cached_heatmap = db_manager.get_cached_heatmap(cache_key)
        if cached_heatmap:
            logger.info(f"Heatmap found in cache: {cache_key[:8]}...")
            return cached_heatmap
    
    # Generate heatmap data
    heatmap_data = []
    
    if heatmap_type_req not in ["order_density", "order_density_organic", "order_density_non_organic", "user_density"]:
        return heatmap_data
    
    df_hm_source = df_orders_filtered.dropna(subset=['customer_latitude', 'customer_longitude'])
    if df_hm_source.empty:
        return heatmap_data
    
    try:
        if heatmap_type_req == "order_density":
            df_hm_source['order_count'] = 1
            df_aggregated = aggregate_heatmap_points_adaptive(
                df_hm_source, 'customer_latitude', 'customer_longitude', 'order_count', zoom_level
            )
        elif heatmap_type_req == "order_density_organic":
            if 'organic' in df_hm_source.columns:
                df_organic = df_hm_source[df_hm_source['organic'] == 1]
                if not df_organic.empty:
                    df_aggregated = aggregate_heatmap_points_adaptive(
                        df_organic.assign(order_count=1), 'customer_latitude', 'customer_longitude', 'order_count', zoom_level
                    )
                else:
                    df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
            else:
                df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
        elif heatmap_type_req == "order_density_non_organic":
            if 'organic' in df_hm_source.columns:
                df_non_organic = df_hm_source[df_hm_source['organic'] == 0]
                if not df_non_organic.empty:
                    df_aggregated = aggregate_heatmap_points_adaptive(
                        df_non_organic.assign(order_count=1), 'customer_latitude', 'customer_longitude', 'order_count', zoom_level
                    )
                else:
                    df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
            else:
                df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
        elif heatmap_type_req == "user_density":
            if 'user_id' in df_hm_source.columns:
                df_aggregated = aggregate_user_heatmap_points_improved(
                    df_hm_source, 'customer_latitude', 'customer_longitude', 'user_id', zoom_level
                )
            else:
                df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
        
        # Apply improved normalization
        if not df_aggregated.empty:
            df_normalized = remove_outliers_and_normalize_improved(df_aggregated, 'value', method='robust')
            if not df_normalized.empty:
                df_normalized['value'] = df_normalized['value_normalized']
                heatmap_data = df_normalized[['lat', 'lng', 'value']].to_dict(orient='records')
        
        # Cache the result
        if db_manager and heatmap_data:
            db_manager.cache_heatmap(
                cache_key, heatmap_type_req, 
                df_orders_filtered['city_name'].iloc[0] if not df_orders_filtered.empty and 'city_name' in df_orders_filtered.columns else 'unknown',
                cache_params['date_range'],
                '',  # business_line
                zoom_level, heatmap_data
            )
            
    except Exception as e:
        logger.error(f"Error generating heatmap data: {e}")
        return []
    
    return heatmap_data

def aggregate_heatmap_points_adaptive(df: pd.DataFrame, lat_col: str, lng_col: str, 
                                    value_col: str, zoom_level: int = 11) -> pd.DataFrame:
    """Adaptive aggregation that adjusts precision based on zoom level"""
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # Adaptive precision based on zoom level
    if zoom_level >= 16:
        precision = 5
    elif zoom_level >= 14:
        precision = 4
    elif zoom_level >= 12:
        precision = 3
    elif zoom_level >= 10:
        precision = 2
    else:
        precision = 1
    
    # Round coordinates
    df_copy['lat_rounded'] = df_copy[lat_col].round(precision)
    df_copy['lng_rounded'] = df_copy[lng_col].round(precision)
    
    # Aggregate with multiple statistics
    aggregated = df_copy.groupby(['lat_rounded', 'lng_rounded']).agg({
        value_col: ['sum', 'count', 'mean']
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['lat', 'lng', 'value_sum', 'value_count', 'value_mean']
    
    # Use sum as primary value, with density weighting
    aggregated['value'] = aggregated['value_sum']
    aggregated['density_weight'] = np.log1p(aggregated['value_count'])
    aggregated['weighted_value'] = aggregated['value'] * (1 + aggregated['density_weight'] * 0.1)
    
    return aggregated[['lat', 'lng', 'weighted_value']].rename(columns={'weighted_value': 'value'})

def aggregate_user_heatmap_points_improved(df: pd.DataFrame, lat_col: str, lng_col: str, 
                                         user_col: str, zoom_level: int = 11) -> pd.DataFrame:
    """Improved user aggregation with better handling of unique users"""
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # Adaptive precision
    if zoom_level >= 16:
        precision = 5
    elif zoom_level >= 14:
        precision = 4
    elif zoom_level >= 12:
        precision = 3
    else:
        precision = 2
    
    df_copy['lat_rounded'] = df_copy[lat_col].round(precision)
    df_copy['lng_rounded'] = df_copy[lng_col].round(precision)
    
    # Count unique users per location
    aggregated = df_copy.groupby(['lat_rounded', 'lng_rounded'])[user_col].nunique().reset_index()
    aggregated.columns = ['lat', 'lng', 'unique_users']
    
    # Apply log transformation for better distribution
    aggregated['value'] = np.log1p(aggregated['unique_users']) * 10
    
    return aggregated[['lat', 'lng', 'value']]

def remove_outliers_and_normalize_improved(df: pd.DataFrame, value_column: str, method: str = 'robust') -> pd.DataFrame:
    """Improved outlier removal and normalization using robust statistical methods"""
    if df.empty or value_column not in df.columns:
        return df
    
    df_copy = df.copy()
    df_copy = df_copy[df_copy[value_column].notna() & (df_copy[value_column] > 0)]
    
    if df_copy.empty:
        logger.warning(f"No valid {value_column} data after removing nulls/zeros")
        return df_copy
    
    values = df_copy[value_column].values
    
    if method == 'robust':
        # Use IQR method for outlier removal
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep some outliers to maintain data integrity
        lower_bound = max(lower_bound, np.percentile(values, 1))
        upper_bound = min(upper_bound, np.percentile(values, 99))
        
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(values))
        threshold = 3
        mask = z_scores < threshold
        df_copy = df_copy[mask]
        values = df_copy[value_column].values
        lower_bound = values.min()
        upper_bound = values.max()
    
    # Apply bounds
    df_copy = df_copy[(df_copy[value_column] >= lower_bound) & (df_copy[value_column] <= upper_bound)]
    
    if df_copy.empty:
        logger.warning(f"No data left after outlier removal for {value_column}")
        return df_copy
    
    # Use log transformation for better distribution
    log_values = np.log1p(df_copy[value_column])
    
    # Robust normalization using percentiles
    p5 = np.percentile(log_values, 5)
    p95 = np.percentile(log_values, 95)
    
    if p95 > p5:
        normalized = ((log_values - p5) / (p95 - p5)) * 100
        normalized = np.clip(normalized, 0, 100)
    else:
        normalized = np.full(len(df_copy), 50)
    
    df_copy[f'{value_column}_normalized'] = normalized
    
    logger.debug(f"{value_column} normalization complete: {len(df_copy)} points")
    return df_copy

def enrich_polygons_with_stats(gdf_polygons: gpd.GeoDataFrame, 
                             name_col: str,
                             df_v_filtered: pd.DataFrame, 
                             df_o_filtered: pd.DataFrame, 
                             df_o_all_for_city: pd.DataFrame) -> gpd.GeoDataFrame:
    """Enrich polygons with vendor and user statistics"""
    if gdf_polygons is None or gdf_polygons.empty:
        return gpd.GeoDataFrame()
    
    enriched_gdf = gdf_polygons.copy()
    
    # Vendor enrichment
    if not df_v_filtered.empty and not df_v_filtered.dropna(subset=['latitude', 'longitude']).empty:
        # Filter out vendors with missing coordinates first
        df_v_valid = df_v_filtered.dropna(subset=['latitude', 'longitude'])
        gdf_v_filtered = gpd.GeoDataFrame(
            df_v_valid,
            geometry=gpd.points_from_xy(df_v_valid.longitude, df_v_valid.latitude),
            crs="EPSG:4326"
        )
        
        joined_vendors = gpd.sjoin(gdf_v_filtered, enriched_gdf, how="inner", predicate="within")
        
        # Total vendor count
        vendor_counts = joined_vendors.groupby(name_col).size().rename('vendor_count')
        enriched_gdf = enriched_gdf.merge(vendor_counts, how='left', left_on=name_col, right_index=True)
        
        # Vendor count by grade
        if 'grade' in joined_vendors.columns:
            grade_counts_series = joined_vendors.groupby([name_col, 'grade'], observed=True).size().unstack(fill_value=0)
            grade_counts_dict = grade_counts_series.apply(
                lambda row: {k: v for k, v in row.items() if v > 0}, axis=1
            ).to_dict()
            enriched_gdf['grade_counts'] = enriched_gdf[name_col].astype(str).map(grade_counts_dict)
        else:
            enriched_gdf['grade_counts'] = None
    else:
        enriched_gdf['vendor_count'] = 0
        enriched_gdf['grade_counts'] = None
    
    enriched_gdf['vendor_count'] = enriched_gdf['vendor_count'].fillna(0).astype(int)
    
    # User enrichment
    has_user_id = 'user_id' in df_o_all_for_city.columns
    if has_user_id:
        # Date-ranged unique users
        if not df_o_filtered.empty and not df_o_filtered.dropna(subset=['customer_latitude', 'customer_longitude']).empty:
            df_o_valid = df_o_filtered.dropna(subset=['customer_latitude', 'customer_longitude'])
            gdf_orders_filtered = gpd.GeoDataFrame(
                df_o_valid,
                geometry=gpd.points_from_xy(df_o_valid.customer_longitude, df_o_valid.customer_latitude),
                crs="EPSG:4326"
            )
            joined_orders_filtered = gpd.sjoin(gdf_orders_filtered, enriched_gdf, how="inner", predicate="within")
            user_counts_filtered = joined_orders_filtered.groupby(name_col, observed=True)['user_id'].nunique().rename('unique_user_count')
            enriched_gdf = enriched_gdf.merge(user_counts_filtered, how='left', left_on=name_col, right_index=True)
        
        # Total unique users
        if not df_o_all_for_city.empty and not df_o_all_for_city.dropna(subset=['customer_latitude', 'customer_longitude']).empty:
            df_o_all_valid = df_o_all_for_city.dropna(subset=['customer_latitude', 'customer_longitude'])
            gdf_orders_all = gpd.GeoDataFrame(
                df_o_all_valid,
                geometry=gpd.points_from_xy(df_o_all_valid.customer_longitude, df_o_all_valid.customer_latitude),
                crs="EPSG:4326"
            )
            joined_orders_all = gpd.sjoin(gdf_orders_all, enriched_gdf, how="inner", predicate="within")
            user_counts_all = joined_orders_all.groupby(name_col, observed=True)['user_id'].nunique().rename('total_unique_user_count')
            enriched_gdf = enriched_gdf.merge(user_counts_all, how='left', left_on=name_col, right_index=True)
    
    enriched_gdf['unique_user_count'] = enriched_gdf.get('unique_user_count', pd.Series(0, index=enriched_gdf.index)).fillna(0).astype(int)
    enriched_gdf['total_unique_user_count'] = enriched_gdf.get('total_unique_user_count', pd.Series(0, index=enriched_gdf.index)).fillna(0).astype(int)
    
    # Population-based metrics
    if 'Pop' in enriched_gdf.columns:
        enriched_gdf['Pop'] = pd.to_numeric(enriched_gdf['Pop'], errors='coerce').fillna(0)
        enriched_gdf['vendor_per_10k_pop'] = enriched_gdf.apply(
            lambda row: (row['vendor_count'] / row['Pop']) * 10000 if row['Pop'] > 0 else 0, axis=1
        )
    
    if 'PopDensity' in enriched_gdf.columns:
        enriched_gdf['PopDensity'] = pd.to_numeric(enriched_gdf['PopDensity'], errors='coerce').fillna(0)
    
    return enriched_gdf

# --- Application Initialization ---

def load_polygon_data():
    """Load polygon data from database at startup"""
    global gdf_marketing_areas, gdf_tehran_region, gdf_tehran_main_districts
    global df_coverage_targets, target_lookup_dict, db_manager
    
    logger.info("Loading polygon data from database...")
    
    try:
        # Ensure src/ data is loaded into database first
        src_loaded = {
            'marketing_areas': db_manager.get_metadata("marketing_areas_loaded"),
            'district_boundaries': db_manager.get_metadata("district_boundaries_loaded"),
            'coverage_targets': db_manager.get_metadata("coverage_targets_loaded")
        }
        
        # Check if src/ data needs to be initialized
        missing_data = [k for k, v in src_loaded.items() if not v]
        if missing_data:
            logger.info(f"Src/ data not found in database: {missing_data}")
            logger.info("Initializing src/ data into database...")
            results = db_manager.initialize_src_data(force_reload=False)
            logger.info(f"Src/ data initialization results: {results}")
        
        # Load marketing areas from database
        cities = ["mashhad", "tehran", "shiraz"]
        for city_name in cities:
            try:
                gdf = db_manager.get_marketing_areas(city_name)
                if not gdf.empty:
                    # Convert category columns to string to avoid issues
                    if 'name' in gdf.columns:
                        gdf['name'] = gdf['name'].astype(str)
                    gdf_marketing_areas[city_name] = gdf
                    logger.info(f"Loaded marketing areas for {city_name}: {len(gdf)} polygons")
                else:
                    logger.warning(f"No marketing areas found for {city_name}")
                    gdf_marketing_areas[city_name] = gpd.GeoDataFrame()
            except Exception as e:
                logger.error(f"Error loading marketing areas for {city_name}: {e}")
                gdf_marketing_areas[city_name] = gpd.GeoDataFrame()
        
        # Load Tehran district boundaries from database
        try:
            gdf_tehran_region = db_manager.get_district_boundaries('tehran', 'region')
            if not gdf_tehran_region.empty:
                # Rename column for compatibility
                if 'name' in gdf_tehran_region.columns:
                    gdf_tehran_region = gdf_tehran_region.rename(columns={'name': 'Name'})
                logger.info(f"Loaded Tehran region districts: {len(gdf_tehran_region)} polygons")
            else:
                logger.warning("No Tehran region districts found")
                gdf_tehran_region = gpd.GeoDataFrame()
        except Exception as e:
            logger.error(f"Error loading Tehran region districts: {e}")
            gdf_tehran_region = gpd.GeoDataFrame()
        
        try:
            gdf_tehran_main_districts = db_manager.get_district_boundaries('tehran', 'main')
            if not gdf_tehran_main_districts.empty:
                # Rename columns for compatibility
                rename_map = {}
                if 'name' in gdf_tehran_main_districts.columns:
                    rename_map['name'] = 'NAME_MAHAL'
                if 'pop' in gdf_tehran_main_districts.columns:
                    rename_map['pop'] = 'Pop'
                if rename_map:
                    gdf_tehran_main_districts = gdf_tehran_main_districts.rename(columns=rename_map)
                logger.info(f"Loaded Tehran main districts: {len(gdf_tehran_main_districts)} polygons")
            else:
                logger.warning("No Tehran main districts found")
                gdf_tehran_main_districts = gpd.GeoDataFrame()
        except Exception as e:
            logger.error(f"Error loading Tehran main districts: {e}")
            gdf_tehran_main_districts = gpd.GeoDataFrame()
        
        # Load coverage targets from database
        try:
            df_coverage_targets = db_manager.get_coverage_targets()
            if not df_coverage_targets.empty:
                # Create target lookup dictionary
                target_lookup_dict = db_manager.get_target_lookup_dict('tehran')
                logger.info(f"Loaded coverage targets: {len(target_lookup_dict)} entries")
            else:
                logger.warning("No coverage targets found")
                df_coverage_targets = pd.DataFrame()
                target_lookup_dict = {}
        except Exception as e:
            logger.error(f"Error loading coverage targets: {e}")
            df_coverage_targets = pd.DataFrame()
            target_lookup_dict = {}
        
    except Exception as e:
        logger.error(f"Error loading polygon data from database: {e}")
        # Fallback to empty dataframes
        for city in ["mashhad", "tehran", "shiraz"]:
            gdf_marketing_areas[city] = gpd.GeoDataFrame()
        gdf_tehran_region = gpd.GeoDataFrame()
        gdf_tehran_main_districts = gpd.GeoDataFrame()
        df_coverage_targets = pd.DataFrame()
        target_lookup_dict = {}

def create_app() -> Flask:
    """Create and configure the Flask application"""
    global db_manager, scheduler, coverage_cache
    
    logger.info("Initializing optimized Tapsi Food Map Dashboard...")
    
    # Initialize database
    db_manager = DatabaseManager(config.DATABASE_PATH)
    logger.info("Database initialized")
    
    # Load polygon data
    load_polygon_data()
    
    # Initialize and start scheduler
    scheduler = init_scheduler(config, db_manager)
    scheduler.start()
    logger.info("Data scheduler started")

    # Initialize coverage grid cache manager
    coverage_cache = init_cache_manager(config, db_manager)
    if config.PRELOAD_COVERAGE_GRIDS:
        coverage_cache.start_preloading()
    
    # Add debug endpoint
    @app.route('/api/debug-globals', methods=['GET'])
    def debug_globals():
        """Debug endpoint to check global variables"""
        return jsonify({
            'gdf_tehran_main_districts_none': gdf_tehran_main_districts is None,
            'gdf_tehran_region_none': gdf_tehran_region is None,
            'gdf_tehran_main_districts_len': len(gdf_tehran_main_districts) if gdf_tehran_main_districts is not None else 0,
            'gdf_tehran_main_districts_cols': list(gdf_tehran_main_districts.columns) if gdf_tehran_main_districts is not None else [],
            'has_pop_column': 'Pop' in gdf_tehran_main_districts.columns if gdf_tehran_main_districts is not None else False
        })
    
    return app

# --- Flask Routes ---

@app.route('/')
def serve_index():
    """Serve the main dashboard page"""
    return send_from_directory('public', 'index.html')

@app.route('/debug')
def serve_debug():
    """Serve debug page for frontend vendor issues"""
    return send_from_directory('.', 'debug_frontend_state.html')

@app.route('/api/initial-data', methods=['GET'])
def get_initial_data():
    """Get initial filter data for the frontend"""
    try:
        # Check if db_manager is initialized
        if db_manager is None:
            logger.error("Database manager not initialized")
            return jsonify({"error": "Database not initialized"}), 500
            
        # Get data from database instead of global variables
        # NOTE: Orders have city_name='nan' in DB, so get all and filter by city_id=2 (tehran)
        sample_orders = db_manager.get_orders(city_name=None).head(10000)  # Get more for filtering
        if not sample_orders.empty and 'city_id' in sample_orders.columns:
            sample_orders = sample_orders[sample_orders['city_id'] == 2].head(1000)  # Tehran city_id=2
        
        sample_vendors = db_manager.get_vendors(city_name="tehran").head(1000)
        
        cities = [{"id": cid, "name": name} for cid, name in CITY_ID_MAP.items()]
        
        # Get business lines from orders, vendors, or use fallback
        business_lines = []
        
        # Try from orders first
        if not sample_orders.empty and 'business_line' in sample_orders.columns:
            business_lines = safe_tolist(sample_orders['business_line'])
        
        # If no business lines from orders, try vendors
        if not business_lines and not sample_vendors.empty and 'business_line' in sample_vendors.columns:
            business_lines = safe_tolist(sample_vendors['business_line'])
        
        # Fallback to known business lines if database is empty
        if not business_lines:
            business_lines = [
                "Restaurant", "Cafe", "Bakery", "Pastry", 
                "Meat Shop", "Fruit Shop", "Ice Cream and Juice Shop"
            ]
            logger.info("Using fallback business lines as no data found in database")
        
        marketing_area_names_by_city = {}
        for city_key, gdf in gdf_marketing_areas.items():
            if not gdf.empty and 'name' in gdf.columns:
                marketing_area_names_by_city[city_key] = sorted(safe_tolist(gdf['name'].astype(str)))
            else:
                marketing_area_names_by_city[city_key] = []
        
        tehran_region_districts = get_district_names_from_gdf(gdf_tehran_region, "Region Tehran")
        tehran_main_districts = get_district_names_from_gdf(gdf_tehran_main_districts, "Main Tehran")
        
        vendor_statuses = []
        if not sample_vendors.empty and 'status_id' in sample_vendors.columns:
            status_series = sample_vendors['status_id'].dropna()
            if not status_series.empty:
                vendor_statuses = sorted([int(x) for x in status_series.unique()])
        
        vendor_grades = []
        if not sample_vendors.empty and 'grade' in sample_vendors.columns:
            vendor_grades = sorted(safe_tolist(sample_vendors['grade'].astype(str)))
        
        return jsonify({
            "cities": cities,
            "business_lines": business_lines,
            "marketing_areas_by_city": marketing_area_names_by_city,
            "tehran_region_districts": tehran_region_districts,
            "tehran_main_districts": tehran_main_districts,
            "vendor_statuses": vendor_statuses,
            "vendor_grades": vendor_grades
        })
        
    except Exception as e:
        logger.error(f"Error in get_initial_data: {e}")
        return jsonify({"error": "Failed to load initial data", "details": str(e)}), 500

@app.route('/api/map-data', methods=['GET'])
def get_map_data():
    """Get filtered map data with optimized database queries"""
    try:
        # Check if db_manager is initialized
        if db_manager is None:
            logger.error("Database manager not initialized")
            return jsonify({"error": "Database not initialized"}), 500
            
        # Debug: Log all request parameters
        logger.info(f"MAP-DATA API called with parameters: {dict(request.args)}")
        
        start_time = time.time()
        
        # Parse request parameters
        city_name = request.args.get('city', default="tehran", type=str)
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        
        start_date = pd.to_datetime(start_date_str) if start_date_str else None
        end_date = pd.to_datetime(end_date_str).replace(hour=23, minute=59, second=59) if end_date_str else None
        
        # Parse business lines - handle 'all' case and empty lists
        business_lines_param = request.args.getlist('business_lines')
        if not business_lines_param or 'all' in [bl.lower() for bl in business_lines_param]:
            selected_business_lines = None  # No filtering - show all business lines
        else:
            selected_business_lines = [bl.strip() for bl in business_lines_param if bl.strip()]
        vendor_codes_input_str = request.args.get('vendor_codes_filter', default="", type=str)
        selected_vendor_codes = [code.strip() for code in vendor_codes_input_str.replace('\n', ',').split(',') if code.strip()]
        
        # Parse vendor status IDs - handle 'all' case
        status_ids_param = request.args.getlist('vendor_status_ids')
        if not status_ids_param or 'all' in [s.lower() for s in status_ids_param]:
            selected_vendor_status_ids = None
        else:
            selected_vendor_status_ids = [int(s.strip()) for s in status_ids_param if s.strip().isdigit()]
        
        # Parse vendor grades - handle 'all' case  
        grades_param = request.args.getlist('vendor_grades')
        if not grades_param or 'all' in [g.lower() for g in grades_param]:
            selected_vendor_grades = None
        else:
            selected_vendor_grades = [g.strip() for g in grades_param if g.strip()]
        
        vendor_visible_str = request.args.get('vendor_visible', default="any", type=str)
        vendor_is_open_str = request.args.get('vendor_is_open', default="any", type=str)
        
        heatmap_type_req = request.args.get('heatmap_type_request', default="none", type=str)
        area_type_display = request.args.get('area_type_display', default="tapsifood_marketing_areas", type=str)
        selected_polygon_sub_types = [s.strip() for s in request.args.getlist('area_sub_type_filter') if s.strip()]
        
        zoom_level = request.args.get('zoom_level', default=11, type=float)

        # Radius modifier parameters
        radius_modifier = request.args.get('radius_modifier', default=1.0, type=float)
        radius_mode = request.args.get('radius_mode', default='percentage', type=str)
        radius_fixed = request.args.get('radius_fixed', default=3.0, type=float)

        center_lat = request.args.get('center_lat', type=float)
        center_lng = request.args.get('center_lng', type=float)
        
        # Force recalculate parameter
        force_recalculate = request.args.get('force_recalculate', default=False, type=bool)
        if force_recalculate:
            logger.info(f"Force recalculate requested for {city_name} coverage grid")
        
        # Get filtered vendors from database
        vendor_visible = int(vendor_visible_str) if vendor_visible_str != "any" else None
        vendor_is_open = int(vendor_is_open_str) if vendor_is_open_str != "any" else None
        
        df_vendors_filtered = db_manager.get_vendors(
            city_name=city_name,
            business_lines=selected_business_lines,
            vendor_codes=selected_vendor_codes if selected_vendor_codes else None,
            status_ids=selected_vendor_status_ids if selected_vendor_status_ids else None,
            grades=selected_vendor_grades if selected_vendor_grades else None,
            visible=vendor_visible,
            is_open=vendor_is_open
        )
        
        # Apply radius modifications to vendors for display
        if not df_vendors_filtered.empty and 'radius' in df_vendors_filtered.columns:
            df_vendors_filtered = apply_radius_modifications_to_vendors(df_vendors_filtered, radius_modifier, radius_mode, radius_fixed)
        
        # Get filtered orders from database
        # NOTE: Orders in DB have city_name='nan', so we filter by city_id after fetching
        df_orders_filtered = db_manager.get_orders(
            city_name=None,  # Get all orders first
            start_date=start_date,
            end_date=end_date,
            business_lines=selected_business_lines,
            vendor_codes=selected_vendor_codes if selected_vendor_codes else None
        )
        
        # Filter by city using city_id mapping
        if city_name != "all" and not df_orders_filtered.empty and 'city_id' in df_orders_filtered.columns:
            city_id = CITY_NAME_TO_ID_MAP.get(city_name)
            if city_id:
                df_orders_filtered = df_orders_filtered[df_orders_filtered['city_id'] == city_id]
        
        # Get all orders for city (for total user counts)
        df_orders_all_for_city = db_manager.get_orders(city_name=None)
        if city_name != "all" and not df_orders_all_for_city.empty and 'city_id' in df_orders_all_for_city.columns:
            city_id = CITY_NAME_TO_ID_MAP.get(city_name)
            if city_id:
                df_orders_all_for_city = df_orders_all_for_city[df_orders_all_for_city['city_id'] == city_id]
        
        # Initialize response data
        vendor_markers = []
        heatmap_data = []
        polygons_geojson = {"type": "FeatureCollection", "features": []}
        coverage_grid_data = []
        
        # Process vendors data
        if not df_vendors_filtered.empty:
            vendor_markers = df_vendors_filtered.replace({np.nan: None}).to_dict(orient='records')
        
        # Process heatmap data
        logger.info(f"Heatmap processing: heatmap_type_req='{heatmap_type_req}', city_name='{city_name}'")
        if heatmap_type_req in ["order_density", "order_density_organic", "order_density_non_organic", "user_density"]:
            heatmap_data = generate_improved_heatmap_data(heatmap_type_req, df_orders_filtered, zoom_level)
        elif heatmap_type_req == "population" and city_name == "tehran":
            # Generate population heatmap
            logger.info(f"API: About to call generate_population_heatmap with area_type={area_type_display}")
            heatmap_data = generate_population_heatmap(area_type_display, selected_polygon_sub_types, zoom_level)
            logger.info(f"API: Population heatmap returned {len(heatmap_data)} points")
        else:
            logger.info(f"No heatmap condition matched. Type: '{heatmap_type_req}', City: '{city_name}'")
        
        # Process coverage grid
        if area_type_display == "coverage_grid":
            vendor_filters = {
                'status_ids': selected_vendor_status_ids,
                'grades': selected_vendor_grades,
                'visible': vendor_visible,
                'is_open': vendor_is_open
            }

            cache_mgr = coverage_cache or get_cache_manager()
            if cache_mgr:
                coverage_grid_data = cache_mgr.get_or_calculate_coverage_grid(
                    city_name, selected_business_lines, vendor_filters,
                    force_recalculate=force_recalculate,
                    radius_modifier=radius_modifier,
                    radius_mode=radius_mode,
                    radius_fixed=radius_fixed,
                    center_lat=center_lat,
                    center_lng=center_lng
                ) or []
            else:
                logger.info(
                    f"Calculating coverage grid directly with radius_modifier={radius_modifier}, radius_mode={radius_mode}"
                )
                coverage_grid_data = calculate_coverage_grid_direct(
                    city_name, selected_business_lines, vendor_filters,
                    radius_modifier, radius_mode, radius_fixed
                ) or []
        
        # Process polygons
        if area_type_display != "none" and area_type_display != "coverage_grid":
            polygons_geojson = get_enriched_polygons(
                area_type_display, city_name, selected_polygon_sub_types,
                df_vendors_filtered, df_orders_filtered, df_orders_all_for_city
            )
        elif area_type_display == "coverage_grid" and city_name in gdf_marketing_areas:
            # Marketing areas overlay for coverage grid
            gdf_to_send = gdf_marketing_areas[city_name].copy()
            if selected_polygon_sub_types and 'name' in gdf_to_send.columns:
                gdf_to_send = gdf_to_send[gdf_to_send['name'].astype(str).isin(selected_polygon_sub_types)]
            
            if not gdf_to_send.empty:
                # Clean GeoDataFrame for JSON serialization
                clean_gdf = gdf_to_send.copy()
                for col in clean_gdf.columns:
                    if col != 'geometry':
                        clean_gdf[col] = clean_gdf[col].astype(object).where(pd.notna(clean_gdf[col]), None)
                polygons_geojson = clean_gdf.__geo_interface__
        
        request_time = time.time() - start_time
        logger.info(f"Map data request processed in {request_time:.2f}s")
        
        response_data = {
            "vendors": vendor_markers,
            "heatmap_data": heatmap_data,
            "polygons": polygons_geojson,
            "coverage_grid": coverage_grid_data,
            "processing_time": request_time,
            "zoom_level": zoom_level,
            "heatmap_type": heatmap_type_req
        }
        
        # Use ujson for faster JSON serialization if available
        try:
            import ujson
            return app.response_class(ujson.dumps(response_data), mimetype='application/json')
        except ImportError:
            return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in get_map_data: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

def generate_population_heatmap(area_type_display: str, selected_polygon_sub_types: List[str], zoom_level: int) -> List[Dict]:
    """Generate population heatmap for Tehran"""
    global gdf_tehran_main_districts, gdf_tehran_region, db_manager
    
    logger.info(f"Population heatmap request: area_type={area_type_display}, sub_types={selected_polygon_sub_types}, zoom={zoom_level}")
    
    # If global variables are None, try to reload from database
    if gdf_tehran_main_districts is None or gdf_tehran_region is None:
        logger.warning("Global district data is None, attempting to reload from database")
        
        if db_manager is not None:
            try:
                if gdf_tehran_main_districts is None:
                    gdf_tehran_main_districts = db_manager.get_district_boundaries('tehran', 'main')
                    if not gdf_tehran_main_districts.empty:
                        # Rename columns for compatibility
                        rename_map = {}
                        if 'name' in gdf_tehran_main_districts.columns:
                            rename_map['name'] = 'NAME_MAHAL'
                        if 'pop' in gdf_tehran_main_districts.columns:
                            rename_map['pop'] = 'Pop'
                        if rename_map:
                            gdf_tehran_main_districts = gdf_tehran_main_districts.rename(columns=rename_map)
                        logger.info(f"Reloaded Tehran main districts: {len(gdf_tehran_main_districts)} polygons")
                
                if gdf_tehran_region is None:
                    gdf_tehran_region = db_manager.get_district_boundaries('tehran', 'region')
                    if not gdf_tehran_region.empty:
                        if 'name' in gdf_tehran_region.columns:
                            gdf_tehran_region = gdf_tehran_region.rename(columns={'name': 'Name'})
                        logger.info(f"Reloaded Tehran region districts: {len(gdf_tehran_region)} polygons")
            except Exception as e:
                logger.error(f"Failed to reload district data: {e}")
    
    gdf_pop_source = None
    
    logger.info(f"Global variables: gdf_tehran_main_districts is None: {gdf_tehran_main_districts is None}")
    logger.info(f"Global variables: gdf_tehran_region is None: {gdf_tehran_region is None}")
    
    if area_type_display == "tehran_main_districts" and gdf_tehran_main_districts is not None:
        gdf_pop_source = gdf_tehran_main_districts
        logger.info(f"Using tehran_main_districts: {len(gdf_pop_source)} districts")
    elif area_type_display == "tehran_region_districts" and gdf_tehran_region is not None:
        gdf_pop_source = gdf_tehran_region
        logger.info(f"Using tehran_region_districts: {len(gdf_pop_source)} districts")
    elif area_type_display == "all_tehran_districts" and gdf_tehran_main_districts is not None:
        gdf_pop_source = gdf_tehran_main_districts
        logger.info(f"Using all_tehran_districts (main): {len(gdf_pop_source)} districts")
    
    if gdf_pop_source is None:
        logger.warning("No population source GDF available")
        return []
    
    if 'Pop' not in gdf_pop_source.columns:
        logger.warning(f"Pop column not found. Available columns: {list(gdf_pop_source.columns)}")
        return []
    
    logger.info(f"Population source has {len(gdf_pop_source)} districts with Pop column")
    
    if selected_polygon_sub_types:
        name_cols_poly = ['Name', 'NAME_MAHAL']
        actual_name_col = next((col for col in name_cols_poly if col in gdf_pop_source.columns), None)
        if actual_name_col:
            gdf_pop_source = gdf_pop_source[gdf_pop_source[actual_name_col].isin(selected_polygon_sub_types)]
    
    # Adjust point density based on zoom level
    base_divisor = 1000
    zoom_multiplier = max(0.1, min(2.0, (zoom_level / 11.0)))
    point_density_divisor = base_divisor / zoom_multiplier
    
    temp_points = []
    for _, row in gdf_pop_source.iterrows():
        population = pd.to_numeric(row['Pop'], errors='coerce')
        if pd.notna(population) and population > 0:
            num_points = int(population / point_density_divisor)
            if num_points > 0:
                # Generate random points within the polygon
                bounds = row['geometry'].bounds
                for _ in range(min(num_points, 1000)):  # Limit points per polygon
                    while True:
                        lat = random.uniform(bounds[1], bounds[3])
                        lng = random.uniform(bounds[0], bounds[2])
                        point = Point(lng, lat)
                        if row['geometry'].contains(point):
                            temp_points.append({'lat': lat, 'lng': lng, 'value': 1})
                            break
    
    return temp_points

def get_enriched_polygons(area_type_display: str, city_name: str, selected_polygon_sub_types: List[str],
                         df_vendors_filtered: pd.DataFrame, df_orders_filtered: pd.DataFrame,
                         df_orders_all_for_city: pd.DataFrame) -> Dict:
    """Get enriched polygon data"""
    final_polygons_gdf = None
    
    if area_type_display == "tapsifood_marketing_areas" and city_name in gdf_marketing_areas:
        final_polygons_gdf = enrich_polygons_with_stats(
            gdf_marketing_areas[city_name], 'name', 
            df_vendors_filtered, df_orders_filtered, df_orders_all_for_city
        )
    elif city_name == "tehran":
        if area_type_display == "tehran_region_districts" and gdf_tehran_region is not None:
            final_polygons_gdf = enrich_polygons_with_stats(
                gdf_tehran_region, 'Name',
                df_vendors_filtered, df_orders_filtered, df_orders_all_for_city
            )
        elif area_type_display == "tehran_main_districts" and gdf_tehran_main_districts is not None:
            final_polygons_gdf = enrich_polygons_with_stats(
                gdf_tehran_main_districts, 'NAME_MAHAL',
                df_vendors_filtered, df_orders_filtered, df_orders_all_for_city
            )
        elif area_type_display == "all_tehran_districts":
            enriched_list = []
            if gdf_tehran_region is not None:
                enriched_list.append(enrich_polygons_with_stats(
                    gdf_tehran_region, 'Name',
                    df_vendors_filtered, df_orders_filtered, df_orders_all_for_city
                ))
            if gdf_tehran_main_districts is not None:
                enriched_list.append(enrich_polygons_with_stats(
                    gdf_tehran_main_districts, 'NAME_MAHAL',
                    df_vendors_filtered, df_orders_filtered, df_orders_all_for_city
                ))
            if enriched_list:
                final_polygons_gdf = pd.concat(enriched_list, ignore_index=True)
    
    if final_polygons_gdf is None or final_polygons_gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    
    # Apply polygon filters
    if selected_polygon_sub_types:
        name_cols_poly = ['name', 'Name', 'NAME_MAHAL']
        actual_name_col = next((col for col in name_cols_poly if col in final_polygons_gdf.columns), None)
        if actual_name_col:
            final_polygons_gdf = final_polygons_gdf[
                final_polygons_gdf[actual_name_col].astype(str).isin(selected_polygon_sub_types)
            ]
    
    if final_polygons_gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    
    # Clean data for JSON serialization
    clean_gdf = final_polygons_gdf.copy()
    for col in clean_gdf.columns:
        if col != 'geometry':
            clean_gdf[col] = clean_gdf[col].astype(object).where(pd.notna(clean_gdf[col]), None)
    
    return clean_gdf.__geo_interface__

@app.route('/api/admin/scheduler-status', methods=['GET'])
def get_scheduler_status():
    """Get scheduler status (admin endpoint)"""
    if scheduler:
        return jsonify(scheduler.get_status())
    else:
        return jsonify({"error": "Scheduler not initialized"}), 500

@app.route('/api/admin/force-update', methods=['POST'])
def force_data_update():
    """Force data update (admin endpoint)"""
    try:
        update_type = request.json.get('type', 'vendors')
        
        if update_type == 'vendors' and scheduler:
            scheduler.force_vendors_update()
            return jsonify({"message": "Vendors update triggered"})
        elif update_type == 'orders' and scheduler:
            scheduler.force_orders_update()
            return jsonify({"message": "Orders update triggered"})
        else:
            return jsonify({"error": "Invalid update type or scheduler not available"}), 400
            
    except Exception as e:
        logger.error(f"Error forcing data update: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/cache-stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics (admin endpoint)"""
    try:
        stats = {}
        
        if db_manager:
            stats['database'] = db_manager.get_database_stats()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/clear-cache', methods=['POST'])
def clear_cache():
    """Clear cache (admin endpoint)"""
    try:
        cache_type = request.json.get('type', 'all')
        
        if cache_type in ("all", "database"):
            try:
                with db_manager.get_connection() as conn:
                    conn.execute("DELETE FROM coverage_grid_cache")
                    conn.execute("DELETE FROM heatmap_cache")
                    conn.commit()
                    logger.info("Cleared database cache")
            except Exception as e:
                logger.error(f"Error clearing database cache: {e}")
            
        return jsonify({"message": f"Cache cleared: {cache_type}"})
            
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": str(e)}), 500

def open_browser():
    """Open web browser to the application URL"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open_new(f"http://127.0.0.1:{config.FLASK_PORT}/")

# --- Application Entry Point ---

if __name__ == '__main__':
    # Create and configure the app
    app = create_app()
    
    # Open browser in development mode
    if config.FLASK_DEBUG and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        # Run the application
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG,
            use_reloader=config.FLASK_DEBUG
        )
    finally:
        # Cleanup on shutdown
        if scheduler:
            scheduler.stop()
        logger.info("Application shutdown complete")