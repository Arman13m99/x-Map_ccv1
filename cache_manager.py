# cache_manager.py - Fixed preloading combinations
import json
import hashlib
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from models import DatabaseManager, generate_cache_key
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class CoverageGridTask:
    """Represents a coverage grid calculation task"""
    city_name: str
    business_lines: List[str]
    vendor_filters: Dict[str, Any]
    priority: int = 1  # 1 = highest, 5 = lowest
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class CoverageGridCacheManager:
    """Manages intelligent caching and preloading of coverage grid calculations"""
    
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.preload_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="coverage_preload")
        self.preload_queue = []
        self.queue_lock = threading.Lock()
        self.is_preloading = False
        
        # In-memory cache for frequently accessed grids
        self.memory_cache = {}
        self.memory_cache_lock = threading.Lock()
        self.max_memory_cache_size = 50
        
        # Fixed combinations based on actual data structure
        try:
            self.common_combinations = self._define_common_combinations()
        except Exception as e:
            logger.error(f"Error defining preload combinations: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.common_combinations = []  # Fallback to empty list
        
    def _define_common_combinations(self) -> List[Dict[str, Any]]:
        """Define common filter combinations based on actual data structure"""
        combinations = []
        
        # Cities to preload
        cities = ["tehran", "mashhad", "shiraz"]
        
        # Coverage grid works without business line filtering when vendor data lacks business lines
        # Use a single "all vendors" category for preloading
        business_lines = [
            ["All"]  # Single category that matches all vendors regardless of business line
        ]
        
        # Simplified vendor filters - focus on realistic data scenarios
        vendor_filters = [
            # Most common filter - active vendors (status 5), any visibility
            {
                'status_ids': [5],
                'grades': None,
                'visible': None,  # Don't filter by visibility - most are 0
                'is_open': None
            },
            # High quality active vendors
            {
                'status_ids': [5],
                'grades': ["A", "A+", "B"],
                'visible': None,
                'is_open': None
            },
            # All active vendors that are open
            {
                'status_ids': [5],
                'grades': None,
                'visible': None,
                'is_open': 1
            }
        ]
        
        # Debug logging
        logger.debug(f"Cities: {cities}")
        logger.debug(f"Business lines: {business_lines}")
        logger.debug(f"Vendor filters: {len(vendor_filters)} filters")
        
        # Generate all combinations
        for city in cities:
            for bl_combo in business_lines:
                for filter_combo in vendor_filters:
                    combinations.append({
                        'city_name': city,
                        'business_lines': bl_combo,
                        'vendor_filters': filter_combo,
                        'priority': self._calculate_priority(city, bl_combo, filter_combo)
                    })
        
        logger.info(f"Defined {len(combinations)} preloading combinations")
        return combinations
    
    def _calculate_priority(self, city: str, business_lines: List[str], filters: Dict) -> int:
        """Calculate priority for preloading (1 = highest, 5 = lowest)"""
        priority = 3  # Default medium priority
        
        # Tehran gets highest priority (most used)
        if city == "tehran":
            priority -= 1
        elif city == "mashhad":
            priority += 0  # Keep medium
        else:  # shiraz
            priority += 1
            
        # Restaurant and Cafe get higher priority (most common)
        if business_lines and any(bl.lower() in ["restaurant", "cafe"] for bl in business_lines):
            priority -= 1
            
        # Single business line gets higher priority (more common use case)
        if business_lines and len(business_lines) == 1:
            priority -= 1
            
        # High quality filters get higher priority
        grades = filters.get('grades', [])
        if grades and set(grades) == {"A", "A+"}:
            priority -= 1
        
        # Open vendors get higher priority    
        if filters.get('is_open') == 1:
            priority -= 1
            
        return max(1, min(5, priority))
    
    def get_or_calculate_coverage_grid(self, 
                                     city_name: str,
                                     business_lines: List[str],
                                     vendor_filters: Dict[str, Any],
                                     force_recalculate: bool = False,
                                     radius_modifier: float = 1.0,
                                     radius_mode: str = 'percentage',
                                     radius_fixed: float = 3.0) -> Optional[List[Dict]]:
        """Get coverage grid from cache or calculate if not available"""
        
        # Generate cache key including radius parameters
        radius_params = {
            'radius_modifier': radius_modifier,
            'radius_mode': radius_mode, 
            'radius_fixed': radius_fixed
        }
        extended_filters = {**vendor_filters, 'radius_params': radius_params}
        cache_key = generate_cache_key(city_name, business_lines, extended_filters)
        
        # Force recalculation if radius is modified (interactive user changes)
        is_radius_modified = (radius_mode == 'percentage' and radius_modifier != 1.0) or (radius_mode == 'fixed')
        if is_radius_modified:
            force_recalculate = True
            logger.info(f"Forcing recalculation due to radius modification: {radius_mode} = {radius_modifier if radius_mode == 'percentage' else radius_fixed}")
        
        # Check memory cache first (fastest)
        if not force_recalculate:
            with self.memory_cache_lock:
                if cache_key in self.memory_cache:
                    logger.info(f"Coverage grid found in memory cache: {cache_key[:8]}...")
                    self._update_memory_cache_access(cache_key)
                    return self.memory_cache[cache_key]['data']
        
        # Check database cache
        if not force_recalculate:
            cached_data = self.db_manager.get_cached_coverage_grid(cache_key)
            if cached_data:
                logger.info(f"Coverage grid found in database cache: {cache_key[:8]}...")
                # Add to memory cache for faster future access
                self._add_to_memory_cache(cache_key, cached_data)
                return cached_data
        
        # If not in cache, add to preload queue for future requests
        self._add_to_preload_queue(city_name, business_lines, vendor_filters)
        
        # Calculate immediately for this request
        logger.info(f"Calculating coverage grid for: {city_name}, BL: {business_lines}")
        grid_data = self._calculate_coverage_grid(city_name, business_lines, vendor_filters, radius_modifier, radius_mode, radius_fixed)
        
        if grid_data:
            # Cache the result
            self.db_manager.cache_coverage_grid(
                cache_key, city_name, 
                ','.join(business_lines) if business_lines else '',
                vendor_filters, grid_data
            )
            self._add_to_memory_cache(cache_key, grid_data)
            
        return grid_data
    
    def _calculate_coverage_grid(self, 
                                city_name: str, 
                                business_lines: List[str],
                                vendor_filters: Dict[str, Any],
                                radius_modifier: float = 1.0,
                                radius_mode: str = 'percentage', 
                                radius_fixed: float = 3.0) -> Optional[List[Dict]]:
        """Calculate coverage grid with optimized vendor fetching"""
        try:
            # Import here to avoid circular imports
            from app_optimized import generate_coverage_grid, calculate_coverage_for_grid_vectorized
            
            # Get filtered vendors from database
            # Don't filter by business_lines if it's ["All"] since vendor data lacks business lines
            bl_filter = None if business_lines == ["All"] else business_lines
            
            vendors_df = self.db_manager.get_vendors(
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
            else:
                logger.info(f"Found {len(vendors_df)} vendors for coverage calculation")
            
            # Generate grid points
            grid_points = generate_coverage_grid(city_name, self.config.GRID_SIZE_METERS)
            if not grid_points:
                logger.warning(f"No grid points generated for city: {city_name}")
                return []
            
            # Log grid size for monitoring (no more limiting)
            logger.info(f"Generated coverage grid with {len(grid_points)} points (200m spacing)")
            
            # Find marketing areas for points using database directly
            point_area_info = self._find_marketing_areas_for_points(grid_points, city_name)
            
            # Apply radius modifications to vendors
            vendors_df = self._apply_radius_modifications(vendors_df, radius_modifier, radius_mode, radius_fixed)
            
            # Calculate coverage
            coverage_results = calculate_coverage_for_grid_vectorized(grid_points, vendors_df, city_name)
            
            # Process results with target-based logic if applicable
            processed_grid_data = self._process_coverage_results(
                coverage_results, point_area_info, business_lines, city_name
            )
            
            logger.info(f"Calculated coverage grid: {len(processed_grid_data)} points with coverage")
            return processed_grid_data
            
        except Exception as e:
            logger.error(f"Error calculating coverage grid: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _find_marketing_areas_for_points(self, points: List[Dict], city_name: str) -> List[Tuple[Optional[str], Optional[str]]]:
        """Find which marketing area each point belongs to using database"""
        try:
            # Load marketing areas from database
            marketing_areas_gdf = self.db_manager.get_marketing_areas(city_name)
            
            if marketing_areas_gdf.empty:
                logger.warning(f"No marketing areas found for city: {city_name}")
                return [(None, None)] * len(points)
            
            logger.info(f"Loaded {len(marketing_areas_gdf)} marketing areas for {city_name}")
            
            from shapely.geometry import Point
            from shapely.strtree import STRtree
            
            area_geoms = marketing_areas_gdf.geometry.values
            area_ids = marketing_areas_gdf['area_id'].values if 'area_id' in marketing_areas_gdf else [None] * len(marketing_areas_gdf)
            area_names = marketing_areas_gdf['name'].values if 'name' in marketing_areas_gdf else [None] * len(marketing_areas_gdf)
            
            # Create spatial index
            tree = STRtree(area_geoms)
            results = []
            
            for point in points:
                point_geom = Point(point['lng'], point['lat'])
                candidate_indices = tree.query(point_geom)
                
                found_area = False
                for idx in candidate_indices:
                    if area_geoms[idx].contains(point_geom):
                        area_id = area_ids[idx] if idx < len(area_ids) else None
                        area_name = area_names[idx] if idx < len(area_names) else None
                        results.append((area_id, area_name))
                        found_area = True
                        break
                        
                if not found_area:
                    results.append((None, None))
            
            # Log success rate for debugging
            found_count = sum(1 for area_id, _ in results if area_id is not None)
            logger.info(f"Marketing area lookup: {found_count}/{len(points)} points matched ({found_count/len(points)*100:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding marketing areas for points: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [(None, None)] * len(points)
    
    def _apply_radius_modifications(self, vendors_df: pd.DataFrame, radius_modifier: float, radius_mode: str, radius_fixed: float) -> pd.DataFrame:
        """Apply radius modifications to vendors DataFrame"""
        if vendors_df.empty or 'radius' not in vendors_df.columns:
            return vendors_df
            
        vendors_modified = vendors_df.copy()
        
        if radius_mode == 'percentage':
            # Apply percentage modifier (e.g., 0.5 = 50% of original radius)
            vendors_modified['radius'] = vendors_df['radius'] * radius_modifier
        elif radius_mode == 'fixed':
            # Set all vendors to fixed radius
            vendors_modified['radius'] = radius_fixed
        
        # Log the radius modification for debugging
        if radius_mode == 'percentage' and radius_modifier != 1.0:
            logger.info(f"Applied {radius_modifier*100:.0f}% radius modifier to {len(vendors_modified)} vendors")
        elif radius_mode == 'fixed':
            logger.info(f"Set fixed radius {radius_fixed}km for {len(vendors_modified)} vendors")
            
        return vendors_modified
    
    def _process_coverage_results(self, 
                                 coverage_results: List[Dict],
                                 point_area_info: List[Tuple],
                                 business_lines: List[str],
                                 city_name: str) -> List[Dict]:
        """Process coverage results with target-based analysis"""
        processed_data = []
        
        # Load target lookup if available
        target_lookup = {}
        if city_name == "tehran" and business_lines and len(business_lines) == 1:
            try:
                # Import here to avoid circular imports
                from models import DatabaseManager
                from config import Config
                config = Config()
                db_manager = DatabaseManager(config.DATABASE_PATH)
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
    
    def _add_to_memory_cache(self, cache_key: str, data: List[Dict]):
        """Add data to memory cache with LRU eviction"""
        with self.memory_cache_lock:
            # Remove oldest entries if cache is full
            if len(self.memory_cache) >= self.max_memory_cache_size:
                # Find least recently used
                oldest_key = min(self.memory_cache.keys(), 
                               key=lambda k: self.memory_cache[k]['last_accessed'])
                del self.memory_cache[oldest_key]
            
            self.memory_cache[cache_key] = {
                'data': data,
                'last_accessed': datetime.now(),
                'access_count': 1
            }
    
    def _update_memory_cache_access(self, cache_key: str):
        """Update access time and count for memory cache entry"""
        if cache_key in self.memory_cache:
            self.memory_cache[cache_key]['last_accessed'] = datetime.now()
            self.memory_cache[cache_key]['access_count'] += 1
    
    def _add_to_preload_queue(self, city_name: str, business_lines: List[str], vendor_filters: Dict[str, Any]):
        """Add a coverage grid calculation to the preload queue"""
        task = CoverageGridTask(
            city_name=city_name,
            business_lines=business_lines,
            vendor_filters=vendor_filters,
            priority=self._calculate_priority(city_name, business_lines, vendor_filters)
        )
        
        with self.queue_lock:
            # Check if similar task already exists
            cache_key = generate_cache_key(city_name, business_lines, vendor_filters)
            existing_keys = [generate_cache_key(t.city_name, t.business_lines, t.vendor_filters) 
                           for t in self.preload_queue]
            
            if cache_key not in existing_keys:
                self.preload_queue.append(task)
                # Sort by priority
                self.preload_queue.sort(key=lambda x: x.priority)
    
    def start_preloading(self):
        """Start preloading common coverage grid combinations"""
        if self.is_preloading:
            logger.warning("Preloading already in progress")
            return
        
        self.is_preloading = True
        
        # Add common combinations to queue
        for combo in self.common_combinations:
            task = CoverageGridTask(**combo)
            with self.queue_lock:
                self.preload_queue.append(task)
        
        # Sort queue by priority
        with self.queue_lock:
            self.preload_queue.sort(key=lambda x: x.priority)
        
        # Start preloading worker
        self.preload_executor.submit(self._preload_worker)
        logger.info(f"Started coverage grid preloading with {len(self.preload_queue)} tasks")
    
    def _preload_worker(self):
        """Worker function for preloading coverage grids"""
        logger.info("Coverage grid preloading worker started")
        
        while self.is_preloading:
            task = None
            
            with self.queue_lock:
                if self.preload_queue:
                    task = self.preload_queue.pop(0)
            
            if task:
                try:
                    # Check if already cached
                    cache_key = generate_cache_key(task.city_name, task.business_lines, task.vendor_filters)
                    
                    if not self.db_manager.get_cached_coverage_grid(cache_key):
                        logger.info(f"Preloading coverage grid: {task.city_name}, BL: {task.business_lines}")
                        
                        grid_data = self._calculate_coverage_grid(
                            task.city_name, task.business_lines, task.vendor_filters
                        )
                        
                        if grid_data:
                            self.db_manager.cache_coverage_grid(
                                cache_key, task.city_name,
                                ','.join(task.business_lines) if task.business_lines else '',
                                task.vendor_filters, grid_data
                            )
                            logger.info(f"Preloaded coverage grid: {cache_key[:8]}... ({len(grid_data)} points)")
                        else:
                            logger.warning(f"No grid data generated for: {task.city_name}, {task.business_lines}")
                    
                except Exception as e:
                    logger.error(f"Error preloading coverage grid: {e}")
                
                # Small delay between calculations to not overwhelm the system
                time.sleep(2)
            else:
                # No tasks in queue, wait before checking again
                time.sleep(10)
        
        logger.info("Coverage grid preloading worker stopped")
    
    def stop_preloading(self):
        """Stop the preloading process"""
        self.is_preloading = False
        self.preload_executor.shutdown(wait=False)
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.memory_cache_lock:
            memory_stats = {
                'size': len(self.memory_cache),
                'max_size': self.max_memory_cache_size,
                'keys': list(self.memory_cache.keys())
            }
        
        with self.queue_lock:
            queue_stats = {
                'pending_tasks': len(self.preload_queue),
                'is_preloading': self.is_preloading
            }
        
        # Database cache stats
        db_stats = {}
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM coverage_grid_cache")
                db_stats['database_cache_size'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(point_count) FROM coverage_grid_cache")
                avg_points = cursor.fetchone()[0]
                db_stats['average_points_per_grid'] = avg_points if avg_points else 0
        except Exception as e:
            logger.error(f"Error getting database cache stats: {e}")
            db_stats = {'error': str(e)}
        
        return {
            'memory_cache': memory_stats,
            'preload_queue': queue_stats,
            'database_cache': db_stats
        }
    
    def clear_cache(self, cache_type: str = "all"):
        """Clear cache (memory, database, or both)"""
        if cache_type in ("all", "memory"):
            with self.memory_cache_lock:
                self.memory_cache.clear()
                logger.info("Cleared memory cache")
        
        if cache_type in ("all", "database"):
            try:
                with self.db_manager.get_connection() as conn:
                    conn.execute("DELETE FROM coverage_grid_cache")
                    conn.commit()
                    logger.info("Cleared database cache")
            except Exception as e:
                logger.error(f"Error clearing database cache: {e}")
    
    def warm_up_cache(self, priority_cities: List[str] = None):
        """Warm up cache with high-priority combinations"""
        if priority_cities is None:
            priority_cities = ["tehran"]
        
        high_priority_combos = [
            combo for combo in self.common_combinations
            if combo['city_name'] in priority_cities and combo['priority'] <= 2
        ]
        
        logger.info(f"Warming up cache with {len(high_priority_combos)} high-priority combinations")
        
        for combo in high_priority_combos:
            try:
                self.get_or_calculate_coverage_grid(
                    combo['city_name'],
                    combo['business_lines'],
                    combo['vendor_filters']
                )
            except Exception as e:
                logger.error(f"Error warming up cache for {combo}: {e}")

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> Optional[CoverageGridCacheManager]:
    """Get the global cache manager instance"""
    return _cache_manager

def init_cache_manager(config: Config, db_manager: DatabaseManager) -> CoverageGridCacheManager:
    """Initialize the global cache manager instance"""
    global _cache_manager
    _cache_manager = CoverageGridCacheManager(config, db_manager)
    return _cache_manager