# config.py - Application configuration
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration with environment variable support"""
    
    # Database settings
    DATABASE_PATH: str = "tapsi_food_data.db"
    
    # Metabase settings
    METABASE_URL: str = "https://metabase.ofood.cloud"
    METABASE_USERNAME: str = "a.mehmandoost@OFOOD.CLOUD"
    METABASE_PASSWORD: str = "Fff322666@"
    ORDER_DATA_QUESTION_ID: int = 5822
    VENDOR_DATA_QUESTION_ID: int = 5045
    
    # Data fetching settings
    WORKER_COUNT: int = 10
    PAGE_SIZE: int = 100000
    
    # Cache settings
    CACHE_CLEANUP_DAYS: int = 7
    MAX_COVERAGE_CACHE_SIZE: int = 1000
    MAX_HEATMAP_CACHE_SIZE: int = 500
    
    # Flask settings
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5001
    FLASK_DEBUG: bool = False
    
    # Performance settings
    ENABLE_QUERY_OPTIMIZATION: bool = True
    PRELOAD_COVERAGE_GRIDS: bool = False  # Disabled - coverage grid calculated directly without caching
    ENABLE_COMPRESSION: bool = True
    
    # Scheduler settings
    VENDORS_UPDATE_INTERVAL_MINUTES: int = 10
    ORDERS_UPDATE_TIME: str = "09:00"  # 9 AM daily
    CACHE_CLEANUP_TIME: str = "02:00"   # 2 AM daily
    
    # Geographic boundaries for grid generation
    CITY_BOUNDARIES = {
        "tehran": {"min_lat": 35.5, "max_lat": 35.85, "min_lng": 51.1, "max_lng": 51.7},
        "mashhad": {"min_lat": 36.15, "max_lat": 36.45, "min_lng": 59.35, "max_lng": 59.8},
        "shiraz": {"min_lat": 29.5, "max_lat": 29.75, "min_lng": 52.4, "max_lng": 52.7}
    }
    
    # Coverage grid settings
    GRID_SIZE_METERS: int = 200
    MAX_GRID_POINTS: int = 100000  # Increased limit to allow proper 200m grid spacing
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        # Override with environment variables if they exist
        self.METABASE_URL = os.getenv("METABASE_URL", self.METABASE_URL)
        self.METABASE_USERNAME = os.getenv("METABASE_USERNAME", self.METABASE_USERNAME)
        self.METABASE_PASSWORD = os.getenv("METABASE_PASSWORD", self.METABASE_PASSWORD)
        
        self.DATABASE_PATH = os.getenv("DATABASE_PATH", self.DATABASE_PATH)
        
        # Convert string env vars to appropriate types
        try:
            self.ORDER_DATA_QUESTION_ID = int(os.getenv("ORDER_DATA_QUESTION_ID", self.ORDER_DATA_QUESTION_ID))
            self.VENDOR_DATA_QUESTION_ID = int(os.getenv("VENDOR_DATA_QUESTION_ID", self.VENDOR_DATA_QUESTION_ID))
            self.WORKER_COUNT = int(os.getenv("WORKER_COUNT", self.WORKER_COUNT))
            self.PAGE_SIZE = int(os.getenv("PAGE_SIZE", self.PAGE_SIZE))
            self.FLASK_PORT = int(os.getenv("FLASK_PORT", self.FLASK_PORT))
            self.VENDORS_UPDATE_INTERVAL_MINUTES = int(os.getenv("VENDORS_UPDATE_INTERVAL_MINUTES", self.VENDORS_UPDATE_INTERVAL_MINUTES))
        except ValueError as e:
            print(f"Warning: Invalid environment variable value: {e}")
            
        # Boolean environment variables
        self.FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
        self.ENABLE_QUERY_OPTIMIZATION = os.getenv("ENABLE_QUERY_OPTIMIZATION", "true").lower() in ("true", "1", "yes")
        self.PRELOAD_COVERAGE_GRIDS = os.getenv("PRELOAD_COVERAGE_GRIDS", "false").lower() in ("true", "1", "yes")  # Default to False
        self.ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() in ("true", "1", "yes")
        
        # String environment variables
        self.FLASK_HOST = os.getenv("FLASK_HOST", self.FLASK_HOST)
        self.ORDERS_UPDATE_TIME = os.getenv("ORDERS_UPDATE_TIME", self.ORDERS_UPDATE_TIME)
        self.CACHE_CLEANUP_TIME = os.getenv("CACHE_CLEANUP_TIME", self.CACHE_CLEANUP_TIME)

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    FLASK_DEBUG = True
    CACHE_CLEANUP_DAYS = 1  # Clean cache more frequently in dev
    PRELOAD_COVERAGE_GRIDS = False  # Disable preloading in dev

class ProductionConfig(Config):
    """Production environment configuration"""
    FLASK_DEBUG = False
    WORKER_COUNT = 12  # More workers for production
    PAGE_SIZE = 150000  # Larger page size for production
    ENABLE_COMPRESSION = True
    PRELOAD_COVERAGE_GRIDS = False  # Disabled - direct calculation is more responsive

class TestingConfig(Config):
    """Testing environment configuration"""
    DATABASE_PATH = ":memory:"  # Use in-memory database for tests
    FLASK_DEBUG = True
    PRELOAD_COVERAGE_GRIDS = False
    VENDORS_UPDATE_INTERVAL_MINUTES = 1  # Faster updates for testing

def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("FLASK_ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Create a singleton config instance
config = get_config()