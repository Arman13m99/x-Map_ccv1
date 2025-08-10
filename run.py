#!/usr/bin/env python3
"""
Main entry point for X-Map Dashboard
Optimized Flask application with coverage grid caching
"""

import os
import sys
import threading
import webbrowser
from app_optimized import create_app
from config import get_config

def open_browser():
    """Open browser after a short delay"""
    import time
    time.sleep(1.5)
    webbrowser.open('http://localhost:5001')

if __name__ == '__main__':
    config = get_config()
    
    # Create and configure the app
    app = create_app()
    
    # Open browser in development mode
    if config.FLASK_DEBUG and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Thread(target=open_browser, daemon=True).start()
    
    print("Starting X-Map Dashboard...")
    print(f"Dashboard URL: http://localhost:{config.FLASK_PORT}")
    
    try:
        # Run the application
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG,
            use_reloader=config.FLASK_DEBUG
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down X-Map Dashboard...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)