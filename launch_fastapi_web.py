#!/usr/bin/env python3
"""
VIGIA Medical AI - FastAPI Web Application Launcher
==================================================

Launch the professional medical web interface.
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Launch the VIGIA FastAPI web application."""
    
    print("🏥 VIGIA Medical AI - Professional Web Interface")
    print("=" * 60)
    print("🚀 Starting FastAPI medical application...")
    print("🎨 Professional medical interface with 9-agent analysis")
    print("🔒 HIPAA-compliant with PHI tokenization")
    print("📱 Responsive design for medical devices")
    print()
    
    # Set environment variables
    os.environ.setdefault("VIGIA_ENV", "development")
    
    try:
        # Import and run the FastAPI app
        from src.web.main import app
        
        print("✅ VIGIA medical system loaded successfully!")
        print("🌐 Web interface starting...")
        print("-" * 60)
        print("📍 Local URL:    http://127.0.0.1:8000")
        print("📋 API Docs:     http://127.0.0.1:8000/api/docs")
        print("🔧 Admin Panel:  http://127.0.0.1:8000/api/redoc")
        print("-" * 60)
        print("🏥 Ready for medical analysis!")
        print()
        
        # Launch FastAPI with uvicorn
        uvicorn.run(
            "src.web.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            access_log=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("   • Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check that the VIGIA medical system is properly set up")
        print("   • Verify that you're in the correct directory")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("   • Check that port 8000 is available")
        print("   • Ensure all VIGIA components are properly configured")
        print("   • Review the error message above for specific issues")

if __name__ == "__main__":
    main()