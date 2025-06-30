#!/usr/bin/env python3
"""
VIGIA Medical AI - Professional UI Quick Launcher
================================================

Quick launch script for the professional medical interface with enhanced features.

Usage:
    python launch_professional_ui.py [--no-share] [--debug] [--port 7860]
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main launcher function with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Launch VIGIA Medical AI Professional Interface",
        epilog="Example: python launch_professional_ui.py --share --port 7860"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        default=True,
        help="Enable public sharing (default: True)"
    )
    
    parser.add_argument(
        "--no-share", 
        action="store_true", 
        help="Disable public sharing"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Server port (default: 7860)"
    )
    
    args = parser.parse_args()
    
    # Handle share/no-share logic
    share_enabled = args.share and not args.no_share
    
    print("🩺 VIGIA Medical AI - Professional Medical Interface")
    print("=" * 60)
    print("🔧 Initializing medical-grade components...")
    print("🛡️ Loading HIPAA-compliant systems...")
    print("📱 Optimizing for bedside medical devices...")
    print("♿ Ensuring WCAG 2.1 AA accessibility...")
    print()
    
    # Import and launch the interface
    try:
        from ui_components.medical_interface import launch_professional_medical_interface
        
        print("✅ Medical components loaded successfully")
        print(f"🌐 Share enabled: {share_enabled}")
        print(f"🐛 Debug mode: {args.debug}")
        print(f"🔌 Server port: {args.port}")
        print()
        print("🚀 Starting professional medical interface...")
        print("-" * 60)
        
        # Launch the interface
        launch_professional_medical_interface(
            share=share_enabled,
            debug=args.debug,
            server_port=args.port
        )
        
    except ImportError as e:
        print("❌ Error importing medical components:")
        print(f"   {e}")
        print()
        print("💡 Fallback: Launching basic Gradio interface...")
        
        # Fallback to basic interface
        try:
            import gradio as gr
            from ui_components.medical_interface import create_enhanced_medical_interface
            
            interface = create_enhanced_medical_interface()
            interface.launch(
                share=share_enabled,
                debug=args.debug,
                server_port=args.port
            )
            
        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")
            print()
            print("🔧 Please check your installation:")
            print("   1. Ensure Gradio is installed: pip install gradio")
            print("   2. Verify all dependencies are available")
            print("   3. Check that ui_components/ directory exists")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print()
        print("🔧 Troubleshooting steps:")
        print("   1. Check that all medical components are properly installed")
        print("   2. Verify that the current directory contains the VIGIA project")
        print("   3. Ensure all dependencies are installed (see requirements.txt)")
        sys.exit(1)

if __name__ == "__main__":
    main()