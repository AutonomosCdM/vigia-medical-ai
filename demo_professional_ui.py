#!/usr/bin/env python3
"""
VIGIA Medical AI - Professional UI Demo
======================================

Quick demo of the professional medical interface.
"""

from ui_components.medical_interface import create_enhanced_medical_interface

def main():
    """Create and launch the professional medical interface."""
    
    print("🩺 VIGIA Medical AI - Professional Medical Interface Demo")
    print("=" * 60)
    print("🔧 Creating enhanced medical interface...")
    
    try:
        # Create the interface
        interface = create_enhanced_medical_interface()
        
        print("✅ Professional medical interface created successfully!")
        print("🏥 Features included:")
        print("   • Interactive Braden Scale risk assessment")
        print("   • HIPAA-compliant medical workflow")
        print("   • Evidence-based clinical decision support")
        print("   • Mobile-optimized for bedside use")
        print("   • WCAG 2.1 AA accessibility compliance")
        print("   • Professional medical styling")
        print()
        print("🚀 Launching interface...")
        print("-" * 60)
        
        # Launch with optimal settings
        interface.launch(
            share=True,  # Enable public sharing
            debug=False,
            server_port=7860,
            quiet=True
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("   • Ensure Gradio is installed: pip install gradio")
        print("   • Check that all dependencies are available")
        print("   • Verify ui_components directory structure")

if __name__ == "__main__":
    main()