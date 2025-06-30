#!/usr/bin/env python3
"""
VIGIA Medical AI - Smart Care Interface Launcher
===============================================

Launch the exact Smart Care interface copy for VIGIA.
"""

print("🏥 VIGIA Medical AI - Smart Care Interface")
print("=" * 50)
print("🎨 Loading exact Smart Care replica...")
print("📱 Modern, clean medical interface")
print("🩺 VIGIA medical terminology")
print("✨ Simple and professional design")
print()

try:
    from ui_components.smartcare_interface import launch_smartcare_interface
    
    print("✅ Smart Care interface loaded!")
    print("🚀 Launching...")
    print("-" * 50)
    
    # Launch the Smart Care interface
    launch_smartcare_interface(
        share=True,
        debug=False,
        server_port=7860
    )
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("🔧 Fallback: Creating basic interface...")
    
    try:
        from ui_components.smartcare_interface import create_smartcare_interface
        
        interface = create_smartcare_interface()
        interface.launch(
            share=True,
            server_port=7860
        )
        
    except Exception as fallback_error:
        print(f"❌ Fallback failed: {fallback_error}")
        print("Please check that all files are installed correctly.")

if __name__ == "__main__":
    pass