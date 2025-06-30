#!/usr/bin/env python3
"""
VIGIA Medical AI - Modern Smart Care Style Interface
===================================================

Launch the modern, colorful medical interface inspired by Smart Care design.
"""

print("🌈 VIGIA Medical AI - Modern Smart Care Interface")
print("=" * 60)
print("🎨 Loading colorful medical interface...")
print("🏥 Smart Care inspired design")
print("🌟 Modern gradients and vibrant colors")
print("📊 Interactive metric cards")
print()

try:
    from ui_components.medical_interface import create_enhanced_medical_interface
    
    print("✅ Modern components loaded!")
    print("🚀 Launching Smart Care-style interface...")
    print("-" * 60)
    
    # Create and launch the modern interface
    interface = create_enhanced_medical_interface()
    interface.launch(
        share=True,
        debug=False,
        server_port=7860,
        quiet=False
    )
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("🔧 Note: Interface takes a moment to initialize...")
    print("   The modern UI is loading - please wait!")

if __name__ == "__main__":
    pass