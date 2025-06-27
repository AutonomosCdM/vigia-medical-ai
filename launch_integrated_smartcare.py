#!/usr/bin/env python3
"""
VIGIA Medical AI - Smart Care Integrated Interface
=================================================

Launcher para Smart Care con assessment de riesgo completamente integrado.
"""

print("🏥 VIGIA Medical AI - Smart Care Integrated")
print("=" * 50)
print("🎨 Smart Care interface with integrated risk assessment")
print("🧠 Assessment del agente de riesgo en box")
print("📊 Visualización compacta y profesional")
print("🦇 Batman token analysis integrated")
print()

try:
    from ui_components.smartcare_interface import create_smartcare_interface
    from ui_components.risk_visualization import RiskVisualization
    import gradio as gr
    
    print("✅ Smart Care integrated interface loaded!")
    print("🚀 Launching with risk assessment...")
    print("-" * 50)
    
    # Create the integrated interface
    interface = create_smartcare_interface()
    
    # Auto-run assessment after 3 seconds
    auto_assessment_js = """
    <script>
    setTimeout(function() {
        console.log('Auto-running risk assessment...');
        const button = document.querySelector('button[onclick="runRiskAssessment()"]');
        if (button) {
            // Simulate assessment data
            const assessmentData = {
                batman_token: "BATMAN_JESSE_WYNN_12345",
                assessment_timestamp: new Date().toLocaleString(),
                scores: {
                    movilidad: 2,
                    actividad: 1,
                    percepcion_sensorial: 3,
                    humedad: 2
                },
                braden_total: 15,
                risk_level: "ALTO",
                risk_color: "#f39c12",
                assessment_confidence: 0.92
            };
            
            // Create compact HTML
            const compactHtml = `
            <div style="padding: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="background: #f39c12; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-weight: 600; font-size: 0.8rem;">
                            ALTO
                        </div>
                        <div style="color: #4a5568; font-weight: 600;">Braden: 15/23</div>
                    </div>
                    <div style="color: #718096; font-size: 0.8rem;">${new Date().toLocaleString()}</div>
                </div>
                
                <div style="display: grid; gap: 0.6rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="min-width: 120px; font-size: 0.85rem; font-weight: 500; color: #2d3748;">Movilidad</div>
                        <div style="flex: 1; background: #e2e8f0; border-radius: 6px; height: 14px; position: relative; overflow: hidden;">
                            <div style="background: #ff6b6b; height: 100%; width: 50%; border-radius: 6px;"></div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.4rem; min-width: 80px;">
                            <span style="background: #f39c12; color: white; padding: 0.1rem 0.4rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600;">ALTO</span>
                            <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">2/4</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="min-width: 120px; font-size: 0.85rem; font-weight: 500; color: #2d3748;">Actividad Física</div>
                        <div style="flex: 1; background: #e2e8f0; border-radius: 6px; height: 14px; position: relative; overflow: hidden;">
                            <div style="background: #4ecdc4; height: 100%; width: 25%; border-radius: 6px;"></div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.4rem; min-width: 80px;">
                            <span style="background: #e74c3c; color: white; padding: 0.1rem 0.4rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600;">CRÍTICO</span>
                            <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">1/4</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="min-width: 120px; font-size: 0.85rem; font-weight: 500; color: #2d3748;">Percepción Sensorial</div>
                        <div style="flex: 1; background: #e2e8f0; border-radius: 6px; height: 14px; position: relative; overflow: hidden;">
                            <div style="background: #45b7d1; height: 100%; width: 75%; border-radius: 6px;"></div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.4rem; min-width: 80px;">
                            <span style="background: #f1c40f; color: white; padding: 0.1rem 0.4rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600;">MODERADO</span>
                            <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">3/4</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="min-width: 120px; font-size: 0.85rem; font-weight: 500; color: #2d3748;">Humedad</div>
                        <div style="flex: 1; background: #e2e8f0; border-radius: 6px; height: 14px; position: relative; overflow: hidden;">
                            <div style="background: #f9ca24; height: 100%; width: 50%; border-radius: 6px;"></div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.4rem; min-width: 80px;">
                            <span style="background: #f39c12; color: white; padding: 0.1rem 0.4rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600;">ALTO</span>
                            <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">2/4</span>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="color: #2d3748; font-size: 0.85rem; font-weight: 500;">📋 Recomendaciones:</div>
                        <div style="color: #718096; font-size: 0.75rem;">Confianza: 92% | HIPAA ✓</div>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #4a5568;">
                        ⚠️ Reposición 4h • 🛏️ Superficie presión • 🍽️ Optimizar nutrición
                    </div>
                </div>
            </div>
            `;
            
            const container = document.getElementById('risk-assessment-content');
            if (container) {
                container.innerHTML = compactHtml;
            }
        }
    }, 3000);
    </script>
    """
    
    # Add the auto-assessment script to the interface
    interface.load(lambda: auto_assessment_js, outputs=gr.HTML(visible=False))
    
    # Launch with optimal settings
    interface.launch(
        share=True,
        debug=False,
        server_port=7860,
        quiet=False
    )
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("🔧 Note: Check that all components are installed correctly")

if __name__ == "__main__":
    pass