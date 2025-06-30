#!/usr/bin/env python3
"""
VIGIA Medical AI - Risk Assessment Visualization Launcher
========================================================

Launcher específico para la visualización del assessment de riesgo
que hace el agente con Batman tokens.
"""

print("🧠 VIGIA Medical AI - Risk Assessment Agent Visualization")
print("=" * 60)
print("🦇 Batman Token Analysis")
print("📊 7 Factores de Riesgo Principales")
print("🔬 Assessment del Agente de Riesgo")
print("📈 Visualización en Barras Horizontales")
print()

try:
    from ui_components.risk_visualization import create_risk_assessment_interface
    
    print("✅ Risk assessment visualization loaded!")
    print("🚀 Launching interactive interface...")
    print("-" * 60)
    
    # Launch the risk assessment visualization
    interface = create_risk_assessment_interface()
    interface.launch(
        share=True,
        debug=False,
        server_port=7862,
        server_name="0.0.0.0"
    )
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("🔧 Fallback: Creating basic interface...")
    
    try:
        import gradio as gr
        
        def simple_assessment(batman_token):
            return f"""
            <div style="padding: 2rem; background: white; border-radius: 12px;">
                <h3>🧠 Assessment Básico</h3>
                <p><strong>Batman Token:</strong> {batman_token}</p>
                <p><strong>Status:</strong> Assessment functionality not available</p>
                <p>Please check that all components are installed correctly.</p>
            </div>
            """
        
        with gr.Blocks(title="VIGIA Risk Assessment - Fallback") as fallback:
            gr.HTML("<h1>🧠 VIGIA Risk Assessment (Fallback Mode)</h1>")
            token_input = gr.Textbox(label="Batman Token", value="BATMAN_12345")
            assess_btn = gr.Button("Run Assessment")
            output = gr.HTML()
            
            assess_btn.click(simple_assessment, token_input, output)
        
        fallback.launch(share=True, server_port=7862)
        
    except Exception as fallback_error:
        print(f"❌ Fallback failed: {fallback_error}")
        print("Please check your Gradio installation.")

if __name__ == "__main__":
    pass