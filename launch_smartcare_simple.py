#!/usr/bin/env python3
"""
VIGIA Medical AI - Smart Care Simple Launcher
============================================

Version simple y funcional de Smart Care con assessment.
"""

import gradio as gr
from ui_components.risk_visualization import RiskVisualization

def create_simple_smartcare():
    """Crea versión simple de Smart Care con assessment funcional."""
    
    # Load CSS
    css = """
    /* Smart Care Simple Styles */
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
    .header { background: linear-gradient(135deg, #ff6b6b 0%, #8b5cf6 100%); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
    .metric-card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .patient-card { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; }
    .assessment-box { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    """
    
    with gr.Blocks(css=css, title="VIGIA Smart Care") as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="margin: 0; display: flex; align-items: center; gap: 1rem;">
                🏥 VIGIA Care
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">
                    Demo Mode
                </span>
            </h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Professional Pressure Injury Detection & Risk Assessment
            </p>
        </div>
        """)
        
        # Metrics
        gr.HTML("""
        <div class="metrics">
            <div class="metric-card" style="background: linear-gradient(135deg, #8b5cf6, #6366f1); color: white;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;">96.2%</div>
                        <div style="opacity: 0.9;">AI Accuracy</div>
                    </div>
                    <div style="font-size: 1.5rem;">📊</div>
                </div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #f59e0b, #f97316); color: white;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;">1.2s</div>
                        <div style="opacity: 0.9;">Analysis Speed</div>
                    </div>
                    <div style="font-size: 1.5rem;">⚡</div>
                </div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #10b981, #059669); color: white;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;">100%</div>
                        <div style="opacity: 0.9;">HIPAA Compliance</div>
                    </div>
                    <div style="font-size: 1.5rem;">🛡️</div>
                </div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #3b82f6, #2563eb); color: white;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;">24/7</div>
                        <div style="opacity: 0.9;">Medical Support</div>
                    </div>
                    <div style="font-size: 1.5rem;">👨‍⚕️</div>
                </div>
            </div>
        </div>
        """)
        
        with gr.Row():
            # Patient Info Column
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="patient-card">
                    <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem;">
                        <div style="width: 80px; height: 80px; border-radius: 50%; background: #e3f2fd; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                            👨
                        </div>
                        <div>
                            <h2 style="margin: 0; color: #2d3748;">Patient</h2>
                            <h3 style="margin: 0.5rem 0; color: #2d3748;">Mr. Jesse Wynn</h3>
                            <button style="background: #f5f5f5; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.8rem;">
                                VIEW PROFILE
                            </button>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.9rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #718096;">Sex:</span>
                            <span style="font-weight: 500;">Male</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #718096;">Check-in:</span>
                            <span style="font-weight: 500;">24 Feb, 2020</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #718096;">Age:</span>
                            <span style="font-weight: 500;">32</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #718096;">Dept:</span>
                            <span style="font-weight: 500;">Wound Care</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #718096;">Blood:</span>
                            <span style="font-weight: 500;">B+</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #718096;">Bed #:</span>
                            <span style="font-weight: 500;">0747</span>
                        </div>
                    </div>
                </div>
                """)
            
            # Assessment Column
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="assessment-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                        <h3 style="margin: 0; color: #2d3748;">🧠 Risk Assessment Agent</h3>
                    </div>
                """)
                
                # Assessment controls
                batman_token = gr.Textbox(
                    label="🦇 Batman Token",
                    value="BATMAN_JESSE_WYNN_12345",
                    placeholder="Enter patient token..."
                )
                
                assess_btn = gr.Button(
                    "🔬 Ejecutar Assessment",
                    variant="primary"
                )
                
                # Assessment output
                assessment_output = gr.HTML(
                    """
                    <div style="text-align: center; padding: 2rem; color: #718096;">
                        <p>📊 Assessment del agente de riesgo</p>
                        <p style="font-size: 0.9rem;">Haga clic en "Ejecutar Assessment" para análisis</p>
                    </div>
                    """
                )
                
                gr.HTML("</div>")
        
        # Assessment function
        def run_assessment(batman_token):
            assessment_data = RiskVisualization.simulate_batman_assessment(batman_token)
            return RiskVisualization.create_compact_assessment_html(assessment_data)
        
        assess_btn.click(
            fn=run_assessment,
            inputs=[batman_token],
            outputs=[assessment_output]
        )
    
    return interface

print("🏥 VIGIA Medical AI - Smart Care Simple")
print("=" * 45)
print("🎨 Simple and functional Smart Care interface")
print("🧠 Integrated risk assessment")
print("📊 Clean and professional design")
print()

try:
    interface = create_simple_smartcare()
    
    print("✅ Simple Smart Care interface created!")
    print("🚀 Launching...")
    print("-" * 45)
    
    interface.launch(
        share=True,
        debug=False,
        server_port=7860
    )
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("🔧 Check that all components are installed correctly")