"""
VIGIA Medical AI - Smart Care Interface Exact Copy
=================================================

Exact replica of Smart Care interface with VIGIA medical terminology.
"""

import gradio as gr
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import random
from .risk_visualization import RiskVisualization

def load_smartcare_css() -> str:
    """Load Smart Care CSS styles."""
    css_path = Path(__file__).parent / "smartcare_styles.css"
    if css_path.exists():
        return css_path.read_text()
    return ""

def create_smartcare_interface() -> gr.Blocks:
    """Create exact Smart Care interface copy for VIGIA."""
    
    css = load_smartcare_css()
    
    with gr.Blocks(
        css=css,
        title="VIGIA Medical AI - Smart Care Interface",
        theme=gr.themes.Base()
    ) as interface:
        
        # Complete Smart Care HTML Structure
        gr.HTML("""
        <div class="smartcare-container">
            <!-- Sidebar - Exact Copy -->
            <div class="smartcare-sidebar">
                <div class="sidebar-header">
                    <div class="sidebar-logo">VIGIA Care</div>
                </div>
                
                <div class="sidebar-gradient">
                    <div class="sidebar-greeting">Good Morning</div>
                    <div class="sidebar-name">Dr. John Doe</div>
                </div>
                
                <nav class="sidebar-nav">
                    <button class="nav-item active">
                        <span class="nav-item-icon">📊</span>
                        Dashboard
                    </button>
                    <button class="nav-item">
                        <span class="nav-item-icon">👥</span>
                        Patients
                    </button>
                    <button class="nav-item">
                        <span class="nav-item-icon">📅</span>
                        Calendar
                    </button>
                    <button class="nav-item">
                        <span class="nav-item-icon">⚙️</span>
                        Settings
                    </button>
                    <button class="nav-item">
                        <span class="nav-item-icon">🆘</span>
                        Support
                    </button>
                </nav>
                
                <div style="position: absolute; bottom: 1rem; left: 0; right: 0;">
                    <button class="nav-item">
                        <span class="nav-item-icon">🚪</span>
                        Log Out
                    </button>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="smartcare-main">
                <!-- Top Header -->
                <div class="smartcare-header">
                    <div class="header-search">
                        <span>🔍</span>
                        <input type="text" placeholder="Search patients here..." />
                    </div>
                    
                    <div class="header-right">
                        <div class="notification-icon">
                            <span>🔔</span>
                            <div class="notification-badge">3</div>
                        </div>
                        
                        <div class="user-profile">
                            <div class="user-avatar">JD</div>
                            <span class="user-name">Dr. John Doe</span>
                            <span>⌄</span>
                        </div>
                    </div>
                </div>
                
                <!-- Content Area -->
                <div style="padding: 2rem;">
                    <!-- Breadcrumb -->
                    <div class="breadcrumb">
                        <a href="#">Patients</a> > Mr. Jesse Wynn
                    </div>
                    
                    <div class="smartcare-content">
                        <!-- Left Column -->
                        <div>
                            <!-- Patient Card -->
                            <div class="patient-card">
                                <div class="patient-header">
                                    <div class="patient-avatar">
                                        <span style="font-size: 2.5rem;">👨</span>
                                    </div>
                                    <div class="patient-info">
                                        <h2>
                                            <span style="font-size: 1.2rem; font-weight: 500;">Patient</span><br>
                                            <span class="patient-name">Mr. Jesse Wynn</span>
                                        </h2>
                                        <button class="view-profile">VIEW PROFILE</button>
                                    </div>
                                </div>
                                
                                <div class="patient-details">
                                    <div class="patient-detail">
                                        <span class="detail-label">Sex:</span>
                                        <span class="detail-value">Male</span>
                                    </div>
                                    <div class="patient-detail">
                                        <span class="detail-label">Check-in:</span>
                                        <span class="detail-value">24 Feb, 2020</span>
                                    </div>
                                    <div class="patient-detail">
                                        <span class="detail-label">Age:</span>
                                        <span class="detail-value">32</span>
                                    </div>
                                    <div class="patient-detail">
                                        <span class="detail-label">Dept:</span>
                                        <span class="detail-value">Wound Care</span>
                                    </div>
                                    <div class="patient-detail">
                                        <span class="detail-label">Blood:</span>
                                        <span class="detail-value">B+</span>
                                    </div>
                                    <div class="patient-detail">
                                        <span class="detail-label">Bed #:</span>
                                        <span class="detail-value">0747</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Medical Data Section -->
                            <div class="medical-data-section">
                                <div class="medical-data-header">Medical Data</div>
                                <div class="medical-graph"></div>
                            </div>
                        </div>
                        
                        <!-- Right Column -->
                        <div>
                            <!-- Metrics Grid -->
                            <div class="metrics-grid">
                                <!-- LPP Grade Card (Dark) -->
                                <div class="metric-card dark">
                                    <div class="metric-header">
                                        <div class="metric-label">LPP</div>
                                        <div class="metric-icon" style="background: #ff6b6b;">❤️</div>
                                    </div>
                                    <div class="metric-value">
                                        2<span class="metric-unit">grade</span>
                                    </div>
                                    <div class="metric-chart">
                                        <svg width="60" height="30" viewBox="0 0 60 30">
                                            <path d="M0,15 Q15,5 30,15 T60,15" stroke="currentColor" stroke-width="1" fill="none" opacity="0.5"/>
                                        </svg>
                                    </div>
                                </div>
                                
                                <!-- Braden Score Card -->
                                <div class="metric-card">
                                    <div class="metric-header">
                                        <div class="metric-label">RISK</div>
                                        <div class="metric-icon" style="background: #ff6b6b;">❤️</div>
                                    </div>
                                    <div class="metric-value">
                                        16<span class="metric-unit">braden</span>
                                    </div>
                                    <div class="metric-chart">
                                        <svg width="60" height="30" viewBox="0 0 60 30">
                                            <path d="M0,20 L15,10 L30,15 L45,8 L60,12" stroke="#ff6b6b" stroke-width="1" fill="none"/>
                                        </svg>
                                    </div>
                                </div>
                                
                                <!-- Pulse Card -->
                                <div class="metric-card">
                                    <div class="metric-header">
                                        <div class="metric-label">Pulse</div>
                                        <div class="metric-icon" style="background: #ffa500;">⚡</div>
                                    </div>
                                    <div class="metric-value">
                                        122<span class="metric-unit">bpm</span>
                                    </div>
                                </div>
                                
                                <!-- Weight Card -->
                                <div class="metric-card">
                                    <div class="metric-header">
                                        <div class="metric-label">Weight</div>
                                        <div class="metric-icon" style="background: #28a745;">⚖️</div>
                                    </div>
                                    <div class="metric-value">
                                        80.0<span class="metric-unit">kg</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Risk Assessment Box -->
                            <div class="analysis-section" id="risk-assessment-box">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <div class="analysis-header">🧠 Risk Assessment Agent</div>
                                    <button onclick="runRiskAssessment()" style="background: #6366f1; color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.8rem; cursor: pointer;">
                                        Ejecutar Assessment
                                    </button>
                                </div>
                                <div id="risk-assessment-content" style="min-height: 150px;">
                                    <div style="text-align: center; padding: 2rem; color: #718096;">
                                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">📊 Assessment del Agente de Riesgo</div>
                                        <div style="font-size: 0.9rem; opacity: 0.8;">Batman Token: BATMAN_JESSE_WYNN_12345</div>
                                        <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">Haga clic en "Ejecutar Assessment" para análisis</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- JavaScript para integración -->
        <script>
        function runRiskAssessment() {
            // Simula el click del botón Gradio hidden
            const button = document.querySelector('button[data-testid="assess-button-hidden"]');
            if (button) {
                button.click();
            }
        }
        
        function updateRiskAssessment(htmlContent) {
            const container = document.getElementById('risk-assessment-content');
            if (container) {
                container.innerHTML = htmlContent;
            }
        }
        </script>
        """)
        
        # Risk Assessment Functionality (Hidden for integration)
        with gr.Row(visible=False):
            batman_token_input = gr.Textbox(
                value="BATMAN_JESSE_WYNN_12345"
            )
            assess_button = gr.Button(
                "Hidden Assessment Button",
                elem_id="assess-button-hidden"
            )
            risk_assessment_output = gr.HTML()
        
        def run_risk_assessment(batman_token):
            """Ejecuta el assessment del agente de riesgo."""
            assessment_data = RiskVisualization.simulate_batman_assessment(batman_token)
            return RiskVisualization.create_compact_assessment_html(assessment_data)
        
        def update_assessment_and_display(batman_token):
            """Ejecuta assessment y actualiza el display integrado."""
            assessment_data = RiskVisualization.simulate_batman_assessment(batman_token)
            compact_html = RiskVisualization.create_compact_assessment_html(assessment_data)
            
            # Simplemente retornamos el HTML - Gradio lo manejará
            return compact_html
        
        assess_button.click(
            fn=update_assessment_and_display,
            inputs=[batman_token_input],
            outputs=[risk_assessment_output]
        )
    
    return interface

def launch_smartcare_interface(
    share: bool = True,
    debug: bool = False,
    server_port: int = 7860
) -> None:
    """Launch Smart Care interface."""
    
    print("🏥 VIGIA Medical AI - Smart Care Interface")
    print("=" * 50)
    print("🎨 Exact replica of Smart Care design")
    print("📱 Clean, modern medical interface")
    print("🩺 VIGIA medical terminology")
    print()
    
    interface = create_smartcare_interface()
    
    interface.launch(
        share=share,
        debug=debug,
        server_port=server_port,
        quiet=not debug
    )

if __name__ == "__main__":
    launch_smartcare_interface()