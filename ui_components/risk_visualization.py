"""
VIGIA Medical AI - Risk Assessment Visualization
===============================================

Visualización interactiva del assessment de riesgo que hace el agente
basado en Batman token y los 7 factores principales de riesgo LPP.
"""

import gradio as gr
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class RiskVisualization:
    """Visualización del assessment de riesgo del agente VIGIA."""
    
    # 7 Factores principales que evalúa el agente de riesgo
    RISK_FACTORS = {
        "movilidad": {
            "name": "Movilidad",
            "description": "Capacidad de cambiar y controlar posición corporal",
            "max_score": 4,
            "color": "#ff6b6b"  # Rojo
        },
        "actividad": {
            "name": "Actividad Física", 
            "description": "Grado de actividad física del paciente",
            "max_score": 4,
            "color": "#4ecdc4"  # Verde azulado
        },
        "percepcion_sensorial": {
            "name": "Percepción Sensorial",
            "description": "Capacidad de responder a molestias relacionadas con presión",
            "max_score": 4,
            "color": "#45b7d1"  # Azul
        },
        "humedad": {
            "name": "Humedad de Piel",
            "description": "Grado de exposición de la piel a humedad",
            "max_score": 4,
            "color": "#f9ca24"  # Amarillo
        },
        "nutricion": {
            "name": "Nutrición",
            "description": "Patrón usual de ingesta de alimentos",
            "max_score": 4,
            "color": "#6c5ce7"  # Morado
        },
        "friccion_cizallamiento": {
            "name": "Fricción y Cizallamiento",
            "description": "Problemas con fricción y fuerzas de cizallamiento",
            "max_score": 3,
            "color": "#fd79a8"  # Rosa
        },
        "condiciones_medicas": {
            "name": "Condiciones Médicas",
            "description": "Factores médicos adicionales (diabetes, circulación, etc.)",
            "max_score": 4,
            "color": "#fdcb6e"  # Naranja
        }
    }
    
    @staticmethod
    def simulate_batman_assessment(batman_token: str = "BATMAN_12345") -> Dict[str, any]:
        """
        Simula el assessment que hace el agente de riesgo con Batman token.
        En producción, esto llamaría al agente real.
        """
        # Simula scores del assessment (en producción vendría del agente)
        assessment_scores = {
            "movilidad": 2,  # Muy limitada
            "actividad": 1,  # Encamado
            "percepcion_sensorial": 3,  # Ligeramente limitada
            "humedad": 2,  # Muy húmeda
            "nutricion": 3,  # Adecuada
            "friccion_cizallamiento": 2,  # Problema potencial
            "condiciones_medicas": 2  # Diabetes + edad
        }
        
        # Calcula score total Braden (6-23)
        braden_total = sum(assessment_scores.values())
        
        # Determina nivel de riesgo
        if braden_total <= 9:
            risk_level = "CRÍTICO"
            risk_color = "#e74c3c"
        elif braden_total <= 12:
            risk_level = "ALTO"
            risk_color = "#f39c12"
        elif braden_total <= 14:
            risk_level = "MODERADO"
            risk_color = "#f1c40f"
        elif braden_total <= 18:
            risk_level = "BAJO"
            risk_color = "#27ae60"
        else:
            risk_level = "MÍNIMO"
            risk_color = "#2ecc71"
        
        return {
            "batman_token": batman_token,
            "assessment_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scores": assessment_scores,
            "braden_total": braden_total,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "assessment_confidence": 0.92
        }
    
    @staticmethod
    def create_compact_assessment_html(assessment_data: Dict[str, any]) -> str:
        """Crea HTML compacto para integrar en Smart Care interface."""
        
        scores = assessment_data["scores"]
        total_score = assessment_data["braden_total"]
        risk_level = assessment_data["risk_level"]
        risk_color = assessment_data["risk_color"]
        
        # Version compacta para integrar en el box
        html = f"""
        <div style="padding: 0;">
            <!-- Header compacto -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="background: {risk_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-weight: 600; font-size: 0.8rem;">
                        {risk_level}
                    </div>
                    <div style="color: #4a5568; font-weight: 600;">Braden: {total_score}/23</div>
                </div>
                <div style="color: #718096; font-size: 0.8rem;">{assessment_data['assessment_timestamp']}</div>
            </div>
            
            <!-- Barras compactas -->
            <div style="display: grid; gap: 0.6rem;">
        """
        
        # Solo los 4 factores más críticos para el espacio compacto
        critical_factors = ["movilidad", "actividad", "percepcion_sensorial", "humedad"]
        
        for factor_key in critical_factors:
            if factor_key in RiskVisualization.RISK_FACTORS:
                factor_info = RiskVisualization.RISK_FACTORS[factor_key]
                score = scores.get(factor_key, 0)
                max_score = factor_info["max_score"]
                percentage = (score / max_score) * 100
                color = factor_info["color"]
                
                # Nivel de riesgo simplificado
                if percentage <= 25:
                    factor_risk = "CRÍTICO"
                    factor_risk_color = "#e74c3c"
                elif percentage <= 50:
                    factor_risk = "ALTO"
                    factor_risk_color = "#f39c12"
                elif percentage <= 75:
                    factor_risk = "MODERADO"
                    factor_risk_color = "#f1c40f"
                else:
                    factor_risk = "BAJO"
                    factor_risk_color = "#27ae60"
                
                html += f"""
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="min-width: 120px; font-size: 0.85rem; font-weight: 500; color: #2d3748;">
                        {factor_info['name']}
                    </div>
                    <div style="flex: 1; background: #e2e8f0; border-radius: 6px; height: 14px; position: relative; overflow: hidden;">
                        <div style="background: {color}; height: 100%; width: {percentage}%; border-radius: 6px;"></div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.4rem; min-width: 80px;">
                        <span style="background: {factor_risk_color}; color: white; padding: 0.1rem 0.4rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600;">
                            {factor_risk}
                        </span>
                        <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">
                            {score}/{max_score}
                        </span>
                    </div>
                </div>
                """
        
        html += """
            </div>
            
            <!-- Recomendaciones compactas -->
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="color: #2d3748; font-size: 0.85rem; font-weight: 500;">
                        📋 Recomendaciones:
                    </div>
                    <div style="color: #718096; font-size: 0.75rem;">
                        Confianza: {:.0%} | HIPAA ✓
                    </div>
                </div>
        """.format(assessment_data['assessment_confidence'])
        
        # Recomendaciones ultra compactas
        if risk_level == "CRÍTICO":
            html += """
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #4a5568;">
                    🚨 Intervención inmediata • 🛏️ Superficie avanzada • 👨‍⚕️ Especialista
                </div>
            """
        elif risk_level == "ALTO":
            html += """
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #4a5568;">
                    ⚠️ Reposición 4h • 🛏️ Superficie presión • 🍽️ Optimizar nutrición
                </div>
            """
        else:
            html += """
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #4a5568;">
                    ✅ Medidas preventivas • 🔍 Evaluación regular • 📚 Educación equipo
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    @staticmethod
    def create_horizontal_bars_html(assessment_data: Dict[str, any]) -> str:
        """Crea HTML para barras horizontales del assessment."""
        
        scores = assessment_data["scores"]
        total_score = assessment_data["braden_total"]
        risk_level = assessment_data["risk_level"]
        risk_color = assessment_data["risk_color"]
        
        # Header con información del assessment
        html = f"""
        <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <div>
                    <h3 style="margin: 0; color: #2d3748; font-size: 1.2rem;">🧠 Assessment del Agente de Riesgo</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #718096; font-size: 0.9rem;">
                        Batman Token: {assessment_data['batman_token']} | {assessment_data['assessment_timestamp']}
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="background: {risk_color}; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; margin-bottom: 0.5rem;">
                        {risk_level}
                    </div>
                    <div style="color: #4a5568; font-weight: 600; font-size: 1.1rem;">
                        Braden: {total_score}/23
                    </div>
                </div>
            </div>
        """
        
        # Barras horizontales para cada factor
        html += '<div style="space-y: 1rem;">'
        
        for factor_key, factor_info in RiskVisualization.RISK_FACTORS.items():
            score = scores.get(factor_key, 0)
            max_score = factor_info["max_score"]
            percentage = (score / max_score) * 100
            color = factor_info["color"]
            
            # Determina el nivel de riesgo para este factor
            if percentage <= 25:
                factor_risk = "CRÍTICO"
                factor_risk_color = "#e74c3c"
            elif percentage <= 50:
                factor_risk = "ALTO"
                factor_risk_color = "#f39c12"
            elif percentage <= 75:
                factor_risk = "MODERADO"
                factor_risk_color = "#f1c40f"
            else:
                factor_risk = "BAJO"
                factor_risk_color = "#27ae60"
            
            html += f"""
            <div style="margin-bottom: 1.2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div>
                        <span style="font-weight: 600; color: #2d3748; font-size: 0.95rem;">{factor_info['name']}</span>
                        <span style="color: #718096; font-size: 0.8rem; margin-left: 0.5rem;">({factor_info['description']})</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="background: {factor_risk_color}; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.7rem; font-weight: 600;">
                            {factor_risk}
                        </span>
                        <span style="font-weight: 600; color: #4a5568; min-width: 3rem; text-align: right;">
                            {score}/{max_score}
                        </span>
                    </div>
                </div>
                <div style="background: #e2e8f0; border-radius: 10px; height: 20px; position: relative; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, {color}, {color}dd); height: 100%; width: {percentage}%; border-radius: 10px; position: relative;">
                        <div style="position: absolute; right: 0.5rem; top: 50%; transform: translateY(-50%); color: white; font-size: 0.7rem; font-weight: 600;">
                            {percentage:.0f}%
                        </div>
                    </div>
                </div>
            </div>
            """
        
        html += '</div>'
        
        # Footer con recomendaciones
        html += f"""
            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                <h4 style="margin: 0 0 0.5rem 0; color: #2d3748; font-size: 1rem;">📋 Recomendaciones del Agente:</h4>
                <ul style="margin: 0; padding-left: 1.5rem; color: #4a5568; font-size: 0.9rem;">
        """
        
        # Recomendaciones basadas en el nivel de riesgo
        if risk_level == "CRÍTICO":
            html += """
                <li>🚨 <strong>Intervención inmediata</strong> - Reposicionamiento cada 2 horas</li>
                <li>🛏️ Superficie de redistribución de presión avanzada</li>
                <li>👨‍⚕️ Evaluación por especialista en heridas</li>
                <li>🔍 Monitoreo continuo de la piel</li>
            """
        elif risk_level == "ALTO":
            html += """
                <li>⚠️ Reposicionamiento cada 4 horas como mínimo</li>
                <li>🛏️ Considerar superficie de redistribución de presión</li>
                <li>🍽️ Optimización nutricional</li>
                <li>📊 Reevaluación diaria del riesgo</li>
            """
        elif risk_level == "MODERADO":
            html += """
                <li>🔄 Reposicionamiento cada 6 horas</li>
                <li>🧴 Cuidado de la piel y manejo de humedad</li>
                <li>📈 Monitoreo de factores de riesgo</li>
                <li>👩‍⚕️ Educación al personal de enfermería</li>
            """
        else:
            html += """
                <li>✅ Mantener medidas preventivas estándar</li>
                <li>🔍 Evaluación semanal de la piel</li>
                <li>📚 Educación sobre prevención</li>
                <li>📊 Seguimiento de factores de riesgo</li>
            """
        
        html += """
                </ul>
                <div style="margin-top: 1rem; padding: 0.75rem; background: #f7fafc; border-radius: 8px; border-left: 4px solid #4299e1;">
                    <p style="margin: 0; color: #2d3748; font-size: 0.85rem;">
                        <strong>🤖 Confianza del Assessment:</strong> {:.1%} | 
                        <strong>🔄 Próxima evaluación:</strong> En 24 horas |
                        <strong>🛡️ HIPAA:</strong> Datos tokenizados
                    </p>
                </div>
            </div>
        </div>
        """.format(assessment_data['assessment_confidence'])
        
        return html

def create_risk_assessment_interface() -> gr.Blocks:
    """Crea interfaz para visualizar el assessment del agente de riesgo."""
    
    with gr.Blocks(title="VIGIA - Risk Assessment Visualization") as interface:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #2d3748; margin-bottom: 0.5rem;">🧠 VIGIA Risk Assessment Agent</h1>
            <p style="color: #718096; font-size: 1.1rem;">Visualización del assessment de riesgo basado en Batman token</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                batman_token_input = gr.Textbox(
                    label="🦇 Batman Token",
                    value="BATMAN_12345",
                    placeholder="Ingrese Batman token del paciente..."
                )
                
                assess_button = gr.Button(
                    "🔬 Ejecutar Assessment",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                risk_visualization = gr.HTML(
                    """
                    <div style="text-align: center; padding: 3rem; color: #718096;">
                        <h3>Esperando assessment...</h3>
                        <p>Haga clic en "Ejecutar Assessment" para ver el análisis del agente de riesgo</p>
                    </div>
                    """
                )
        
        def run_assessment(batman_token):
            """Ejecuta el assessment y retorna la visualización."""
            assessment_data = RiskVisualization.simulate_batman_assessment(batman_token)
            return RiskVisualization.create_horizontal_bars_html(assessment_data)
        
        assess_button.click(
            fn=run_assessment,
            inputs=[batman_token_input],
            outputs=[risk_visualization]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_risk_assessment_interface()
    interface.launch(share=True, server_port=7862)