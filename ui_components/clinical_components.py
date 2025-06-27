#!/usr/bin/env python3
"""
VIGIA Medical AI - Specialized Clinical UI Components
===================================================

Specialized clinical components for advanced medical workflows including:
- Clinical assessment forms
- Medical imaging annotations
- Risk stratification tools
- Care plan interfaces
- Quality metrics displays

Designed for healthcare professionals with clinical workflow optimization.
"""

import gradio as gr
import json
from typing import Dict, List, Any, Optional, Tuple
import datetime
import uuid

class ClinicalComponents:
    """
    Specialized clinical UI components for advanced medical workflows.
    
    Provides medical-specific interface elements optimized for:
    - Clinical decision making
    - Medical documentation
    - Risk assessment workflows
    - Care plan management
    - Quality assurance metrics
    """
    
    def __init__(self):
        """Initialize clinical components with medical styling."""
        self.clinical_colors = {
            'assessment_blue': '#1565C0',
            'risk_red': '#C62828',
            'care_green': '#2E7D32',
            'quality_purple': '#7B1FA2',
            'documentation_gray': '#455A64'
        }
    
    def create_clinical_assessment_form(self) -> gr.HTML:
        """
        Create comprehensive clinical assessment form component.
        
        Returns:
            gr.HTML: Clinical assessment form interface
        """
        assessment_html = """
        <div class="clinical-assessment-form" style="background: linear-gradient(135deg, #E3F2FD 0%, #FFFFFF 100%); border: 2px solid #1565C0; border-radius: 16px; padding: 24px; margin: 20px 0;">
            <h3 style="color: #1565C0; font-size: 20px; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;">
                📋 Comprehensive Clinical Assessment
            </h3>
            
            <!-- Braden Scale Risk Assessment -->
            <div class="assessment-section" style="margin-bottom: 24px; padding: 16px; background: rgba(255,255,255,0.7); border-radius: 12px;">
                <h4 style="color: #1565C0; font-size: 16px; font-weight: 600; margin-bottom: 16px;">🎯 Braden Scale Risk Assessment</h4>
                <div class="braden-grid" style="display: grid; grid-template-columns: 2fr 1fr; gap: 12px; align-items: center;">
                    <div>
                        <strong>1. Sensory Perception</strong><br>
                        <small>Ability to respond meaningfully to pressure-related discomfort</small>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="braden-btn" onclick="setBradenScore('sensory', 1)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">1</button>
                        <button class="braden-btn" onclick="setBradenScore('sensory', 2)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">2</button>
                        <button class="braden-btn" onclick="setBradenScore('sensory', 3)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">3</button>
                        <button class="braden-btn" onclick="setBradenScore('sensory', 4)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">4</button>
                    </div>
                    
                    <div>
                        <strong>2. Moisture</strong><br>
                        <small>Degree to which skin is exposed to moisture</small>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="braden-btn" onclick="setBradenScore('moisture', 1)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">1</button>
                        <button class="braden-btn" onclick="setBradenScore('moisture', 2)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">2</button>
                        <button class="braden-btn" onclick="setBradenScore('moisture', 3)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">3</button>
                        <button class="braden-btn" onclick="setBradenScore('moisture', 4)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">4</button>
                    </div>
                    
                    <div>
                        <strong>3. Activity</strong><br>
                        <small>Degree of physical activity</small>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="braden-btn" onclick="setBradenScore('activity', 1)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">1</button>
                        <button class="braden-btn" onclick="setBradenScore('activity', 2)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">2</button>
                        <button class="braden-btn" onclick="setBradenScore('activity', 3)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">3</button>
                        <button class="braden-btn" onclick="setBradenScore('activity', 4)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">4</button>
                    </div>
                    
                    <div>
                        <strong>4. Mobility</strong><br>
                        <small>Ability to change and control body position</small>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="braden-btn" onclick="setBradenScore('mobility', 1)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">1</button>
                        <button class="braden-btn" onclick="setBradenScore('mobility', 2)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">2</button>
                        <button class="braden-btn" onclick="setBradenScore('mobility', 3)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">3</button>
                        <button class="braden-btn" onclick="setBradenScore('mobility', 4)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">4</button>
                    </div>
                    
                    <div>
                        <strong>5. Nutrition</strong><br>
                        <small>Usual food intake pattern</small>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="braden-btn" onclick="setBradenScore('nutrition', 1)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">1</button>
                        <button class="braden-btn" onclick="setBradenScore('nutrition', 2)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">2</button>
                        <button class="braden-btn" onclick="setBradenScore('nutrition', 3)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">3</button>
                        <button class="braden-btn" onclick="setBradenScore('nutrition', 4)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">4</button>
                    </div>
                    
                    <div>
                        <strong>6. Friction & Shear</strong><br>
                        <small>Friction when moved or positioned</small>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="braden-btn" onclick="setBradenScore('friction', 1)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">1</button>
                        <button class="braden-btn" onclick="setBradenScore('friction', 2)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">2</button>
                        <button class="braden-btn" onclick="setBradenScore('friction', 3)" style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer;">3</button>
                    </div>
                </div>
                
                <div style="margin-top: 20px; padding: 16px; background: #FFF3E0; border-radius: 8px; border-left: 4px solid #FF8F00;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 16px;">Total Braden Score:</span>
                        <span id="total-braden-score" style="font-size: 24px; font-weight: 700; color: #FF8F00;">18</span>
                    </div>
                    <div style="margin-top: 8px;">
                        <span style="font-weight: 600;">Risk Level: </span>
                        <span id="risk-level" style="font-weight: 600; color: #2E7D32;">Low Risk (18-23)</span>
                    </div>
                    <div style="margin-top: 4px; font-size: 12px; color: #666;">
                        18-23: Low Risk | 15-17: Moderate Risk | 13-14: High Risk | ≤12: Very High Risk
                    </div>
                </div>
            </div>
            
            <!-- Clinical Context -->
            <div class="assessment-section" style="margin-bottom: 24px; padding: 16px; background: rgba(255,255,255,0.7); border-radius: 12px;">
                <h4 style="color: #1565C0; font-size: 16px; font-weight: 600; margin-bottom: 16px;">🩺 Clinical Context</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px;">
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Primary Diagnosis:</label>
                        <input type="text" placeholder="e.g., Acute myocardial infarction" style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Current Medications:</label>
                        <input type="text" placeholder="e.g., Anticoagulants, steroids" style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Mobility Status:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>Ambulatory</option>
                            <option>Limited mobility</option>
                            <option>Bedrest</option>
                            <option>Chair/wheelchair bound</option>
                        </select>
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Nutritional Status:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>Excellent</option>
                            <option>Adequate</option>
                            <option>Probably inadequate</option>
                            <option>Very poor</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        let bradenScores = {
            sensory: 3,
            moisture: 3,
            activity: 3,
            mobility: 3,
            nutrition: 3,
            friction: 3
        };
        
        function setBradenScore(category, score) {
            bradenScores[category] = score;
            
            // Update visual selection
            document.querySelectorAll('.braden-btn').forEach(btn => {
                btn.style.background = 'white';
                btn.style.color = 'black';
            });
            
            event.target.style.background = '#1565C0';
            event.target.style.color = 'white';
            
            // Calculate total
            let total = Object.values(bradenScores).reduce((a, b) => a + b, 0);
            document.getElementById('total-braden-score').textContent = total;
            
            // Update risk level
            let riskLevel, riskColor;
            if (total >= 18) {
                riskLevel = 'Low Risk (18-23)';
                riskColor = '#2E7D32';
            } else if (total >= 15) {
                riskLevel = 'Moderate Risk (15-17)';
                riskColor = '#FF8F00';
            } else if (total >= 13) {
                riskLevel = 'High Risk (13-14)';
                riskColor = '#F57C00';
            } else {
                riskLevel = 'Very High Risk (≤12)';
                riskColor = '#C62828';
            }
            
            let riskElement = document.getElementById('risk-level');
            riskElement.textContent = riskLevel;
            riskElement.style.color = riskColor;
        }
        </script>
        """
        return gr.HTML(assessment_html)
    
    def create_wound_assessment_tool(self) -> gr.HTML:
        """
        Create specialized wound assessment tool.
        
        Returns:
            gr.HTML: Wound assessment interface
        """
        wound_html = """
        <div class="wound-assessment-tool" style="background: linear-gradient(135deg, #FFEBEE 0%, #FFFFFF 100%); border: 2px solid #C62828; border-radius: 16px; padding: 24px; margin: 20px 0;">
            <h3 style="color: #C62828; font-size: 20px; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;">
                🩹 Advanced Wound Assessment Tool
            </h3>
            
            <!-- Wound Characteristics -->
            <div class="wound-section" style="margin-bottom: 24px; padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px;">
                <h4 style="color: #C62828; font-size: 16px; font-weight: 600; margin-bottom: 16px;">📏 Wound Characteristics</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Length (cm):</label>
                        <input type="number" placeholder="0.0" step="0.1" style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Width (cm):</label>
                        <input type="number" placeholder="0.0" step="0.1" style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Depth (cm):</label>
                        <input type="number" placeholder="0.0" step="0.1" style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Shape:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>Round/Oval</option>
                            <option>Irregular</option>
                            <option>Linear</option>
                            <option>Tunneling</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <!-- Wound Bed Assessment -->
            <div class="wound-section" style="margin-bottom: 24px; padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px;">
                <h4 style="color: #C62828; font-size: 16px; font-weight: 600; margin-bottom: 16px;">🔍 Wound Bed Assessment</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px;">
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Tissue Type:</label>
                        <div style="display: flex; flex-direction: column; gap: 8px;">
                            <label style="font-weight: normal;"><input type="checkbox"> Granulation tissue</label>
                            <label style="font-weight: normal;"><input type="checkbox"> Epithelial tissue</label>
                            <label style="font-weight: normal;"><input type="checkbox"> Slough</label>
                            <label style="font-weight: normal;"><input type="checkbox"> Necrotic tissue</label>
                        </div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Exudate Amount:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>None</option>
                            <option>Scant</option>
                            <option>Small</option>
                            <option>Moderate</option>
                            <option>Large</option>
                        </select>
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Exudate Type:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>Serous</option>
                            <option>Serosanguinous</option>
                            <option>Sanguinous</option>
                            <option>Purulent</option>
                        </select>
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Odor:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>None</option>
                            <option>Mild</option>
                            <option>Moderate</option>
                            <option>Strong</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <!-- Periwound Assessment -->
            <div class="wound-section" style="margin-bottom: 20px; padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px;">
                <h4 style="color: #C62828; font-size: 16px; font-weight: 600; margin-bottom: 16px;">🔄 Periwound Assessment</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Color:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>Pink</option>
                            <option>Red</option>
                            <option>White/Pale</option>
                            <option>Darkened</option>
                        </select>
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Condition:</label>
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            <label style="font-weight: normal;"><input type="checkbox"> Intact</label>
                            <label style="font-weight: normal;"><input type="checkbox"> Macerated</label>
                            <label style="font-weight: normal;"><input type="checkbox"> Excoriated</label>
                            <label style="font-weight: normal;"><input type="checkbox"> Indurated</label>
                        </div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Temperature:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>Normal</option>
                            <option>Warm</option>
                            <option>Hot</option>
                            <option>Cool</option>
                        </select>
                    </div>
                    <div>
                        <label style="display: block; font-weight: 600; margin-bottom: 8px;">Edema:</label>
                        <select style="width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px;">
                            <option>None</option>
                            <option>Mild</option>
                            <option>Moderate</option>
                            <option>Severe</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
        """
        return gr.HTML(wound_html)
    
    def create_care_plan_interface(self) -> gr.HTML:
        """
        Create care plan management interface.
        
        Returns:
            gr.HTML: Care plan interface
        """
        care_plan_html = """
        <div class="care-plan-interface" style="background: linear-gradient(135deg, #E8F5E8 0%, #FFFFFF 100%); border: 2px solid #2E7D32; border-radius: 16px; padding: 24px; margin: 20px 0;">
            <h3 style="color: #2E7D32; font-size: 20px; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;">
                📋 Individualized Care Plan
            </h3>
            
            <!-- Immediate Interventions -->
            <div class="care-section" style="margin-bottom: 24px; padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px;">
                <h4 style="color: #2E7D32; font-size: 16px; font-weight: 600; margin-bottom: 16px;">⚡ Immediate Interventions (0-2 hours)</h4>
                <div class="intervention-list">
                    <div class="intervention-item" style="display: flex; align-items: center; gap: 12px; padding: 8px; border-bottom: 1px solid #E0E0E0;">
                        <input type="checkbox" style="width: 18px; height: 18px;">
                        <span style="flex: 1;">Pressure relief - reposition patient off affected area</span>
                        <span style="font-size: 12px; color: #666;">Due: Now</span>
                    </div>
                    <div class="intervention-item" style="display: flex; align-items: center; gap: 12px; padding: 8px; border-bottom: 1px solid #E0E0E0;">
                        <input type="checkbox" style="width: 18px; height: 18px;">
                        <span style="flex: 1;">Skin assessment documentation</span>
                        <span style="font-size: 12px; color: #666;">Due: 30 min</span>
                    </div>
                    <div class="intervention-item" style="display: flex; align-items: center; gap: 12px; padding: 8px; border-bottom: 1px solid #E0E0E0;">
                        <input type="checkbox" style="width: 18px; height: 18px;">
                        <span style="flex: 1;">Physician notification (Grade 2+)</span>
                        <span style="font-size: 12px; color: #666;">Due: 1 hour</span>
                    </div>
                    <div class="intervention-item" style="display: flex; align-items: center; gap: 12px; padding: 8px;">
                        <input type="checkbox" style="width: 18px; height: 18px;">
                        <span style="flex: 1;">Pain assessment and management</span>
                        <span style="font-size: 12px; color: #666;">Due: 2 hours</span>
                    </div>
                </div>
            </div>
            
            <!-- Ongoing Care -->
            <div class="care-section" style="margin-bottom: 24px; padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px;">
                <h4 style="color: #2E7D32; font-size: 16px; font-weight: 600; margin-bottom: 16px;">🔄 Ongoing Care (Daily)</h4>
                <div class="intervention-list">
                    <div class="intervention-item" style="display: flex; align-items: center; gap: 12px; padding: 8px; border-bottom: 1px solid #E0E0E0;">
                        <input type="checkbox" style="width: 18px; height: 18px;">
                        <span style="flex: 1;">Repositioning schedule - every 2 hours</span>
                        <span style="font-size: 12px; color: #666;">Q2H</span>
                    </div>
                    <div class="intervention-item" style="display: flex; align-items: center; gap: 12px; padding: 8px; border-bottom: 1px solid #E0E0E0;">
                        <input type="checkbox" style="width: 18px; height: 18px;">
                        <span style="flex: 1;">Skin moisturization and barrier protection</span>
                        <span style="font-size: 12px; color: #666;">BID</span>
                    </div>
                    <div class="intervention-item" style="display: flex; align-items: center; gap: 12px; padding: 8px; border-bottom: 1px solid #E0E0E0;">
                        <input type="checkbox" style="width: 18px; height: 18px;">
                        <span style="flex: 1;">Nutritional assessment and support</span>
                        <span style="font-size: 12px; color: #666;">Daily</span>
                    </div>
                    <div class="intervention-item" style="display: flex; align-items: center; gap: 12px; padding: 8px;">
                        <input type="checkbox" style="width: 18px; height: 18px;">
                        <span style="flex: 1;">Pressure-reducing surface evaluation</span>
                        <span style="font-size: 12px; color: #666;">Daily</span>
                    </div>
                </div>
            </div>
            
            <!-- Goals and Outcomes -->
            <div class="care-section" style="margin-bottom: 20px; padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px;">
                <h4 style="color: #2E7D32; font-size: 16px; font-weight: 600; margin-bottom: 16px;">🎯 Goals and Expected Outcomes</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">
                    <div style="padding: 12px; background: #F1F8E9; border-radius: 8px; border-left: 4px solid #2E7D32;">
                        <h5 style="margin: 0 0 8px 0; color: #2E7D32;">Short-term (24-48 hours)</h5>
                        <ul style="margin: 0; padding-left: 16px; font-size: 14px;">
                            <li>No progression of existing pressure injury</li>
                            <li>No new pressure injuries develop</li>
                            <li>Patient comfort maintained</li>
                        </ul>
                    </div>
                    <div style="padding: 12px; background: #F1F8E9; border-radius: 8px; border-left: 4px solid #2E7D32;">
                        <h5 style="margin: 0 0 8px 0; color: #2E7D32;">Long-term (1-2 weeks)</h5>
                        <ul style="margin: 0; padding-left: 16px; font-size: 14px;">
                            <li>Pressure injury healing progression</li>
                            <li>Risk factors minimized</li>
                            <li>Patient/family education completed</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """
        return gr.HTML(care_plan_html)
    
    def create_quality_metrics_display(self, metrics: Optional[Dict] = None) -> gr.HTML:
        """
        Create quality metrics and outcomes display.
        
        Args:
            metrics: Quality metrics data
            
        Returns:
            gr.HTML: Quality metrics display
        """
        if not metrics:
            metrics = {
                'prevention_rate': 92.5,
                'detection_accuracy': 89.2,
                'treatment_adherence': 87.8,
                'healing_rate': 76.3,
                'patient_satisfaction': 94.1
            }
        
        quality_html = f"""
        <div class="quality-metrics-display" style="background: linear-gradient(135deg, #F3E5F5 0%, #FFFFFF 100%); border: 2px solid #7B1FA2; border-radius: 16px; padding: 24px; margin: 20px 0;">
            <h3 style="color: #7B1FA2; font-size: 20px; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;">
                📊 Quality Metrics & Outcomes
            </h3>
            
            <!-- Key Performance Indicators -->
            <div class="metrics-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
                <div class="metric-card" style="padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px; text-align: center; border-left: 4px solid #2E7D32;">
                    <h4 style="margin: 0 0 8px 0; color: #2E7D32; font-size: 14px; font-weight: 600;">Prevention Rate</h4>
                    <div style="font-size: 28px; font-weight: 700; color: #2E7D32; margin-bottom: 4px;">{metrics['prevention_rate']}%</div>
                    <div style="font-size: 12px; color: #666;">Target: ≥90%</div>
                </div>
                
                <div class="metric-card" style="padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px; text-align: center; border-left: 4px solid #1565C0;">
                    <h4 style="margin: 0 0 8px 0; color: #1565C0; font-size: 14px; font-weight: 600;">Detection Accuracy</h4>
                    <div style="font-size: 28px; font-weight: 700; color: #1565C0; margin-bottom: 4px;">{metrics['detection_accuracy']}%</div>
                    <div style="font-size: 12px; color: #666;">Target: ≥85%</div>
                </div>
                
                <div class="metric-card" style="padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px; text-align: center; border-left: 4px solid #FF8F00;">
                    <h4 style="margin: 0 0 8px 0; color: #FF8F00; font-size: 14px; font-weight: 600;">Treatment Adherence</h4>
                    <div style="font-size: 28px; font-weight: 700; color: #FF8F00; margin-bottom: 4px;">{metrics['treatment_adherence']}%</div>
                    <div style="font-size: 12px; color: #666;">Target: ≥80%</div>
                </div>
                
                <div class="metric-card" style="padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px; text-align: center; border-left: 4px solid #C62828;">
                    <h4 style="margin: 0 0 8px 0; color: #C62828; font-size: 14px; font-weight: 600;">Healing Rate</h4>
                    <div style="font-size: 28px; font-weight: 700; color: #C62828; margin-bottom: 4px;">{metrics['healing_rate']}%</div>
                    <div style="font-size: 12px; color: #666;">Target: ≥70%</div>
                </div>
                
                <div class="metric-card" style="padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px; text-align: center; border-left: 4px solid #7B1FA2;">
                    <h4 style="margin: 0 0 8px 0; color: #7B1FA2; font-size: 14px; font-weight: 600;">Patient Satisfaction</h4>
                    <div style="font-size: 28px; font-weight: 700; color: #7B1FA2; margin-bottom: 4px;">{metrics['patient_satisfaction']}%</div>
                    <div style="font-size: 12px; color: #666;">Target: ≥90%</div>
                </div>
            </div>
            
            <!-- Benchmark Comparison -->
            <div class="benchmark-section" style="margin-bottom: 20px; padding: 16px; background: rgba(255,255,255,0.8); border-radius: 12px;">
                <h4 style="color: #7B1FA2; font-size: 16px; font-weight: 600; margin-bottom: 16px;">📈 Benchmark Comparison</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px;">
                    <div style="padding: 12px; background: #E8F5E8; border-radius: 8px;">
                        <h5 style="margin: 0 0 8px 0; color: #2E7D32;">vs. National Average</h5>
                        <div style="font-size: 14px;">
                            <span style="color: #2E7D32; font-weight: 600;">+5.2%</span> above national average
                        </div>
                    </div>
                    <div style="padding: 12px; background: #E3F2FD; border-radius: 8px;">
                        <h5 style="margin: 0 0 8px 0; color: #1565C0;">vs. Peer Hospitals</h5>
                        <div style="font-size: 14px;">
                            <span style="color: #1565C0; font-weight: 600;">Top 15%</span> performance quartile
                        </div>
                    </div>
                    <div style="padding: 12px; background: #F3E5F5; border-radius: 8px;">
                        <h5 style="margin: 0 0 8px 0; color: #7B1FA2;">Trend Analysis</h5>
                        <div style="font-size: 14px;">
                            <span style="color: #2E7D32; font-weight: 600;">↗ +2.8%</span> improvement (30 days)
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        return gr.HTML(quality_html)

def create_clinical_components():
    """
    Factory function to create clinical components instance.
    
    Returns:
        ClinicalComponents: Configured clinical components
    """
    return ClinicalComponents()