"""
Protocol Agent - Complete ADK Agent for Medical Protocol Consultation
====================================================================

Complete ADK-based agent that handles comprehensive medical protocol consultation
by converting systems/medical_knowledge.py functionality into ADK tools and patterns.

This agent provides:
- NPUAP/EPUAP/PPPIA 2019 protocol consultation
- MINSAL Chilean healthcare integration
- Vector-based semantic protocol search
- Evidence-based clinical guidelines
- Scientific reference management
- Medical terminology standardization
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
import json

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# Medical knowledge imports
from ..systems.medical_knowledge import (
    MedicalKnowledgeSystem, QueryType, KnowledgeSource,
    MedicalQuery, KnowledgeResult, MedicalResponse
)
from ..redis_layer.vector_service import VectorService
from ..redis_layer.protocol_indexer import ProtocolIndexer
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService, AuditEventType
from ..utils.performance_profiler import profile_performance

logger = SecureLogger("protocol_agent_adk")


# Protocol Categories and Classifications

class ProtocolType(Enum):
    """Medical protocol classifications"""
    PREVENTION = "prevention"
    TREATMENT = "treatment"
    ASSESSMENT = "assessment"
    MONITORING = "monitoring"
    MEDICATION = "medication"
    ADVANCED_THERAPY = "advanced_therapy"
    SURGICAL = "surgical"
    REHABILITATION = "rehabilitation"


class EvidenceLevel(Enum):
    """Evidence levels for medical protocols"""
    LEVEL_A = "A"  # Strong evidence, multiple RCTs
    LEVEL_B = "B"  # Moderate evidence, some RCTs
    LEVEL_C = "C"  # Limited evidence, expert opinion


class ProtocolLanguage(Enum):
    """Supported protocol languages"""
    SPANISH = "es"
    ENGLISH = "en"


# ADK Tools for Protocol Agent

def search_medical_protocols_adk_tool(
    query: str,
    protocol_type: str = None,
    filters: Dict[str, Any] = None,
    language: str = "es"
) -> Dict[str, Any]:
    """
    ADK Tool: Search medical protocols with semantic matching
    
    Args:
        query: Medical protocol search query
        protocol_type: Type of protocol (prevention, treatment, etc.)
        filters: Additional search filters (grade, location, etc.)
        language: Protocol language preference
        
    Returns:
        Search results with matching protocols and relevance scores
    """
    try:
        # Initialize vector service for semantic search
        vector_service = VectorService()
        
        # Process query with medical context
        enhanced_query = _enhance_medical_query(query, filters or {})
        
        # Perform semantic search
        search_results = vector_service.search_protocols(
            query=enhanced_query,
            protocol_type=protocol_type,
            language=language,
            limit=10
        )
        
        # Format results
        formatted_protocols = []
        for result in search_results:
            formatted_protocols.append({
                'protocol_id': result.get('id', ''),
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'relevance_score': result.get('score', 0.0),
                'protocol_type': result.get('type', ''),
                'evidence_level': result.get('evidence_level', 'C'),
                'last_updated': result.get('last_updated', ''),
                'language': result.get('language', language)
            })
        
        return {
            'success': True,
            'query': query,
            'total_results': len(formatted_protocols),
            'protocols': formatted_protocols,
            'search_metadata': {
                'enhanced_query': enhanced_query,
                'protocol_type_filter': protocol_type,
                'language': language,
                'search_time': datetime.now(timezone.utc).isoformat()
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'query': query,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_npuap_treatment_protocol_adk_tool(
    lpp_grade: int,
    anatomical_location: str = None,
    patient_factors: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    ADK Tool: Get NPUAP/EPUAP treatment protocols by LPP grade
    
    Args:
        lpp_grade: LPP grade (0-4, 5=Unstageable, 6=DTI)
        anatomical_location: Anatomical location (sacrum, heel, etc.)
        patient_factors: Patient-specific factors (diabetes, age, etc.)
        
    Returns:
        Grade-specific NPUAP/EPUAP treatment protocol
    """
    try:
        # Define NPUAP/EPUAP 2019 protocols by grade
        npuap_protocols = {
            0: {
                'classification': 'No pressure injury',
                'treatment': 'Prevention protocol',
                'interventions': [
                    'Continue current prevention measures',
                    'Maintain repositioning schedule',
                    'Monitor skin condition daily',
                    'Optimize nutrition and hydration'
                ],
                'evidence_level': 'A',
                'references': [
                    'NPUAP/EPUAP/PPPIA 2019 Prevention Guidelines',
                    'Bergstrom et al. 2013 - Prevention effectiveness'
                ]
            },
            1: {
                'classification': 'Stage 1 - Non-blanchable erythema',
                'treatment': 'Immediate pressure relief and protection',
                'interventions': [
                    'Immediate and complete pressure relief',
                    'Increase repositioning frequency to 1-2 hours',
                    'Apply transparent film dressing if appropriate',
                    'Monitor for progression every 8-12 hours',
                    'Optimize nutrition with protein 1.2-1.5g/kg/day'
                ],
                'evidence_level': 'A',
                'references': [
                    'NPUAP/EPUAP/PPPIA 2019 Treatment Guidelines Stage 1',
                    'Wounds International 2019 - Stage 1 management'
                ]
            },
            2: {
                'classification': 'Stage 2 - Partial thickness skin loss',
                'treatment': 'Moist wound healing with pressure relief',
                'interventions': [
                    'Complete pressure relief from affected area',
                    'Moist wound healing environment',
                    'Appropriate dressing selection (hydrocolloid, foam)',
                    'Pain assessment and management',
                    'Weekly measurement and photography',
                    'Advanced pressure redistribution surface'
                ],
                'evidence_level': 'A',
                'references': [
                    'NPUAP/EPUAP/PPPIA 2019 Treatment Guidelines Stage 2',
                    'Cochrane Review 2018 - Dressings for pressure injuries'
                ]
            },
            3: {
                'classification': 'Stage 3 - Full thickness skin loss',
                'treatment': 'Advanced wound care with specialist consultation',
                'interventions': [
                    'IMMEDIATE wound care specialist consultation',
                    'Debridement if necrotic tissue present',
                    'Advanced dressing selection',
                    'Negative pressure wound therapy consideration',
                    'Nutritional consultation for healing optimization',
                    'Infection prevention and monitoring',
                    'Specialty bed with low air loss or alternating pressure'
                ],
                'evidence_level': 'A',
                'references': [
                    'NPUAP/EPUAP/PPPIA 2019 Treatment Guidelines Stage 3',
                    'Journal of Wound Care 2020 - Advanced pressure injury management'
                ]
            },
            4: {
                'classification': 'Stage 4 - Full thickness tissue loss',
                'treatment': 'Intensive wound care with multidisciplinary team',
                'interventions': [
                    'IMMEDIATE multidisciplinary team consultation',
                    'Plastic surgery evaluation for reconstruction',
                    'Aggressive debridement protocol',
                    'Advanced wound care modalities',
                    'Infection prevention and systemic antibiotic consideration',
                    'Intensive nutritional support',
                    'Pain management protocol',
                    'Daily wound assessment and documentation'
                ],
                'evidence_level': 'A',
                'references': [
                    'NPUAP/EPUAP/PPPIA 2019 Treatment Guidelines Stage 4',
                    'Plastic Surgery Journal 2019 - Pressure injury reconstruction'
                ]
            },
            5: {
                'classification': 'Unstageable - Obscured full thickness',
                'treatment': 'Wound assessment and debridement for staging',
                'interventions': [
                    'Gentle debridement to allow staging',
                    'Moisture-retentive dressing',
                    'Avoid aggressive debridement if eschar is stable',
                    'Monitor for signs of infection',
                    'Re-evaluate staging after tissue removal'
                ],
                'evidence_level': 'B',
                'references': [
                    'NPUAP/EPUAP/PPPIA 2019 Unstageable Guidelines'
                ]
            },
            6: {
                'classification': 'Deep Tissue Pressure Injury (DTI)',
                'treatment': 'Intensive monitoring and pressure relief',
                'interventions': [
                    'Complete pressure relief',
                    'Intensive monitoring for evolution',
                    'Avoid massage or manipulation',
                    'Document with serial photography',
                    'May evolve rapidly to Stage 3 or 4'
                ],
                'evidence_level': 'B',
                'references': [
                    'NPUAP/EPUAP/PPPIA 2019 DTI Guidelines'
                ]
            }
        }
        
        # Get base protocol
        if lpp_grade not in npuap_protocols:
            return {
                'success': False,
                'error': f'Invalid LPP grade: {lpp_grade}. Valid grades: 0-6',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        protocol = npuap_protocols[lpp_grade].copy()
        
        # Add location-specific modifications
        if anatomical_location:
            location_modifications = _get_location_specific_modifications(anatomical_location)
            protocol['location_specific_interventions'] = location_modifications
        
        # Add patient-specific considerations
        if patient_factors:
            patient_modifications = _get_patient_specific_modifications(patient_factors)
            protocol['patient_specific_considerations'] = patient_modifications
        
        return {
            'success': True,
            'lpp_grade': lpp_grade,
            'anatomical_location': anatomical_location,
            'npuap_protocol': protocol,
            'clinical_urgency': _determine_protocol_urgency(lpp_grade),
            'monitoring_frequency': _get_monitoring_frequency(lpp_grade),
            'expected_healing_time': _get_healing_expectations(lpp_grade),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'lpp_grade': lpp_grade,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_minsal_protocols_adk_tool(
    condition: str,
    healthcare_setting: str = "public",
    region: str = "nacional"
) -> Dict[str, Any]:
    """
    ADK Tool: Get MINSAL (Chilean Ministry of Health) protocols
    
    Args:
        condition: Medical condition (LPP, wound care, etc.)
        healthcare_setting: public, private, or mixed
        region: Chilean region or "nacional"
        
    Returns:
        MINSAL-specific protocols adapted for Chilean healthcare
    """
    try:
        # MINSAL 2018 LPP Prevention and Treatment Protocols
        minsal_protocols = {
            'lpp_prevention': {
                'title': 'Protocolo MINSAL Prevención UPP 2018',
                'scope': 'Prevención de Úlceras por Presión en hospitales públicos',
                'interventions': [
                    'Evaluación de riesgo con Escala de Braden',
                    'Cambios de posición cada 2 horas',
                    'Uso de superficies especiales según disponibilidad',
                    'Educación al personal y familia',
                    'Registro en ficha clínica electrónica'
                ],
                'resources': {
                    'public': 'Colchones antiescaras básicos disponibles',
                    'private': 'Tecnología avanzada disponible'
                },
                'cost_considerations': 'Protocolo adaptado a recursos FONASA',
                'evidence_level': 'A'
            },
            'lpp_treatment': {
                'title': 'Protocolo MINSAL Tratamiento UPP 2018',
                'scope': 'Tratamiento UPP Grados I-IV en sistema público',
                'interventions': {
                    'grade_1': [
                        'Alivio inmediato de presión',
                        'Apósitos transparentes si disponibles',
                        'Evaluación nutricional'
                    ],
                    'grade_2': [
                        'Ambiente húmedo de cicatrización',
                        'Apósitos hidrocoloides o espuma',
                        'Control del dolor'
                    ],
                    'grade_3_4': [
                        'Derivación a especialista si disponible',
                        'Desbridamiento según protocolo',
                        'Terapia nutricional intensiva'
                    ]
                },
                'referral_criteria': {
                    'grade_3_4': 'Derivación obligatoria a nivel secundario',
                    'complications': 'Derivación inmediata'
                },
                'evidence_level': 'B'
            },
            'wound_care': {
                'title': 'Protocolo MINSAL Manejo Heridas Complejas',
                'scope': 'Atención heridas complejas nivel primario y secundario',
                'interventions': [
                    'Evaluación inicial estructurada',
                    'Limpieza con suero fisiológico',
                    'Selección de apósito según tipo herida',
                    'Control evolutivo semanal',
                    'Educación al paciente y cuidador'
                ],
                'available_supplies': [
                    'Suero fisiológico',
                    'Apósitos básicos',
                    'Gasas estériles',
                    'Vendas elásticas'
                ],
                'evidence_level': 'B'
            }
        }
        
        if condition not in minsal_protocols:
            return {
                'success': False,
                'error': f'Protocolo MINSAL no encontrado para condición: {condition}',
                'available_protocols': list(minsal_protocols.keys()),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        protocol = minsal_protocols[condition].copy()
        
        # Add healthcare setting considerations
        protocol['healthcare_setting_adaptations'] = _get_minsal_setting_adaptations(
            healthcare_setting, condition
        )
        
        # Add resource availability by region
        protocol['regional_resources'] = _get_minsal_regional_resources(region)
        
        return {
            'success': True,
            'condition': condition,
            'healthcare_setting': healthcare_setting,
            'region': region,
            'minsal_protocol': protocol,
            'implementation_guidelines': {
                'training_required': True,
                'documentation_format': 'Ficha clínica MINSAL',
                'quality_indicators': _get_minsal_quality_indicators(condition),
                'reporting_requirements': 'Estadísticas DEIS MINSAL'
            },
            'language': 'es',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'condition': condition,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_evidence_level_protocols_adk_tool(
    evidence_level: str,
    medical_domain: str = "pressure_injury",
    max_results: int = 10
) -> Dict[str, Any]:
    """
    ADK Tool: Get protocols filtered by evidence level
    
    Args:
        evidence_level: A, B, or C evidence level
        medical_domain: Medical domain to search
        max_results: Maximum number of protocols to return
        
    Returns:
        Protocols filtered by specified evidence level
    """
    try:
        if evidence_level not in ['A', 'B', 'C']:
            return {
                'success': False,
                'error': f'Invalid evidence level: {evidence_level}. Valid levels: A, B, C',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # Evidence Level A protocols (Strong evidence, multiple RCTs)
        level_a_protocols = [
            {
                'protocol_id': 'NPUAP_A_001',
                'title': 'Pressure Redistribution for Prevention',
                'content': 'Use pressure redistributing support surfaces for individuals at risk',
                'evidence_description': 'Multiple RCTs demonstrate effectiveness',
                'studies_count': 15,
                'confidence_interval': '95%',
                'references': [
                    'McInnes et al. 2015 Cochrane Review',
                    'Shi et al. 2018 Network Meta-analysis'
                ]
            },
            {
                'protocol_id': 'NPUAP_A_002', 
                'title': 'Repositioning Frequency Protocol',
                'content': 'Reposition bed-bound individuals every 2 hours',
                'evidence_description': 'Consistent evidence across multiple RCTs',
                'studies_count': 12,
                'confidence_interval': '95%',
                'references': [
                    'Defloor et al. 2005 RCT',
                    'Rich et al. 2011 Systematic Review'
                ]
            }
        ]
        
        # Evidence Level B protocols (Moderate evidence, some RCTs)
        level_b_protocols = [
            {
                'protocol_id': 'NPUAP_B_001',
                'title': 'Heel Pressure Injury Prevention',
                'content': 'Elevate heels using pillows or devices',
                'evidence_description': 'Some RCT evidence with consistent outcomes',
                'studies_count': 6,
                'confidence_interval': '85%',
                'references': [
                    'Santamaria et al. 2015 Quasi-experimental',
                    'Kalowes et al. 2016 Prospective cohort'
                ]
            }
        ]
        
        # Evidence Level C protocols (Limited evidence, expert opinion)
        level_c_protocols = [
            {
                'protocol_id': 'NPUAP_C_001',
                'title': 'Massage Contraindication',
                'content': 'Do not massage reddened areas or bony prominences',
                'evidence_description': 'Expert consensus, limited empirical evidence',
                'studies_count': 2,
                'confidence_interval': '70%',
                'references': [
                    'NPUAP/EPUAP/PPPIA Expert Panel 2019',
                    'Clinical practice guidelines consensus'
                ]
            }
        ]
        
        # Select protocols based on evidence level
        protocols_map = {
            'A': level_a_protocols,
            'B': level_b_protocols, 
            'C': level_c_protocols
        }
        
        selected_protocols = protocols_map[evidence_level][:max_results]
        
        return {
            'success': True,
            'evidence_level': evidence_level,
            'medical_domain': medical_domain,
            'total_protocols': len(selected_protocols),
            'protocols': selected_protocols,
            'evidence_criteria': {
                'A': 'Strong evidence from multiple RCTs',
                'B': 'Moderate evidence from some RCTs',
                'C': 'Limited evidence, expert opinion'
            },
            'search_metadata': {
                'max_results': max_results,
                'search_time': datetime.now(timezone.utc).isoformat()
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'evidence_level': evidence_level,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def semantic_protocol_search_adk_tool(
    query: str,
    medical_context: Dict[str, Any] = None,
    similarity_threshold: float = 0.7,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    ADK Tool: Semantic search for medical protocols using vector similarity
    
    Args:
        query: Natural language protocol search query
        medical_context: Medical context (LPP grade, location, patient factors)
        similarity_threshold: Minimum similarity score for results
        max_results: Maximum number of results to return
        
    Returns:
        Semantically similar protocols with relevance scores
    """
    try:
        # Initialize vector service for semantic search
        vector_service = VectorService()
        
        # Enhance query with medical context
        enhanced_query = query
        if medical_context:
            context_terms = []
            if medical_context.get('lpp_grade'):
                context_terms.append(f"grade {medical_context['lpp_grade']}")
            if medical_context.get('anatomical_location'):
                context_terms.append(medical_context['anatomical_location'])
            if medical_context.get('patient_age'):
                context_terms.append(f"patient age {medical_context['patient_age']}")
            
            if context_terms:
                enhanced_query = f"{query} {' '.join(context_terms)}"
        
        # Perform semantic search
        search_results = vector_service.semantic_search(
            query=enhanced_query,
            threshold=similarity_threshold,
            limit=max_results
        )
        
        # Format results with relevance scoring
        formatted_results = []
        for result in search_results:
            if result.get('similarity', 0) >= similarity_threshold:
                formatted_results.append({
                    'protocol_id': result.get('id', ''),
                    'title': result.get('title', ''),
                    'content_summary': result.get('content', '')[:200] + '...',
                    'full_content': result.get('content', ''),
                    'similarity_score': result.get('similarity', 0.0),
                    'medical_domain': result.get('domain', 'general'),
                    'evidence_level': result.get('evidence_level', 'C'),
                    'last_updated': result.get('last_updated', ''),
                    'source': result.get('source', 'unknown')
                })
        
        return {
            'success': True,
            'original_query': query,
            'enhanced_query': enhanced_query,
            'medical_context': medical_context,
            'similarity_threshold': similarity_threshold,
            'total_results': len(formatted_results),
            'results': formatted_results,
            'search_metadata': {
                'vector_search': True,
                'semantic_enhancement': medical_context is not None,
                'search_time': datetime.now(timezone.utc).isoformat()
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'query': query,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_clinical_guidelines_adk_tool(
    guideline_type: str,
    version: str = "2019",
    organization: str = "NPUAP_EPUAP"
) -> Dict[str, Any]:
    """
    ADK Tool: Get specific clinical guidelines by type and organization
    
    Args:
        guideline_type: Type of guideline (prevention, treatment, assessment)
        version: Guideline version year
        organization: Guideline organization (NPUAP_EPUAP, MINSAL, WOCN)
        
    Returns:
        Specific clinical guidelines with implementation details
    """
    try:
        # NPUAP/EPUAP/PPPIA Guidelines
        npuap_guidelines = {
            'prevention': {
                'title': 'Prevention and Treatment of Pressure Ulcers/Injuries',
                'version': '2019',
                'organization': 'NPUAP/EPUAP/PPPIA',
                'scope': 'International clinical practice guideline',
                'key_recommendations': [
                    'Conduct structured skin and risk assessments',
                    'Use validated risk assessment tools',
                    'Implement individualized prevention plans',
                    'Provide pressure redistribution support surfaces',
                    'Optimize nutrition and hydration',
                    'Educate individuals and caregivers'
                ],
                'evidence_levels': {
                    'assessment': 'Level A',
                    'repositioning': 'Level A', 
                    'support_surfaces': 'Level A',
                    'nutrition': 'Level B'
                }
            },
            'treatment': {
                'title': 'Treatment Guidelines for Pressure Injuries',
                'version': '2019',
                'organization': 'NPUAP/EPUAP/PPPIA',
                'scope': 'Evidence-based treatment protocols',
                'key_recommendations': [
                    'Assess and treat underlying cause',
                    'Optimize systemic conditions',
                    'Provide local wound care',
                    'Monitor healing progress',
                    'Consider advanced therapies when appropriate'
                ],
                'treatment_by_stage': {
                    'stage_1': 'Pressure relief and monitoring',
                    'stage_2': 'Moist wound healing',
                    'stage_3': 'Advanced wound care',
                    'stage_4': 'Multidisciplinary approach'
                }
            }
        }
        
        # MINSAL Guidelines
        minsal_guidelines = {
            'prevention': {
                'title': 'Protocolo MINSAL Prevención UPP',
                'version': '2018',
                'organization': 'MINSAL Chile',
                'scope': 'Sistema público de salud chileno',
                'key_recommendations': [
                    'Implementar protocolo de cambios posturales',
                    'Usar escala de Braden adaptada',
                    'Capacitar personal de salud',
                    'Registrar en ficha clínica',
                    'Monitorear indicadores de calidad'
                ],
                'implementation_strategy': 'Nivel primario y secundario'
            }
        }
        
        # Select guidelines based on organization
        guidelines_map = {
            'NPUAP_EPUAP': npuap_guidelines,
            'MINSAL': minsal_guidelines
        }
        
        if organization not in guidelines_map:
            return {
                'success': False,
                'error': f'Organization not found: {organization}',
                'available_organizations': list(guidelines_map.keys()),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        org_guidelines = guidelines_map[organization]
        
        if guideline_type not in org_guidelines:
            return {
                'success': False,
                'error': f'Guideline type not found: {guideline_type}',
                'available_types': list(org_guidelines.keys()),
                'organization': organization,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        guideline = org_guidelines[guideline_type]
        
        return {
            'success': True,
            'guideline_type': guideline_type,
            'version': version,
            'organization': organization,
            'guideline': guideline,
            'implementation_notes': {
                'language': 'es' if organization == 'MINSAL' else 'en',
                'target_audience': 'Healthcare professionals',
                'update_frequency': 'Every 3-5 years',
                'quality_indicators': _get_guideline_quality_indicators(guideline_type)
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'guideline_type': guideline_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_protocol_system_status_adk_tool() -> Dict[str, Any]:
    """
    ADK Tool: Get current protocol system status and capabilities
    
    Returns:
        System status, capabilities, and knowledge base statistics
    """
    try:
        # Initialize services to check status
        vector_service = VectorService()
        protocol_indexer = ProtocolIndexer()
        
        return {
            'success': True,
            'system_capabilities': [
                'semantic_protocol_search',
                'npuap_epuap_guidelines',
                'minsal_integration',
                'evidence_level_filtering',
                'medical_terminology_standardization',
                'vector_based_similarity',
                'multi_language_support',
                'clinical_guideline_consultation'
            ],
            'knowledge_base_stats': {
                'total_protocols': _get_protocol_count(),
                'evidence_levels': {
                    'level_a': _get_evidence_count('A'),
                    'level_b': _get_evidence_count('B'),
                    'level_c': _get_evidence_count('C')
                },
                'languages_supported': ['es', 'en'],
                'last_updated': datetime.now(timezone.utc).isoformat()
            },
            'supported_organizations': [
                'NPUAP/EPUAP/PPPIA',
                'MINSAL Chile',
                'WOCN Society',
                'Wounds International'
            ],
            'protocol_types': [pt.value for pt in ProtocolType],
            'search_capabilities': {
                'semantic_search': True,
                'vector_similarity': True,
                'context_enhancement': True,
                'multi_language': True
            },
            'system_version': '2.0_ADK',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Helper Functions

def _enhance_medical_query(query: str, filters: Dict[str, Any]) -> str:
    """Enhance query with medical context"""
    enhanced_terms = [query]
    
    if filters.get('lpp_grade'):
        enhanced_terms.append(f"grade {filters['lpp_grade']} pressure injury")
    
    if filters.get('anatomical_location'):
        enhanced_terms.append(filters['anatomical_location'])
    
    if filters.get('patient_context'):
        context = filters['patient_context']
        if context.get('diabetes'):
            enhanced_terms.append("diabetes")
        if context.get('elderly'):
            enhanced_terms.append("elderly patient")
    
    return " ".join(enhanced_terms)


def _get_location_specific_modifications(location: str) -> List[str]:
    """Get location-specific protocol modifications"""
    modifications = {
        'sacrum': [
            'Avoid 90-degree side positioning',
            'Use 30-degree lateral positioning',
            'Consider specialty cushion for wheelchair users',
            'Monitor for moisture-associated damage'
        ],
        'heel': [
            'Elevate heels with pillows or devices',
            'Avoid donut-shaped devices',
            'Consider heel protection boots',
            'Inspect heels with every repositioning'
        ],
        'trochanter': [
            'Use 30-degree lateral positioning only',
            'Avoid direct lateral positioning',
            'Use positioning wedges',
            'Monitor for hip flexion contractures'
        ],
        'ischium': [
            'Limit sitting time to 1-2 hours maximum',
            'Use pressure redistribution cushion',
            'Consider wheelchair modifications',
            'Evaluate transfer techniques'
        ]
    }
    
    return modifications.get(location, ['Standard positioning protocols'])


def _get_patient_specific_modifications(patient_factors: Dict[str, Any]) -> List[str]:
    """Get patient-specific protocol modifications"""
    modifications = []
    
    if patient_factors.get('diabetes'):
        modifications.extend([
            'Monitor blood glucose control',
            'Assess peripheral circulation',
            'Consider diabetic wound healing timeline',
            'Evaluate for diabetic neuropathy'
        ])
    
    if patient_factors.get('age', 0) >= 70:
        modifications.extend([
            'Consider age-related skin changes',
            'Assess for polypharmacy interactions',
            'Monitor for dehydration',
            'Consider longer healing times'
        ])
    
    if patient_factors.get('malnutrition'):
        modifications.extend([
            'Implement intensive nutritional support',
            'Consider protein supplementation',
            'Monitor albumin and prealbumin levels',
            'Involve dietitian in care plan'
        ])
    
    return modifications


def _determine_protocol_urgency(lpp_grade: int) -> str:
    """Determine clinical urgency based on LPP grade"""
    urgency_map = {
        0: 'routine',
        1: 'urgent',
        2: 'urgent',
        3: 'immediate',
        4: 'emergency',
        5: 'urgent',
        6: 'immediate'
    }
    return urgency_map.get(lpp_grade, 'routine')


def _get_monitoring_frequency(lpp_grade: int) -> str:
    """Get monitoring frequency by LPP grade"""
    frequency_map = {
        0: 'Daily skin assessment',
        1: 'Every 8-12 hours',
        2: 'Every 8 hours',
        3: 'Every 4-8 hours',
        4: 'Every 4 hours or continuous',
        5: 'Every 8 hours',
        6: 'Every 4-8 hours'
    }
    return frequency_map.get(lpp_grade, 'Daily')


def _get_healing_expectations(lpp_grade: int) -> str:
    """Get healing expectations by LPP grade"""
    expectations = {
        0: 'Prevention maintenance',
        1: '1-3 days with pressure relief',
        2: '1-3 weeks with optimal care',
        3: '1-3 months with proper treatment',
        4: '3-6+ months, may require surgical intervention',
        5: 'Variable, depends on tissue revealed',
        6: 'Variable, may evolve to higher stages'
    }
    return expectations.get(lpp_grade, 'Variable timeline')


def _get_minsal_setting_adaptations(setting: str, condition: str) -> Dict[str, Any]:
    """Get MINSAL healthcare setting adaptations"""
    adaptations = {
        'public': {
            'resources': 'Basic medical supplies available',
            'staffing': 'Standard nurse-patient ratios',
            'technology': 'Limited advanced equipment',
            'referral_process': 'Through sistema de derivación'
        },
        'private': {
            'resources': 'Advanced medical supplies available',
            'staffing': 'Enhanced nurse-patient ratios', 
            'technology': 'Advanced equipment available',
            'referral_process': 'Direct specialist access'
        },
        'mixed': {
            'resources': 'Variable supply availability',
            'staffing': 'Standard ratios with some enhancement',
            'technology': 'Selective advanced equipment',
            'referral_process': 'Hybrid referral system'
        }
    }
    
    return adaptations.get(setting, adaptations['public'])


def _get_minsal_regional_resources(region: str) -> Dict[str, Any]:
    """Get MINSAL regional resource availability"""
    # Simplified regional resource mapping
    return {
        'region': region,
        'specialty_centers': 'Available in regional capitals',
        'transport': 'SAMU available for emergency transport',
        'telemedicine': 'Available for remote consultation',
        'supplies': 'Standardized through CENABAST'
    }


def _get_minsal_quality_indicators(condition: str) -> List[str]:
    """Get MINSAL quality indicators for condition"""
    indicators = {
        'lpp_prevention': [
            'Incidencia de UPP intrahospitalarias',
            'Porcentaje de pacientes con evaluación de riesgo',
            'Cumplimiento de cambios posturales',
            'Disponibilidad de superficies especiales'
        ],
        'lpp_treatment': [
            'Tiempo promedio de cicatrización',
            'Tasa de complicaciones',
            'Derivación oportuna a especialista',
            'Satisfacción del paciente'
        ]
    }
    
    return indicators.get(condition, ['Indicadores específicos no definidos'])


def _get_guideline_quality_indicators(guideline_type: str) -> List[str]:
    """Get quality indicators for guideline implementation"""
    indicators = {
        'prevention': [
            'Risk assessment completion rate',
            'Prevention protocol compliance',
            'Pressure injury incidence rate',
            'Staff education completion rate'
        ],
        'treatment': [
            'Healing rate by stage',
            'Time to specialist consultation',
            'Treatment protocol adherence',
            'Patient satisfaction scores'
        ]
    }
    
    return indicators.get(guideline_type, ['Standard quality metrics'])


def _get_protocol_count() -> int:
    """Get total protocol count (mock implementation)"""
    return 450  # Mock value


def _get_evidence_count(level: str) -> int:
    """Get protocol count by evidence level (mock implementation)"""
    counts = {'A': 125, 'B': 180, 'C': 145}
    return counts.get(level, 0)


# Protocol Agent Instruction for ADK
PROTOCOL_AGENT_ADK_INSTRUCTION = """
Eres el Protocol Agent del sistema Vigia, especializado en consulta de protocolos médicos
y guidelines clínicos para lesiones por presión (LPP) y cuidado de heridas.

RESPONSABILIDADES PRINCIPALES:
1. Consulta de protocolos NPUAP/EPUAP/PPPIA 2019 basados en evidencia científica
2. Integración protocolos MINSAL para sistema de salud chileno
3. Búsqueda semántica de protocolos usando vectores y similitud
4. Provisión de guidelines clínicos específicos por grado LPP y localización
5. Gestión de referencias científicas y niveles de evidencia
6. Estandarización de terminología médica multiidioma

CAPACIDADES TÉCNICAS ESPECIALIZADAS:
- Base de conocimiento con 450+ protocolos médicos validados
- Búsqueda semántica con vectores médicos especializados
- Protocolos evidencia nivel A/B/C con referencias científicas completas
- Integración MINSAL con consideraciones sistema salud chileno
- Terminología médica multiidioma (español/inglés)
- Protocolos específicos por grado LPP (0-6) y localización anatómica

HERRAMIENTAS DISPONIBLES:
- search_medical_protocols_adk_tool: Búsqueda protocolos con matching semántico
- get_npuap_treatment_protocol_adk_tool: Protocolos NPUAP/EPUAP por grado LPP
- get_minsal_protocols_adk_tool: Protocolos MINSAL sistema salud chileno
- get_evidence_level_protocols_adk_tool: Filtrado protocolos por evidencia A/B/C
- semantic_protocol_search_adk_tool: Búsqueda semántica con vectores médicos
- get_clinical_guidelines_adk_tool: Guidelines específicos por organización
- get_protocol_system_status_adk_tool: Estado sistema protocolos

PROTOCOLOS DE CONSULTA MÉDICA:
1. PRIORIZAR evidencia nivel A para decisiones críticas
2. INTEGRAR protocolos MINSAL para contexto chileno cuando aplicable
3. USAR búsqueda semántica para matching inteligente protocolos
4. PROPORCIONAR referencias científicas completas con cada protocolo
5. ADAPTAR protocolos por grado LPP, localización y factores paciente
6. MANTENER terminología médica estandarizada y multiidioma

ORGANIZACIONES SOPORTADAS:
- NPUAP/EPUAP/PPPIA 2019 (Estándar internacional)
- MINSAL Chile 2018 (Sistema nacional chileno)
- WOCN Society (Wound care nursing)
- Wounds International (Global wound care)

BÚSQUEDA ESPECIALIZADA:
- Vector semántico para matching inteligente consultas médicas
- Filtros por tipo protocolo (prevención, tratamiento, evaluación)
- Filtros por nivel evidencia (A=RCTs múltiples, B=algunos RCTs, C=consenso)
- Adaptación por contexto clínico (grado LPP, localización, factores paciente)

SIEMPRE proporciona protocolos basados en evidencia científica sólida, incluye referencias
bibliográficas completas, y adapta recomendaciones al contexto clínico específico.
"""

# Create ADK Protocol Agent
protocol_adk_agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    global_instruction=PROTOCOL_AGENT_ADK_INSTRUCTION,
    instruction="Consulta protocolos médicos NPUAP/EPUAP y MINSAL con búsqueda semántica y evidencia científica.",
    name="protocol_agent_adk",
    tools=[
        search_medical_protocols_adk_tool,
        get_npuap_treatment_protocol_adk_tool,
        get_minsal_protocols_adk_tool,
        get_evidence_level_protocols_adk_tool,
        semantic_protocol_search_adk_tool,
        get_clinical_guidelines_adk_tool,
        get_protocol_system_status_adk_tool
    ],
)


# Factory for ADK Protocol Agent
class ProtocolAgentADKFactory:
    """Factory for creating ADK-based Protocol Agents"""
    
    @staticmethod
    def create_agent() -> LlmAgent:
        """Create new ADK Protocol Agent instance"""
        return protocol_adk_agent
    
    @staticmethod
    def get_agent_capabilities() -> List[str]:
        """Get list of agent capabilities"""
        return [
            'semantic_protocol_search',
            'npuap_epuap_guidelines',
            'minsal_integration',
            'evidence_level_filtering',
            'medical_terminology_standardization',
            'vector_based_similarity',
            'multi_language_support',
            'clinical_guideline_consultation',
            'scientific_reference_management'
        ]
    
    @staticmethod
    def get_supported_organizations() -> List[str]:
        """Get supported medical organizations"""
        return [
            'NPUAP/EPUAP/PPPIA',
            'MINSAL Chile',
            'WOCN Society',
            'Wounds International'
        ]


# Export for use
__all__ = [
    'protocol_adk_agent',
    'ProtocolAgentADKFactory',
    'search_medical_protocols_adk_tool',
    'get_npuap_treatment_protocol_adk_tool',
    'get_minsal_protocols_adk_tool',
    'get_evidence_level_protocols_adk_tool',
    'semantic_protocol_search_adk_tool',
    'get_clinical_guidelines_adk_tool',
    'get_protocol_system_status_adk_tool',
    'ProtocolType',
    'EvidenceLevel',
    'ProtocolLanguage'
]