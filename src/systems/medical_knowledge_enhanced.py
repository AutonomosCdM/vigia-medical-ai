"""
Medical Knowledge System Enhanced - Capa 3: Sistema de Conocimiento Médico Ampliado
Sistema especializado para consultas médicas y protocolos con base de conocimiento completa.

Responsabilidades:
- Responder consultas médicas estructuradas
- Buscar protocolos clínicos validados
- Proporcionar información de medicamentos
- Mantener base de conocimiento actualizada
- Garantizar información médica precisa
- Incluir tratamientos avanzados y cirugía
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

from ..core.input_packager import StandardizedInput
from ..core.medical_dispatcher import TriageDecision
from ..redis_layer.vector_service import VectorService
from ..redis_layer.protocol_indexer import ProtocolIndexer
from ..utils.secure_logger import SecureLogger
from ..ai.medgemma_client import MedGemmaClient

logger = SecureLogger("medical_knowledge_enhanced")


class QueryType(Enum):
    """Tipos de consulta médica."""
    PROTOCOL_SEARCH = "protocol_search"
    MEDICATION_INFO = "medication_info"
    CLINICAL_GUIDELINE = "clinical_guideline"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    DIAGNOSTIC_SUPPORT = "diagnostic_support"
    PREVENTION_PROTOCOL = "prevention_protocol"
    ADVANCED_THERAPY = "advanced_therapy"
    SURGICAL_CONSULTATION = "surgical_consultation"
    GENERAL_MEDICAL = "general_medical"


class KnowledgeSource(Enum):
    """Fuentes de conocimiento médico."""
    CLINICAL_PROTOCOLS = "clinical_protocols"
    MEDICAL_LITERATURE = "medical_literature"
    TREATMENT_GUIDELINES = "treatment_guidelines"
    MEDICATION_DATABASE = "medication_database"
    BEST_PRACTICES = "best_practices"
    INSTITUTIONAL_POLICIES = "institutional_policies"
    AI_GENERATED = "ai_generated"


@dataclass
class MedicalQuery:
    """Consulta médica estructurada."""
    session_id: str
    query_text: str
    query_type: QueryType
    context: Optional[str] = None
    patient_context: Optional[str] = None  # Anonimizado
    urgency: str = "routine"
    language: str = "es"


@dataclass
class KnowledgeResult:
    """Resultado de búsqueda en base de conocimiento."""
    query_id: str
    found: bool
    confidence: float
    sources: List[KnowledgeSource]
    content: Dict[str, Any]
    references: List[str]
    last_updated: Optional[datetime]
    validation_status: str
    ai_enhanced: bool = False


@dataclass
class MedicalResponse:
    """Respuesta médica estructurada."""
    session_id: str
    query: MedicalQuery
    knowledge_result: KnowledgeResult
    response_text: str
    clinical_recommendations: List[str]
    follow_up_suggestions: List[str]
    disclaimers: List[str]
    evidence_level: str
    response_time: float
    timestamp: datetime


class EnhancedMedicalKnowledgeSystem:
    """
    Sistema de conocimiento médico especializado con IA generativa integrada.
    """
    
    def __init__(self, 
                 vector_service: Optional[VectorService] = None,
                 protocol_indexer: Optional[ProtocolIndexer] = None,
                 medgemma_client: Optional[MedGemmaClient] = None):
        """
        Inicializar sistema de conocimiento médico enhazado.
        
        Args:
            vector_service: Servicio de búsqueda vectorial
            protocol_indexer: Indexador de protocolos médicos
            medgemma_client: Cliente MedGemma para IA generativa
        """
        self.vector_service = vector_service or VectorService()
        self.protocol_indexer = protocol_indexer or ProtocolIndexer()
        self.medgemma_client = medgemma_client
        
        # Base de conocimiento estructurado completa
        self.knowledge_base = self._initialize_comprehensive_knowledge_base()
        
        # Configuración de respuestas
        self.response_config = {
            "max_response_length": 3000,
            "include_references": True,
            "include_disclaimers": True,
            "require_evidence": True,
            "enable_ai_enhancement": True
        }
        
        # Disclaimers médicos obligatorios
        self.medical_disclaimers = [
            "Esta información es solo para fines educativos",
            "No reemplaza el criterio médico profesional",
            "Consulte siempre con un profesional de la salud",
            "Información basada en protocolos actuales al momento de consulta"
        ]
        
        logger.audit("enhanced_medical_knowledge_initialized", {
            "component": "layer3_enhanced_medical_knowledge",
            "knowledge_sources": len(self.knowledge_base),
            "ai_integration": self.medgemma_client is not None,
            "medical_compliance": True,
            "disclaimers_active": True
        })
    
    async def initialize(self):
        """Inicializar servicios del sistema."""
        await self.vector_service.initialize()
        await self.protocol_indexer.initialize()
        
        if self.medgemma_client:
            try:
                await self.medgemma_client.validate_connection()
                logger.info("MedGemma integration active")
            except Exception as e:
                logger.warning(f"MedGemma integration disabled: {e}")
                self.medgemma_client = None
        
        logger.audit("enhanced_medical_knowledge_ready", {
            "vector_service": "active",
            "protocol_indexer": "active",
            "ai_enhancement": self.medgemma_client is not None
        })
    
    async def process(self, 
                     standardized_input: StandardizedInput,
                     triage_decision: TriageDecision) -> Dict[str, Any]:
        """
        Procesar consulta médica con capacidades enhanced.
        
        Args:
            standardized_input: Input estandarizado
            triage_decision: Decisión del triage engine
            
        Returns:
            Dict con respuesta médica estructurada
        """
        session_id = standardized_input.session_id
        start_time = datetime.now(timezone.utc)
        
        try:
            # Estructurar consulta médica
            medical_query = await self._structure_medical_query(
                standardized_input, 
                triage_decision
            )
            
            # Buscar en base de conocimiento estructurada
            structured_result = await self._search_knowledge_base(medical_query)
            
            # Enhazar con AI si está disponible y es necesario
            enhanced_result = await self._enhance_with_ai(medical_query, structured_result)
            
            # Generar respuesta médica
            medical_response = await self._generate_medical_response(
                medical_query,
                enhanced_result,
                start_time
            )
            
            # Validar respuesta médica
            validation_result = await self._validate_medical_response(medical_response)
            if not validation_result["valid"]:
                raise ValueError(f"Medical response validation failed: {validation_result['reason']}")
            
            # Log exitoso (sin contenido médico sensible)
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.audit("enhanced_medical_query_processed", {
                "session_id": session_id,
                "query_type": medical_query.query_type.value,
                "knowledge_found": enhanced_result.found,
                "confidence": enhanced_result.confidence,
                "ai_enhanced": enhanced_result.ai_enhanced,
                "response_time": response_time,
                "evidence_level": medical_response.evidence_level
            })
            
            return {
                "success": True,
                "medical_response": self._serialize_medical_response(medical_response),
                "processing_time": response_time,
                "next_steps": self._determine_next_steps(medical_response),
                "follow_up_required": len(medical_response.follow_up_suggestions) > 0,
                "ai_enhanced": enhanced_result.ai_enhanced
            }
            
        except Exception as e:
            logger.error("enhanced_medical_processing_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": str(e),
                "fallback_response": self._generate_fallback_response(session_id),
                "escalation_required": True
            }
    
    async def _enhance_with_ai(self, 
                             query: MedicalQuery, 
                             structured_result: KnowledgeResult) -> KnowledgeResult:
        """Enhazar resultado con AI si es necesario y está disponible."""
        
        # Si no hay MedGemma o ya se encontró información suficiente
        if not self.medgemma_client or (structured_result.found and structured_result.confidence > 0.8):
            return structured_result
        
        try:
            # Determinar si necesita enhancement basado en el tipo de consulta
            needs_ai = self._needs_ai_enhancement(query, structured_result)
            
            if not needs_ai:
                return structured_result
            
            logger.info(f"Enhancing medical query with AI: {query.query_type.value}")
            
            # Preparar contexto para MedGemma
            context = self._prepare_ai_context(query, structured_result)
            
            # Consultar MedGemma
            ai_response = await self.medgemma_client.analyze_medical_query(
                query_text=query.query_text,
                context=context,
                query_type=query.query_type.value
            )
            
            if ai_response and ai_response.success:
                # Combinar resultados estructurados con AI
                enhanced_content = self._combine_structured_and_ai(
                    structured_result.content,
                    ai_response.analysis
                )
                
                return KnowledgeResult(
                    query_id=structured_result.query_id,
                    found=True,
                    confidence=max(structured_result.confidence, 0.85),  # AI boost
                    sources=structured_result.sources + [KnowledgeSource.AI_GENERATED],
                    content=enhanced_content,
                    references=structured_result.references + ["MedGemma Analysis"],
                    last_updated=datetime.now(timezone.utc),
                    validation_status="ai_enhanced",
                    ai_enhanced=True
                )
            
        except Exception as e:
            logger.warning(f"AI enhancement failed, using structured result: {e}")
        
        return structured_result
    
    def _needs_ai_enhancement(self, query: MedicalQuery, result: KnowledgeResult) -> bool:
        """Determinar si una consulta necesita enhancement con AI."""
        
        # Siempre enhazar consultas complejas
        complex_types = [
            QueryType.ADVANCED_THERAPY,
            QueryType.SURGICAL_CONSULTATION,
            QueryType.DIAGNOSTIC_SUPPORT
        ]
        
        if query.query_type in complex_types:
            return True
        
        # Enhazar si no se encontró información o confianza baja
        if not result.found or result.confidence < 0.6:
            return True
        
        # Enhazar si es urgente y necesita más detalle
        if query.urgency in ["emergency", "urgent"]:
            return True
        
        return False
    
    def _prepare_ai_context(self, query: MedicalQuery, result: KnowledgeResult) -> str:
        """Preparar contexto para MedGemma."""
        context_parts = []
        
        if query.context:
            context_parts.append(f"Contexto clínico: {query.context}")
        
        if result.found:
            context_parts.append("Información estructurada disponible:")
            context_parts.append(json.dumps(result.content, indent=2, ensure_ascii=False))
        
        if query.urgency != "routine":
            context_parts.append(f"Urgencia: {query.urgency}")
        
        return "\n\n".join(context_parts)
    
    def _combine_structured_and_ai(self, 
                                 structured: Dict[str, Any], 
                                 ai_content: Dict[str, Any]) -> Dict[str, Any]:
        """Combinar contenido estructurado con análisis de AI."""
        combined = structured.copy()
        
        # Agregar análisis de AI
        if "ai_analysis" not in combined:
            combined["ai_analysis"] = {}
        
        combined["ai_analysis"] = ai_content.get("clinical_analysis", {})
        
        # Agregar recomendaciones de AI si no hay estructuradas
        if not combined.get("clinical_recommendations") and ai_content.get("recommendations"):
            combined["ai_recommendations"] = ai_content["recommendations"]
        
        # Agregar consideraciones adicionales
        if ai_content.get("additional_considerations"):
            combined["ai_considerations"] = ai_content["additional_considerations"]
        
        return combined
    
    def _initialize_comprehensive_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar base de conocimiento médico completa y ampliada."""
        return {
            "lpp_protocols": {
                "prevention": {
                    "title": "Protocolo de Prevención de LPP",
                    "content": {
                        "risk_assessment": [
                            "Evaluación Braden Scale cada 24h",
                            "Identificar factores de riesgo específicos",
                            "Documentar en historia clínica"
                        ],
                        "prevention_measures": [
                            "Cambios posturales cada 2 horas",
                            "Superficies de redistribución de presión",
                            "Cuidados de la piel especializada",
                            "Nutrición adecuada y hidratación"
                        ],
                        "monitoring": [
                            "Inspección diaria de la piel",
                            "Documentación fotográfica si hay cambios",
                            "Reevaluación de riesgo semanal"
                        ]
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01",
                    "references": [
                        "NPUAP/EPUAP Guidelines 2019",
                        "WOCN Society Guidelines"
                    ]
                },
                "treatment_grade1": {
                    "title": "Tratamiento LPP Grado 1",
                    "content": {
                        "immediate_actions": [
                            "Alivio inmediato de presión",
                            "No masajear área eritematosa",
                            "Proteger con apósito transparente"
                        ],
                        "ongoing_care": [
                            "Continuar redistribución de presión",
                            "Monitoreo cada 8 horas",
                            "Hidratación de piel circundante"
                        ],
                        "expected_outcome": "Resolución en 24-72 horas con manejo adecuado"
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01"
                },
                "treatment_grade2": {
                    "title": "Tratamiento LPP Grado 2",
                    "content": {
                        "wound_care": [
                            "Limpieza con suero fisiológico",
                            "Apósito hidrocoloide o hidrogel",
                            "Protección de bordes"
                        ],
                        "monitoring": [
                            "Evaluación cada 24-48 horas",
                            "Medición de dimensiones",
                            "Documentación fotográfica"
                        ],
                        "complications_watch": [
                            "Signos de infección",
                            "Aumento de tamaño",
                            "Deterioro del lecho"
                        ]
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01"
                },
                "treatment_grade3": {
                    "title": "Tratamiento LPP Grado 3 - Pérdida Total del Grosor de la Piel",
                    "content": {
                        "immediate_assessment": [
                            "Evaluación de profundidad y extensión",
                            "Identificación de tejidos comprometidos",
                            "Evaluación vascular y neurológica",
                            "Cultivo si signos de infección"
                        ],
                        "wound_care": [
                            "Desbridamiento quirúrgico si necesario",
                            "Limpieza con suero fisiológico a presión",
                            "Apósitos de espuma o alginato según exudado",
                            "Terapia de presión negativa (VAC) si indicada"
                        ],
                        "pain_management": [
                            "Analgesia previa a curaciones",
                            "Técnicas de distracción",
                            "Evaluación regular escala EVA"
                        ],
                        "nutrition_support": [
                            "Evaluación nutricional completa",
                            "Suplementación proteica",
                            "Vitamina C y Zinc",
                            "Hidratación adecuada"
                        ],
                        "monitoring": [
                            "Evaluación diaria",
                            "Medición semanal con fotografía",
                            "Signos vitales si fiebre",
                            "Laboratorios si sospecha infección"
                        ]
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01",
                    "references": [
                        "NPUAP/EPUAP/PPPIA Guidelines 2019",
                        "Cochrane Reviews Wound Care 2023"
                    ]
                },
                "treatment_grade4": {
                    "title": "Tratamiento LPP Grado 4 - Pérdida Total con Exposición Ósea",
                    "content": {
                        "emergency_measures": [
                            "Evaluación quirúrgica urgente",
                            "Hemocultivos y cultivos de herida",
                            "Antibioterapia empírica si signos sistémicos",
                            "Manejo del dolor intensivo"
                        ],
                        "surgical_management": [
                            "Desbridamiento quirúrgico extenso",
                            "Evaluación de viabilidad ósea",
                            "Consideración de colgajos o injertos",
                            "Osteomielitis: biopsia ósea"
                        ],
                        "advanced_therapies": [
                            "Terapia de presión negativa",
                            "Matriz dérmica acelular",
                            "Factores de crecimiento",
                            "Oxígeno hiperbárico si disponible"
                        ],
                        "multidisciplinary_care": [
                            "Cirugía plástica/vascular",
                            "Infectología",
                            "Nutrición clínica",
                            "Medicina del dolor",
                            "Trabajo social"
                        ]
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01"
                },
                "assessment_tools": {
                    "title": "Herramientas de Evaluación LPP",
                    "content": {
                        "braden_scale": {
                            "description": "Escala de riesgo de LPP más utilizada",
                            "domains": [
                                "Percepción sensorial (1-4)",
                                "Humedad de la piel (1-4)",
                                "Actividad (1-4)",
                                "Movilidad (1-4)",
                                "Nutrición (1-4)",
                                "Fricción y deslizamiento (1-3)"
                            ],
                            "interpretation": {
                                "≤9": "Riesgo muy alto",
                                "10-12": "Riesgo alto",
                                "13-14": "Riesgo moderado",
                                "15-18": "Riesgo bajo",
                                "19-23": "Sin riesgo"
                            }
                        },
                        "push_tool": {
                            "description": "Pressure Ulcer Scale for Healing",
                            "parameters": [
                                "Longitud x anchura",
                                "Profundidad",
                                "Cantidad de exudado",
                                "Tipo de tejido",
                                "Bordes de la herida"
                            ]
                        }
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01"
                }
            },
            "medications": {
                "topical_antibiotics": {
                    "title": "Antibióticos Tópicos para LPP",
                    "content": {
                        "indications": [
                            "LPP con signos locales de infección",
                            "Carga bacteriana elevada confirmada",
                            "Falla en cicatrización > 2 semanas"
                        ],
                        "contraindications": [
                            "Alergia conocida al principio activo",
                            "LPP sin signos de infección",
                            "Uso profiláctico rutinario"
                        ],
                        "common_agents": [
                            "Mupirocina 2% - Primera línea",
                            "Ácido fusídico - Alternativa",
                            "Sulfadiazina de plata - LPP extensas"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                },
                "systemic_antibiotics": {
                    "title": "Antibióticos Sistémicos en LPP Infectadas",
                    "content": {
                        "indications": [
                            "Celulitis perilesional",
                            "Signos sistémicos de infección",
                            "Osteomielitis sospechada/confirmada",
                            "Bacteremia asociada"
                        ],
                        "empirical_therapy": [
                            "Amoxicilina-clavulánico 875/125mg c/8h",
                            "Clindamicina 300-450mg c/6h (alérgicos)",
                            "Ceftriaxona 1-2g c/24h (casos severos)"
                        ],
                        "targeted_therapy": [
                            "S. aureus MSSA: Cefazolina 1g c/8h",
                            "S. aureus MRSA: Vancomicina 15-20mg/kg c/12h",
                            "P. aeruginosa: Piperacilina-tazobactam 4.5g c/6h"
                        ],
                        "duration": "7-14 días según respuesta clínica"
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01"
                },
                "pain_management": {
                    "title": "Manejo del Dolor en LPP",
                    "content": {
                        "background_pain": [
                            "Paracetamol 500-1000mg c/6-8h",
                            "Ibuprofeno 400-600mg c/8h",
                            "Tramadol 50-100mg c/8h (dolor moderado)",
                            "Morfina 5-10mg c/4h PRN (dolor severo)"
                        ],
                        "procedural_pain": [
                            "Lidocaína tópica 30min antes",
                            "Morfina IR 30-60min antes",
                            "Técnicas de distracción",
                            "Posicionamiento cómodo"
                        ],
                        "topical_analgesics": [
                            "Lidocaína gel 2%",
                            "Morfina tópica 0.1-0.2%",
                            "Ketamina gel 1%"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                },
                "nutritional_supplements": {
                    "title": "Suplementos Nutricionales para Cicatrización",
                    "content": {
                        "protein_supplements": [
                            "Proteína: 1.2-1.5g/kg/día",
                            "Arginina: 4.5g tid x 3 semanas",
                            "Colágeno hidrolizado: 15g/día"
                        ],
                        "vitamins_minerals": [
                            "Vitamina C: 500-1000mg/día",
                            "Zinc: 15-30mg/día",
                            "Vitamina A: 10,000 UI/día",
                            "Vitamina E: 400 UI/día"
                        ],
                        "specialized_formulas": [
                            "Cubitan® (Nestlé) - específico para heridas",
                            "Juven® (Abbott) - arginina + colágeno",
                            "Vivonex® - péptidos"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                }
            },
            "guidelines": {
                "documentation": {
                    "title": "Documentación Clínica de LPP",
                    "content": {
                        "required_elements": [
                            "Localización anatómica exacta",
                            "Dimensiones (largo x ancho x profundidad)",
                            "Grado/estadio según NPUAP",
                            "Descripción del lecho",
                            "Cantidad y tipo de exudado",
                            "Estado de bordes y piel perilesional"
                        ],
                        "frequency": [
                            "Inicial: Documentación completa",
                            "Seguimiento: Cada 24-48h",
                            "Cambios significativos: Inmediato"
                        ]
                    },
                    "evidence_level": "regulatory_requirement",
                    "last_updated": "2024-12-01"
                },
                "quality_indicators": {
                    "title": "Indicadores de Calidad en Prevención de LPP",
                    "content": {
                        "process_indicators": [
                            "% pacientes con evaluación riesgo en primeras 8h",
                            "% pacientes alto riesgo con plan prevención",
                            "% cambios posturales documentados",
                            "% evaluaciones Braden completadas"
                        ],
                        "outcome_indicators": [
                            "Incidencia de LPP adquiridas en institución",
                            "Prevalencia de LPP por servicio",
                            "Tiempo promedio de cicatrización",
                            "% LPP que progresan a grado superior"
                        ],
                        "benchmarks": {
                            "incidencia_meta": "< 5% en UCI, < 2% en ward",
                            "evaluacion_riesgo": "> 95% en primeras 8h",
                            "documentacion": "> 90% completa"
                        }
                    },
                    "evidence_level": "best_practice",
                    "last_updated": "2024-12-01"
                },
                "infection_control": {
                    "title": "Control de Infecciones en LPP",
                    "content": {
                        "standard_precautions": [
                            "Higiene de manos antes y después",
                            "Uso de guantes estériles para curaciones",
                            "Técnica aséptica en procedimientos",
                            "Manejo adecuado de material contaminado"
                        ],
                        "contact_precautions": [
                            "MRSA: aislamiento de contacto",
                            "VRE: precauciones especiales",
                            "Pseudomonas MDR: cohorte si posible"
                        ],
                        "environmental_control": [
                            "Desinfección de superficies",
                            "Manejo seguro de ropa de cama",
                            "Ventilación adecuada del cuarto"
                        ]
                    },
                    "evidence_level": "regulatory_requirement",
                    "last_updated": "2024-12-01"
                }
            },
            "advanced_therapies": {
                "negative_pressure": {
                    "title": "Terapia de Presión Negativa (VAC)",
                    "content": {
                        "indications": [
                            "LPP grado 3-4 con lecho limpio",
                            "Heridas con exudado abundante",
                            "Preparación de lecho para cirugía",
                            "Post-quirúrgicas complicadas"
                        ],
                        "contraindications": [
                            "Tejido necrótico no desbridado",
                            "Fístulas enterocutáneas",
                            "Vasos sanguíneos expuestos",
                            "Infección activa no controlada"
                        ],
                        "parameters": [
                            "Presión: 75-125 mmHg continua",
                            "Modo intermitente: 5min ON, 2min OFF",
                            "Cambio de apósito: cada 2-3 días",
                            "Duración: hasta granulación completa"
                        ]
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01"
                },
                "growth_factors": {
                    "title": "Factores de Crecimiento y Terapias Avanzadas",
                    "content": {
                        "pdgf_therapy": [
                            "Becaplermin gel 0.01%",
                            "Aplicación diaria en capa fina",
                            "Indicado en úlceras diabéticas",
                            "Eficacia en LPP limitada"
                        ],
                        "platelet_rich_plasma": [
                            "Concentrado autólogo de plaquetas",
                            "Rico en factores de crecimiento",
                            "Aplicación semanal",
                            "Evidencia preliminar prometedora"
                        ],
                        "stem_cell_therapy": [
                            "Células madre mesenquimales",
                            "Terapia experimental",
                            "Resultados variables",
                            "Requiere protocolo de investigación"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                },
                "hyperbaric_oxygen": {
                    "title": "Oxigenoterapia Hiperbárica",
                    "content": {
                        "mechanism": [
                            "Aumenta tensión de O2 en tejidos",
                            "Estimula angiogénesis",
                            "Efecto bactericida",
                            "Mejora migración de neutrófilos"
                        ],
                        "indications": [
                            "LPP grado 3-4 refractarias",
                            "Osteomielitis asociada",
                            "Falla de injertos/colgajos",
                            "Radionecrosis"
                        ],
                        "protocol": [
                            "2.0-2.5 ATA por 90-120 minutos",
                            "20-40 sesiones según respuesta",
                            "5 días por semana",
                            "Evaluación semanal de progreso"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                }
            },
            "surgical_interventions": {
                "debridement": {
                    "title": "Desbridamiento Quirúrgico de LPP",
                    "content": {
                        "indications": [
                            "Tejido necrótico extenso",
                            "Signos de infección profunda",
                            "Falta de progreso con manejo conservador",
                            "Preparación para cirugía reconstructiva"
                        ],
                        "techniques": [
                            "Sharp debridement: bisturí/tijeras",
                            "Hydrosurgical: Versajet®",
                            "Ultrasónico: MIST Therapy®",
                            "Láser: CO2 o Er:YAG"
                        ],
                        "post_operative": [
                            "Hemostasia cuidadosa",
                            "Irrigación copiosa",
                            "Apósito absorbente primario",
                            "Antibióticos según cultivo"
                        ]
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01"
                },
                "reconstruction": {
                    "title": "Cirugía Reconstructiva de LPP",
                    "content": {
                        "flap_selection": [
                            "Sacro: colgajo glúteo superior",
                            "Isquion: colgajo gracilis/semimembranoso",
                            "Trocánter: colgajo tensor fascia lata",
                            "Occipital: colgajo trapecio"
                        ],
                        "principles": [
                            "Resección de prominencias óseas",
                            "Cierre por planos",
                            "Evitar tensión excesiva",
                            "Drenaje cerrado"
                        ],
                        "post_operative_care": [
                            "Reposo absoluto 2-3 semanas",
                            "Progresión gradual de movilidad",
                            "Fisioterapia especializada",
                            "Seguimiento a largo plazo"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                }
            }
        }
    
    # Los demás métodos son idénticos a la clase original, solo agregar métodos específicos para AI
    async def _structure_medical_query(self, 
                                     standardized_input: StandardizedInput,
                                     triage_decision: TriageDecision) -> MedicalQuery:
        """Estructurar consulta médica desde input."""
        text_content = standardized_input.raw_content.get("text", "")
        
        # Determinar tipo de consulta
        query_type = self._classify_query_type(text_content, triage_decision)
        
        # Extraer contexto clínico (anonimizado)
        clinical_context = self._extract_clinical_context(text_content)
        
        # Determinar urgencia
        urgency = self._determine_query_urgency(text_content, triage_decision)
        
        # Detectar idioma
        language = self._detect_language(text_content)
        
        return MedicalQuery(
            session_id=standardized_input.session_id,
            query_text=text_content,
            query_type=query_type,
            context=clinical_context,
            urgency=urgency,
            language=language
        )
    
    async def _search_knowledge_base(self, query: MedicalQuery) -> KnowledgeResult:
        """Buscar en base de conocimiento médico."""
        query_id = hashlib.sha256(f"{query.session_id}_{query.query_text}".encode()).hexdigest()[:16]
        
        try:
            # Búsqueda vectorial semántica
            vector_results = await self.vector_service.search_similar(
                query.query_text,
                k=5,
                filter_by_type="medical_protocol"
            )
            
            # Búsqueda en protocolos indexados
            protocol_results = await self.protocol_indexer.search_protocols(
                query.query_text,
                query_type=query.query_type.value
            )
            
            # Buscar en knowledge base estructurada
            structured_results = await self._search_structured_knowledge(query)
            
            # Combinar y rankear resultados
            combined_results = self._combine_search_results(
                vector_results, 
                protocol_results, 
                structured_results
            )
            
            if not combined_results:
                return KnowledgeResult(
                    query_id=query_id,
                    found=False,
                    confidence=0.0,
                    sources=[],
                    content={},
                    references=[],
                    last_updated=None,
                    validation_status="not_found"
                )
            
            # Tomar mejor resultado
            best_result = combined_results[0]
            
            return KnowledgeResult(
                query_id=query_id,
                found=True,
                confidence=best_result.get("confidence", 0.0),
                sources=best_result.get("sources", []),
                content=best_result.get("content", {}),
                references=best_result.get("references", []),
                last_updated=best_result.get("last_updated"),
                validation_status="validated"
            )
            
        except Exception as e:
            logger.error("knowledge_search_failed", {
                "query_id": query_id,
                "error": str(e)
            })
            
            return KnowledgeResult(
                query_id=query_id,
                found=False,
                confidence=0.0,
                sources=[],
                content={"error": str(e)},
                references=[],
                last_updated=None,
                validation_status="error"
            )
    
    def _classify_query_type(self, text: str, triage_decision: TriageDecision) -> QueryType:
        """Clasificar tipo de consulta médica."""
        text_lower = text.lower()
        
        # Palabras clave por tipo
        type_keywords = {
            QueryType.PROTOCOL_SEARCH: ["protocolo", "protocol", "procedimiento", "guía"],
            QueryType.MEDICATION_INFO: ["medicamento", "fármaco", "droga", "medication", "drug"],
            QueryType.CLINICAL_GUIDELINE: ["guía clínica", "guideline", "recomendación"],
            QueryType.TREATMENT_RECOMMENDATION: ["tratamiento", "treatment", "manejo", "therapy"],
            QueryType.DIAGNOSTIC_SUPPORT: ["diagnóstico", "diagnosis", "síntomas", "symptoms"],
            QueryType.PREVENTION_PROTOCOL: ["prevención", "prevention", "profilaxis", "prophylaxis"],
            QueryType.ADVANCED_THERAPY: ["terapia avanzada", "vac", "presión negativa", "oxígeno hiperbárico"],
            QueryType.SURGICAL_CONSULTATION: ["cirugía", "surgery", "colgajo", "flap", "desbridamiento"]
        }
        
        # Buscar coincidencias
        for query_type, keywords in type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return query_type
        
        # Default basado en contexto del triage
        if "lpp" in triage_decision.reason.lower() or "pressure" in triage_decision.reason.lower():
            return QueryType.TREATMENT_RECOMMENDATION
        
        return QueryType.GENERAL_MEDICAL
    
    def _extract_clinical_context(self, text: str) -> Optional[str]:
        """Extraer contexto clínico anonimizado."""
        text_lower = text.lower()
        
        clinical_contexts = {
            "lpp": ["lpp", "lesión por presión", "pressure injury", "ulcer"],
            "wound_care": ["herida", "wound", "curación", "healing"],
            "infection": ["infección", "infection", "bacteria", "sepsis"],
            "pain_management": ["dolor", "pain", "analgesia", "analgésico"],
            "nutrition": ["nutrición", "nutrition", "alimentación", "diet"],
            "surgery": ["cirugía", "surgery", "operación", "colgajo"]
        }
        
        for context, keywords in clinical_contexts.items():
            if any(keyword in text_lower for keyword in keywords):
                return context
        
        return None
    
    def _determine_query_urgency(self, text: str, triage_decision: TriageDecision) -> str:
        """Determinar urgencia de la consulta."""
        urgency_keywords = {
            "emergency": ["emergencia", "emergency", "crítico", "critical"],
            "urgent": ["urgente", "urgent", "pronto", "soon"],
            "routine": ["rutina", "routine", "cuando", "when"]
        }
        
        text_lower = text.lower()
        
        for urgency, keywords in urgency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return urgency
        
        # Basado en flags del triage
        if "emergency" in triage_decision.flags:
            return "emergency"
        elif "urgent" in triage_decision.flags:
            return "urgent"
        
        return "routine"
    
    def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto."""
        spanish_indicators = ["protocolo", "tratamiento", "medicamento", "lesión"]
        english_indicators = ["protocol", "treatment", "medication", "injury"]
        
        text_lower = text.lower()
        
        spanish_count = sum(1 for indicator in spanish_indicators if indicator in text_lower)
        english_count = sum(1 for indicator in english_indicators if indicator in text_lower)
        
        return "es" if spanish_count >= english_count else "en"
    
    async def _search_structured_knowledge(self, query: MedicalQuery) -> List[Dict[str, Any]]:
        """Buscar en knowledge base estructurada."""
        results = []
        query_lower = query.query_text.lower()
        
        # Buscar en cada categoría
        for category, items in self.knowledge_base.items():
            for item_key, item_data in items.items():
                # Calcular relevancia
                relevance = self._calculate_relevance(query_lower, item_data)
                
                if relevance > 0.3:  # Umbral mínimo
                    results.append({
                        "confidence": relevance,
                        "content": item_data["content"],
                        "sources": [KnowledgeSource.CLINICAL_PROTOCOLS],
                        "references": item_data.get("references", []),
                        "last_updated": item_data.get("last_updated"),
                        "evidence_level": item_data.get("evidence_level", "moderate"),
                        "title": item_data.get("title", item_key)
                    })
        
        # Ordenar por relevancia
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:3]  # Top 3
    
    def _calculate_relevance(self, query: str, item_data: Dict[str, Any]) -> float:
        """Calcular relevancia de un item para la consulta."""
        relevance = 0.0
        
        # Buscar en título
        title = item_data.get("title", "").lower()
        if any(word in title for word in query.split()):
            relevance += 0.5
        
        # Buscar en contenido
        content = str(item_data.get("content", "")).lower()
        query_words = query.split()
        matching_words = sum(1 for word in query_words if word in content)
        relevance += (matching_words / len(query_words)) * 0.4
        
        # Bonus por evidencia alta
        if item_data.get("evidence_level") == "high":
            relevance += 0.1
        
        return min(1.0, relevance)
    
    def _combine_search_results(self, 
                               vector_results: List[Dict[str, Any]],
                               protocol_results: List[Dict[str, Any]], 
                               structured_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combinar y rankear resultados de búsqueda."""
        all_results = []
        
        # Agregar resultados vectoriales
        for result in vector_results:
            all_results.append({
                **result,
                "source_type": "vector_search"
            })
        
        # Agregar resultados de protocolos
        for result in protocol_results:
            all_results.append({
                **result,
                "source_type": "protocol_index"
            })
        
        # Agregar resultados estructurados
        for result in structured_results:
            all_results.append({
                **result,
                "source_type": "structured_knowledge"
            })
        
        # Ordenar por confianza
        all_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return all_results[:5]  # Top 5
    
    async def _generate_medical_response(self,
                                       query: MedicalQuery,
                                       knowledge_result: KnowledgeResult,
                                       start_time: datetime) -> MedicalResponse:
        """Generar respuesta médica estructurada."""
        try:
            if not knowledge_result.found:
                response_text = self._generate_no_knowledge_response(query)
                clinical_recommendations = []
                follow_up_suggestions = ["Consultar con especialista médico"]
                evidence_level = "N/A"
            else:
                # Generar respuesta basada en conocimiento encontrado
                response_text = self._format_medical_response(
                    query, 
                    knowledge_result.content
                )
                
                clinical_recommendations = self._extract_clinical_recommendations(
                    knowledge_result.content
                )
                
                follow_up_suggestions = self._generate_follow_up_suggestions(
                    query,
                    knowledge_result.content
                )
                
                evidence_level = knowledge_result.content.get("evidence_level", "moderate")
            
            # Calcular tiempo de respuesta
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return MedicalResponse(
                session_id=query.session_id,
                query=query,
                knowledge_result=knowledge_result,
                response_text=response_text,
                clinical_recommendations=clinical_recommendations,
                follow_up_suggestions=follow_up_suggestions,
                disclaimers=self.medical_disclaimers.copy(),
                evidence_level=evidence_level,
                response_time=response_time,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error("medical_response_generation_failed", {
                "session_id": query.session_id,
                "error": str(e)
            })
            raise
    
    def _format_medical_response(self, 
                                query: MedicalQuery, 
                                content: Dict[str, Any]) -> str:
        """Formatear respuesta médica."""
        response_parts = []
        
        # Título si está disponible
        if "title" in content:
            response_parts.append(f"## {content['title']}\n")
        
        # Contenido principal
        if isinstance(content, dict):
            for section, data in content.items():
                if section in ["title", "evidence_level", "last_updated", "references", "ai_analysis", "ai_recommendations", "ai_considerations"]:
                    continue
                
                response_parts.append(f"### {section.replace('_', ' ').title()}")
                
                if isinstance(data, list):
                    for item in data:
                        response_parts.append(f"• {item}")
                elif isinstance(data, str):
                    response_parts.append(data)
                elif isinstance(data, dict):
                    for key, value in data.items():
                        response_parts.append(f"**{key.replace('_', ' ').title()}:** {value}")
                
                response_parts.append("")  # Línea en blanco
        
        # Agregar análisis de AI si existe
        if "ai_analysis" in content:
            response_parts.append("### Análisis AI Enhanced")
            ai_analysis = content["ai_analysis"]
            for key, value in ai_analysis.items():
                response_parts.append(f"**{key.replace('_', ' ').title()}:** {value}")
            response_parts.append("")
        
        # Agregar recomendaciones de AI si existen
        if "ai_recommendations" in content:
            response_parts.append("### Recomendaciones AI")
            for rec in content["ai_recommendations"]:
                response_parts.append(f"• {rec}")
            response_parts.append("")
        
        # Nivel de evidencia
        if "evidence_level" in content:
            response_parts.append(f"**Nivel de evidencia:** {content['evidence_level']}")
        
        response_text = "\n".join(response_parts)
        
        # Truncar si es muy largo
        if len(response_text) > self.response_config["max_response_length"]:
            response_text = response_text[:self.response_config["max_response_length"]] + "..."
        
        return response_text
    
    def _extract_clinical_recommendations(self, content: Dict[str, Any]) -> List[str]:
        """Extraer recomendaciones clínicas del contenido."""
        recommendations = []
        
        # Buscar en secciones específicas
        recommendation_keys = [
            "immediate_actions", "ongoing_care", "prevention_measures",
            "treatment", "monitoring", "follow_up", "emergency_measures",
            "wound_care", "surgical_management", "ai_recommendations"
        ]
        
        for key in recommendation_keys:
            if key in content:
                data = content[key]
                if isinstance(data, list):
                    recommendations.extend(data)
                elif isinstance(data, str):
                    recommendations.append(data)
        
        return recommendations[:5]  # Máximo 5 recomendaciones
    
    def _generate_follow_up_suggestions(self, 
                                      query: MedicalQuery,
                                      content: Dict[str, Any]) -> List[str]:
        """Generar sugerencias de seguimiento."""
        suggestions = []
        
        # Basado en tipo de consulta
        if query.query_type == QueryType.TREATMENT_RECOMMENDATION:
            suggestions.extend([
                "Evaluar respuesta al tratamiento en 24-48 horas",
                "Documentar evolución en historia clínica"
            ])
        elif query.query_type == QueryType.MEDICATION_INFO:
            suggestions.extend([
                "Verificar alergias conocidas",
                "Monitorear efectos adversos"
            ])
        elif query.query_type == QueryType.ADVANCED_THERAPY:
            suggestions.extend([
                "Evaluación por especialista en heridas",
                "Considerar criterios de inclusión/exclusión"
            ])
        elif query.query_type == QueryType.SURGICAL_CONSULTATION:
            suggestions.extend([
                "Interconsulta a cirugía plástica",
                "Evaluación preoperatoria completa"
            ])
        
        # Basado en contenido
        if "monitoring" in content:
            suggestions.append("Implementar plan de monitoreo continuo")
        
        if "complications_watch" in content:
            suggestions.append("Vigilar signos de complicaciones")
        
        return suggestions[:3]  # Máximo 3 sugerencias
    
    def _generate_no_knowledge_response(self, query: MedicalQuery) -> str:
        """Generar respuesta cuando no se encuentra conocimiento."""
        if query.language == "es":
            return (
                f"No se encontró información específica para su consulta sobre '{query.query_text}'. "
                "Se recomienda:\n"
                "• Consultar con el médico tratante\n"
                "• Revisar protocolos institucionales actualizados\n"
                "• Considerar consulta con especialista si es necesario"
            )
        else:
            return (
                f"No specific information found for your query about '{query.query_text}'. "
                "Recommendations:\n"
                "• Consult with attending physician\n"
                "• Review current institutional protocols\n"
                "• Consider specialist consultation if needed"
            )
    
    async def _validate_medical_response(self, response: MedicalResponse) -> Dict[str, Any]:
        """Validar respuesta médica antes de envío."""
        try:
            # Verificar que tiene disclaimers
            if not response.disclaimers:
                return {
                    "valid": False,
                    "reason": "Missing required medical disclaimers"
                }
            
            # Verificar longitud de respuesta
            if len(response.response_text) < 50:
                return {
                    "valid": False,
                    "reason": "Response too short to be medically meaningful"
                }
            
            # Verificar que no contiene información potencialmente peligrosa
            dangerous_keywords = ["automedication", "self-medication", "stop taking"]
            if any(keyword in response.response_text.lower() for keyword in dangerous_keywords):
                return {
                    "valid": False,
                    "reason": "Response contains potentially dangerous medical advice"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    def _serialize_medical_response(self, response: MedicalResponse) -> Dict[str, Any]:
        """Serializar respuesta médica."""
        return {
            "session_id": response.session_id,
            "query": {
                "text": response.query.query_text,
                "type": response.query.query_type.value,
                "context": response.query.context,
                "urgency": response.query.urgency,
                "language": response.query.language
            },
            "knowledge_result": {
                "found": response.knowledge_result.found,
                "confidence": response.knowledge_result.confidence,
                "sources": [s.value for s in response.knowledge_result.sources],
                "validation_status": response.knowledge_result.validation_status,
                "ai_enhanced": response.knowledge_result.ai_enhanced
            },
            "response_text": response.response_text,
            "clinical_recommendations": response.clinical_recommendations,
            "follow_up_suggestions": response.follow_up_suggestions,
            "disclaimers": response.disclaimers,
            "evidence_level": response.evidence_level,
            "response_time": response.response_time,
            "timestamp": response.timestamp.isoformat()
        }
    
    def _determine_next_steps(self, response: MedicalResponse) -> List[str]:
        """Determinar próximos pasos basados en la respuesta."""
        next_steps = []
        
        # Basado en urgencia
        if response.query.urgency == "emergency":
            next_steps.append("immediate_medical_consultation")
        elif response.query.urgency == "urgent":
            next_steps.append("priority_medical_review")
        
        # Basado en tipo de consulta
        if response.query.query_type == QueryType.TREATMENT_RECOMMENDATION:
            next_steps.append("implement_treatment_plan")
        elif response.query.query_type == QueryType.MEDICATION_INFO:
            next_steps.append("verify_prescription_compatibility")
        elif response.query.query_type == QueryType.ADVANCED_THERAPY:
            next_steps.append("specialist_evaluation_required")
        elif response.query.query_type == QueryType.SURGICAL_CONSULTATION:
            next_steps.append("surgical_team_consultation")
        
        # Si no se encontró conocimiento
        if not response.knowledge_result.found:
            next_steps.append("escalate_to_medical_expert")
        
        return next_steps
    
    def _generate_fallback_response(self, session_id: str) -> Dict[str, Any]:
        """Generar respuesta de fallback en caso de error."""
        return {
            "session_id": session_id,
            "response_text": (
                "Disculpe, ocurrió un error procesando su consulta médica. "
                "Por favor, contacte directamente con el equipo médico para "
                "asistencia inmediata."
            ),
            "clinical_recommendations": [
                "Contactar médico tratante",
                "Usar canales de comunicación de respaldo"
            ],
            "disclaimers": self.medical_disclaimers,
            "error_occurred": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class EnhancedMedicalKnowledgeFactory:
    """Factory para crear instancias del sistema de conocimiento médico enhanced."""
    
    @staticmethod
    async def create_system(config: Optional[Dict[str, Any]] = None) -> EnhancedMedicalKnowledgeSystem:
        """Crear sistema de conocimiento médico enhanced."""
        config = config or {}
        
        # Crear servicios
        vector_service = VectorService() if config.get("use_vector_search", True) else None
        protocol_indexer = ProtocolIndexer() if config.get("use_protocol_indexer", True) else None
        medgemma_client = MedGemmaClient() if config.get("enable_ai_enhancement", True) else None
        
        system = EnhancedMedicalKnowledgeSystem(
            vector_service=vector_service,
            protocol_indexer=protocol_indexer,
            medgemma_client=medgemma_client
        )
        
        await system.initialize()
        return system