"""
Medical Knowledge System - Capa 3: Sistema de Conocimiento Médico
Sistema especializado para consultas médicas y protocolos.

Responsabilidades:
- Responder consultas médicas estructuradas
- Buscar protocolos clínicos validados
- Proporcionar información de medicamentos
- Mantener base de conocimiento actualizada
- Garantizar información médica precisa
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
# from ..redis_layer.vector_service import VectorService
# from ..redis_layer.protocol_indexer import ProtocolIndexer
from ..utils.secure_logger import SecureLogger

logger = SecureLogger("medical_knowledge_system")


class QueryType(Enum):
    """Tipos de consulta médica."""
    PROTOCOL_SEARCH = "protocol_search"
    MEDICATION_INFO = "medication_info"
    CLINICAL_GUIDELINE = "clinical_guideline"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    DIAGNOSTIC_SUPPORT = "diagnostic_support"
    PREVENTION_PROTOCOL = "prevention_protocol"
    GENERAL_MEDICAL = "general_medical"


class KnowledgeSource(Enum):
    """Fuentes de conocimiento médico."""
    CLINICAL_PROTOCOLS = "clinical_protocols"
    MEDICAL_LITERATURE = "medical_literature"
    TREATMENT_GUIDELINES = "treatment_guidelines"
    MEDICATION_DATABASE = "medication_database"
    BEST_PRACTICES = "best_practices"
    INSTITUTIONAL_POLICIES = "institutional_policies"


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


class MedicalKnowledgeSystem:
    """
    Sistema de conocimiento médico especializado.
    """
    
    def __init__(self, 
                 vector_service: Optional[Any] = None,
                 protocol_indexer: Optional[Any] = None):
        """
        Inicializar sistema de conocimiento médico.
        
        Args:
            vector_service: Servicio de búsqueda vectorial
            protocol_indexer: Indexador de protocolos médicos
        """
        # Mock service for testing\n        self.vector_service = vector_service or None  # VectorService()
        self.protocol_indexer = protocol_indexer or None  # ProtocolIndexer()
        
        # Base de conocimiento estructurado
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Configuración de respuestas
        self.response_config = {
            "max_response_length": 2000,
            "include_references": True,
            "include_disclaimers": True,
            "require_evidence": True
        }
        
        # Disclaimers médicos obligatorios
        self.medical_disclaimers = [
            "Esta información es solo para fines educativos",
            "No reemplaza el criterio médico profesional",
            "Consulte siempre con un profesional de la salud",
            "Información basada en protocolos actuales al momento de consulta"
        ]
        
        logger.audit("medical_knowledge_system_initialized", {
            "component": "layer3_medical_knowledge",
            "knowledge_sources": len(self.knowledge_base),
            "medical_compliance": True,
            "disclaimers_active": True
        })
    
    async def initialize(self):
        """Inicializar servicios del sistema."""
        if self.vector_service:
            await self.vector_service.initialize()
        if self.protocol_indexer:
            await self.protocol_indexer.initialize()
        
        logger.audit("medical_knowledge_system_ready", {
            "vector_service": "active" if self.vector_service else "disabled",
            "protocol_indexer": "active" if self.protocol_indexer else "disabled"
        })
    
    async def process_medical_query(self,
                                  standardized_input: StandardizedInput,
                                  triage_decision: TriageDecision) -> Dict[str, Any]:
        """
        Procesar consulta médica.
        
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
            
            # Buscar en base de conocimiento
            knowledge_result = await self._search_knowledge_base(medical_query)
            
            # Generar respuesta médica
            medical_response = await self._generate_medical_response(
                medical_query,
                knowledge_result,
                start_time
            )
            
            # Validar respuesta médica
            validation_result = await self._validate_medical_response(medical_response)
            if not validation_result["valid"]:
                raise ValueError(f"Medical response validation failed: {validation_result['reason']}")
            
            # Log exitoso (sin contenido médico sensible)
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.audit("medical_query_processed", {
                "session_id": session_id,
                "query_type": medical_query.query_type.value,
                "knowledge_found": knowledge_result.found,
                "confidence": knowledge_result.confidence,
                "response_time": response_time,
                "evidence_level": medical_response.evidence_level
            })
            
            return {
                "success": True,
                "medical_response": self._serialize_medical_response(medical_response),
                "processing_time": response_time,
                "next_steps": self._determine_next_steps(medical_response),
                "follow_up_required": len(medical_response.follow_up_suggestions) > 0
            }
            
        except Exception as e:
            logger.error("medical_knowledge_processing_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": str(e),
                "fallback_response": self._generate_fallback_response(session_id),
                "escalation_required": True
            }
    
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
    
    def _initialize_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar base de conocimiento estructurada."""
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
                    "last_updated": "2024-01-01",
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
                    "last_updated": "2024-01-01"
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
                    "last_updated": "2024-01-01"
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
                    "last_updated": "2024-01-01"
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
                    "last_updated": "2024-01-01"
                }
            },
            "advanced_therapies": {
                "negative_pressure": {
                    "title": "Terapia de Presión Negativa (VAC)",
                    "content": {
                        "indications": [
                            "LPP grado 3-4 con lecho limpio",
                            "Heridas con exudado abundante",
                            "Preparaci\u00f3n de lecho para cirug\u00eda",
                            "Post-quir\u00fargicas complicadas"
                        ],
                        "contraindications": [
                            "Tejido necr\u00f3tico no desbridado",
                            "F\u00edstulas enterocut\u00e1neas",
                            "Vasos sangu\u00edneos expuestos",
                            "Infecci\u00f3n activa no controlada"
                        ],
                        "parameters": [
                            "Presi\u00f3n: 75-125 mmHg continua",
                            "Modo intermitente: 5min ON, 2min OFF",
                            "Cambio de ap\u00f3sito: cada 2-3 d\u00edas",
                            "Duraci\u00f3n: hasta granulaci\u00f3n completa"
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
                            "Aplicaci\u00f3n diaria en capa fina",
                            "Indicado en \u00falceras diab\u00e9ticas",
                            "Eficacia en LPP limitada"
                        ],
                        "platelet_rich_plasma": [
                            "Concentrado aut\u00f3logo de plaquetas",
                            "Rico en factores de crecimiento",
                            "Aplicaci\u00f3n semanal",
                            "Evidencia preliminar prometedora"
                        ],
                        "stem_cell_therapy": [
                            "C\u00e9lulas madre mesenquimales",
                            "Terapia experimental",
                            "Resultados variables",
                            "Requiere protocolo de investigaci\u00f3n"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                },
                "hyperbaric_oxygen": {
                    "title": "Oxigenoterapia Hiperb\u00e1rica",
                    "content": {
                        "mechanism": [
                            "Aumenta tensi\u00f3n de O2 en tejidos",
                            "Estimula angiog\u00e9nesis",
                            "Efecto bactericida",
                            "Mejora migraci\u00f3n de neutr\u00f3filos"
                        ],
                        "indications": [
                            "LPP grado 3-4 refractarias",
                            "Osteomielitis asociada",
                            "Falla de injertos/colgajos",
                            "Radionecrosis"
                        ],
                        "protocol": [
                            "2.0-2.5 ATA por 90-120 minutos",
                            "20-40 sesiones seg\u00fan respuesta",
                            "5 d\u00edas por semana",
                            "Evaluaci\u00f3n semanal de progreso"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                }
            },
            "surgical_interventions": {
                "debridement": {
                    "title": "Desbridamiento Quir\u00fargico de LPP",
                    "content": {
                        "indications": [
                            "Tejido necr\u00f3tico extenso",
                            "Signos de infecci\u00f3n profunda",
                            "Falta de progreso con manejo conservador",
                            "Preparaci\u00f3n para cirug\u00eda reconstructiva"
                        ],
                        "techniques": [
                            "Sharp debridement: bistur\u00ed/tijeras",
                            "Hydrosurgical: Versajet\u00ae",
                            "Ultras\u00f3nico: MIST Therapy\u00ae",
                            "L\u00e1ser: CO2 o Er:YAG"
                        ],
                        "post_operative": [
                            "Hemostasia cuidadosa",
                            "Irrigaci\u00f3n copiosa",
                            "Ap\u00f3sito absorbente primario",
                            "Antibi\u00f3ticos seg\u00fan cultivo"
                        ]
                    },
                    "evidence_level": "high",
                    "last_updated": "2024-12-01"
                },
                "reconstruction": {
                    "title": "Cirug\u00eda Reconstructiva de LPP",
                    "content": {
                        "flap_selection": [
                            "Sacro: colgajo gl\u00fateo superior",
                            "Isquion: colgajo gracilis/semimembranoso",
                            "Troc\u00e1nter: colgajo tensor fascia lata",
                            "Occipital: colgajo trapecio"
                        ],
                        "principles": [
                            "Resecci\u00f3n de prominencias \u00f3seas",
                            "Cierre por planos",
                            "Evitar tensi\u00f3n excesiva",
                            "Drenaje cerrado"
                        ],
                        "post_operative_care": [
                            "Reposo absoluto 2-3 semanas",
                            "Progresi\u00f3n gradual de movilidad",
                            "Fisioterapia especializada",
                            "Seguimiento a largo plazo"
                        ]
                    },
                    "evidence_level": "moderate",
                    "last_updated": "2024-12-01"
                }
            }
        }
    
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
            QueryType.PREVENTION_PROTOCOL: ["prevención", "prevention", "profilaxis", "prophylaxis"]
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
            "nutrition": ["nutrición", "nutrition", "alimentación", "diet"]
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
                if section in ["title", "evidence_level", "last_updated", "references"]:
                    continue
                
                response_parts.append(f"### {section.replace('_', ' ').title()}")
                
                if isinstance(data, list):
                    for item in data:
                        response_parts.append(f"• {item}")
                elif isinstance(data, str):
                    response_parts.append(data)
                
                response_parts.append("")  # Línea en blanco
        
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
            "treatment", "monitoring", "follow_up"
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
        elif query.query_type == QueryType.PROTOCOL_SEARCH:
            suggestions.extend([
                "Revisar protocolo institucional",
                "Capacitar personal en aplicación"
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
                "validation_status": response.knowledge_result.validation_status
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


class MedicalKnowledgeFactory:
    """Factory para crear instancias del sistema de conocimiento médico."""
    
    @staticmethod
    async def create_system(config: Optional[Dict[str, Any]] = None) -> MedicalKnowledgeSystem:
        """Crear sistema de conocimiento médico."""
        config = config or {}
        
        # Crear servicios
        vector_service = VectorService() if config.get("use_vector_search", True) else None
        protocol_indexer = ProtocolIndexer() if config.get("use_protocol_indexer", True) else None
        
        system = MedicalKnowledgeSystem(
            vector_service=vector_service,
            protocol_indexer=protocol_indexer
        )
        
        await system.initialize()
        return system