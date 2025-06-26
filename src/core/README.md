# El Cerebro del Sistema - Módulo de Orquestación Médica (Capa 2)

## Principio Fundamental: Orquestación Inteligente sin Decisiones Finales

El módulo "cerebro" es la capa de orquestación médica que recibe inputs de la Capa 1, toma decisiones de enrutamiento y coordina el procesamiento entre sistemas especializados. **Orquesta pero NO diagnostica**.

## Cómo funciona el Cerebro del Sistema

### 1. **Recepción desde Input Queue**
El cerebro recibe paquetes encriptados de la cola temporal:
- Desencripta el `StandardizedInput`
- Valida la sesión (timeout de 15 minutos)
- Extrae el contenido para análisis

### 2. **Medical Dispatcher - El Orquestador Principal**
Toma decisiones de enrutamiento basándose en el contenido:

```python
Rutas disponibles:
├── CLINICAL_IMAGE → Procesamiento de imágenes médicas
├── MEDICAL_QUERY → Consultas de conocimiento médico
├── HUMAN_REVIEW → Casos ambiguos para revisión humana
├── EMERGENCY → Escalamiento inmediato
└── INVALID → Input no procesable
```

**Lógica de decisión:**
- ¿Tiene imagen + código paciente? → `CLINICAL_IMAGE`
- ¿Es consulta de texto médico? → `MEDICAL_QUERY`
- ¿Contiene palabras de emergencia? → `EMERGENCY`
- ¿Es ambiguo o incompleto? → `HUMAN_REVIEW`

### 3. **Triage Engine - Evaluación de Urgencia**
Analiza el contenido para determinar prioridad clínica:

```python
Niveles de urgencia:
├── EMERGENCY (< 5 min) → "dolor severo", "sangrado", "urgente"
├── URGENT (< 30 min) → "herida abierta", "infección"
├── PRIORITY (< 2 hrs) → "enrojecimiento", "leve dolor"
└── ROUTINE (< 24 hrs) → consultas generales
```

**Con MedGemma integrado:**
- Análisis de lenguaje natural médico
- Detección de síntomas complejos
- Evaluación contextual del riesgo

### 4. **Procesamiento de Imágenes**
Cuando recibe una imagen médica:

```python
Flujo de procesamiento:
1. Descarga imagen desde URL (Twilio)
   ↓
2. Validación técnica (formato, tamaño)
   ↓
3. Preprocesamiento (normalización, mejora)
   ↓
4. Detección YOLO (identifica LPP)
   ↓
5. Análisis con MedGemma (si disponible)
   ↓
6. Generación de reporte clínico
```

### 5. **Session Manager - Aislamiento Temporal**
Gestiona el ciclo de vida de cada interacción:
- Crea sesiones únicas con timeout
- Mantiene estado temporal durante procesamiento
- Elimina automáticamente datos después del timeout
- Garantiza que no persista información sensible

### 6. **Decisiones que TOMA vs DELEGA**

**El Cerebro TOMA estas decisiones:**
- ✅ A qué sistema especializado enviar el input
- ✅ Qué nivel de urgencia asignar
- ✅ Si requiere revisión humana
- ✅ Si la calidad del input es suficiente
- ✅ Qué información adicional necesita

**El Cerebro DELEGA estas decisiones:**
- ❌ Diagnóstico específico de LPP → `ClinicalProcessingSystem`
- ❌ Interpretación médica profunda → `MedGemma`
- ❌ Protocolos de tratamiento → `MedicalKnowledgeSystem`
- ❌ Decisiones clínicas finales → `HumanReviewQueue`

## Analogía: El Triaje de Urgencias

El cerebro funciona como el **sistema de triaje en urgencias**:
- Recibe a todos los pacientes que llegan
- Evalúa rápidamente la gravedad
- Decide a qué especialista enviar
- Asigna prioridad de atención
- **NO realiza** el tratamiento médico

## Componentes del Cerebro

### **medical_dispatcher.py**
- Orquestador principal
- Toma decisiones de enrutamiento
- Coordina entre sistemas

### **triage_engine.py**
- Motor de evaluación de urgencia
- Detecta indicadores clínicos
- Integra con MedGemma

### **session_manager.py**
- Gestión de sesiones temporales
- Control de timeouts
- Limpieza automática

### **unified_image_processor.py**
- Pipeline unificado de visión
- Preprocesamiento y YOLO
- Enriquecimiento de resultados

## Flujo de Datos Completo

```
Input Queue (encriptado)
         ↓
   Medical Dispatcher
         ↓
    Triage Engine
         ↓
  ┌──────┴──────┐
  │  Decisión   │
  │ de Ruta     │
  └──────┬──────┘
         ↓
┌────────┴────────┬────────────┬──────────────┐
│                 │            │              │
▼                 ▼            ▼              ▼
Clinical      Medical      Human         Emergency
Processing    Knowledge    Review        Handler
│                 │            │              │
└────────┬────────┴────────────┴──────────────┘
         ▼
   Notificaciones
   Almacenamiento
   Seguimiento
```

## Integración con MedGemma

MedGemma mejora las capacidades del cerebro:
```python
# Entrada: Observaciones clínicas + detecciones visuales
{
    "clinical_observations": "Eritema en región sacra",
    "visual_findings": {
        "detections": [...],
        "confidence": 0.87
    }
}

# Salida: Análisis médico estructurado
{
    "lpp_grade": "Grado 2",
    "clinical_urgency": "moderada",
    "recommendations": [
        "Alivio de presión inmediato",
        "Aplicación de apósito"
    ]
}
```

## Garantías de Seguridad

1. **Separación de responsabilidades**: Orquesta pero no diagnostica
2. **Aislamiento temporal**: Datos se eliminan después de 15 minutos
3. **Trazabilidad completa**: Todo se audita sin exponer PII
4. **Sin almacenamiento permanente**: Solo coordina flujos
5. **Escalamiento humano**: Casos dudosos siempre van a revisión

## Configuración Requerida

Variables de entorno:
- `REDIS_URL`: Para gestión de sesiones
- `GOOGLE_API_KEY`: Para integración con MedGemma
- `ENCRYPTION_KEY`: Para datos en tránsito
- `TIMEOUT_MINUTES`: Timeout de sesiones (default: 15)

## Estado Actual de Implementación

✅ **Implementado:**
- Medical Dispatcher con rutas de decisión
- Triage Engine con detección de urgencias
- Session Manager con timeouts
- Unified Image Processor con YOLO
- Integración con MedGemma
- Enrutamiento a sistemas especializados

⚠️ **En Desarrollo:**
- Análisis de sentimientos para respuestas adaptativas
- Métricas de confianza más granulares
- Cache de decisiones frecuentes

## Principios de Diseño

1. **Orquestación sin opinión**: Coordina pero no diagnostica
2. **Decisiones rápidas**: < 100ms para enrutamiento
3. **Escalabilidad**: Puede manejar múltiples inputs concurrentes
4. **Resiliencia**: Fallback a revisión humana si hay dudas
5. **Transparencia**: Todas las decisiones son auditables