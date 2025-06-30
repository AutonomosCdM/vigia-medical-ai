# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
VIGIA Medical AI v1.0 is a production-ready medical-grade pressure injury (LPP) detection system with Google Cloud ADK agent coordination, HIPAA-compliant PHI tokenization, and dual AI engines (MONAI + YOLOv5) for 95% detection accuracy.

## Core Architecture

### Production System Structure
- **src/**: Medical modules with clean agent-based architecture
- **scripts/**: Installation, validation, and medical AI setup tools
- **tests/**: Comprehensive medical functionality testing
- **demo/**: Professional medical interfaces with public sharing
- **docs/**: Architecture documentation and Eraser.io diagrams

### Key Components
- **Agents** (`src/agents/`): 9 specialized medical agents with Google Cloud ADK
- **AI Engines** (`src/ai/`): MedGemma 27B local + Hume AI integration + Medical guardrails
- **Core Systems** (`src/core/`): Async pipeline, PHI tokenization, medical dispatcher
- **CV Pipeline** (`src/cv_pipeline/`): MONAI primary + YOLOv5 backup detection
- **Medical Systems** (`src/systems/`): Evidence-based NPUAP/EPUAP decision engine
- **Communication** (`src/messaging/`): WhatsApp patient + Slack medical team integration
- **Security** (`src/security/`): PHI tokenization for HIPAA compliance
- **Storage** (`src/storage/`): Medical image storage with audit trails
- **Database** (`src/db/`): Production PostgreSQL training database with NPUAP grading
- **ML Tracking** (`src/ml/`): Advanced model performance monitoring with drift detection
- **Synthetic Data** (`src/synthetic/`): HIPAA-compliant patient data generation
- **Pipeline** (`src/pipeline/`): Automated retraining pipeline with Celery
- **Monitoring** (`src/monitoring/`): Real-time response monitoring and escalation

## Essential Commands

### Quick Start
```bash
# One-command installation (hackathon simplicity)
./install.sh

# Professional medical demo with sharing
python final_demo.py

# Full medical system
python demo/launch_medical_demo.py
```

### Development Setup
```bash
# Setup MedGemma medical AI (primary method)
python scripts/setup_medgemma_ollama.py --install-ollama
python scripts/setup_medgemma_ollama.py --model 27b --install

# Alternative MedGemma setup
python scripts/setup_medgemma.py

# System validation
python scripts/validate_medical_system.py
python scripts/validate_post_refactor_simple.py --verbose
```

### Advanced Infrastructure Commands

#### Database & Training Pipeline
```bash
# Initialize training database (PostgreSQL + Redis)
python -c "from src.db.training_database import training_db; asyncio.run(training_db.initialize())"

# Generate synthetic medical data
python -c "from src.synthetic.patient_generator import SyntheticPatientGenerator; generator = SyntheticPatientGenerator(); asyncio.run(generator.generate_patient_cohort(100))"

# Track model performance
python -c "from src.ml.model_tracking import model_tracker; asyncio.run(model_tracker.generate_performance_dashboard())"

# Run automated retraining pipeline
python -c "from src.pipeline.training_pipeline import training_pipeline; asyncio.run(training_pipeline.trigger_retraining('accuracy_drop'))"
```

#### Medical Dataset Integration
```bash
# Install dataset integration dependencies
pip install datasets>=2.18.0 huggingface-hub>=0.21.0 scikit-learn>=1.4.0

# Test dataset integration system
python scripts/test_dataset_integration.py

# Integrate all medical datasets (production)
python scripts/integrate_medical_datasets.py --datasets all

# Integrate specific datasets with sample size
python scripts/integrate_medical_datasets.py --datasets ham10000 skincap --sample-size 1000

# Test mode integration (small samples)
python scripts/integrate_medical_datasets.py --test-mode --verbose

# Integration with custom output directory
python scripts/integrate_medical_datasets.py --output-dir ./data/custom_datasets --validation-split 0.3
```

#### Medical Guardrails & Safety
```bash
# Test medical guardrails
python -c "from src.ai.medical_guardrails import MedicalGuardrails; guardrails = MedicalGuardrails(); asyncio.run(guardrails.validate_response('Grade 4 pressure injury requires immediate surgical intervention'))"

# Monitor response safety
python -c "from src.monitoring.response_monitoring import ResponseMonitor; monitor = ResponseMonitor(); asyncio.run(monitor.start_monitoring())"

# Check fallback mechanisms
python -c "from src.ai.fallback_handler import FallbackHandler; fallback = FallbackHandler(); asyncio.run(fallback.get_safe_response('emergency_protocol'))"
```

### Testing
```bash
# Run test suite with different modes
./scripts/run_tests.sh                         # Default comprehensive tests
./scripts/run_tests.sh unit                    # Unit tests only
./scripts/run_tests.sh medical                 # Medical functionality tests
./scripts/run_tests.sh integration             # Integration tests
./scripts/run_tests.sh quick                   # Quick validation (smoke + basic)
./scripts/run_tests.sh coverage               # Tests with coverage report

# Individual test commands
pytest tests/ -v                               # All tests
pytest tests/test_medical_functionality.py -v  # Medical validation
pytest tests/test_shared_utilities.py -v       # Utilities validation
pytest tests/test_medical_simple.py -v         # Simple medical tests
```

## Medical Development Standards

### Professional Medical Code Patterns
```python
# 9-Agent Medical Coordination
from src.agents.master_medical_orchestrator import MasterMedicalOrchestrator
from src.agents.agent_factory import AgentFactory

orchestrator = MasterMedicalOrchestrator()
result = await orchestrator.process_medical_case_async(batman_token, image_path)

# PHI Tokenization (HIPAA Compliance)
from src.core.phi_tokenization_client import PHITokenizationClient
tokenizer = PHITokenizationClient()
batman_token = await tokenizer.create_token_async(hospital_mrn, patient_data)

# Medical Detection Pipeline (Dual Engine)
from src.cv_pipeline.adaptive_medical_detector import AdaptiveMedicalDetector
detector = AdaptiveMedicalDetector()
assessment = await detector.detect_medical_condition_async(image_path, batman_token)

# Agent Factory for Specialized Processing
from src.agents.agent_factory import create_specialized_agent
agent = create_specialized_agent("risk_assessment")
analysis = await agent.analyze_medical_case(batman_token, medical_data)

# Training Database with NPUAP Grading
from src.db.training_database import TrainingDatabase, NPUAPGrade
db = TrainingDatabase()
await db.initialize()
await db.store_medical_image(image_path, NPUAPGrade.STAGE_2, batman_token)

# Medical Guardrails for LLM Safety
from src.ai.medical_guardrails import MedicalGuardrails
guardrails = MedicalGuardrails()
safety_result = await guardrails.validate_medical_response(llm_response, context)

# Model Performance Tracking
from src.ml.model_tracking import ModelPerformanceTracker
tracker = ModelPerformanceTracker()
await tracker.track_model_inference(model_version, batman_token, prediction)

# Synthetic Patient Generation
from src.synthetic.patient_generator import SyntheticPatientGenerator
generator = SyntheticPatientGenerator()
synthetic_patient = await generator.generate_realistic_patient(risk_level="high")
```

### Medical Compliance Requirements
- **PHI Tokenization MANDATORY**: Use Batman tokens instead of real patient data
- **9-Agent Coordination**: Google Cloud ADK for medical workflow orchestration
- **Evidence-Based Decisions**: NPUAP/EPUAP/PPPIA 2019 guidelines implementation
- **Medical Audit Trail**: Complete decision traceability for regulatory compliance
- **Local AI Processing**: MedGemma 27B runs locally for HIPAA compliance
- **Medical Guardrails REQUIRED**: All LLM responses must pass safety validation
- **Model Performance Monitoring**: Continuous drift detection and safety tracking
- **Synthetic Data Only**: Use synthetic patients for training and testing
- **Database NPUAP Compliance**: All medical images stored with proper grading
- **Automated Escalation**: Safety violations trigger immediate medical review

## Production Architecture

### 9-Agent Medical Coordination
- **Master Medical Orchestrator**: Central coordination with A2A protocol
- **Image Analysis Agent**: MONAI + YOLOv5 medical imaging
- **Clinical Assessment Agent**: Evidence-based medical evaluation
- **Risk Assessment Agent**: Medical risk stratification
- **MONAI Review Agent**: Specialized medical imaging review
- **Diagnostic Agent**: Multi-agent diagnostic fusion
- **Protocol Agent**: NPUAP/EPUAP guidelines implementation
- **Communication Agent**: Patient + medical team coordination
- **Workflow Orchestration Agent**: Async medical pipeline management

### Security & Compliance Architecture
- **3-Layer Security**: Input isolation → Medical orchestration → Specialized processing
- **PHI Tokenization**: Bruce Wayne → Batman tokens (15-min sessions)
- **Google Cloud ADK**: A2A protocol for secure agent communication
- **Complete Audit Trail**: Medical decision traceability with timestamps
- **Local AI Stack**: MedGemma 27B + MONAI + YOLOv5 (no external data transfer)

### Communication Flow
```
Patient (WhatsApp) → PHI Tokenization → 9-Agent Analysis → Medical Team (Slack) → Patient Response
                                ↓
                        Google Cloud ADK Coordination
```

## Key Technologies
- **AI/ML**: PyTorch 2.3.0, MONAI, YOLOv5 7.0.13, MedGemma Local
- **Backend**: FastAPI 0.110.2, Celery 5.3.6, Redis 5.0.4
- **Frontend**: Gradio 4.20+ with public sharing capabilities
- **Database**: Supabase 2.4.2, Vector search with FAISS
- **Security**: Cryptography, PHI tokenization, comprehensive audit trails
- **Integration**: Google Cloud ADK, Twilio WhatsApp, Slack API
- **Cloud**: Professional deployment with Render.com support

## Professional Development Guidelines

### Requirements Structure
- **requirements.txt**: Production dependencies with pinned versions
- **pyproject.toml**: Package configuration with medical-specific development tools
  - Black code formatting, pytest testing, mypy type checking
  - Optional medical analysis dependencies (scikit-learn, matplotlib)
  - Professional package structure for `vigia-medical-ai` v1.0.0
- Professional medical-grade dependency management
- HIPAA-compliant packages and security validations

### Code Quality & Validation
```bash
# Post-development validation (run after changes)
python scripts/validate_post_refactor_simple.py --verbose
python scripts/validate_medical_system.py

# Code quality validation
pylint src/

# Coverage reporting
./scripts/run_tests.sh coverage    # Generates htmlcov/ directory
```

### Medical Interface Options

#### Core Medical Interfaces
1. **Professional Demo**: `python final_demo.py` (Gradio with sharing)
2. **Full Medical System**: `python demo/launch_medical_demo.py`
3. **Simple Test Demo**: `python test_demo.py`

#### Smart Care Interface Collection
4. **Smart Care Basic**: `python launch_smartcare.py` (Smart Care replica interface)
5. **Integrated Smart Care**: `python launch_integrated_smartcare.py` (Smart Care + risk assessment)
6. **Simple Smart Care**: `python launch_smartcare_simple.py` (Simplified Smart Care variant)

#### Professional Interface Variants
7. **Professional UI**: `python launch_professional_ui.py` (Advanced CLI options)
8. **Quick Professional**: `python demo_professional_ui.py` (Quick professional demo)
9. **Modern Interface**: `python launch_modern_ui.py` (Modern UI variant)

#### Specialized Assessment Tools
10. **Risk Assessment**: `python launch_risk_assessment.py` (Dedicated risk visualization on port 7862)

#### Professional Interface CLI Options
```bash
# Advanced professional interface with full control
python launch_professional_ui.py --share --port 7860 --debug

# Disable public sharing
python launch_professional_ui.py --no-share

# Custom port configuration
python launch_professional_ui.py --port 8080
```

### Testing Standards
- Medical functionality validation
- Shared utilities testing
- Professional test organization in `tests/`
- Medical compliance verification

## Documentation & Diagrams

### Architecture Diagrams (`docs/diagrams/`)
Professional Eraser.io diagrams for complete system understanding:

1. **System Overview** (`01_system_overview.eraser`) - 5-minute understanding
2. **Technical Architecture** (`02_technical_architecture.eraser`) - Google Cloud ADK stack
3. **Critical User Flow** (`03_critical_user_flow.eraser`) - Sub-3 minute response workflow  
4. **Data Model** (`04_data_model.eraser`) - PHI tokenization + audit trail
5. **External Integrations** (`05_external_integrations.eraser`) - Dependencies & risks
6. **Deployment Infrastructure** (`06_deployment_infrastructure.eraser`) - Production setup
7. **Entity States** (`07_entity_states.eraser`) - Medical session lifecycle

```bash
# View diagram usage guide
cat docs/diagrams/README_DIAGRAMS.md

# Edit diagrams at eraser.io
# 1. Copy code from .eraser files
# 2. Go to eraser.io → Diagram-as-code  
# 3. Paste and edit visually
```

## Current System Status

### ✅ **PRODUCTION ARCHITECTURE COMPLETE**
- **Agent Coordination**: 9 specialized medical agents with Google Cloud ADK
- **HIPAA Compliance**: PHI tokenization with Batman token system (15-min sessions)
- **Medical AI Stack**: MONAI primary + YOLOv5 backup + MedGemma 27B local
- **Evidence-Based**: NPUAP/EPUAP 2019 clinical guidelines implementation
- **Clean Architecture**: Agent factory pattern with comprehensive testing
- **Medical Guardrails**: LLM safety validation with automatic escalation
- **Training Infrastructure**: PostgreSQL database with NPUAP grading system
- **Performance Monitoring**: Real-time model tracking with drift detection
- **Synthetic Data Pipeline**: HIPAA-compliant patient generation with Braden scoring
- **Automated Retraining**: Celery-based pipeline with trigger mechanisms

### 📊 **Latest Achievements (Commit: 4f94c19)**
- **Agent Architecture**: Complete cleanup and production optimization
- **Import Structure**: 100% consistency (vigia_detect → src)
- **9 Core Agents**: Risk Assessment (1,496 lines), Master Orchestrator (1,800+ lines)
- **Zero Dependencies**: Eliminated circular imports and missing modules
- **Production Ready**: Full medical system validation with Batman tokenization

### 🚀 **Key Features**
- **One-Command Install**: `./install.sh` for complete system setup
- **Professional Demo**: `python final_demo.py` with public sharing
- **Comprehensive Testing**: Medical functionality validation with coverage
- **Documentation**: Architecture diagrams in `/docs/diagrams/` with Eraser.io syntax
- **Production Deployment**: HIPAA-compliant infrastructure ready

This repository delivers production-grade medical AI capabilities with professional architecture and comprehensive agent coordination.