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
- **AI Engines** (`src/ai/`): MedGemma 27B local + Hume AI integration
- **Core Systems** (`src/core/`): Async pipeline, PHI tokenization, medical dispatcher
- **CV Pipeline** (`src/cv_pipeline/`): MONAI primary + YOLOv5 backup detection
- **Medical Systems** (`src/systems/`): Evidence-based NPUAP/EPUAP decision engine
- **Communication** (`src/messaging/`): WhatsApp patient + Slack medical team integration
- **Security** (`src/security/`): PHI tokenization for HIPAA compliance
- **Storage** (`src/storage/`): Medical image storage with audit trails

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
```

### Medical Compliance Requirements
- **PHI Tokenization MANDATORY**: Use Batman tokens instead of real patient data
- **9-Agent Coordination**: Google Cloud ADK for medical workflow orchestration
- **Evidence-Based Decisions**: NPUAP/EPUAP/PPPIA 2019 guidelines implementation
- **Medical Audit Trail**: Complete decision traceability for regulatory compliance
- **Local AI Processing**: MedGemma 27B runs locally for HIPAA compliance

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
1. **Professional Demo**: `python final_demo.py` (Gradio with sharing)
2. **Full Medical System**: `python demo/launch_medical_demo.py`
3. **Simple Test Demo**: `python test_demo.py`
4. **Testing Interface**: Medical functionality validation

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