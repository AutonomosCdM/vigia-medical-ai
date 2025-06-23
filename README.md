# 🩺 VIGIA Medical AI

[![Medical Grade](https://img.shields.io/badge/medical-grade-critical)](https://github.com/AutonomosCdM/vigia-medical-ai-hackathon)
[![HIPAA Compliant](https://img.shields.io/badge/HIPAA-compliant-blue)](https://www.hhs.gov/hipaa/index.html)
[![Google Cloud ADK](https://img.shields.io/badge/Google_Cloud-ADK_Agents-4285f4)](https://cloud.google.com/adk)
[![Medical AI](https://img.shields.io/badge/AI-MedGemma_27B-purple)](https://ollama.ai/medgemma)
[![Production Ready](https://img.shields.io/badge/status-production_ready-success)](./install.sh)

## 🚀 **Quick Start**

### ⚡ One-Command Installation
```bash
./install.sh
```

### 🌐 **Live Demo**
```bash
python final_demo.py
```
**Professional medical interface with public sharing URL for external access**

### 🔬 **Full System**
```bash
python demo/launch_medical_demo.py
```
**Complete medical AI system ready in minutes with web interface at `http://localhost:7860`**

---

**VIGIA Medical AI** is a production-ready pressure injury detection system that implements real medical protocols for healthcare environments. Built with HIPAA compliance and evidence-based clinical decision making.

## 🏥 **Medical Capabilities**

### 🎯 **Clinical Features**
- **NPUAP/EPUAP 2019 Guidelines**: Grade 4 → "Urgent surgical evaluation" 
- **Evidence-Based Medicine**: Level A/B/C recommendations with scientific references
- **Medical Audit Trail**: Complete decision traceability for regulatory compliance
- **Safety-First Design**: Low confidence cases escalate to human medical review

### 🔒 **HIPAA Compliance & Security**
- **PHI Tokenization**: Hospital data → Processing tokens for privacy protection
- **3-Layer Architecture**: Input isolation → Medical orchestration → Specialized processing
- **Local Medical AI**: MedGemma 27B runs locally, no external medical data transfer
- **Comprehensive Audit**: Every medical decision fully traceable with timestamps

### 🧠 **Advanced AI Architecture**
- **Google Cloud ADK**: Multi-agent coordination for medical workflows
- **MONAI Primary**: Medical imaging framework optimized for healthcare
- **YOLOv5 Backup**: Computer vision fallback for robust detection
- **Bidirectional Communication**: WhatsApp patients ↔ Slack medical teams

### 💬 **Communication Integration**
- **Patient Communication**: WhatsApp bot for image submission and results
- **Medical Team Collaboration**: Slack integration for physician coordination  
- **Async Processing**: Celery-based pipeline prevents communication timeouts
- **Multi-Modal Analysis**: Image + voice + patient context integration

---

## 🏗️ **System Architecture**

### 🔐 **3-Layer Security Design**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Layer 1       │    │     Layer 2      │    │      Layer 3        │
│ Input Isolation │───▶│ Medical Routing  │───▶│ Specialized Medical │
│ (WhatsApp Bot)  │    │ (PHI Tokenization│    │ (LPP Detection +    │
│ No medical data │    │  + Triage)       │    │  Clinical Processing│
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

### 🧠 **Medical AI Stack**
- **Primary Detection**: MONAI medical imaging framework
- **Backup Detection**: YOLOv5 computer vision  
- **Clinical AI**: MedGemma 27B local medical language model
- **Decision Engine**: Evidence-based NPUAP/EPUAP 2019 protocols

### 💬 **Communication Flow**
```
Patient (WhatsApp) → Medical Analysis → Physician Review (Slack) → Patient Response
```

---

## 📋 **Technical Excellence**

### 🎯 **System Capabilities**
1. **Medical Functionality**: Implements clinical guidelines and protocols
2. **Production Security**: HIPAA-compliant with PHI tokenization and audit trails
3. **Advanced Architecture**: Google Cloud ADK multi-agent coordination
4. **Healthcare Integration**: Bidirectional patient-physician communication
5. **Regulatory Ready**: Complete medical decision audit trail for compliance

### 🏆 **Technical Highlights**
- **Medical Decision Engine**: NPUAP recommendations with clinical protocols
- **PHI Protection**: Tokenization system for patient privacy protection
- **Agent Coordination**: Google Cloud ADK for complex medical workflow orchestration
- **Evidence-Based**: Level A/B/C medical evidence classification with scientific references
- **Safety Mechanisms**: Automatic escalation for low-confidence medical decisions

### ⚡ **Web Interface Features**
- Upload medical images → Get clinical analysis
- View NPUAP/EPUAP compliance recommendations  
- Complete audit trail for regulatory requirements
- Bidirectional patient-physician communication flow

---

## 🚀 **Installation & Setup**

### Prerequisites
- **Python 3.11+** (recommended)
- **Docker** (optional, for enhanced features)
- **macOS or Linux** (Windows WSL2 supported)

### One-Command Setup
```bash
# Clone and install
git clone https://github.com/AutonomosCdM/vigia-medical-ai-hackathon.git
cd vigia-medical-ai-hackathon
./install.sh

# System automatically:
# ✅ Detects your OS (macOS/Linux)
# ✅ Installs Redis, Ollama, MedGemma 27B
# ✅ Configures medical environment
# ✅ Runs health checks
# ✅ Launches interface at http://localhost:7860
```

### Manual Installation (Advanced)
```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install medical AI
python scripts/setup_medgemma.py --install-ollama
python scripts/setup_medgemma.py --model 27b --install

# Start services
python scripts/start_services.py

# Launch interface
python demo/launch_medical_demo.py
```

---

## 🔬 **Medical Validation**

### Comprehensive Health Checks
```bash
python scripts/validate_medical_system.py
```

**Expected Output:**
```
✅ Medical Decision Engine: Real NPUAP Grade 4 → Emergency
✅ CV Pipeline Framework: MONAI + YOLOv5 ready
✅ PHI Tokenization: HIPAA-compliant client configured
✅ Redis Medical Database: Connected and operational
✅ MedGemma Medical AI: 27B model loaded and ready
✅ Medical Audit Trail: Complete traceability enabled
✅ Web Interface: Medical analysis interface ready
```

### Real Medical Test Cases
The system demonstrates actual medical decision-making:

**Grade 4 Pressure Injury:**
- **Clinical Decision**: "Urgent surgical evaluation required"
- **Timeline**: "Immediate intervention within 15 minutes"
- **Evidence**: "NPUAP Strong Recommendation 7.1"
- **Escalation**: "High priority medical team notification"

---

## 🏗️ **Project Structure**

```
vigia-medical-ai-hackathon/
├── 📖 README.md (This file)
├── ⚡ install.sh (One-command installer)
├── 🔧 requirements.txt (Python dependencies)
├── 
├── 🧠 src/
│   ├── medical/ (NPUAP/EPUAP decision engine)
│   ├── cv_pipeline/ (MONAI + YOLOv5 detection)
│   ├── agents/ (Google Cloud ADK integration)
│   ├── security/ (PHI tokenization + HIPAA)
│   └── communication/ (WhatsApp ↔ Slack)
├── 
├── 🎭 demo/ (Web medical interface)
├── 🧪 tests/ (Medical functionality validation)
├── 📚 docs/ (Architecture & medical compliance)
├── 🔧 scripts/ (Installation & health checks)
└── 🐳 docker/ (Production deployment)
```

---

## 🏆 **Medical Standards & Compliance**

### 🏥 **Medical Standards**
- **NPUAP/EPUAP/PPPIA 2019**: International pressure injury clinical guidelines
- **Evidence-Based Medicine**: Level A/B/C scientific recommendations
- **Medical Device Standards**: IEC 62304 software lifecycle processes
- **Clinical Safety**: ISO 14971 medical device risk management

### 🔒 **Security & Privacy**
- **HIPAA Compliance**: Health Insurance Portability and Accountability Act
- **PHI Protection**: Protected Health Information tokenization
- **SOC 2 Type II**: Security, availability, and confidentiality controls
- **ISO 27001**: Information security management systems

### 🏅 **Technical Standards**
- **Google Cloud ADK**: Advanced agent development kit integration
- **FHIR Compatibility**: Fast Healthcare Interoperability Resources
- **HL7 Standards**: Health Level Seven International messaging
- **DICOM Support**: Digital Imaging and Communications in Medicine

---

## 🤝 **Contributing & Contact**

### 🎯 **Development Team**
- **Medical AI**: NPUAP-compliant clinical decision engine
- **Computer Vision**: MONAI medical imaging + YOLOv5 backup
- **Agent Architecture**: Google Cloud ADK multi-agent coordination  
- **Security Engineering**: HIPAA PHI tokenization and audit systems
- **Healthcare Integration**: Bidirectional patient-physician communication

### 📧 **Contact**
- **GitHub**: [vigia-medical-ai-hackathon](https://github.com/AutonomosCdM/vigia-medical-ai-hackathon)
- **Interface**: `http://localhost:7860` (after installation)
- **Documentation**: `./docs/` (comprehensive medical and technical docs)

---

## 📜 **License & Disclaimer**

### 🏥 **Medical Use Disclaimer**
This system is designed for **healthcare professional use** and **educational purposes**. Not intended for direct patient diagnosis without medical professional oversight. All medical decisions should be validated by qualified healthcare providers.

### 📄 **License**
Licensed under MIT License with medical use provisions. See [LICENSE](LICENSE) for details.

---

**🩺 Built for healthcare. Secured for compliance. Ready for production.**