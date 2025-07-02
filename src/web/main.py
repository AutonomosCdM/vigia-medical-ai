#!/usr/bin/env python3
"""
VIGIA Medical AI - Professional FastAPI Web Application
======================================================

Production-ready medical interface for pressure injury detection and analysis.
Inspired by modern medical platforms with VIGIA-specific functionality.
"""

from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uvicorn
import os
import json
import logging
from pathlib import Path

# Import VIGIA components
try:
    from ..agents.master_medical_orchestrator import MasterMedicalOrchestrator
    from ..core.phi_tokenization_client import PHITokenizationClient
    from ..medical.risk_assessment import RiskAssessment
    VIGIA_SYSTEM_AVAILABLE = True
except ImportError:
    VIGIA_SYSTEM_AVAILABLE = False
    logging.warning("VIGIA medical system not available - running in demo mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app = FastAPI(
    title="VIGIA Medical AI",
    description="Professional medical interface for pressure injury detection",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for medical device compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

# Create directories if they don't exist
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Add custom template functions
def format_timedelta(dt):
    """Format datetime for template display."""
    from datetime import datetime
    if isinstance(dt, datetime):
        now = datetime.now()
        diff = now - dt
        days = diff.days
        if days == 0:
            return "Today"
        elif days == 1:
            return "Yesterday"
        else:
            return f"{days} days ago"
    return "Unknown"

def format_date(dt):
    """Format date for template display."""
    if dt:
        return dt.strftime("%B %d, %Y")
    return "Unknown"

# VIGIA helper object for templates
class VigiaTemplateHelpers:
    @staticmethod
    def formatDate(dt):
        return format_date(dt)
    
    @staticmethod
    def formatTimedelta(dt):
        return format_timedelta(dt)

templates.env.globals['now'] = datetime.now
templates.env.globals['format_timedelta'] = format_timedelta
templates.env.globals['format_date'] = format_date
templates.env.globals['VIGIA'] = VigiaTemplateHelpers()

# Data models
class PatientProfile(BaseModel):
    """Patient profile data model."""
    id: str
    name: str
    age: int
    diabetes: bool = False
    hypertension: bool = False
    mobility_score: int = Field(ge=1, le=4, description="Braden mobility score")
    batman_token: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class MedicalCase(BaseModel):
    """Medical case data model."""
    id: str
    patient_id: str
    status: str = Field(default="new", description="new, analyzing, reviewed, completed")
    lpp_grade: Optional[int] = Field(ge=1, le=4, description="LPP Grade 1-4")
    confidence_score: Optional[float] = Field(ge=0, le=1)
    risk_level: str = Field(default="unknown", description="low, medium, high, critical")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class AgentAnalysis(BaseModel):
    """Agent analysis result model."""
    agent_name: str
    status: str = Field(description="pending, processing, completed, error")
    confidence: Optional[float] = Field(ge=0, le=1)
    result: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class VoiceAnalysis(BaseModel):
    """Voice analysis data model."""
    case_id: str
    transcript: Optional[str] = None
    emotions: Optional[Dict[str, float]] = None
    pain_indicators: Optional[List[str]] = None
    audio_duration_seconds: Optional[float] = None
    hume_ai_results: Optional[Dict[str, Any]] = None

# In-memory storage (replace with database in production)
patients_db: Dict[str, PatientProfile] = {}
cases_db: Dict[str, MedicalCase] = {}
agent_results_db: Dict[str, List[AgentAnalysis]] = {}
voice_analysis_db: Dict[str, VoiceAnalysis] = {}

# Initialize VIGIA components
if VIGIA_SYSTEM_AVAILABLE:
    orchestrator = MasterMedicalOrchestrator()
    phi_client = PHITokenizationClient()
    risk_assessment = RiskAssessment()
else:
    orchestrator = None
    phi_client = None
    risk_assessment = None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - replace with proper medical auth in production."""
    # For demo purposes, accept any token
    return {"id": "demo_user", "role": "physician", "name": "Dr. Demo"}

# Main routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main medical dashboard."""
    # Get recent cases for dashboard
    recent_cases = list(cases_db.values())[:3]  # Get first 3 cases
    recent_patients = [patients_db.get(case.patient_id) for case in recent_cases]
    
    # Create combined list for template
    recent_case_data = []
    for case in recent_cases:
        patient = patients_db.get(case.patient_id)
        if patient:
            recent_case_data.append({
                'case': case,
                'patient': patient
            })
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "VIGIA Medical AI Dashboard",
        "vigia_available": VIGIA_SYSTEM_AVAILABLE,
        "recent_case_data": recent_case_data
    })

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    """Simple test page without external dependencies."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VIGIA Medical AI - Test Page</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 40px; background: #f0f0f0; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .success { color: #16a34a; font-weight: bold; }
            .info { color: #2563eb; margin: 10px 0; }
            .card { background: #f8fafc; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #2563eb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 VIGIA Medical AI - System Test</h1>
            <p class="success">✅ FastAPI Server is Running Successfully!</p>
            
            <div class="card">
                <h3>📊 System Status</h3>
                <p class="info">• Web Server: Active</p>
                <p class="info">• Templates: Loading</p>
                <p class="info">• API Endpoints: Available</p>
                <p class="info">• VIGIA System: Demo Mode</p>
            </div>
            
            <div class="card">
                <h3>🔗 Available Pages</h3>
                <p><a href="/">Main Dashboard</a> - Professional medical interface</p>
                <p><a href="/api/docs">API Documentation</a> - Interactive API docs</p>
                <p><a href="/api/v1/health">Health Check</a> - System health JSON</p>
            </div>
            
            <div class="card">
                <h3>🎯 Next Steps</h3>
                <p>1. Refresh the main dashboard page</p>
                <p>2. Ensure internet connection for external resources</p>
                <p>3. Try the API endpoints for backend functionality</p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/patient/{patient_id}", response_class=HTMLResponse)
async def patient_profile(request: Request, patient_id: str):
    """Patient profile page."""
    patient = patients_db.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return templates.TemplateResponse("patient_profile.html", {
        "request": request,
        "patient": patient,
        "title": f"Patient: {patient.name}"
    })

@app.get("/cases", response_class=HTMLResponse)
async def cases_queue(request: Request):
    """Cases queue management page."""
    return templates.TemplateResponse("cases_queue.html", {
        "request": request,
        "title": "Cases Queue",
        "vigia_available": VIGIA_SYSTEM_AVAILABLE
    })

@app.get("/cases/new", response_class=HTMLResponse)
async def new_case_form(request: Request):
    """New case creation form."""
    all_patients = list(patients_db.values())
    return templates.TemplateResponse("new_case.html", {
        "request": request,
        "title": "New Patient Case",
        "patients": all_patients,
        "vigia_available": VIGIA_SYSTEM_AVAILABLE
    })

@app.get("/patients", response_class=HTMLResponse)
async def patients_list(request: Request):
    """Patients list and management page."""
    all_patients = list(patients_db.values())
    patient_cases = {}
    
    # Get case counts and latest case for each patient
    for patient in all_patients:
        patient_case_list = [case for case in cases_db.values() if case.patient_id == patient.id]
        patient_cases[patient.id] = {
            "total_cases": len(patient_case_list),
            "latest_case": patient_case_list[-1] if patient_case_list else None
        }
    
    return templates.TemplateResponse("patients_list.html", {
        "request": request,
        "title": "Patients",
        "vigia_available": VIGIA_SYSTEM_AVAILABLE,
        "patients": all_patients,
        "patient_cases": patient_cases
    })

@app.get("/voice", response_class=HTMLResponse)
async def voice_analysis_page(request: Request):
    """Voice analysis interface page."""
    return templates.TemplateResponse("voice_analysis.html", {
        "request": request,
        "title": "Voice Analysis",
        "vigia_available": VIGIA_SYSTEM_AVAILABLE
    })

@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Medical reports page."""
    return templates.TemplateResponse("reports.html", {
        "request": request,
        "title": "Reports",
        "vigia_available": VIGIA_SYSTEM_AVAILABLE
    })

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """System settings page."""
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "title": "Settings",
        "vigia_available": VIGIA_SYSTEM_AVAILABLE
    })

@app.get("/case/{case_id}/analysis", response_class=HTMLResponse)
async def case_analysis(request: Request, case_id: str):
    """Case analysis page with 9-agent results."""
    case = cases_db.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    patient = patients_db.get(case.patient_id)
    agent_results = agent_results_db.get(case_id, [])
    voice_data = voice_analysis_db.get(case_id)
    
    return templates.TemplateResponse("case_analysis.html", {
        "request": request,
        "case": case,
        "patient": patient,
        "agent_results": agent_results,
        "voice_analysis": voice_data,
        "title": f"Case Analysis: {case.id}"
    })

# API endpoints
@app.get("/ping")
def ultra_basic_ping():
    """ULTRA basic ping for AWS App Runner - fastest possible response."""
    import time
    return {"ping": "ok", "ts": int(time.time())}

@app.get("/health")
async def simple_health_check():
    """Lightweight health check for AWS App Runner - no dependencies."""
    return {"status": "ok"}

@app.get("/api/v1/health")
async def detailed_health_check():
    """Health check endpoint with AgentOps monitoring status."""
    
    # Check monitoring status
    monitoring_status = "inactive"
    agentops_status = "not_initialized"
    monitoring_details = {}
    
    try:
        # Import monitoring components in try-catch for deployment resilience
        from monitoring.aws_monitoring_init import check_monitoring_health
        monitoring_health = check_monitoring_health()
        monitoring_status = monitoring_health.get("monitoring_status", "inactive")
        agentops_status = "active" if monitoring_health.get("agentops_initialized") else "inactive"
        monitoring_details = monitoring_health
    except ImportError:
        # Monitoring not available
        monitoring_details = {"error": "monitoring_components_not_available"}
    except Exception as e:
        monitoring_details = {"error": str(e)}
    
    # AWS environment detection
    aws_deployment = os.getenv('AWS_DEPLOYMENT', 'false').lower() == 'true'
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vigia_system": VIGIA_SYSTEM_AVAILABLE,
        "version": "1.0.0",
        "environment": "aws_app_runner" if aws_deployment else "local",
        "monitoring": {
            "status": monitoring_status,
            "agentops": agentops_status,
            "details": monitoring_details
        },
        "deployment": {
            "platform": "aws_app_runner" if aws_deployment else "local",
            "compute": "1_vCPU_2GB" if aws_deployment else "variable",
            "url": "https://fid3rd9z3z.us-east-1.awsapprunner.com" if aws_deployment else "localhost"
        }
    }

@app.get("/api/v1/patients")
async def list_patients(user = Depends(get_current_user)):
    """List all patients."""
    return list(patients_db.values())

@app.post("/api/v1/patients")
async def create_patient(patient: PatientProfile, user = Depends(get_current_user)):
    """Create a new patient."""
    # Generate Batman token if VIGIA system is available
    if VIGIA_SYSTEM_AVAILABLE and phi_client:
        try:
            batman_token = await phi_client.create_token_async(patient.id, patient.dict())
            patient.batman_token = batman_token
        except Exception as e:
            logger.warning(f"Failed to create Batman token: {e}")
    
    patients_db[patient.id] = patient
    return patient

@app.get("/api/v1/cases")
async def list_cases(status: Optional[str] = None, user = Depends(get_current_user)):
    """List medical cases with optional status filter."""
    cases = list(cases_db.values())
    if status:
        cases = [case for case in cases if case.status == status]
    return cases

@app.post("/api/v1/cases")
async def create_case(case: MedicalCase, user = Depends(get_current_user)):
    """Create a new medical case."""
    cases_db[case.id] = case
    agent_results_db[case.id] = []
    return case

@app.post("/cases/new")
async def create_new_case(request: Request):
    """Handle new case form submission."""
    import uuid
    from datetime import datetime
    
    form_data = await request.form()
    
    # Generate new case ID
    case_id = f"VIG-{datetime.now().strftime('%Y-%m')}-{str(uuid.uuid4())[:8].upper()}"
    
    # Create new case
    new_case = MedicalCase(
        id=case_id,
        patient_id=form_data.get("patient_id"),
        status="new",
        lpp_grade=None,
        confidence_score=None,
        risk_level="unknown",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Store in database
    cases_db[case_id] = new_case
    agent_results_db[case_id] = []
    
    return {"success": True, "case_id": case_id, "message": "Case created successfully"}

@app.post("/api/v1/cases/{case_id}/analyze")
async def analyze_case(case_id: str, user = Depends(get_current_user)):
    """Trigger 9-agent analysis for a case."""
    case = cases_db.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Update case status
    case.status = "analyzing"
    case.updated_at = datetime.now()
    
    if VIGIA_SYSTEM_AVAILABLE and orchestrator:
        try:
            # Real VIGIA analysis
            patient = patients_db.get(case.patient_id)
            if patient and patient.batman_token:
                result = await orchestrator.process_medical_case_async(
                    patient.batman_token, 
                    None  # Image path would be provided
                )
                
                # Update case with results
                case.lpp_grade = result.get("lpp_grade")
                case.confidence_score = result.get("confidence")
                case.risk_level = result.get("risk_level", "medium")
                case.status = "completed"
        except Exception as e:
            logger.error(f"VIGIA analysis failed: {e}")
            case.status = "error"
    else:
        # Demo analysis results
        import random
        import time
        
        agent_names = [
            "Image Analysis Agent",
            "Voice Analysis Agent", 
            "Clinical Assessment Agent",
            "Risk Assessment Agent",
            "Diagnostic Agent",
            "MONAI Review Agent",
            "Protocol Agent",
            "Communication Agent",
            "Workflow Orchestration Agent"
        ]
        
        # Simulate agent processing
        agent_results = []
        for agent_name in agent_names:
            result = AgentAnalysis(
                agent_name=agent_name,
                status="completed",
                confidence=random.uniform(0.75, 0.98),
                result={
                    "finding": f"Analysis completed by {agent_name}",
                    "recommendation": "Continue monitoring"
                },
                processing_time_ms=random.randint(500, 2000)
            )
            agent_results.append(result)
        
        agent_results_db[case_id] = agent_results
        
        # Update case
        case.lpp_grade = random.randint(1, 3)
        case.confidence_score = random.uniform(0.85, 0.95)
        case.risk_level = random.choice(["low", "medium", "high"])
        case.status = "completed"
    
    case.updated_at = datetime.now()
    return {"message": "Analysis completed", "case": case}

@app.get("/api/v1/cases/{case_id}/agents")
async def get_agent_results(case_id: str, user = Depends(get_current_user)):
    """Get 9-agent analysis results for a case."""
    results = agent_results_db.get(case_id, [])
    return results

@app.post("/api/v1/media/upload")
async def upload_media(
    case_id: str,
    file: UploadFile = File(...),
    media_type: str = "image",
    user = Depends(get_current_user)
):
    """Upload medical media (images/audio) for a case."""
    # Validate case exists
    if case_id not in cases_db:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Create uploads directory
    uploads_dir = static_dir / "uploads" / case_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = uploads_dir / file.filename
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Return file info
    return {
        "filename": file.filename,
        "media_type": media_type,
        "size": len(content),
        "url": f"/static/uploads/{case_id}/{file.filename}",
        "uploaded_at": datetime.now().isoformat()
    }

@app.post("/api/v1/voice/analyze")
async def analyze_voice(
    case_id: str,
    audio_file: UploadFile = File(...),
    user = Depends(get_current_user)
):
    """Analyze voice for emotional and pain indicators."""
    # Demo voice analysis (replace with real Hume AI integration)
    voice_analysis = VoiceAnalysis(
        case_id=case_id,
        transcript="Patient reports discomfort and difficulty sleeping due to pressure point pain.",
        emotions={
            "distress": 0.73,
            "discomfort": 0.68,
            "fatigue": 0.45,
            "anxiety": 0.32
        },
        pain_indicators=["discomfort", "difficulty sleeping", "pressure point"],
        audio_duration_seconds=45.2,
        hume_ai_results={
            "confidence": 0.89,
            "dominant_emotion": "distress",
            "pain_level_detected": "moderate"
        }
    )
    
    voice_analysis_db[case_id] = voice_analysis
    return voice_analysis

if __name__ == "__main__":
    # Check deployment environment
    aws_deployment = os.environ.get("AWS_DEPLOYMENT", "false").lower() == "true"
    spaces_deployment = os.environ.get("SPACES_DEPLOYMENT", "false").lower() == "true"
    
    if aws_deployment:
        # AWS App Runner configuration
        port = int(os.environ.get("PORT", 7860))
        host = "0.0.0.0"
        reload = False
        logger.info(f"🚀 VIGIA Medical AI starting on AWS App Runner: {host}:{port}")
    elif spaces_deployment:
        # Hugging Face Spaces configuration
        port = int(os.environ.get("PORT", 7860))
        host = "0.0.0.0"
        reload = False
        logger.info(f"🚀 VIGIA Medical AI starting on Hugging Face Spaces: {host}:{port}")
    else:
        # Development server configuration
        port = 8000
        host = "0.0.0.0"
        reload = True
        logger.info(f"🧪 VIGIA Medical AI starting in development mode: {host}:{port}")
    
    uvicorn.run(
        "main:app" if not (spaces_deployment or aws_deployment) else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        timeout_keep_alive=65 if aws_deployment else 5
    )
# Initialize demo data
def initialize_demo_data():
    """Initialize demo data for the medical interface."""
    from datetime import datetime, timedelta
    
    # Demo patients
    demo_patients = [
        PatientProfile(
            id="PAT-001",
            name="María González",
            age=78,
            diabetes=True,
            hypertension=False,
            mobility_score=2,
            batman_token="BATMAN_MG_78_DIABETES",
            created_at=datetime.now() - timedelta(days=5)
        ),
        PatientProfile(
            id="PAT-002", 
            name="Carlos Mendoza",
            age=65,
            diabetes=False,
            hypertension=True,
            mobility_score=3,
            batman_token="BATMAN_CM_65_HYPERTENSION",
            created_at=datetime.now() - timedelta(days=3)
        ),
        PatientProfile(
            id="PAT-003",
            name="Ana Rodriguez",
            age=82,
            diabetes=True,
            hypertension=True,
            mobility_score=1,
            batman_token="BATMAN_AR_82_MULTIPLE",
            created_at=datetime.now() - timedelta(days=1)
        )
    ]
    
    for patient in demo_patients:
        patients_db[patient.id] = patient
    
    # Demo cases
    demo_cases = [
        MedicalCase(
            id="VIG-2024-001",
            patient_id="PAT-001",
            status="analyzing",
            lpp_grade=2,
            confidence_score=0.84,
            risk_level="high",
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now() - timedelta(minutes=30)
        ),
        MedicalCase(
            id="VIG-2024-002",
            patient_id="PAT-002",
            status="completed",
            lpp_grade=1,
            confidence_score=0.87,
            risk_level="medium",
            created_at=datetime.now() - timedelta(days=1),
            updated_at=datetime.now() - timedelta(hours=6)
        ),
        MedicalCase(
            id="VIG-2024-003",
            patient_id="PAT-003",
            status="reviewing",
            lpp_grade=4,
            confidence_score=0.96,
            risk_level="critical",
            created_at=datetime.now() - timedelta(hours=1),
            updated_at=datetime.now() - timedelta(minutes=15)
        )
    ]
    
    for case in demo_cases:
        cases_db[case.id] = case
    
    logger.info(f"Initialized demo data: {len(demo_patients)} patients, {len(demo_cases)} cases")

# Initialize demo data on startup
initialize_demo_data()

