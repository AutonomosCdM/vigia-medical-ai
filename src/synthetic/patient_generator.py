"""
VIGIA Medical AI - Synthetic Patient Generator
==============================================

HIPAA-compliant synthetic patient data generation compatible with PHI tokenization.
Generates realistic medical profiles, demographics, and clinical scenarios.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import json
import random
import hashlib
from faker import Faker
import numpy as np

from ..core.phi_tokenization_client import PHITokenizationClient
from ..db.training_database import NPUAPGrade
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService

logger = SecureLogger(__name__)

class RiskLevel(Enum):
    """Patient risk levels for pressure injuries"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class MobilityLevel(Enum):
    """Patient mobility levels"""
    FULLY_MOBILE = "fully_mobile"
    SLIGHTLY_LIMITED = "slightly_limited"
    VERY_LIMITED = "very_limited"
    COMPLETELY_IMMOBILE = "completely_immobile"

@dataclass
class MedicalCondition:
    """Medical condition with ICD-10 coding"""
    name: str
    icd10_code: str
    severity: str  # mild, moderate, severe
    onset_date: datetime
    status: str  # active, resolved, chronic

@dataclass
class Medication:
    """Patient medication"""
    name: str
    dosage: str
    frequency: str
    route: str
    start_date: datetime
    end_date: Optional[datetime] = None
    indication: str = ""

@dataclass
class VitalSigns:
    """Patient vital signs"""
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    respiratory_rate: int
    temperature_celsius: float
    oxygen_saturation: int
    pain_score: int  # 0-10 scale
    recorded_at: datetime

@dataclass
class BradenScore:
    """Braden Scale assessment for pressure injury risk"""
    sensory_perception: int  # 1-4
    moisture: int  # 1-4
    activity: int  # 1-4
    mobility: int  # 1-4
    nutrition: int  # 1-4
    friction_shear: int  # 1-3
    total_score: int
    risk_level: RiskLevel
    assessed_by: str
    assessment_date: datetime

@dataclass
class SyntheticPatient:
    """Complete synthetic patient profile"""
    # Basic Demographics
    patient_id: str
    batman_token: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    age: int
    gender: str
    ethnicity: str
    preferred_language: str
    
    # Medical Information
    medical_record_number: str
    admission_date: datetime
    primary_diagnosis: str
    secondary_diagnoses: List[str]
    medical_conditions: List[MedicalCondition]
    current_medications: List[Medication]
    allergies: List[str]
    
    # Clinical Assessment
    braden_score: BradenScore
    mobility_level: MobilityLevel
    current_location: str  # room/bed number
    care_unit: str
    attending_physician: str
    primary_nurse: str
    
    # Risk Factors
    pressure_injury_history: bool
    previous_surgery: bool
    diabetes: bool
    cardiovascular_disease: bool
    malnutrition_risk: bool
    cognitive_impairment: bool
    
    # Current Status
    vital_signs: VitalSigns
    skin_assessment_notes: str
    current_pressure_injuries: List[Dict[str, Any]]
    prevention_measures: List[str]
    
    # Metadata
    created_at: datetime
    synthetic_profile_version: str

class SyntheticPatientGenerator:
    """Generator for realistic synthetic patient data"""
    
    def __init__(self):
        self.fake = Faker('en_US')
        self.phi_client = PHITokenizationClient()
        self.audit_service = AuditService()
        
        # Medical data distributions based on real clinical data
        self.age_distribution = {
            'pediatric': (0, 17, 0.05),      # 5% pediatric
            'young_adult': (18, 39, 0.15),   # 15% young adults
            'middle_aged': (40, 64, 0.35),   # 35% middle-aged
            'elderly': (65, 84, 0.35),       # 35% elderly
            'very_elderly': (85, 100, 0.10)  # 10% very elderly
        }
        
        self.gender_distribution = {
            'Female': 0.52,
            'Male': 0.47,
            'Other': 0.01
        }
        
        self.ethnicity_distribution = {
            'White': 0.60,
            'Hispanic/Latino': 0.18,
            'Black/African American': 0.13,
            'Asian': 0.06,
            'American Indian/Alaska Native': 0.02,
            'Other': 0.01
        }
        
        # Common medical conditions affecting pressure injury risk
        self.medical_conditions = {
            'diabetes_mellitus': {
                'icd10': 'E11.9',
                'prevalence': 0.25,
                'risk_multiplier': 1.5
            },
            'cardiovascular_disease': {
                'icd10': 'I25.9',
                'prevalence': 0.30,
                'risk_multiplier': 1.3
            },
            'chronic_kidney_disease': {
                'icd10': 'N18.6',
                'prevalence': 0.15,
                'risk_multiplier': 1.4
            },
            'malnutrition': {
                'icd10': 'E46',
                'prevalence': 0.20,
                'risk_multiplier': 2.0
            },
            'dementia': {
                'icd10': 'F03.90',
                'prevalence': 0.12,
                'risk_multiplier': 1.8
            },
            'spinal_cord_injury': {
                'icd10': 'S14.109A',
                'prevalence': 0.03,
                'risk_multiplier': 3.0
            }
        }
        
        # Common medications affecting healing
        self.medications = {
            'steroids': ['Prednisone', 'Methylprednisolone', 'Hydrocortisone'],
            'anticoagulants': ['Warfarin', 'Heparin', 'Rivaroxaban'],
            'diabetes': ['Metformin', 'Insulin', 'Glipizide'],
            'blood_pressure': ['Lisinopril', 'Amlodipine', 'Metoprolol'],
            'pain': ['Morphine', 'Oxycodone', 'Acetaminophen'],
            'antibiotics': ['Cephalexin', 'Clindamycin', 'Vancomycin']
        }
        
        # Clinical units with different risk profiles
        self.care_units = {
            'ICU': {'base_risk': 'high', 'mobility_factor': 0.3},
            'Medical': {'base_risk': 'moderate', 'mobility_factor': 0.6},
            'Surgical': {'base_risk': 'moderate', 'mobility_factor': 0.5},
            'Orthopedic': {'base_risk': 'moderate', 'mobility_factor': 0.4},
            'Rehabilitation': {'base_risk': 'low', 'mobility_factor': 0.8},
            'Long-term Care': {'base_risk': 'high', 'mobility_factor': 0.3}
        }
        
        logger.info("SyntheticPatientGenerator initialized")
    
    async def generate_patient(self, 
                              risk_profile: RiskLevel = None,
                              age_group: str = None,
                              include_pressure_injuries: bool = None) -> SyntheticPatient:
        """Generate a complete synthetic patient"""
        
        try:
            # Generate basic demographics
            demographics = self._generate_demographics(age_group)
            
            # Generate medical record number and IDs
            mrn = self._generate_mrn()
            patient_id = f"PAT_{random.randint(100000, 999999)}"
            
            # Create Batman token for PHI compliance
            phi_data = {
                'mrn': mrn,
                'name': f"{demographics['first_name']} {demographics['last_name']}",
                'dob': demographics['date_of_birth'].isoformat(),
                'patient_type': 'synthetic'
            }
            
            batman_token = await self.phi_client.create_token_async("HOSPITAL_MAIN", phi_data)
            
            # Generate medical conditions based on age and risk factors
            medical_conditions = self._generate_medical_conditions(demographics['age'])
            
            # Generate Braden score and risk assessment
            braden_score = self._generate_braden_score(demographics['age'], medical_conditions, risk_profile)
            
            # Generate medications based on conditions
            medications = self._generate_medications(medical_conditions)
            
            # Generate vital signs
            vital_signs = self._generate_vital_signs(demographics['age'], medical_conditions)
            
            # Generate current pressure injuries if specified
            current_injuries = []
            if include_pressure_injuries or (include_pressure_injuries is None and random.random() < 0.3):
                current_injuries = self._generate_pressure_injuries(braden_score.risk_level)
            
            # Select care unit
            care_unit = self._select_care_unit(braden_score.risk_level, medical_conditions)
            
            # Generate clinical staff
            staff = self._generate_clinical_staff()
            
            # Create complete patient profile
            patient = SyntheticPatient(
                # Basic Demographics
                patient_id=patient_id,
                batman_token=batman_token,
                first_name=demographics['first_name'],
                last_name=demographics['last_name'],
                date_of_birth=demographics['date_of_birth'],
                age=demographics['age'],
                gender=demographics['gender'],
                ethnicity=demographics['ethnicity'],
                preferred_language=demographics.get('language', 'English'),
                
                # Medical Information
                medical_record_number=mrn,
                admission_date=self._generate_admission_date(),
                primary_diagnosis=self._select_primary_diagnosis(medical_conditions),
                secondary_diagnoses=self._select_secondary_diagnoses(medical_conditions),
                medical_conditions=medical_conditions,
                current_medications=medications,
                allergies=self._generate_allergies(),
                
                # Clinical Assessment
                braden_score=braden_score,
                mobility_level=self._determine_mobility_level(braden_score),
                current_location=f"{care_unit}-{random.randint(101, 450)}",
                care_unit=care_unit,
                attending_physician=staff['physician'],
                primary_nurse=staff['nurse'],
                
                # Risk Factors
                pressure_injury_history=random.random() < 0.15,
                previous_surgery=random.random() < 0.4,
                diabetes=any(c.name == 'diabetes_mellitus' for c in medical_conditions),
                cardiovascular_disease=any(c.name == 'cardiovascular_disease' for c in medical_conditions),
                malnutrition_risk=any(c.name == 'malnutrition' for c in medical_conditions),
                cognitive_impairment=any(c.name == 'dementia' for c in medical_conditions),
                
                # Current Status
                vital_signs=vital_signs,
                skin_assessment_notes=self._generate_skin_assessment(current_injuries),
                current_pressure_injuries=current_injuries,
                prevention_measures=self._generate_prevention_measures(braden_score.risk_level),
                
                # Metadata
                created_at=datetime.now(),
                synthetic_profile_version="1.0"
            )
            
            # Log patient generation
            await self.audit_service.log_activity(
                activity_type="synthetic_patient_generated",
                batman_token=batman_token,
                details={
                    'patient_id': patient_id,
                    'risk_level': braden_score.risk_level.value,
                    'age': demographics['age'],
                    'care_unit': care_unit,
                    'pressure_injuries': len(current_injuries)
                }
            )
            
            logger.info(f"Generated synthetic patient: {patient_id} (Risk: {braden_score.risk_level.value})")
            
            return patient
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic patient: {e}")
            raise
    
    def _generate_demographics(self, age_group: str = None) -> Dict[str, Any]:
        """Generate realistic demographic data"""
        
        # Select age group and generate age
        if age_group:
            if age_group in self.age_distribution:
                min_age, max_age, _ = self.age_distribution[age_group]
                age = random.randint(min_age, max_age)
            else:
                age = random.randint(18, 85)
        else:
            # Weighted random selection
            age_group_choice = np.random.choice(
                list(self.age_distribution.keys()),
                p=[prob for _, _, prob in self.age_distribution.values()]
            )
            min_age, max_age, _ = self.age_distribution[age_group_choice]
            age = random.randint(min_age, max_age)
        
        # Generate other demographics
        gender = np.random.choice(
            list(self.gender_distribution.keys()),
            p=list(self.gender_distribution.values())
        )
        
        ethnicity = np.random.choice(
            list(self.ethnicity_distribution.keys()),
            p=list(self.ethnicity_distribution.values())
        )
        
        # Generate names and birth date
        if gender == 'Female':
            first_name = self.fake.first_name_female()
        elif gender == 'Male':
            first_name = self.fake.first_name_male()
        else:
            first_name = self.fake.first_name()
        
        last_name = self.fake.last_name()
        
        # Calculate birth date
        birth_date = datetime.now() - timedelta(days=age * 365 + random.randint(0, 365))
        
        return {
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity,
            'date_of_birth': birth_date,
            'language': 'Spanish' if ethnicity == 'Hispanic/Latino' and random.random() < 0.7 else 'English'
        }
    
    def _generate_mrn(self) -> str:
        """Generate medical record number"""
        return f"MRN{random.randint(1000000, 9999999)}"
    
    def _generate_medical_conditions(self, age: int) -> List[MedicalCondition]:
        """Generate realistic medical conditions based on age"""
        
        conditions = []
        
        # Age-based condition probability
        age_factor = min(age / 80, 1.0)  # Normalize to 0-1
        
        for condition_name, condition_data in self.medical_conditions.items():
            base_prevalence = condition_data['prevalence']
            age_adjusted_prevalence = base_prevalence * (0.5 + age_factor)
            
            if random.random() < age_adjusted_prevalence:
                severity = random.choices(
                    ['mild', 'moderate', 'severe'],
                    weights=[0.5, 0.4, 0.1]
                )[0]
                
                onset_date = datetime.now() - timedelta(
                    days=random.randint(30, 365 * 5)
                )
                
                condition = MedicalCondition(
                    name=condition_name,
                    icd10_code=condition_data['icd10'],
                    severity=severity,
                    onset_date=onset_date,
                    status='active' if random.random() < 0.8 else 'chronic'
                )
                conditions.append(condition)
        
        return conditions
    
    def _generate_braden_score(self, 
                              age: int, 
                              medical_conditions: List[MedicalCondition],
                              target_risk: RiskLevel = None) -> BradenScore:
        """Generate realistic Braden scale assessment"""
        
        # Base scores (higher is better)
        base_scores = {
            'sensory_perception': random.randint(2, 4),
            'moisture': random.randint(2, 4),
            'activity': random.randint(1, 4),
            'mobility': random.randint(1, 4),
            'nutrition': random.randint(2, 4),
            'friction_shear': random.randint(1, 3)
        }
        
        # Adjust scores based on age
        if age > 75:
            base_scores['mobility'] = max(1, base_scores['mobility'] - 1)
            base_scores['activity'] = max(1, base_scores['activity'] - 1)
        
        # Adjust scores based on medical conditions
        condition_names = [c.name for c in medical_conditions]
        
        if 'diabetes_mellitus' in condition_names:
            base_scores['nutrition'] = max(1, base_scores['nutrition'] - 1)
        
        if 'malnutrition' in condition_names:
            base_scores['nutrition'] = max(1, base_scores['nutrition'] - 2)
        
        if 'spinal_cord_injury' in condition_names:
            base_scores['mobility'] = 1
            base_scores['activity'] = 1
            base_scores['sensory_perception'] = max(1, base_scores['sensory_perception'] - 1)
        
        if 'dementia' in condition_names:
            base_scores['sensory_perception'] = max(1, base_scores['sensory_perception'] - 1)
        
        # Adjust for target risk level if specified
        if target_risk:
            total_score = sum(base_scores.values())
            
            target_ranges = {
                RiskLevel.LOW: (19, 23),
                RiskLevel.MODERATE: (15, 18),
                RiskLevel.HIGH: (13, 14),
                RiskLevel.VERY_HIGH: (6, 12)
            }
            
            target_min, target_max = target_ranges[target_risk]
            
            if total_score < target_min:
                # Increase scores randomly
                while sum(base_scores.values()) < target_min:
                    factor = random.choice(['mobility', 'activity', 'nutrition'])
                    if base_scores[factor] < (3 if factor == 'friction_shear' else 4):
                        base_scores[factor] += 1
            
            elif total_score > target_max:
                # Decrease scores randomly
                while sum(base_scores.values()) > target_max:
                    factor = random.choice(['mobility', 'activity', 'sensory_perception'])
                    if base_scores[factor] > 1:
                        base_scores[factor] -= 1
        
        total_score = sum(base_scores.values())
        
        # Determine risk level
        if total_score <= 9:
            risk_level = RiskLevel.VERY_HIGH
        elif total_score <= 12:
            risk_level = RiskLevel.HIGH
        elif total_score <= 14:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        return BradenScore(
            sensory_perception=base_scores['sensory_perception'],
            moisture=base_scores['moisture'],
            activity=base_scores['activity'],
            mobility=base_scores['mobility'],
            nutrition=base_scores['nutrition'],
            friction_shear=base_scores['friction_shear'],
            total_score=total_score,
            risk_level=risk_level,
            assessed_by=f"Nurse {self.fake.last_name()}",
            assessment_date=datetime.now() - timedelta(hours=random.randint(1, 24))
        )
    
    def _generate_medications(self, medical_conditions: List[MedicalCondition]) -> List[Medication]:
        """Generate medications based on medical conditions"""
        
        medications = []
        condition_names = [c.name for c in medical_conditions]
        
        # Condition-specific medications
        if 'diabetes_mellitus' in condition_names:
            med_name = random.choice(self.medications['diabetes'])
            medications.append(Medication(
                name=med_name,
                dosage="500mg" if med_name == "Metformin" else "10 units",
                frequency="BID" if med_name == "Metformin" else "AC",
                route="PO" if med_name == "Metformin" else "SubQ",
                start_date=datetime.now() - timedelta(days=random.randint(30, 365)),
                indication="Diabetes management"
            ))
        
        if 'cardiovascular_disease' in condition_names:
            med_name = random.choice(self.medications['blood_pressure'])
            medications.append(Medication(
                name=med_name,
                dosage="5mg",
                frequency="Daily",
                route="PO",
                start_date=datetime.now() - timedelta(days=random.randint(30, 365)),
                indication="Hypertension"
            ))
        
        # Pain medications for high-risk patients
        if random.random() < 0.4:
            med_name = random.choice(self.medications['pain'])
            medications.append(Medication(
                name=med_name,
                dosage="5mg",
                frequency="Q4H PRN",
                route="PO",
                start_date=datetime.now() - timedelta(days=random.randint(1, 30)),
                indication="Pain management"
            ))
        
        return medications
    
    def _generate_vital_signs(self, age: int, medical_conditions: List[MedicalCondition]) -> VitalSigns:
        """Generate realistic vital signs"""
        
        # Base vital signs
        systolic = random.randint(110, 140)
        diastolic = random.randint(60, 90)
        heart_rate = random.randint(60, 100)
        resp_rate = random.randint(12, 20)
        temperature = round(random.uniform(36.1, 37.2), 1)
        o2_sat = random.randint(95, 100)
        pain_score = random.randint(0, 3)
        
        # Adjust for age
        if age > 70:
            systolic += random.randint(10, 30)
            heart_rate += random.randint(5, 15)
        
        # Adjust for conditions
        condition_names = [c.name for c in medical_conditions]
        
        if 'cardiovascular_disease' in condition_names:
            systolic += random.randint(20, 40)
            diastolic += random.randint(10, 20)
        
        if 'chronic_kidney_disease' in condition_names:
            systolic += random.randint(15, 25)
        
        return VitalSigns(
            systolic_bp=min(systolic, 180),
            diastolic_bp=min(diastolic, 110),
            heart_rate=min(heart_rate, 120),
            respiratory_rate=min(resp_rate, 24),
            temperature_celsius=temperature,
            oxygen_saturation=o2_sat,
            pain_score=pain_score,
            recorded_at=datetime.now() - timedelta(hours=random.randint(1, 8))
        )
    
    def _generate_pressure_injuries(self, risk_level: RiskLevel) -> List[Dict[str, Any]]:
        """Generate current pressure injuries based on risk level"""
        
        injuries = []
        
        # Probability of having pressure injuries
        injury_probability = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MODERATE: 0.2,
            RiskLevel.HIGH: 0.4,
            RiskLevel.VERY_HIGH: 0.6
        }
        
        if random.random() < injury_probability[risk_level]:
            num_injuries = random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05])[0]
            
            common_locations = [
                'sacrum', 'heel', 'trochanter', 'ischial_tuberosity',
                'elbow', 'occiput', 'shoulder_blade'
            ]
            
            for _ in range(num_injuries):
                # Higher risk = more severe injuries
                if risk_level == RiskLevel.VERY_HIGH:
                    stage_weights = [0.1, 0.2, 0.3, 0.25, 0.1, 0.03, 0.02]
                elif risk_level == RiskLevel.HIGH:
                    stage_weights = [0.2, 0.35, 0.25, 0.15, 0.05, 0, 0]
                else:
                    stage_weights = [0.4, 0.4, 0.15, 0.05, 0, 0, 0]
                
                stage = random.choices(
                    ['0', '1', '2', '3', '4', 'U', 'DTI'],
                    weights=stage_weights
                )[0]
                
                injury = {
                    'location': random.choice(common_locations),
                    'npuap_stage': stage,
                    'size_length_cm': round(random.uniform(0.5, 8.0), 1),
                    'size_width_cm': round(random.uniform(0.5, 6.0), 1),
                    'depth_cm': round(random.uniform(0.1, 3.0), 1) if stage in ['3', '4'] else 0,
                    'drainage': random.choice(['none', 'minimal', 'moderate', 'heavy']),
                    'odor': random.choice(['none', 'mild', 'strong']),
                    'pain_level': random.randint(0, 8),
                    'identified_date': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                    'treatment_plan': self._generate_treatment_plan(stage)
                }
                
                injuries.append(injury)
        
        return injuries
    
    def _generate_treatment_plan(self, stage: str) -> List[str]:
        """Generate treatment plan based on injury stage"""
        
        treatments = {
            '0': ['Monitor skin', 'Position changes q2h', 'Moisture management'],
            '1': ['Protective dressing', 'Position changes q2h', 'Assess nutrition'],
            '2': ['Hydrocolloid dressing', 'Position changes q2h', 'Nutrition consult'],
            '3': ['Wound assessment', 'Debridement PRN', 'Nutrition support', 'Pressure relief'],
            '4': ['Surgical consult', 'Aggressive debridement', 'Pressure relief', 'Nutrition support'],
            'U': ['Gentle cleansing', 'Assessment for debridement', 'Pressure relief'],
            'DTI': ['Monitor progression', 'Pressure relief', 'Document changes']
        }
        
        return treatments.get(stage, ['Standard wound care'])
    
    async def generate_patient_cohort(self, 
                                     cohort_size: int,
                                     risk_distribution: Dict[RiskLevel, float] = None) -> List[SyntheticPatient]:
        """Generate a cohort of synthetic patients"""
        
        if not risk_distribution:
            risk_distribution = {
                RiskLevel.LOW: 0.4,
                RiskLevel.MODERATE: 0.3,
                RiskLevel.HIGH: 0.2,
                RiskLevel.VERY_HIGH: 0.1
            }
        
        patients = []
        
        for i in range(cohort_size):
            # Select risk level based on distribution
            risk_level = np.random.choice(
                list(risk_distribution.keys()),
                p=list(risk_distribution.values())
            )
            
            try:
                patient = await self.generate_patient(
                    risk_profile=risk_level,
                    include_pressure_injuries=None  # Let it decide randomly
                )
                patients.append(patient)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Generated {i + 1}/{cohort_size} synthetic patients")
                    
            except Exception as e:
                logger.error(f"Failed to generate patient {i + 1}: {e}")
                continue
        
        logger.info(f"Generated cohort of {len(patients)} synthetic patients")
        return patients
    
    def export_patient_to_dict(self, patient: SyntheticPatient) -> Dict[str, Any]:
        """Export patient to dictionary for serialization"""
        
        patient_dict = asdict(patient)
        
        # Convert datetime objects to ISO strings
        for key, value in patient_dict.items():
            if isinstance(value, datetime):
                patient_dict[key] = value.isoformat()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, datetime):
                                item[k] = v.isoformat()
        
        return patient_dict
    
    def get_generation_statistics(self, patients: List[SyntheticPatient]) -> Dict[str, Any]:
        """Get statistics about generated patient cohort"""
        
        if not patients:
            return {}
        
        # Age distribution
        ages = [p.age for p in patients]
        
        # Risk distribution
        risk_counts = {}
        for patient in patients:
            risk_level = patient.braden_score.risk_level.value
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        # Gender distribution
        gender_counts = {}
        for patient in patients:
            gender_counts[patient.gender] = gender_counts.get(patient.gender, 0) + 1
        
        # Pressure injury statistics
        patients_with_injuries = sum(1 for p in patients if p.current_pressure_injuries)
        total_injuries = sum(len(p.current_pressure_injuries) for p in patients)
        
        return {
            'total_patients': len(patients),
            'age_statistics': {
                'mean': np.mean(ages),
                'median': np.median(ages),
                'min': min(ages),
                'max': max(ages)
            },
            'risk_distribution': risk_counts,
            'gender_distribution': gender_counts,
            'pressure_injury_statistics': {
                'patients_with_injuries': patients_with_injuries,
                'total_injuries': total_injuries,
                'injury_rate': patients_with_injuries / len(patients)
            },
            'care_unit_distribution': {
                unit: sum(1 for p in patients if p.care_unit == unit)
                for unit in set(p.care_unit for p in patients)
            }
        }


# Utility functions
def _select_primary_diagnosis(self, medical_conditions: List[MedicalCondition]) -> str:
    """Select primary diagnosis from conditions"""
    if not medical_conditions:
        return "General medical care"
    
    # Select most severe condition as primary
    severity_weights = {'severe': 3, 'moderate': 2, 'mild': 1}
    primary = max(medical_conditions, 
                 key=lambda c: severity_weights.get(c.severity, 1))
    
    return primary.name.replace('_', ' ').title()

def _select_secondary_diagnoses(self, medical_conditions: List[MedicalCondition]) -> List[str]:
    """Select secondary diagnoses"""
    return [c.name.replace('_', ' ').title() for c in medical_conditions[1:4]]

def _generate_allergies(self) -> List[str]:
    """Generate patient allergies"""
    common_allergies = [
        'Penicillin', 'Latex', 'Shellfish', 'Nuts', 'Iodine',
        'Sulfa drugs', 'Adhesive tape', 'Morphine'
    ]
    
    num_allergies = random.choices([0, 1, 2, 3], weights=[0.6, 0.25, 0.1, 0.05])[0]
    return random.sample(common_allergies, num_allergies)

def _determine_mobility_level(self, braden_score: BradenScore) -> MobilityLevel:
    """Determine mobility level from Braden score"""
    mobility_score = braden_score.mobility
    
    if mobility_score == 1:
        return MobilityLevel.COMPLETELY_IMMOBILE
    elif mobility_score == 2:
        return MobilityLevel.VERY_LIMITED
    elif mobility_score == 3:
        return MobilityLevel.SLIGHTLY_LIMITED
    else:
        return MobilityLevel.FULLY_MOBILE

def _select_care_unit(self, risk_level: RiskLevel, medical_conditions: List[MedicalCondition]) -> str:
    """Select appropriate care unit"""
    condition_names = [c.name for c in medical_conditions]
    
    if 'spinal_cord_injury' in condition_names:
        return 'ICU'
    elif risk_level == RiskLevel.VERY_HIGH:
        return random.choice(['ICU', 'Medical'])
    elif risk_level == RiskLevel.HIGH:
        return random.choice(['Medical', 'Surgical'])
    else:
        return random.choice(['Medical', 'Surgical', 'Rehabilitation'])

def _generate_clinical_staff(self) -> Dict[str, str]:
    """Generate clinical staff names"""
    return {
        'physician': f"Dr. {self.fake.last_name()}",
        'nurse': f"{self.fake.first_name()} {self.fake.last_name()}, RN"
    }

def _generate_admission_date(self) -> datetime:
    """Generate admission date"""
    return datetime.now() - timedelta(days=random.randint(1, 14))

def _generate_skin_assessment(self, current_injuries: List[Dict[str, Any]]) -> str:
    """Generate skin assessment notes"""
    if not current_injuries:
        return "Skin intact, no areas of concern noted. Patient education provided on pressure injury prevention."
    
    injury_count = len(current_injuries)
    locations = [injury['location'] for injury in current_injuries]
    
    return f"Patient has {injury_count} pressure injury(ies) noted at {', '.join(locations)}. See individual wound assessments for details. Continuing prevention measures."

def _generate_prevention_measures(self, risk_level: RiskLevel) -> List[str]:
    """Generate prevention measures based on risk level"""
    base_measures = [
        "Assess skin daily",
        "Reposition every 2 hours",
        "Keep skin clean and dry",
        "Use pressure-redistributing surfaces"
    ]
    
    if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        base_measures.extend([
            "Turn every 2 hours around the clock",
            "Nutritional assessment and support",
            "Heel protection devices",
            "Frequent skin assessment"
        ])
    
    if risk_level == RiskLevel.VERY_HIGH:
        base_measures.extend([
            "Specialty bed/mattress",
            "Turn team protocols",
            "Wound care specialist consult"
        ])
    
    return base_measures

# Add missing methods to class
SyntheticPatientGenerator._select_primary_diagnosis = _select_primary_diagnosis
SyntheticPatientGenerator._select_secondary_diagnoses = _select_secondary_diagnoses
SyntheticPatientGenerator._generate_allergies = _generate_allergies
SyntheticPatientGenerator._determine_mobility_level = _determine_mobility_level
SyntheticPatientGenerator._select_care_unit = _select_care_unit
SyntheticPatientGenerator._generate_clinical_staff = _generate_clinical_staff
SyntheticPatientGenerator._generate_admission_date = _generate_admission_date
SyntheticPatientGenerator._generate_skin_assessment = _generate_skin_assessment
SyntheticPatientGenerator._generate_prevention_measures = _generate_prevention_measures

# Singleton instance
synthetic_patient_generator = SyntheticPatientGenerator()

__all__ = [
    'SyntheticPatientGenerator',
    'SyntheticPatient',
    'RiskLevel',
    'MobilityLevel',
    'BradenScore',
    'MedicalCondition',
    'Medication',
    'VitalSigns',
    'synthetic_patient_generator'
]