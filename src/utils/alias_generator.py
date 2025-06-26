"""
Patient Alias Generator Module

Generates secure patient aliases for PHI tokenization.
Converts hospital PHI to safe codenames (Bruce Wayne → Batman).
"""

import hashlib
import secrets
import logging
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class AliasTheme(Enum):
    """Themes for alias generation"""
    SUPERHEROES = "superheroes"
    MYTHOLOGY = "mythology"
    LITERATURE = "literature"
    ASTRONOMY = "astronomy"
    ANIMALS = "animals"


class PatientAliasGenerator:
    """
    Generates secure, memorable aliases for patient PHI tokenization.
    
    Converts hospital PHI (Bruce Wayne) to processing aliases (Batman)
    for secure dual-database architecture.
    """
    
    # Superhero aliases for patient anonymization
    SUPERHERO_ALIASES = [
        "Batman", "Superman", "Wonder Woman", "Flash", "Green Lantern",
        "Aquaman", "Cyborg", "Martian Manhunter", "Green Arrow", "Black Canary",
        "Hawkman", "Hawkgirl", "Atom", "Firestorm", "Zatanna",
        "Captain Marvel", "Supergirl", "Batgirl", "Robin", "Nightwing",
        "Red Robin", "Red Hood", "Batwoman", "Oracle", "Huntress",
        "Question", "Blue Beetle", "Booster Gold", "Steel", "Power Girl",
        "Starfire", "Raven", "Beast Boy", "Cyborg", "Terra",
        "Spider-Man", "Iron Man", "Captain America", "Thor", "Hulk",
        "Black Widow", "Hawkeye", "Falcon", "Winter Soldier", "Ant-Man",
        "Wasp", "Doctor Strange", "Scarlet Witch", "Vision", "Quicksilver",
        "Black Panther", "Captain Marvel", "Star-Lord", "Gamora", "Rocket",
        "Groot", "Drax", "Mantis", "Nebula", "Yondu",
        "Wolverine", "Cyclops", "Storm", "Jean Grey", "Beast",
        "Iceman", "Angel", "Nightcrawler", "Colossus", "Rogue",
        "Jubilee", "Psylocke", "Emma Frost", "Mystique", "Magneto"
    ]
    
    MYTHOLOGY_ALIASES = [
        "Apollo", "Artemis", "Athena", "Zeus", "Hera",
        "Poseidon", "Hades", "Demeter", "Hestia", "Ares",
        "Aphrodite", "Hephaestus", "Hermes", "Dionysus", "Persephone",
        "Thor", "Odin", "Freya", "Loki", "Balder",
        "Frigg", "Heimdall", "Tyr", "Vidar", "Vali"
    ]
    
    LITERATURE_ALIASES = [
        "Sherlock", "Watson", "Moriarty", "Poirot", "Marple",
        "Holmes", "Gatsby", "Atticus", "Hermione", "Gandalf",
        "Aragorn", "Legolas", "Gimli", "Frodo", "Samwise",
        "Bilbo", "Thorin", "Smaug", "Galadriel", "Elrond"
    ]
    
    ASTRONOMY_ALIASES = [
        "Orion", "Cassiopeia", "Andromeda", "Perseus", "Draco",
        "Lyra", "Vega", "Sirius", "Polaris", "Rigel",
        "Betelgeuse", "Aldebaran", "Spica", "Antares", "Canopus",
        "Arcturus", "Capella", "Procyon", "Achernar", "Hadar"
    ]
    
    ANIMAL_ALIASES = [
        "Phoenix", "Dragon", "Eagle", "Wolf", "Lion",
        "Tiger", "Bear", "Falcon", "Hawk", "Raven",
        "Owl", "Fox", "Deer", "Horse", "Dolphin",
        "Whale", "Shark", "Octopus", "Butterfly", "Bee"
    ]
    
    THEME_ALIASES = {
        AliasTheme.SUPERHEROES: SUPERHERO_ALIASES,
        AliasTheme.MYTHOLOGY: MYTHOLOGY_ALIASES,
        AliasTheme.LITERATURE: LITERATURE_ALIASES,
        AliasTheme.ASTRONOMY: ASTRONOMY_ALIASES,
        AliasTheme.ANIMALS: ANIMAL_ALIASES
    }
    
    def __init__(self, theme: AliasTheme = AliasTheme.SUPERHEROES, seed: Optional[str] = None):
        """
        Initialize alias generator
        
        Args:
            theme: Theme for alias generation
            seed: Optional seed for deterministic generation
        """
        self.theme = theme
        self.aliases = self.THEME_ALIASES[theme].copy()
        self.used_aliases: Dict[str, str] = {}  # hospital_mrn -> alias
        self.reverse_mapping: Dict[str, str] = {}  # alias -> hospital_mrn
        
        # For deterministic generation (testing)
        self.seed = seed
        if seed:
            self._deterministic_mode = True
            self._seed_generator = self._create_seeded_generator(seed)
        else:
            self._deterministic_mode = False
        
        logger.info(f"Patient alias generator initialized with {theme.value} theme")
    
    def generate_alias(self, hospital_mrn: str, patient_data: Optional[Dict] = None) -> str:
        """
        Generate secure alias for patient
        
        Args:
            hospital_mrn: Hospital medical record number
            patient_data: Additional patient data for alias generation
            
        Returns:
            Generated alias (e.g., "Batman")
        """
        try:
            # Check if alias already exists
            if hospital_mrn in self.used_aliases:
                existing_alias = self.used_aliases[hospital_mrn]
                logger.info(f"Returning existing alias for {hospital_mrn}: {existing_alias}")
                return existing_alias
            
            # Generate new alias
            if self._deterministic_mode:
                alias = self._generate_deterministic_alias(hospital_mrn, patient_data)
            else:
                alias = self._generate_random_alias(hospital_mrn, patient_data)
            
            # Store mappings
            self.used_aliases[hospital_mrn] = alias
            self.reverse_mapping[alias] = hospital_mrn
            
            logger.info(f"Generated alias for {hospital_mrn}: {alias}")
            return alias
            
        except Exception as e:
            logger.error(f"Failed to generate alias for {hospital_mrn}: {e}")
            # Fallback to hash-based alias
            return self._generate_fallback_alias(hospital_mrn)
    
    def get_hospital_mrn(self, alias: str) -> Optional[str]:
        """
        Get hospital MRN from alias
        
        Args:
            alias: Patient alias
            
        Returns:
            Hospital MRN if found
        """
        return self.reverse_mapping.get(alias)
    
    def get_alias(self, hospital_mrn: str) -> Optional[str]:
        """
        Get alias for hospital MRN
        
        Args:
            hospital_mrn: Hospital medical record number
            
        Returns:
            Alias if found
        """
        return self.used_aliases.get(hospital_mrn)
    
    def is_alias_available(self, alias: str) -> bool:
        """Check if alias is available for use"""
        return alias not in self.reverse_mapping
    
    def get_available_aliases(self) -> List[str]:
        """Get list of available aliases"""
        used_set = set(self.reverse_mapping.keys())
        return [alias for alias in self.aliases if alias not in used_set]
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get alias usage statistics"""
        return {
            'total_aliases': len(self.aliases),
            'used_aliases': len(self.used_aliases),
            'available_aliases': len(self.aliases) - len(self.used_aliases),
            'usage_percentage': round((len(self.used_aliases) / len(self.aliases)) * 100, 2)
        }
    
    def _generate_random_alias(self, hospital_mrn: str, patient_data: Optional[Dict] = None) -> str:
        """Generate random alias from available pool"""
        available_aliases = self.get_available_aliases()
        
        if not available_aliases:
            # All aliases used, generate hash-based fallback
            logger.warning("All aliases exhausted, using fallback generation")
            return self._generate_fallback_alias(hospital_mrn)
        
        # Use secure random selection
        alias = secrets.choice(available_aliases)
        return alias
    
    def _generate_deterministic_alias(self, hospital_mrn: str, patient_data: Optional[Dict] = None) -> str:
        """Generate deterministic alias for testing"""
        available_aliases = self.get_available_aliases()
        
        if not available_aliases:
            return self._generate_fallback_alias(hospital_mrn)
        
        # Use hash for deterministic selection
        hash_value = hashlib.sha256(f"{self.seed}{hospital_mrn}".encode()).hexdigest()
        index = int(hash_value[:8], 16) % len(available_aliases)
        return available_aliases[index]
    
    def _generate_fallback_alias(self, hospital_mrn: str) -> str:
        """Generate fallback alias when pool is exhausted"""
        # Create hash-based alias with theme prefix
        hash_value = hashlib.sha256(hospital_mrn.encode()).hexdigest()[:8]
        theme_prefix = self.theme.value.capitalize()[:4]  # e.g., "Supe" for superheroes
        return f"{theme_prefix}{hash_value}"
    
    def _create_seeded_generator(self, seed: str):
        """Create seeded random generator for testing"""
        import random
        generator = random.Random(seed)
        return generator
    
    def clear_mappings(self):
        """Clear all alias mappings (for testing)"""
        self.used_aliases.clear()
        self.reverse_mapping.clear()
        logger.info("Alias mappings cleared")
    
    def export_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Export alias mappings for backup/audit
        
        Returns:
            Dictionary with mapping data
        """
        return {
            'hospital_to_alias': self.used_aliases.copy(),
            'alias_to_hospital': self.reverse_mapping.copy(),
            'theme': self.theme.value,
            'stats': self.get_usage_stats()
        }
    
    def import_mappings(self, mappings: Dict[str, Dict[str, str]]):
        """
        Import alias mappings from backup
        
        Args:
            mappings: Mapping data from export_mappings()
        """
        try:
            self.used_aliases = mappings.get('hospital_to_alias', {})
            self.reverse_mapping = mappings.get('alias_to_hospital', {})
            logger.info(f"Imported {len(self.used_aliases)} alias mappings")
        except Exception as e:
            logger.error(f"Failed to import mappings: {e}")
            raise
    
    def validate_alias_integrity(self) -> bool:
        """
        Validate alias mapping integrity
        
        Returns:
            True if mappings are consistent
        """
        try:
            # Check bidirectional mapping consistency
            for mrn, alias in self.used_aliases.items():
                if self.reverse_mapping.get(alias) != mrn:
                    logger.error(f"Mapping inconsistency: {mrn} -> {alias}")
                    return False
            
            for alias, mrn in self.reverse_mapping.items():
                if self.used_aliases.get(mrn) != alias:
                    logger.error(f"Reverse mapping inconsistency: {alias} -> {mrn}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Alias integrity validation failed: {e}")
            return False