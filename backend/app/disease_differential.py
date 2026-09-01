"""Avian Disease Differential Diagnosis & Clinical Guidance Module.
Evaluates CNN risk scores, acoustic biomarkers, and spectral patterns to generate
evidence-grounded disease differentials and on-farm biosecurity checklists.
"""

from typing import Dict, Any, List


DISEASE_KNOWLEDGE_BASE = [
    {
        "id": "ibv",
        "name": "Infectious Bronchitis (IBV)",
        "pathogen": "Avian Coronavirus (Gammacoronavirus)",
        "transmission": "Airborne aerosol, direct contact, contaminated equipment",
        "acoustic_hallmark": "Wet tracheal rales, rapid flock-wide snicking, coughing bursts",
        "typical_freq_range": "2000 Hz – 3800 Hz",
        "is_notifiable": False,
        "key_symptoms": [
            "Excessive nasal discharge & watery/foamy eyes",
            "Severe drop in egg production with wrinkled, soft, or misshapen shells",
            "Lethargy, huddling near heat sources, ruffled feathers",
            "Coughing and head-shaking to dislodge tracheal mucus",
        ],
        "biosecurity_actions": [
            "Immediately isolate birds exhibiting active rales/snicks into a warm quarantine shed.",
            "Increase coop ventilation while strictly avoiding cold drafts.",
            "Add supportive electrolytes and vitamin A/D supplements to drinking water.",
            "Disinfect water fonts and feeders twice daily with avian-safe Virkon/iodine solutions.",
        ],
    },
    {
        "id": "crd",
        "name": "Chronic Respiratory Disease (CRD)",
        "pathogen": "Mycoplasma gallisepticum (MG)",
        "transmission": "Egg transmission (transovarian), direct bird contact, aerosol",
        "acoustic_hallmark": "Persistent dry wheezing, nocturnal rattling during roosting",
        "typical_freq_range": "2400 Hz – 4200 Hz",
        "is_notifiable": False,
        "key_symptoms": [
            "Persistent nighttime wheezing and rattling in the roost",
            "Swollen facial sinuses (swollen eyes with cheesy exudate)",
            "Foamy discharge at the corner of the eye (conjunctivitis)",
            "Slow weight gain, poor feed conversion, and emaciation",
        ],
        "biosecurity_actions": [
            "Quarantine new stock for 30 days before introducing to established flock.",
            "Reduce stocking density and improve ammonia ventilation (keep ammonia <15 ppm).",
            "Consult a poultry veterinarian for targeted antibiotic therapy (e.g. tylosin/oxytetracycline).",
            "Do not use infected stock for breeding due to vertical egg transmission.",
        ],
    },
    {
        "id": "coryza",
        "name": "Infectious Coryza",
        "pathogen": "Avibacterium paragallinarum (Bacterium)",
        "transmission": "Direct bird-to-bird contact, contaminated drinking water",
        "acoustic_hallmark": "Labored snoring respiration, nasal snicking with muffled harmonics",
        "typical_freq_range": "1600 Hz – 3200 Hz",
        "is_notifiable": False,
        "key_symptoms": [
            "Characteristic acute facial swelling (edema of face & wattles)",
            "Foul-smelling, sticky nasal discharge adhering to beak and feathers",
            "Partially or fully glued-shut eyelids from thick discharge",
            "Sudden sharp drop in water and feed consumption",
        ],
        "biosecurity_actions": [
            "Immediately segregate swollen-faced birds; sanitize all communal waterers.",
            "Administer clean, chlorinated drinking water to break the transmission loop.",
            "Thoroughly clean crusted nostrils and eyes with warm saline wash.",
            "Depopulate or strictly isolate chronic carriers, as recovered birds remain lifetime reservoirs.",
        ],
    },
    {
        "id": "ndv",
        "name": "Newcastle Disease (Respiratory / Viscerotropic)",
        "pathogen": "Avian Paramyxovirus Serotype 1 (APMV-1)",
        "transmission": "Highly contagious via droppings, aerosol, wild birds, boots",
        "acoustic_hallmark": "High-distress gasping chirps, extreme hoarseness, severe vocal exhaustion",
        "typical_freq_range": "2800 Hz – 5000 Hz",
        "is_notifiable": True,
        "key_symptoms": [
            "Gasping with open beak, severe respiratory distress",
            "Bright green watery diarrhea and sudden lethargy",
            "Neurological signs in later stages: twisted neck (torticollis), circling, paralysis",
            "High flock mortality within 48–72 hours",
        ],
        "biosecurity_actions": [
            "CRITICAL: Report suspected outbreaks immediately to your local government animal health officer.",
            "Implement complete farm lockdown: no movement of birds, eggs, manure, or equipment.",
            "Maintain strict boot disinfection baths with virucidal disinfectants at coop entrances.",
            "Keep domestic poultry strictly isolated from wild waterfowl and migratory birds.",
        ],
    },
    {
        "id": "aspergillosis",
        "name": "Aspergillosis (Brooder Pneumonia)",
        "pathogen": "Aspergillus fumigatus (Fungus/Mold)",
        "transmission": "Inhalation of mold spores from damp bedding, moldy feed, or incubators",
        "acoustic_hallmark": "Silent/faint open-mouth gasping with dry inspiratory clicks",
        "typical_freq_range": "3000 Hz – 4800 Hz",
        "is_notifiable": False,
        "key_symptoms": [
            "Rapid, silent open-mouth breathing ('silent gaspers') without gurgling rales",
            "Bluish discoloration of the comb/skin (cyanosis from lack of oxygen)",
            "Absence of nasal discharge or foul odor (distinguishes from Coryza/IBV)",
            "Increased thirst accompanied by severe loss of appetite and somnolence",
        ],
        "biosecurity_actions": [
            "Immediately strip and replace all damp, caked, or moldy coop litter with clean dry pine shavings.",
            "Inspect feed bins and discard any caked or mold-smelling grain or pellets.",
            "Improve air exchange to reduce airborne fungal spore concentrations.",
            "Disinfect brooding facilities with approved antifungal agents before placing new chicks.",
        ],
    },
]


def generate_disease_differential(
    risk_score: float,
    biomarkers: Dict[str, float],
    status: str = "calibrated",
) -> Dict[str, Any]:
    """
    Computes evidence-grounded differential diagnosis probabilities, acoustic rationales,
    and priority physical inspection checklists based on flock acoustic biomarkers.
    """
    rale_pct = biomarkers.get("rale_intensity_pct", 20.0)
    centroid = biomarkers.get("spectral_centroid_hz", 1500.0)
    density = biomarkers.get("event_density_pct", 25.0)

    # If the flock is healthy, all respiratory diseases have low likelihood
    if risk_score < 35.0:
        differentials = []
        for d in DISEASE_KNOWLEDGE_BASE:
            differentials.append({
                "disease_id": d["id"],
                "name": d["name"],
                "pathogen": d["pathogen"],
                "likelihood": "Low",
                "probability_pct": max(2, int(risk_score * 0.2)),
                "acoustic_rationale": "Acoustic parameters within healthy baseline ranges. No characteristic pathognomonic rales or wheezing detected.",
                "is_notifiable": d["is_notifiable"],
                "key_symptoms": d["key_symptoms"][:2],
                "biosecurity_actions": d["biosecurity_actions"][:2],
            })

        return {
            "flock_clinical_status": "Healthy / Normal Respiratory Profile",
            "primary_concern": "No immediate respiratory pathology indicated.",
            "differentials": differentials,
            "overall_biosecurity_advice": [
                "Maintain clean, dry bedding and ensure adequate fresh air exchange.",
                "Continue routine acoustic monitoring once or twice weekly.",
                "Ensure clean, chlorinated drinking water is available at all times.",
            ],
        }

    # Otherwise, score each differential against biomarker signatures
    differentials = []

    # 1. Infectious Bronchitis (IBV) score: Driven by high rale energy and high event density
    ibv_score = min(95, int(risk_score * 0.65 + (rale_pct / 100.0) * 25.0 + (density / 100.0) * 15.0))
    differentials.append({
        "disease_id": "ibv",
        "name": "Infectious Bronchitis (IBV)",
        "pathogen": "Avian Coronavirus",
        "likelihood": "High" if ibv_score >= 70 else ("Moderate" if ibv_score >= 45 else "Possible"),
        "probability_pct": ibv_score,
        "acoustic_rationale": f"High rale/wheeze energy ({rale_pct:.1f}%) and repeated coughing bursts ({density:.1f}% density) strongly correspond to wet tracheal exudate characteristic of IBV.",
        "is_notifiable": False,
        "key_symptoms": DISEASE_KNOWLEDGE_BASE[0]["key_symptoms"],
        "biosecurity_actions": DISEASE_KNOWLEDGE_BASE[0]["biosecurity_actions"],
    })

    # 2. Chronic Respiratory Disease (CRD / Mycoplasma) score: Driven by high centroid and sustained rales
    crd_score = min(92, int(risk_score * 0.60 + (centroid / 3000.0) * 25.0 + (rale_pct / 100.0) * 15.0))
    differentials.append({
        "disease_id": "crd",
        "name": "Chronic Respiratory Disease (CRD)",
        "pathogen": "Mycoplasma gallisepticum",
        "likelihood": "High" if crd_score >= 70 else ("Moderate" if crd_score >= 45 else "Possible"),
        "probability_pct": crd_score,
        "acoustic_rationale": f"Elevated spectral centroid ({centroid:.0f} Hz) and persistent wheezing harmonics reflect chronic upper respiratory rattling and air sacculitis.",
        "is_notifiable": False,
        "key_symptoms": DISEASE_KNOWLEDGE_BASE[1]["key_symptoms"],
        "biosecurity_actions": DISEASE_KNOWLEDGE_BASE[1]["biosecurity_actions"],
    })

    # 3. Infectious Coryza score: Moderate rales with harmonic muffling
    coryza_score = min(88, int(risk_score * 0.50 + (rale_pct / 100.0) * 20.0 + (35 if risk_score > 60 else 10)))
    differentials.append({
        "disease_id": "coryza",
        "name": "Infectious Coryza",
        "pathogen": "Avibacterium paragallinarum",
        "likelihood": "High" if coryza_score >= 70 else ("Moderate" if coryza_score >= 45 else "Possible"),
        "probability_pct": coryza_score,
        "acoustic_rationale": "Muffled acoustic harmonics and nasal snicking sounds indicate acute nasal passage blockage and sinus inflammation.",
        "is_notifiable": False,
        "key_symptoms": DISEASE_KNOWLEDGE_BASE[2]["key_symptoms"],
        "biosecurity_actions": DISEASE_KNOWLEDGE_BASE[2]["biosecurity_actions"],
    })

    # 4. Newcastle Disease (NDV - Respiratory) score: High distress, severe centroid shift
    ndv_score = min(75, int(risk_score * 0.45 + (centroid / 3500.0) * 30.0))
    differentials.append({
        "disease_id": "ndv",
        "name": "Newcastle Disease (Respiratory Form)",
        "pathogen": "Avian Paramyxovirus 1 (APMV-1)",
        "likelihood": "Moderate" if (ndv_score >= 50 and risk_score > 75) else "Possible",
        "probability_pct": ndv_score,
        "acoustic_rationale": "Extreme vocal fragmentation and high-frequency gasping distress patterns. Requires clinical differentiation from IBV.",
        "is_notifiable": True,
        "key_symptoms": DISEASE_KNOWLEDGE_BASE[3]["key_symptoms"],
        "biosecurity_actions": DISEASE_KNOWLEDGE_BASE[3]["biosecurity_actions"],
    })

    # 5. Aspergillosis score: High spectral flatness and high frequencies
    asp_score = min(70, int(risk_score * 0.40 + (centroid / 3500.0) * 20.0 + (biomarkers.get("spectral_flatness", 0.01) * 200.0)))
    differentials.append({
        "disease_id": "aspergillosis",
        "name": "Aspergillosis (Brooder Pneumonia)",
        "pathogen": "Aspergillus fumigatus",
        "likelihood": "Moderate" if asp_score >= 50 else "Possible",
        "probability_pct": asp_score,
        "acoustic_rationale": "High-frequency dry inspiratory clicks with reduced overall vocal volume consistent with lower respiratory mycotic granulomas.",
        "is_notifiable": False,
        "key_symptoms": DISEASE_KNOWLEDGE_BASE[4]["key_symptoms"],
        "biosecurity_actions": DISEASE_KNOWLEDGE_BASE[4]["biosecurity_actions"],
    })

    # Sort differentials by probability percentage descending
    differentials.sort(key=lambda x: x["probability_pct"], reverse=True)

    return {
        "flock_clinical_status": "Active Respiratory Distress Detected",
        "primary_concern": f"Acoustic profile indicates elevated risk predominantly consistent with {differentials[0]['name']}.",
        "differentials": differentials,
        "overall_biosecurity_advice": [
            "Immediately quarantine birds exhibiting active audible wheezing or nasal discharge.",
            "Perform physical head, eye, and nasal inspections on representative birds using the symptom checklist below.",
            "Contact a licensed avian veterinarian or agricultural extension officer for diagnostic swabbing.",
            "Sanitize all communal water lines and increase dry airflow through the coop.",
        ],
    }
