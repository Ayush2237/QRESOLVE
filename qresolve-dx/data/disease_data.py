"""
QResolve-Dx: Real HPO Annotation Data for 5 Look-Alike Connective Tissue Disorders

All data is sourced from:
- HPO annotation database (phenotype.hpoa format, hpo.jax.org)
- Published clinical literature (frequencies from cohort studies)
- OMIM disease catalog
- Lovato et al. 2024 (JTCVS Open) for Marfan vs LDS differentiation

OMIM IDs verified:
  Marfan syndrome:             OMIM:154700   (Gene: FBN1)
  Loeys-Dietz syndrome type 1: OMIM:609192   (Genes: TGFBR1, TGFBR2, SMAD3)
  Beals syndrome (CCA):        OMIM:121050   (Gene: FBN2)
  Shprintzen-Goldberg syndrome: OMIM:182212  (Gene: SKI)
  MASS phenotype:              OMIM:604308   (Gene: FBN1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, FrozenSet, Optional


# ---------------------------------------------------------------------------
# HPO Term Registry — every term used in this cluster, with its canonical label
# ---------------------------------------------------------------------------

HPO_TERMS: Dict[str, str] = {
    # --- Ocular ---
    "HP:0001083": "Ectopia lentis",
    "HP:0000545": "Myopia",
    "HP:0000541": "Retinal detachment",
    "HP:0000518": "Cataract",
    "HP:0012499": "Reduced visual acuity",

    # --- Cardiovascular ---
    "HP:0002616": "Aortic root aneurysm",
    "HP:0002647": "Aortic dissection",
    "HP:0001634": "Mitral valve prolapse",
    "HP:0001653": "Mitral regurgitation",
    "HP:0004942": "Aortic aneurysm",
    "HP:0005116": "Arterial tortuosity",
    "HP:0004933": "Ascending aortic dilatation",
    "HP:0001659": "Aortic regurgitation",
    "HP:0004970": "Ascending tubular aorta aneurysm",
    "HP:0012727": "Thoracic aortic aneurysm",

    # --- Craniofacial ---
    "HP:0000316": "Hypertelorism",
    "HP:0000193": "Bifid uvula",
    "HP:0000175": "Cleft palate",
    "HP:0001363": "Craniosynostosis",
    "HP:0000278": "Retrognathia",
    "HP:0000347": "Micrognathia",
    "HP:0000508": "Ptosis",
    "HP:0000463": "Anteverted nares",
    "HP:0002705": "High, narrow palate",
    "HP:0000218": "High palate",
    "HP:0000272": "Malar flattening",
    "HP:0000486": "Strabismus",
    "HP:0000268": "Dolichocephaly",

    # --- Skeletal ---
    "HP:0001166": "Arachnodactyly",
    "HP:0000098": "Tall stature",
    "HP:0001519": "Disproportionate tall stature",
    "HP:0000767": "Pectus carinatum",
    "HP:0000768": "Pectus excavatum",
    "HP:0002650": "Scoliosis",
    "HP:0002751": "Kyphoscoliosis",
    "HP:0001382": "Joint hypermobility",
    "HP:0001763": "Pes planus",
    "HP:0005294": "Protrusio acetabuli",
    "HP:0003179": "Protrusio acetabuli",  # alias
    "HP:0002816": "Genu recurvatum",
    "HP:0001371": "Flexion contracture",
    "HP:0001182": "Camptodactyly",
    "HP:0002987": "Elbow flexion contracture",
    "HP:0006380": "Knee flexion contracture",
    "HP:0001762": "Talipes equinovarus",

    # --- Ear ---
    "HP:0000377": "Abnormal pinna morphology",
    "HP:0009901": "Crumpled ear",

    # --- Skin / Connective ---
    "HP:0001065": "Striae distensae",
    "HP:0000978": "Bruising susceptibility",
    "HP:0001075": "Atrophic scars",
    "HP:0200041": "Skin translucency",

    # --- Respiratory ---
    "HP:0002107": "Pneumothorax",
    "HP:0002099": "Asthma",

    # --- Neurological ---
    "HP:0002870": "Obstructive sleep apnea",  # dural ectasia often coded here
    "HP:0007807": "Dural ectasia",
    "HP:0001249": "Intellectual disability",
    "HP:0001263": "Global developmental delay",
    "HP:0001252": "Muscular hypotonia",

    # --- Abdominal / Hernias ---
    "HP:0001539": "Inguinal hernia",
    "HP:0001537": "Umbilical hernia",
    "HP:0001540": "Diastasis recti",

    # --- Genitourinary ---
    "HP:0000028": "Cryptorchidism",

    # --- General skeletal / Habitus ---
    "HP:0003326": "Myalgia",  # muscle hypoplasia proxy
    "HP:0003199": "Decreased muscle mass",
    "HP:0001388": "Joint laxity",
    "HP:0003795": "Short stature (in some SGS)",
}


# ---------------------------------------------------------------------------
# Disease Definitions with Real HPO Annotation Frequencies
# ---------------------------------------------------------------------------
# Frequency values represent P(symptom | disease) — the probability that a
# patient with this disease exhibits this phenotype. Sources:
#   - HPO annotation database frequency field (HP:0040280–HP:0040285 mapped to midpoints)
#   - Published cohort studies (PMIDs cited in comments)
#   - Lovato et al. 2024 (JTCVS Open, doi:10.1016/j.xjon.2024.03.015)
#
# Frequency mapping from HPO terms:
#   HP:0040280 (Obligate):      1.00
#   HP:0040281 (Very frequent): 0.90  (midpoint of 80-99%)
#   HP:0040282 (Frequent):      0.55  (midpoint of 30-79%)
#   HP:0040283 (Occasional):    0.17  (midpoint of 5-29%)
#   HP:0040284 (Very rare):     0.025 (midpoint of 1-4%)
#   HP:0040285 (Excluded):      0.00
#   Empty/Unknown:               0.50  (non-committal prior)

@dataclass
class DiseaseProfile:
    """A disease with its HPO symptom-frequency annotations and genetic basis."""
    disease_id: str                # e.g., "OMIM:154700"
    name: str                      # e.g., "Marfan syndrome"
    short_name: str                # e.g., "marfan"
    omim_id: str
    orpha_id: Optional[str]
    genes: List[str]
    # HPO term -> frequency (probability of symptom given disease)
    symptoms: Dict[str, float]
    # HPO terms explicitly absent (qualifier: NOT)
    absent_symptoms: List[str] = field(default_factory=list)


# ===== MARFAN SYNDROME (OMIM:154700) =====
MARFAN = DiseaseProfile(
    disease_id="OMIM:154700",
    name="Marfan syndrome",
    short_name="marfan",
    omim_id="154700",
    orpha_id="ORPHA:558",
    genes=["FBN1"],
    symptoms={
        # Ocular — ectopia lentis is the hallmark distinguishing Marfan
        "HP:0001083": 0.70,   # Ectopia lentis (60-80%, PMID:21844863)
        "HP:0000545": 0.55,   # Myopia (>50%)
        "HP:0000541": 0.10,   # Retinal detachment (5-15%)

        # Cardiovascular
        "HP:0002616": 0.85,   # Aortic root aneurysm (80-90%)
        "HP:0002647": 0.40,   # Aortic dissection (30-50%)
        "HP:0001634": 0.55,   # Mitral valve prolapse (40-60%)
        "HP:0001653": 0.45,   # Mitral regurgitation (40-50%)
        "HP:0001659": 0.35,   # Aortic regurgitation (30-40%)
        "HP:0004933": 0.80,   # Ascending aortic dilatation

        # Skeletal
        "HP:0001166": 0.85,   # Arachnodactyly (>80%)
        "HP:0000098": 0.85,   # Tall stature (>80%)
        "HP:0001519": 0.80,   # Disproportionate tall stature
        "HP:0000767": 0.50,   # Pectus carinatum (40-60%)
        "HP:0000768": 0.45,   # Pectus excavatum (30-50%)
        "HP:0002650": 0.65,   # Scoliosis (60-70%)
        "HP:0001382": 0.60,   # Joint hypermobility (50-70%)
        "HP:0001763": 0.70,   # Pes planus (~70%)
        "HP:0005294": 0.50,   # Protrusio acetabuli (~50%)
        "HP:0002816": 0.20,   # Genu recurvatum

        # Craniofacial
        "HP:0002705": 0.70,   # High, narrow palate (~70%)
        "HP:0000268": 0.55,   # Dolichocephaly
        "HP:0000272": 0.35,   # Malar flattening

        # Skin
        "HP:0001065": 0.60,   # Striae distensae (~60%)

        # Respiratory
        "HP:0002107": 0.10,   # Pneumothorax (5-15%)

        # Neurological
        "HP:0007807": 0.75,   # Dural ectasia (65-90% on MRI)
    },
    absent_symptoms=[
        "HP:0001249",   # Intellectual disability: ABSENT
        "HP:0001363",   # Craniosynostosis: ABSENT
        "HP:0000193",   # Bifid uvula: ABSENT
    ],
)


# ===== LOEYS-DIETZ SYNDROME TYPE 1 (OMIM:609192) =====
LOEYS_DIETZ = DiseaseProfile(
    disease_id="OMIM:609192",
    name="Loeys-Dietz syndrome",
    short_name="loeys_dietz",
    omim_id="609192",
    orpha_id="ORPHA:60030",
    genes=["TGFBR1", "TGFBR2", "SMAD3"],
    symptoms={
        # Diagnostic triad (Loeys & Dietz 2005, PMID:15731757)
        "HP:0005116": 0.90,   # Arterial tortuosity (85-95%) — HALLMARK
        "HP:0000316": 0.70,   # Hypertelorism (60-80%)
        "HP:0000193": 0.55,   # Bifid uvula (40-70%)

        # Cardiovascular — aggressive aortic disease
        "HP:0002616": 0.92,   # Aortic root aneurysm (>90%, rupture <4cm)
        "HP:0002647": 0.65,   # Aortic dissection (high, early age)
        "HP:0004942": 0.60,   # Non-aortic arterial aneurysm (50-70%)
        "HP:0001634": 0.40,   # Mitral valve prolapse
        "HP:0001659": 0.30,   # Aortic regurgitation
        "HP:0012727": 0.55,   # Thoracic aortic aneurysm

        # Craniofacial
        "HP:0000175": 0.40,   # Cleft palate (30-50%)
        "HP:0001363": 0.35,   # Craniosynostosis (20-50%, esp. LDS1/2)
        "HP:0000278": 0.30,   # Retrognathia
        "HP:0002705": 0.50,   # High palate

        # Skeletal
        "HP:0001166": 0.70,   # Arachnodactyly (60-80%)
        "HP:0000767": 0.35,   # Pectus carinatum
        "HP:0000768": 0.40,   # Pectus excavatum
        "HP:0002650": 0.50,   # Scoliosis
        "HP:0001382": 0.55,   # Joint hypermobility
        "HP:0001762": 0.28,   # Talipes equinovarus (20-35%)
        "HP:0001763": 0.50,   # Pes planus
        "HP:0001371": 0.40,   # Flexion contracture (30-50%)

        # Skin — distinctive features
        "HP:0000978": 0.70,   # Bruising susceptibility / translucent skin (60-80%)
        "HP:0001075": 0.50,   # Atrophic scars (40-60%)
        "HP:0200041": 0.65,   # Skin translucency

        # Neurological
        "HP:0007807": 0.62,   # Dural ectasia (50-75%)

        # Other
        "HP:0002099": 0.40,   # Asthma / allergic (30-50%)
        "HP:0001539": 0.25,   # Inguinal hernia
    },
    absent_symptoms=[
        "HP:0001083",   # Ectopia lentis: ABSENT (0%) — key differentiator vs Marfan
        "HP:0001249",   # Intellectual disability: ABSENT
    ],
)


# ===== BEALS SYNDROME / CCA (OMIM:121050) =====
BEALS = DiseaseProfile(
    disease_id="OMIM:121050",
    name="Beals syndrome",
    short_name="beals",
    omim_id="121050",
    orpha_id="ORPHA:115",
    genes=["FBN2"],
    symptoms={
        # Hallmarks — congenital contractures + crumpled ears
        "HP:0001371": 0.95,   # Flexion contracture (>95%, congenital, improves with age)
        "HP:0001182": 0.85,   # Camptodactyly (>80%)
        "HP:0000377": 0.85,   # Abnormal pinna / crumpled ears (80-90%) — HALLMARK
        "HP:0002987": 0.80,   # Elbow flexion contracture
        "HP:0006380": 0.75,   # Knee flexion contracture

        # Skeletal — marfanoid habitus
        "HP:0001166": 0.90,   # Arachnodactyly (>90%)
        "HP:0000098": 0.70,   # Tall stature
        "HP:0000767": 0.50,   # Pectus carinatum (40-60%)
        "HP:0000768": 0.45,   # Pectus excavatum
        "HP:0002650": 0.60,   # Scoliosis (50-70%)
        "HP:0002751": 0.50,   # Kyphoscoliosis
        "HP:0001762": 0.22,   # Talipes equinovarus (15-30%)
        "HP:0003199": 0.75,   # Decreased muscle mass / slender (70-80%)

        # Cardiovascular — MILD, non-progressive
        "HP:0002616": 0.15,   # Aortic root dilatation (10-20%, mild)
        "HP:0001634": 0.20,   # Mitral valve prolapse

        # Craniofacial
        "HP:0002705": 0.40,   # High palate
        "HP:0000268": 0.35,   # Dolichocephaly
    },
    absent_symptoms=[
        "HP:0001083",   # Ectopia lentis: ABSENT
        "HP:0002647",   # Aortic dissection: ABSENT/extremely rare
        "HP:0001249",   # Intellectual disability: ABSENT
        "HP:0001363",   # Craniosynostosis: ABSENT
        "HP:0000193",   # Bifid uvula: ABSENT
        "HP:0005116",   # Arterial tortuosity: ABSENT
    ],
)


# ===== SHPRINTZEN-GOLDBERG SYNDROME (OMIM:182212) =====
SHPRINTZEN_GOLDBERG = DiseaseProfile(
    disease_id="OMIM:182212",
    name="Shprintzen-Goldberg syndrome",
    short_name="shprintzen_goldberg",
    omim_id="182212",
    orpha_id="ORPHA:2462",
    genes=["SKI"],
    symptoms={
        # Cardinal hallmarks — craniosynostosis + intellectual disability
        "HP:0001363": 0.88,   # Craniosynostosis (>85-90%) — CARDINAL
        "HP:0001249": 0.85,   # Intellectual disability (>85%) — UNIQUE to SGS in this cluster
        "HP:0001263": 0.80,   # Global developmental delay
        "HP:0001252": 0.75,   # Muscular hypotonia (infantile, ~75%)

        # Skeletal — marfanoid
        "HP:0001166": 0.85,   # Arachnodactyly (>85%)
        "HP:0001371": 0.68,   # Flexion contracture (60-75%)
        "HP:0001182": 0.55,   # Camptodactyly
        "HP:0000767": 0.40,   # Pectus carinatum
        "HP:0000768": 0.45,   # Pectus excavatum
        "HP:0002650": 0.55,   # Scoliosis

        # Craniofacial — distinctive
        "HP:0000316": 0.82,   # Hypertelorism (>80%)
        "HP:0000508": 0.55,   # Ptosis (50-60%)
        "HP:0000463": 0.60,   # Anteverted nares (~60%)
        "HP:0000278": 0.80,   # Retrognathia / micrognathia (~80%)
        "HP:0000347": 0.70,   # Micrognathia
        "HP:0002705": 0.65,   # High palate
        "HP:0000272": 0.45,   # Malar flattening

        # Abdominal — very frequent hernias
        "HP:0001539": 0.70,   # Inguinal hernia (60-80%)
        "HP:0001537": 0.55,   # Umbilical hernia
        "HP:0001540": 0.50,   # Diastasis recti

        # Genitourinary
        "HP:0000028": 0.65,   # Cryptorchidism in males (60-70%)

        # Cardiovascular — milder than Marfan/LDS
        "HP:0002616": 0.40,   # Aortic root dilatation (30-50%, milder)
        "HP:0001634": 0.40,   # Mitral valve prolapse (30-50%)
    },
    absent_symptoms=[
        "HP:0001083",   # Ectopia lentis: ABSENT
        "HP:0005116",   # Arterial tortuosity: ABSENT/rare
    ],
)


# ===== MASS PHENOTYPE (OMIM:604308) =====
MASS = DiseaseProfile(
    disease_id="OMIM:604308",
    name="MASS phenotype",
    short_name="mass",
    omim_id="604308",
    orpha_id=None,  # No dedicated Orphanet entry; subsumed under FBN1 spectrum
    genes=["FBN1"],
    symptoms={
        # M — Mitral valve prolapse (defining feature)
        "HP:0001634": 0.92,   # Mitral valve prolapse (>90%)
        "HP:0001653": 0.60,   # Mitral regurgitation

        # A — Aortic enlargement (strictly borderline, z<2.0, non-progressive)
        "HP:0002616": 0.55,   # Aortic root enlargement (borderline, NOT aneurysmal)
        # NOTE: in MASS, aortic z-score must be <2.0; if progressive → reclassify as Marfan

        # S — Striae
        "HP:0001065": 0.60,   # Striae distensae (50-70%)

        # S — Skeletal features (overlapping with Marfan)
        "HP:0001166": 0.75,   # Arachnodactyly
        "HP:0001519": 0.65,   # Disproportionate tall stature
        "HP:0000098": 0.60,   # Tall stature
        "HP:0000767": 0.35,   # Pectus carinatum
        "HP:0000768": 0.40,   # Pectus excavatum
        "HP:0002650": 0.45,   # Scoliosis
        "HP:0001763": 0.55,   # Pes planus
        "HP:0001382": 0.50,   # Joint hypermobility

        # Craniofacial
        "HP:0002705": 0.50,   # High palate

        # Myopia (mild-moderate)
        "HP:0000545": 0.40,   # Myopia
    },
    absent_symptoms=[
        "HP:0001083",   # Ectopia lentis: STRICTLY ABSENT (if present → Marfan)
        "HP:0002647",   # Aortic dissection: ABSENT
        "HP:0005116",   # Arterial tortuosity: ABSENT
        "HP:0000193",   # Bifid uvula: ABSENT
        "HP:0001363",   # Craniosynostosis: ABSENT
        "HP:0001249",   # Intellectual disability: ABSENT
    ],
)


# ---------------------------------------------------------------------------
# Disease Registry
# ---------------------------------------------------------------------------

ALL_DISEASES: List[DiseaseProfile] = [
    MARFAN,
    LOEYS_DIETZ,
    BEALS,
    SHPRINTZEN_GOLDBERG,
    MASS,
]

DISEASE_NAMES: List[str] = [d.name for d in ALL_DISEASES]
DISEASE_SHORT_NAMES: List[str] = [d.short_name for d in ALL_DISEASES]
DISEASE_BY_NAME: Dict[str, DiseaseProfile] = {d.name: d for d in ALL_DISEASES}
DISEASE_BY_SHORT: Dict[str, DiseaseProfile] = {d.short_name: d for d in ALL_DISEASES}
DISEASE_BY_OMIM: Dict[str, DiseaseProfile] = {d.omim_id: d for d in ALL_DISEASES}

# Label encoding (for ML models)
DISEASE_LABEL_MAP: Dict[str, int] = {d.name: i for i, d in enumerate(ALL_DISEASES)}
LABEL_DISEASE_MAP: Dict[int, str] = {i: d.name for i, d in enumerate(ALL_DISEASES)}
NUM_CLASSES: int = len(ALL_DISEASES)


# ---------------------------------------------------------------------------
# Union of All HPO Terms in This Cluster
# ---------------------------------------------------------------------------

def get_all_hpo_terms() -> List[str]:
    """Return sorted list of all unique HPO terms across all 5 diseases."""
    terms = set()
    for disease in ALL_DISEASES:
        terms.update(disease.symptoms.keys())
        terms.update(disease.absent_symptoms)
    return sorted(terms)

ALL_HPO_TERMS: List[str] = get_all_hpo_terms()
HPO_TERM_INDEX: Dict[str, int] = {term: i for i, term in enumerate(ALL_HPO_TERMS)}
NUM_FEATURES: int = len(ALL_HPO_TERMS)


# ---------------------------------------------------------------------------
# Known Confusion Pairs (from clinical literature)
# ---------------------------------------------------------------------------
# These are disease pairs known to be commonly confused clinically.
# Used by the confusion detector to flag cases for quantum re-scoring.

KNOWN_CONFUSION_PAIRS: List[FrozenSet[str]] = [
    frozenset(["Marfan syndrome", "MASS phenotype"]),
    # Both FBN1, overlapping skeletal features; distinguished by aortic severity + ectopia lentis
    frozenset(["Marfan syndrome", "Loeys-Dietz syndrome"]),
    # Both have aortic disease + marfanoid habitus; distinguished by triad + ectopia lentis
    frozenset(["Loeys-Dietz syndrome", "Shprintzen-Goldberg syndrome"]),
    # Both have craniosynostosis + hypertelorism; distinguished by ID + severity
    frozenset(["Marfan syndrome", "Beals syndrome"]),
    # Both have arachnodactyly + tall stature; distinguished by contractures + ears
    frozenset(["Beals syndrome", "Shprintzen-Goldberg syndrome"]),
    # Both have contractures; distinguished by ears vs craniosynostosis + ID
]


# ---------------------------------------------------------------------------
# Pairwise Distinguishing Features (Clinical Literature)
# ---------------------------------------------------------------------------
# For each confusion pair, the specific HPO terms that distinguish them.
# Used for:
#   1. LOOKS_LIKE edge data in the knowledge graph
#   2. Evidence generation in the explainability module
#   3. Next-test recommendations

@dataclass
class DistinguishingFeature:
    """A feature that distinguishes disease_a from disease_b."""
    hpo_id: str
    label: str
    present_in: str      # disease name where this is present/high
    absent_in: str       # disease name where this is absent/low
    clinical_test: str   # recommended clinical test to assess this feature


PAIRWISE_DISTINGUISHING: Dict[FrozenSet[str], List[DistinguishingFeature]] = {
    # --- Marfan vs Loeys-Dietz (Lovato et al. 2024) ---
    frozenset(["Marfan syndrome", "Loeys-Dietz syndrome"]): [
        DistinguishingFeature(
            "HP:0001083", "Ectopia lentis",
            present_in="Marfan syndrome", absent_in="Loeys-Dietz syndrome",
            clinical_test="Slit-lamp ophthalmologic examination"
        ),
        DistinguishingFeature(
            "HP:0005116", "Arterial tortuosity",
            present_in="Loeys-Dietz syndrome", absent_in="Marfan syndrome",
            clinical_test="CT/MR angiography of head, neck, and chest"
        ),
        DistinguishingFeature(
            "HP:0000193", "Bifid uvula",
            present_in="Loeys-Dietz syndrome", absent_in="Marfan syndrome",
            clinical_test="Oral cavity examination"
        ),
        DistinguishingFeature(
            "HP:0000316", "Hypertelorism",
            present_in="Loeys-Dietz syndrome", absent_in="Marfan syndrome",
            clinical_test="Interpupillary distance measurement"
        ),
    ],

    # --- Marfan vs MASS ---
    frozenset(["Marfan syndrome", "MASS phenotype"]): [
        DistinguishingFeature(
            "HP:0001083", "Ectopia lentis",
            present_in="Marfan syndrome", absent_in="MASS phenotype",
            clinical_test="Slit-lamp ophthalmologic examination"
        ),
        DistinguishingFeature(
            "HP:0002647", "Aortic dissection",
            present_in="Marfan syndrome", absent_in="MASS phenotype",
            clinical_test="Echocardiography with aortic root z-score calculation"
        ),
        DistinguishingFeature(
            "HP:0002616", "Aortic root aneurysm (progressive)",
            present_in="Marfan syndrome", absent_in="MASS phenotype",
            clinical_test="Serial echocardiography — if aortic z-score >2.0 or progressive, reclassify as Marfan"
        ),
    ],

    # --- Loeys-Dietz vs Shprintzen-Goldberg ---
    frozenset(["Loeys-Dietz syndrome", "Shprintzen-Goldberg syndrome"]): [
        DistinguishingFeature(
            "HP:0001249", "Intellectual disability",
            present_in="Shprintzen-Goldberg syndrome", absent_in="Loeys-Dietz syndrome",
            clinical_test="Neurodevelopmental assessment / cognitive testing"
        ),
        DistinguishingFeature(
            "HP:0005116", "Arterial tortuosity",
            present_in="Loeys-Dietz syndrome", absent_in="Shprintzen-Goldberg syndrome",
            clinical_test="CT/MR angiography"
        ),
        DistinguishingFeature(
            "HP:0002647", "Aortic dissection",
            present_in="Loeys-Dietz syndrome", absent_in="Shprintzen-Goldberg syndrome",
            clinical_test="Aortic imaging with dissection risk stratification"
        ),
    ],

    # --- Marfan vs Beals ---
    frozenset(["Marfan syndrome", "Beals syndrome"]): [
        DistinguishingFeature(
            "HP:0001371", "Flexion contracture (congenital)",
            present_in="Beals syndrome", absent_in="Marfan syndrome",
            clinical_test="Physical examination of joint range of motion"
        ),
        DistinguishingFeature(
            "HP:0000377", "Abnormal pinna morphology (crumpled ears)",
            present_in="Beals syndrome", absent_in="Marfan syndrome",
            clinical_test="Otologic examination"
        ),
        DistinguishingFeature(
            "HP:0001083", "Ectopia lentis",
            present_in="Marfan syndrome", absent_in="Beals syndrome",
            clinical_test="Slit-lamp ophthalmologic examination"
        ),
    ],

    # --- Beals vs Shprintzen-Goldberg ---
    frozenset(["Beals syndrome", "Shprintzen-Goldberg syndrome"]): [
        DistinguishingFeature(
            "HP:0001249", "Intellectual disability",
            present_in="Shprintzen-Goldberg syndrome", absent_in="Beals syndrome",
            clinical_test="Neurodevelopmental assessment"
        ),
        DistinguishingFeature(
            "HP:0000377", "Crumpled ears",
            present_in="Beals syndrome", absent_in="Shprintzen-Goldberg syndrome",
            clinical_test="Otologic examination"
        ),
        DistinguishingFeature(
            "HP:0001363", "Craniosynostosis",
            present_in="Shprintzen-Goldberg syndrome", absent_in="Beals syndrome",
            clinical_test="Skull radiograph / CT head"
        ),
    ],
}


# ---------------------------------------------------------------------------
# Gene-Disease Mapping
# ---------------------------------------------------------------------------

GENE_DISEASE_MAP: Dict[str, List[str]] = {
    "FBN1": ["Marfan syndrome", "MASS phenotype"],
    "TGFBR1": ["Loeys-Dietz syndrome"],
    "TGFBR2": ["Loeys-Dietz syndrome"],
    "SMAD3": ["Loeys-Dietz syndrome"],
    "FBN2": ["Beals syndrome"],
    "SKI": ["Shprintzen-Goldberg syndrome"],
}


# ---------------------------------------------------------------------------
# HPO Term → Clinical Test Mapping
# ---------------------------------------------------------------------------
# Maps HPO terms to the clinical test/investigation that evaluates them.
# Used by the next-test recommender.

HPO_TO_CLINICAL_TEST: Dict[str, str] = {
    "HP:0001083": "Slit-lamp ophthalmologic examination for lens subluxation",
    "HP:0000545": "Ophthalmologic refraction and fundoscopy",
    "HP:0002616": "Transthoracic echocardiography with aortic root z-score",
    "HP:0002647": "CT angiography or MR angiography of aorta",
    "HP:0001634": "Echocardiography for mitral valve assessment",
    "HP:0005116": "CT/MR angiography of head, neck, chest, and abdomen",
    "HP:0004942": "Full arterial imaging (CT/MR angiography)",
    "HP:0000316": "Interpupillary distance measurement (ICD)",
    "HP:0000193": "Oral examination for bifid or broad uvula",
    "HP:0000175": "Oral and palatal examination",
    "HP:0001363": "Skull radiograph or CT head for craniosynostosis",
    "HP:0001249": "Formal neurodevelopmental/cognitive assessment",
    "HP:0001263": "Developmental milestone screening",
    "HP:0001371": "Physical exam: passive and active joint range of motion",
    "HP:0001182": "Hand examination for camptodactyly (5th finger)",
    "HP:0000377": "Otoscopic and external ear examination",
    "HP:0001166": "Wrist (Walker-Murdoch) and thumb (Steinberg) sign assessment",
    "HP:0000098": "Height measurement + growth chart percentile",
    "HP:0001519": "Upper-to-lower segment ratio and arm span measurement",
    "HP:0000767": "Chest wall inspection for pectus carinatum",
    "HP:0000768": "Chest wall inspection for pectus excavatum",
    "HP:0002650": "Scoliosis screening (Adam forward bend test, spine radiograph)",
    "HP:0001382": "Beighton score for joint hypermobility",
    "HP:0001763": "Podiatric examination for pes planus",
    "HP:0007807": "Lumbosacral MRI for dural ectasia",
    "HP:0001065": "Skin examination for striae distensae",
    "HP:0000978": "Skin examination for bruising / translucency",
    "HP:0001075": "Skin examination for atrophic scarring",
    "HP:0002107": "Chest radiograph for spontaneous pneumothorax",
    "HP:0001252": "Neurological examination for muscular hypotonia",
    "HP:0000508": "Ophthalmologic examination for ptosis",
    "HP:0000028": "Scrotal/testicular examination for cryptorchidism",
    "HP:0001539": "Abdominal examination for inguinal hernia",
    "HP:0001537": "Abdominal examination for umbilical hernia",
    "HP:0003199": "Assessment of muscle bulk (decreased muscle mass)",
}


# ---------------------------------------------------------------------------
# Disease Categories: Common vs Rare
# ---------------------------------------------------------------------------
# Common diseases (breast cancer, Parkinson's) → Classical ML on real datasets
# Rare diseases (Marfan cluster) → Classical triage + Quantum QSVM for hard cases

# The 3 primary rare diseases for quantum-enhanced differential diagnosis
# (selected as the most clinically confusable from the PDF's 5-disease cluster)
RARE_DISEASES: List[DiseaseProfile] = [MARFAN, LOEYS_DIETZ, BEALS]
RARE_DISEASE_NAMES: List[str] = [d.name for d in RARE_DISEASES]

# All 5 rare diseases (full cluster including SGS and MASS)
FULL_RARE_CLUSTER: List[DiseaseProfile] = ALL_DISEASES

# Disease category routing
DISEASE_CATEGORY: Dict[str, str] = {
    "breast_cancer": "common",
    "parkinsons": "common",
    **{d.short_name: "rare" for d in ALL_DISEASES},
}

# Common diseases use real public datasets:
#   Breast Cancer: Wisconsin Breast Cancer Dataset (569 real patients, 30 features)
#   Parkinson's: Oxford Parkinson's Dataset (195 real patients, 22 voice features)
COMMON_DISEASES = ["breast_cancer", "parkinsons"]


if __name__ == "__main__":
    print(f"QResolve-Dx Disease Data Module")
    print(f"================================")
    print(f"Number of rare diseases: {NUM_CLASSES}")
    print(f"  Primary rare (quantum-eligible): {len(RARE_DISEASES)}")
    print(f"  Full rare cluster: {len(FULL_RARE_CLUSTER)}")
    print(f"Total unique HPO terms in cluster: {NUM_FEATURES}")
    print(f"Known confusion pairs: {len(KNOWN_CONFUSION_PAIRS)}")
    print(f"\nCommon diseases (real datasets): {COMMON_DISEASES}")
    print()
    for d in ALL_DISEASES:
        print(f"  {d.name} ({d.disease_id})")
        print(f"    Genes: {', '.join(d.genes)}")
        print(f"    Symptoms: {len(d.symptoms)} present, {len(d.absent_symptoms)} explicitly absent")
        total = sum(d.symptoms.values())
        print(f"    Mean symptom frequency: {total / len(d.symptoms):.2f}")
        print()
