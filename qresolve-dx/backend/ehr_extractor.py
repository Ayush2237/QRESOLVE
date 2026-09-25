"""
QResolve-Dx: Automated EHR PDF Extraction Module
================================================
Extracts free-text clinical notes, diagnoses, laboratory values, and
phenotypic descriptions from patient Electronic Health Record (EHR) PDF documents.

Uses PyPDF / pypdf to extract raw text, normalizes medical terminology,
and segments the record into standardized clinical sections (HPI, Physical Exam,
Imaging, Laboratory findings).
"""

from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Any
import pypdf


def extract_text_from_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Extract raw text and structured clinical sections from a PDF document.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        dict containing:
          - full_text: Complete concatenated text string.
          - page_count: Total number of pages.
          - sections: Extracted medical sections (History, Exam, Imaging, Impression).
          - metadata: PDF document metadata.
    """
    if not pdf_bytes:
        return {
            "full_text": "",
            "page_count": 0,
            "sections": {},
            "metadata": {},
        }

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    page_count = len(reader.pages)
    page_texts: List[str] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        page_texts.append(text)

    full_text = "\n\n".join(page_texts).strip()

    # Normalize whitespace while preserving line breaks
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    # Segment common clinical headings
    sections = {}
    section_patterns = {
        "chief_complaint": r"(?:Chief Complaint|Reason for Visit|Presenting Problem)[\s:]+(.*?)(?=\n[A-Z][a-zA-Z\s]+:|$)",
        "history_of_present_illness": r"(?:History of Present Illness|HPI)[\s:]+(.*?)(?=\n[A-Z][a-zA-Z\s]+:|$)",
        "physical_examination": r"(?:Physical Examination|Physical Exam|Objective)[\s:]+(.*?)(?=\n[A-Z][a-zA-Z\s]+:|$)",
        "imaging_and_diagnostics": r"(?:Imaging|Radiology|Diagnostic Studies|Echocardiogram)[\s:]+(.*?)(?=\n[A-Z][a-zA-Z\s]+:|$)",
        "assessment_and_plan": r"(?:Assessment|Impression|Diagnosis|Plan)[\s:]+(.*?)(?=\n[A-Z][a-zA-Z\s]+:|$)",
    }

    for sec_name, pattern in section_patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[sec_name] = match.group(1).strip()

    metadata = {}
    if reader.metadata:
        for k, v in reader.metadata.items():
            clean_k = str(k).lstrip("/")
            metadata[clean_k] = str(v)

    return {
        "full_text": full_text,
        "page_count": page_count,
        "sections": sections,
        "metadata": metadata,
    }


def generate_sample_ehr_pdf(case_type: str = "marfan") -> bytes:
    """
    Generate an authentic synthetic patient medical record PDF for demonstration.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6
    )
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569'),
    )
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
    )

    story = []
    story.append(Paragraph("ALL INDIA INSTITUTE OF MEDICAL SCIENCES (AIIMS) / ABDM NETWORK", title_style))
    story.append(Paragraph("Department of Medical Genetics & Cardiovascular Medicine — Clinical Summary Record", sub_style))
    story.append(Paragraph("Patient Health ID (ABHA): 91-8273-1928-01 | E-Hospital Record Ref: AIIMS-DEL-2026-9912", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=12))

    if case_type == "marfan":
        story.append(Paragraph("Chief Complaint", h2_style))
        story.append(Paragraph("28-year-old male evaluated for exertional dyspnea, tall slender habitus, and chest wall deformity.", body_style))
        story.append(Spacer(1, 6))

        story.append(Paragraph("History of Present Illness", h2_style))
        story.append(Paragraph(
            "Patient notes a history of rapid longitudinal skeletal growth in adolescence. Family history is significant "
            "for aortic dissection in paternal uncle at age 34. Patient denies acute chest pain or presyncope.", body_style
        ))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Physical Examination & Phenotypic Findings", h2_style))
        story.append(Paragraph(
            "Height: 196 cm, Weight: 74 kg. Arm span to height ratio: 1.08 (exceeds normal threshold of 1.05). "
            "Musculoskeletal exam demonstrates bilateral arachnodactyly with positive Steinberg thumb sign and Walker-Murdoch wrist sign. "
            "Thoracic examination reveals prominent pectus excavatum. Bilateral pes planus and joint hypermobility noted.", body_style
        ))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Imaging & Diagnostic Studies", h2_style))
        story.append(Paragraph(
            "Transthoracic Echocardiogram (TTE): Severe bulbous dilatation of the sinuses of Valsalva measuring 46 mm (Z-score 3.2), "
            "consistent with aortic root aneurysm. Mild mitral valve prolapse with trace regurgitation. "
            "Ophthalmologic Slit-Lamp Exam: Subluxation of the crystalline lens (ectopia lentis) observed superiorly in both eyes.", body_style
        ))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Clinical Assessment", h2_style))
        story.append(Paragraph(
            "Clinical presentation fulfills revised Ghent nosology criteria for Marfan syndrome (FBN1-related disorder). "
            "High suspicion of connective tissue disorder with systemic score >= 7. Immediate beta-blocker therapy indicated.", body_style
        ))
    else:
        story.append(Paragraph("Chief Complaint", h2_style))
        story.append(Paragraph("19-year-old male referred for evaluation of craniofacial dysmorphism and aortic dilatation.", body_style))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Physical Examination", h2_style))
        story.append(Paragraph(
            "Hypertelorism, broad or bifid uvula, craniosynostosis history, and cervical arterial tortuosity observed on contrast MRA.", body_style
        ))

    doc.build(story)
    return buf.getvalue()
