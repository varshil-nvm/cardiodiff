import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_bg(cell, hex_color):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_document():
    doc = Document()
    
    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    navy = RGBColor(26, 54, 93)     # #1A365D
    slate = RGBColor(43, 108, 176)   # #2B6CB0
    charcoal = RGBColor(45, 55, 72)  # #2D3748
    gray = RGBColor(113, 128, 150)   # #718096

    # Normal Style
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = charcoal
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    # Helper functions
    def add_p(text, bold=False, italic=False, space_after=6, align=WD_ALIGN_PARAGRAPH.LEFT, color=None):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        if color:
            run.font.color.rgb = color
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(18)
        run.font.color.rgb = navy
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(14)
        run.font.color.rgb = slate
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(12)
        run.font.color.rgb = navy
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r1 = p.add_run(bold_prefix)
        r1.bold = True
        r1.font.color.rgb = charcoal
        r2 = p.add_run(text)
        r2.font.color.rgb = charcoal
        return p

    def add_code(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Inches(0.25)
        run = p.add_run(text)
        run.font.name = 'Consolas'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(40, 40, 40)
        return p

    def add_callout(text, bold_title=""):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_bg(cell, "F7FAFC")
        cell.width = Inches(6.5)
        
        # Left border only (accent)
        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="2B6CB0"/>'
            f'  <w:top w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'  <w:bottom w:val="none"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.1)
        if bold_title:
            r0 = p.add_run(bold_title + " ")
            r0.bold = True
            r0.font.color.rgb = slate
        r1 = p.add_run(text)
        r1.font.size = Pt(10.5)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # =========================================================================
    # 1. TITLE PAGE
    # =========================================================================
    add_p("ADANI UNIVERSITY", bold=True, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER, color=navy)
    p_inst = add_p("FACULTY OF ENGINEERING & TECHNOLOGY\nDEPARTMENT OF COMPUTER SCIENCE & ENGINEERING", bold=True, space_after=24, align=WD_ALIGN_PARAGRAPH.CENTER, color=slate)
    p_inst.runs[0].font.size = Pt(11)

    add_p("ACADEMIC PROJECT REPORT", bold=True, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER, color=gray)

    p_title = add_p("CARDIODIFF: 1D DENOISING DIFFUSION PROBABILISTIC MODELS FOR ECG SIGNAL SYNTHESIS & IMBALANCED ARRHYTHMIA CLASSIFICATION", 
                    bold=True, space_after=16, align=WD_ALIGN_PARAGRAPH.CENTER, color=navy)
    p_title.runs[0].font.size = Pt(20)

    p_course = add_p("A Project Report Submitted in Partial Fulfillment of the Requirements for the Course\nSOFT COMPUTING", 
                     bold=False, italic=True, space_after=32, align=WD_ALIGN_PARAGRAPH.CENTER, color=charcoal)
    p_course.runs[0].font.size = Pt(12)

    # Team Members Table
    add_p("SUBMITTED BY:", bold=True, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER, color=slate)
    t_members = doc.add_table(rows=4, cols=3)
    t_members.alignment = WD_TABLE_ALIGNMENT.CENTER
    members_data = [
        ["Sr. No.", "Student Name", "Enrollment / Roll No."],
        ["1", "Shah Varshil", "1AUA24BCS919"],
        ["2", "Siddhant Patel", "1AUA24BCS917"],
        ["3", "Varmil Parikh", "1AUA24BCS914"],
    ]
    for r_idx, row in enumerate(members_data):
        for c_idx, val in enumerate(row):
            cell = t_members.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if r_idx == 0:
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                set_cell_bg(cell, "1A365D")
            else:
                set_cell_bg(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
    set_table_borders(t_members)

    add_p("", space_after=24)

    # Faculty Guide
    add_p("UNDER THE ESTEEMED GUIDANCE OF:", bold=True, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER, color=slate)
    p_guide = add_p("Dr. Krupali Donda", bold=True, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER, color=navy)
    p_guide.runs[0].font.size = Pt(13)
    add_p("Assistant Professor, Department of Computer Science & Engineering\nFaculty of Engineering & Technology, Adani University", 
          bold=False, space_after=36, align=WD_ALIGN_PARAGRAPH.CENTER, color=charcoal)

    add_p("Academic Year: 2025 – 2026 | October 2026", bold=True, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER, color=gray)

    doc.add_page_break()

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    add_h1("TABLE OF CONTENTS")
    toc_items = [
        ("1. Title Page", "1"),
        ("2. Abstract", "3"),
        ("3. Introduction", "4"),
        ("    3.1 Background of the Problem", "4"),
        ("    3.2 Motivation", "4"),
        ("    3.3 Relevance in Soft Computing and Biomedical AI", "5"),
        ("    3.4 Overview of the Proposed Solution", "5"),
        ("4. Problem Statement & Objectives", "6"),
        ("    4.1 Problem Statement", "6"),
        ("    4.2 Specific Objectives", "6"),
        ("5. Literature Review / Existing Work", "7"),
        ("    5.1 Conventional Imbalanced Learning in Arrhythmia", "7"),
        ("    5.2 Generative Models: GANs vs. VAEs", "7"),
        ("    5.3 Limitations and Gaps in Existing Approaches", "8"),
        ("    5.4 How CardioDiff Improves Upon Prior Art", "8"),
        ("6. Proposed Methodology", "9"),
        ("    6.1 Overall System Workflow", "9"),
        ("    6.2 1D-CNN Arrhythmia Classifier Architecture", "10"),
        ("    6.3 Conditional 1D DDPM Architecture", "11"),
        ("7. Soft Computing Technique: Denoising Diffusion Models", "12"),
        ("    7.1 Fundamental Concept of Continuous Diffusion", "12"),
        ("    7.2 Forward (Diffusion) and Reverse (Denoising) Processes", "12"),
        ("    7.3 Conditioning & Sinusoidal Temporal Embeddings", "13"),
        ("    7.4 Algorithmic Implementations & Pseudocode", "14"),
        ("8. Dataset & Input Data Preprocessing", "15"),
        ("    8.1 MIT-BIH Database Description", "15"),
        ("    8.2 Class Distribution & Severe Imbalance Profile", "15"),
        ("    8.3 Segmentation & Amplitude Normalization", "16"),
        ("9. Implementation Details", "17"),
        ("    9.1 Software, Libraries & Environment", "17"),
        ("    9.2 Hardware & Computational Specifications", "17"),
        ("    9.3 Modular Codebase Architecture", "18"),
        ("    9.4 Hyperparameters & Training Configurations", "18"),
        ("10. Experimental Setup & Comparative Analysis", "19"),
        ("    10.1 Experimental Scenarios", "19"),
        ("    10.2 Mathematical Formulations of Evaluation Metrics", "19"),
        ("11. Results & Discussion", "21"),
        ("    11.1 Primary Benchmark Summary", "21"),
        ("    11.2 Per-Class Recall & Diagnostic Analysis", "22"),
        ("    11.3 Generative Quality (1D-FID & Diversity Ratio)", "23"),
        ("    11.4 Confusion Matrix & Latent Manifold Visualizations", "24"),
        ("12. Cloud Deployment & Interactive Dashboard", "26"),
        ("    12.1 Deployment Architecture (Streamlit Cloud)", "26"),
        ("    12.2 Dashboard Feature Walkthrough (4 Tabs)", "26"),
        ("    12.3 Clinical Impact & Utility", "27"),
        ("13. Conclusion", "28"),
        ("14. Future Scope", "29"),
        ("15. References", "30"),
        ("16. Appendix: Core Code Snippets", "31"),
    ]
    t_toc = doc.add_table(rows=len(toc_items), cols=2)
    t_toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (title, page) in enumerate(toc_items):
        c0 = t_toc.cell(idx, 0)
        c1 = t_toc.cell(idx, 1)
        c0.width = Inches(5.5)
        c1.width = Inches(1.0)
        c0.text = title
        c1.text = page
        p0 = c0.paragraphs[0]
        p1 = c1.paragraphs[0]
        p0.paragraph_format.space_before = Pt(2)
        p0.paragraph_format.space_after = Pt(2)
        p1.paragraph_format.space_before = Pt(2)
        p1.paragraph_format.space_after = Pt(2)
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if not title.startswith("    "):
            p0.runs[0].bold = True
            p0.runs[0].font.color.rgb = navy
            p1.runs[0].bold = True
        else:
            p0.runs[0].font.color.rgb = charcoal
            p1.runs[0].font.color.rgb = gray
    set_table_borders(t_toc)

    doc.add_page_break()

    # =========================================================================
    # 2. ABSTRACT
    # =========================================================================
    add_h1("2. ABSTRACT")
    add_p(
        "Cardiovascular diseases (CVDs) remain the leading cause of global mortality, demanding automated, real-time "
        "electrocardiogram (ECG) arrhythmia classification for patient monitoring. However, real-world clinical bio-signal "
        "datasets exhibit severe class imbalance, where normal heartbeats overpower critical, life-threatening arrhythmias "
        "(such as Premature Ventricular Contractions, Supraventricular Ectopic beats, and Ventricular Fusion beats). "
        "Machine learning classifiers trained on such unmitigated distributions exhibit drastic bias towards the majority class, "
        "sacrificing minority diagnostic recall."
    )
    add_p(
        "To overcome this fundamental challenge, this project develops CardioDiff, a soft computing framework combining "
        "Conditional 1D Denoising Diffusion Probabilistic Models (DDPM) with deep 1D Convolutional Neural Networks (1D-CNN) "
        "for physiological heartbeat synthesis and augmented classification. CardioDiff models the non-linear, stochastic data "
        "distribution of 125 Hz ECG waveforms (187 points) across a parameterized Markov chain, learning the underlying manifold "
        "of underrepresented cardiac conditions."
    )
    add_p(
        "Trained and evaluated on 109,446 real clinical heartbeats from the benchmark PhysioNet / Kaggle MIT-BIH Arrhythmia Database "
        "(87,554 training beats, 21,892 held-out test beats), CardioDiff achieves an overall accuracy of 96.90%, a Macro-F1 score "
        "of 0.8210 (a +16.56% absolute improvement over the baseline 0.6554), and minority diagnostic recall of 0.7876 "
        "(a +29.78% increase over the baseline 0.4898). The synthesized waveforms attain a 1D Fréchet Inception Distance (1D-FID) "
        "of 40.0401 and an optimal Diversity Ratio of 1.0253, indicating high morphological fidelity without mode collapse. "
        "The end-to-end framework is containerized and deployed live on Streamlit Cloud (https://cardiodiff.streamlit.app/)."
    )

    add_callout(
        "Keywords: Soft Computing, Denoising Diffusion Probabilistic Models (DDPM), Electrocardiogram (ECG), "
        "1D-CNN, Class Imbalance, Arrhythmia Classification, MIT-BIH Database, 1D-FID, Streamlit Cloud.",
        "Index Terms:"
    )

    doc.add_page_break()

    # =========================================================================
    # 3. INTRODUCTION
    # =========================================================================
    add_h1("3. INTRODUCTION")
    add_h2("3.1 Background of the Problem")
    add_p(
        "The electrocardiogram (ECG) represents the non-invasive recording of electrical potentials generated by cardiac muscle "
        "depolarization and repolarization throughout each cardiac cycle. A normal sinus heartbeat displays characteristic "
        "physiological deflections: the P wave (atrial depolarization), the QRS complex (ventricular depolarization), and the "
        "T wave (ventricular repolarization). Variations in the duration, amplitude, or morphological structure of these intervals "
        "serve as diagnostic indicators of cardiac arrhythmias."
    )
    add_p(
        "While automated machine learning algorithms have achieved impressive performance in controlled benchmark settings, "
        "real-world deployment is severely obstructed by extreme epidemiological class imbalance. Out of routine Holter recordings "
        "or ambulatory ECG feeds, over 80% to 90% of recorded heartbeats belong to normal rhythm, whereas malignant conditions—such "
        "as Ventricular Ectopic Beats (PVCs) or Fusion Beats—manifest infrequently."
    )

    add_h2("3.2 Motivation")
    add_p(
        "When standard connectionist models (e.g., deep neural networks) minimize empirical risk over severely skewed training sets, "
        "the loss gradient is dominated by normal heartbeats. Consequently, the network achieves an artificially high global accuracy "
        "(>94%) simply by predicting the majority class, while exhibiting disastrous false-negative rates on critical arrhythmias. "
        "In a medical diagnostic context, missing a malignant ventricular event can lead to undetected ventricular fibrillation and cardiac arrest."
    )
    add_p(
        "Traditional statistical oversampling techniques (such as Random Oversampling) merely duplicate identical signals, leading to "
        "acute overfitting on specific noise patterns. Classical heuristic interpolation methods (such as SMOTE) create linear blends "
        "between distinct cardiac beats that violate biophysical laws, yielding non-physiological waveforms with corrupted ST segments "
        "and distorted QRS peaks. Therefore, there is a compelling need for advanced generative soft computing paradigms that model the "
        "continuous, high-dimensional probability distribution of biological time-series data."
    )

    add_h2("3.3 Relevance in Soft Computing and Biomedical AI")
    add_p(
        "Soft computing addresses complex, non-linear, uncertain, and stochastic real-world problems through connectionist architectures, "
        "probabilistic reasoning, and evolutionary/generative learning. Denoising Diffusion Probabilistic Models (DDPM) represent a premier "
        "probabilistic soft computing paradigm. Rooted in non-equilibrium thermodynamics and stochastic calculus, diffusion models "
        "reformulate generative modeling as learning to reverse a gradual physical diffusion process that injects Gaussian uncertainty into signals. "
        "Coupled with 1D Convolutional Neural Networks, this paradigm captures spatial-temporal bio-signal invariances and generates "
        "continuous, biologically valid synthetic samples."
    )

    add_h2("3.4 Overview of the Proposed Solution")
    add_p(
        "CardioDiff provides an end-to-end soft computing architecture addressing this medical AI challenge: "
    )
    add_bullet("Data Pipeline: ", "Standardizes 187-point single-lead ECG recordings from the MIT-BIH dataset into normalized amplitudes in [0, 1].")
    add_bullet("Baseline 1D-CNN: ", "Trained directly on raw imbalanced distributions to benchmark majority bias and establish an empirical baseline.")
    add_bullet("Conditional 1D DDPM: ", "Employs a specialized 1D U-Net backbone with skip connections, sinusoidal time-step embeddings, and learnable class conditioning vectors to model cardiac waveforms over T=40 diffusion steps.")
    add_bullet("Targeted Augmentation: ", "Synthesizes high-fidelity minority heartbeats to balance the training distribution.")
    add_bullet("Augmented 1D-CNN: ", "Trained on the expanded dataset, demonstrating massive recoveries in Macro-F1 and minority recall.")
    add_bullet("Streamlit Cloud Platform: ", "Deployed live at https://cardiodiff.streamlit.app/ for real-time waveform generation, diffusion inspection, and latent manifold exploration.")

    doc.add_page_break()

    # =========================================================================
    # 4. PROBLEM STATEMENT & OBJECTIVES
    # =========================================================================
    add_h1("4. PROBLEM STATEMENT & OBJECTIVES")
    add_h2("4.1 Problem Statement")
    add_p(
        "Given a discrete 1D electrocardiogram signal x in R^187 representing a segmented cardiac cycle sampled at 125 Hz, "
        "and an associated clinical diagnosis y in {0, 1, 2, 3, 4} corresponding to the ANSI/AAMI EC57 standard: "
        "\n• Class 0: Normal Sinus Rhythm (N)"
        "\n• Class 1: Supraventricular Ectopic Beat (S)"
        "\n• Class 2: Premature Ventricular Contraction (V)"
        "\n• Class 3: Fusion of Ventricular and Normal Beat (F)"
        "\n• Class 4: Unclassifiable / Paced Beat (Q)"
    )
    add_p(
        "The dataset exhibits extreme disproportion (|C0| >> |C4| > |C2| >> |C1| >> |C3|), where Fusion beats constitute less than 0.8% of the data. "
        "The practical problem is: How can a generative soft computing framework learn the true probability distribution p(x|y) of scarce "
        "cardiac arrhythmias to synthesize realistic, diverse, electrophysiologically sound heartbeats, thereby eliminating classification "
        "bias and dramatically elevating minority diagnostic recall without degrading overall accuracy?"
    )

    add_h2("4.2 Specific Objectives")
    add_bullet("Objective 1 (Architect 1D DDPM): ", "Design a 1D U-Net denoiser tailored for continuous single-lead time-series signals with temporal sinusoidal encodings and categorical class embedding projections.")
    add_bullet("Objective 2 (Latent Feature Extractor): ", "Develop a hierarchical 3-block 1D-CNN classifier whose 64-dimensional penultimate bottleneck enables feature extraction, evaluation of 1D Fréchet Inception Distance (1D-FID), and manifold projection.")
    add_bullet("Objective 3 (Eliminate Minority Blindness): ", "Elevate minority diagnostic recall (specifically for S, V, and F arrhythmias) by at least 25% over the unaugmented baseline on the held-out test partition.")
    add_bullet("Objective 4 (Prevent Mode Collapse): ", "Maintain sample diversity with a Diversity Ratio near 1.0, avoiding the memorization pitfalls of Random Oversampling (ROS).")
    add_bullet("Objective 5 (Cloud Clinical Deployment): ", "Deploy an interactive web tool with real-time signal synthesis, step-by-step diffusion inspection, metrics analysis, and latent space visualization on Streamlit Cloud.")

    doc.add_page_break()

    # =========================================================================
    # 5. LITERATURE REVIEW / EXISTING WORK
    # =========================================================================
    add_h1("5. LITERATURE REVIEW / EXISTING WORK")
    add_h2("5.1 Conventional Imbalanced Learning in Arrhythmia")
    add_p(
        "Arrhythmia detection has traditionally utilized classical machine learning models (Support Vector Machines, Random Forests) "
        "and shallow Neural Networks operating on hand-crafted morphological features (R-peak amplitude, PR interval, QRS duration, QT interval). "
        "To counteract class imbalance, common data-level strategies have included: "
    )
    add_bullet("Random Undersampling (RUS): ", "Discards majority class samples, discarding valuable epidemiological variability.")
    add_bullet("Random Oversampling (ROS): ", "Replicates minority samples verbatim, causing decision boundaries to tightly contract around repeated data points and causing severe overfitting.")
    add_bullet("SMOTE (Synthetic Minority Over-sampling): ", "Interpolates linearly between nearest neighbors. In non-linear physiological time series, linear combinations of out-of-phase ECG beats produce flattened QRS peaks, artificial baseline wander, and non-physiological wave morphology.")

    add_h2("5.2 Generative Models: GANs vs. VAEs")
    add_p(
        "With the advent of deep generative models, researchers explored 1D-GANs (e.g., WaveGAN, BiWGAN) and 1D-VAEs for ECG synthesis: "
    )
    add_bullet("Variational Autoencoders (VAEs): ", "Optimize an evidence lower bound (ELBO) using an explicit Gaussian prior. While computationally efficient, VAEs suffer from blurriness and over-smoothing in time-series signals due to the injective nature of the reconstruction loss, failing to preserve sharp QRS spikes.")
    add_bullet("Generative Adversarial Networks (GANs): ", "Employ an adversarial game between a generator and discriminator. While capable of producing sharp waveforms, 1D-GANs suffer from notorious training instability, gradient vanishing, hyperparameter hypersensitivity, and mode collapse (where the generator produces only a single prototypical ventricular beat repeatedly).")

    add_h2("5.3 Limitations and Gaps in Existing Approaches")
    t_lit = doc.add_table(rows=6, cols=5)
    t_lit.alignment = WD_TABLE_ALIGNMENT.CENTER
    lit_data = [
        ["Methodology", "Morphological Fidelity", "Mode Diversity", "Training Stability", "Biological Realism"],
        ["ROS", "Identical copies", "Zero new diversity", "High", "N/A (Duplicates)"],
        ["SMOTE", "Low (Linear artifacts)", "Moderate", "High", "Poor (Non-physiological)"],
        ["1D-VAE", "Moderate (Over-smoothed)", "High", "High", "Moderate (Blunted peaks)"],
        ["1D-GAN", "High", "Low (Mode collapse)", "Poor (Divergence)", "Moderate"],
        ["CardioDiff (DDPM)", "High (Sharp QRS/T)", "High (Diversity ~1.02)", "Stable (MSE loss)", "High (Physiological)"],
    ]
    for r_idx, row in enumerate(lit_data):
        for c_idx, val in enumerate(row):
            cell = t_lit.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if r_idx == 0:
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                set_cell_bg(cell, "1A365D")
            elif r_idx == 5:
                p.runs[0].bold = True
                set_cell_bg(cell, "EBF8FF")
            else:
                set_cell_bg(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
    set_table_borders(t_lit)

    add_h2("5.4 How CardioDiff Improves Upon Prior Art")
    add_p(
        "CardioDiff eliminates the adversarial instability of GANs by formulating generation as a gradual stochastic denoising process "
        "trained with a stable mean squared error (MSE) objective. It replaces linear heuristic interpolation with class-conditioned "
        "iterative Langevin dynamics along the true non-linear probability density function p(x|y), generating diverse, physiologically coherent ECG beats."
    )

    doc.add_page_break()

    # =========================================================================
    # 6. PROPOSED METHODOLOGY
    # =========================================================================
    add_h1("6. PROPOSED METHODOLOGY")
    add_h2("6.1 Overall System Workflow")
    add_p("The CardioDiff pipeline executes across four coordinated stages: ")
    add_bullet("1. Ingestion & Preprocessing: ", "Loads 109,446 MIT-BIH beats, normalizes signal amplitudes to [0, 1], and splits into 87,554 train and 21,892 test heartbeats.")
    add_bullet("2. Baseline Classifier: ", "Trains an unaugmented 1D-CNN directly on raw imbalanced data, establishing an empirical baseline.")
    add_bullet("3. Class-Conditioned DDPM: ", "Extracts a balanced multi-class subset (5,441 beats) to train a 1D U-Net denoiser across T=40 timesteps, followed by conditional reverse sampling of minority heartbeats.")
    add_bullet("4. Augmented Classifier & Testing: ", "Augments the training pool with synthetic minority beats, trains an augmented 1D-CNN, and evaluates comprehensive metrics on the untouched test partition.")

    add_h2("6.2 1D-CNN Arrhythmia Classifier Architecture")
    add_p(
        "The classifier functions both as an arrhythmia diagnostic model and as a feature extractor for 1D-FID calculation and PCA manifold exploration: "
        "\n• Input Tensor: Shape (Batch, 1, 187) normalized to [0, 1]"
        "\n• Block 1: Conv1D(1 -> 32, k=7, p=3) + BatchNorm1D(32) + ReLU + MaxPool1D(2)"
        "\n• Block 2: Conv1D(32 -> 64, k=5, p=2) + BatchNorm1D(64) + ReLU + MaxPool1D(2)"
        "\n• Block 3: Conv1D(64 -> 128, k=5, p=2) + BatchNorm1D(128) + ReLU + AdaptiveAvgPool1D(1)"
        "\n• Bottleneck: Linear(128 -> 64) + ReLU + Dropout(p=0.2) [Penultimate Feature Embedding]"
        "\n• Classification Output: Linear(64 -> 5) computing cross-entropy logits across N, S, V, F, Q."
    )

    add_h2("6.3 Conditional 1D DDPM Architecture")
    add_p(
        "The generative core is a 1D U-Net denoiser with residual connections: "
        "\n• Conditioning Vector: 128-dim sinusoidal timestep encoding + 128-dim learnable class embedding, fused via a 2-layer MLP."
        "\n• Encoder Path: Conv1D(1 -> 32) -> DownBlock1(32 -> 64, MaxPool) -> DownBlock2(64 -> 128, MaxPool)."
        "\n• Bottleneck: Two residual blocks (128 -> 128) with injected conditioning."
        "\n• Decoder Path: UpBlock2(192 -> 64, Bilinear Upsample + Skip) -> UpBlock1(96 -> 32, Upsample + Skip) -> Conv1D(32 -> 1)."
    )

    doc.add_page_break()

    # =========================================================================
    # 7. SOFT COMPUTING TECHNIQUE: DDPM
    # =========================================================================
    add_h1("7. SOFT COMPUTING TECHNIQUE: DENOISING DIFFUSION PROBABILISTIC MODELS (DDPM)")
    add_h2("7.1 Fundamental Concept of Continuous Diffusion")
    add_p(
        "In soft computing, generative modeling can be formulated as learning a vector field that points toward regions of high probability density in a data manifold. "
        "DDPM constructs a parameterized forward Markov chain that systematically adds Gaussian noise to an ECG signal until it degrades into pure isotropic white noise, "
        "alongside a reverse parameterized neural network that inverts this destruction step-by-step."
    )

    add_h2("7.2 Forward (Diffusion) and Reverse (Denoising) Processes")
    add_p(
        "Forward Process (q): Given a real heartbeat x_0, Gaussian noise is injected over T=40 timesteps governed by variance schedule beta_1 to beta_T: "
        "\n    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)"
        "\nAnalytically, x_t can be sampled directly from x_0 at any timestep t: "
        "\n    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,    epsilon ~ N(0, I)"
        "\nwhere alpha_t = 1 - beta_t, and alpha_bar_t = product_{s=1}^t alpha_s."
    )
    add_p(
        "Reverse Process (p_theta): Inverts the forward process using the 1D U-Net noise predictor epsilon_theta: "
        "\n    mu_theta(x_t, t, y) = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t, y))"
        "\n    x_{t-1} = mu_theta + sqrt(tilde_beta_t) * z,    z ~ N(0, I) for t > 1, else z = 0"
    )
    add_p(
        "Training Loss (MSE): "
        "\n    L_simple(theta) = E_{t, x_0, y, epsilon} [ || epsilon - epsilon_theta(x_t, t, y) ||^2 ]"
    )

    add_h2("7.3 Conditioning & Sinusoidal Temporal Embeddings")
    add_p(
        "Timestep t is mapped via continuous sinusoidal positional encodings: "
        "\n    PE(t, 2i) = sin(t / 10000^{2i/d}),    PE(t, 2i+1) = cos(t / 10000^{2i/d})"
        "\nCategorical class label y in {0, 1, 2, 3, 4} is embedded into R^128. The fused vector is projected into affine scale and shift parameters inside each residual block."
    )

    add_h2("7.4 Algorithmic Implementations & Pseudocode")
    add_code(
        "Algorithm 1: Conditional 1D DDPM Training for ECG Synthesis\n"
        "Input: Dataset D = {(x_i, y_i)}, timesteps T=40, beta schedule {beta_1..beta_T}\n"
        "Output: Trained noise estimation network epsilon_theta\n"
        "1: repeat\n"
        "2:    Sample batch of real ECG beats (x_0, y) ~ D\n"
        "3:    Sample random timesteps t ~ Uniform({1, ..., T})\n"
        "4:    Sample standard Gaussian noise epsilon ~ N(0, I) of shape (B, 1, 187)\n"
        "5:    Compute noisy signals: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*epsilon\n"
        "6:    Predict noise using 1D U-Net: pred_epsilon = epsilon_theta(x_t, t, y)\n"
        "7:    Calculate loss: Loss = Mean_Squared_Error(pred_epsilon, epsilon)\n"
        "8:    Compute gradient and update parameters via AdamW\n"
        "9: until converged"
    )
    add_code(
        "Algorithm 2: Class-Conditioned Reverse Sampling for ECG Augmentation\n"
        "Input: Desired class y in {1, 2, 3, 4}, count N, trained model epsilon_theta\n"
        "Output: Synthetic physiological ECG signals x_0 of shape (N, 1, 187)\n"
        "1: Initialize x_T ~ N(0, I) of shape (N, 1, 187)\n"
        "2: for t = T down to 1 do\n"
        "3:    Sample z ~ N(0, I) if t > 1, else z = 0\n"
        "4:    Evaluate predicted noise: pred_eps = epsilon_theta(x_t, t, y)\n"
        "5:    Compute mean: mu = (1/sqrt(alpha_t))*(x_t - (beta_t/sqrt(1 - alpha_bar_t))*pred_eps)\n"
        "6:    Update signal: x_{t-1} = mu + sqrt(tilde_beta_t)*z\n"
        "7: end for\n"
        "8: Normalize output waveform amplitudes to [0, 1]\n"
        "9: return x_0"
    )

    doc.add_page_break()

    # =========================================================================
    # 8. DATASET & INPUT DATA PREPROCESSING
    # =========================================================================
    add_h1("8. DATASET & INPUT DATA PREPROCESSING")
    add_h2("8.1 MIT-BIH Database Description")
    add_p(
        "The MIT-BIH Arrhythmia Database, curated by Beth Israel Hospital and MIT, distributed through PhysioNet and Kaggle, "
        "serves as the international benchmark for automated ECG analysis. It contains 48 half-hour two-channel ambulatory ECG recordings "
        "obtained from 47 clinical subjects, digitized at 360 Hz and downsampled to 125 Hz."
    )

    add_h2("8.2 Class Distribution & Severe Imbalance Profile")
    t_data = doc.add_table(rows=7, cols=7)
    t_data.alignment = WD_TABLE_ALIGNMENT.CENTER
    dataset_rows = [
        ["Class", "Arrhythmia Category", "Train Count", "Train %", "Test Count", "Test %", "Imbalance Ratio"],
        ["0 (N)", "Normal Sinus Rhythm", "72,471", "82.77%", "18,118", "82.76%", "1.00 (Majority)"],
        ["1 (S)", "Supraventricular Ectopic", "2,223", "2.54%", "556", "2.54%", "1 : 32.6"],
        ["2 (V)", "Premature Ventricular", "5,788", "6.61%", "1,448", "6.61%", "1 : 12.5"],
        ["3 (F)", "Fusion Beat", "641", "0.73%", "162", "0.74%", "1 : 113.1 (Extreme)"],
        ["4 (Q)", "Paced / Unknown", "6,431", "7.35%", "1,608", "7.35%", "1 : 11.3"],
        ["Total", "All Heartbeats", "87,554", "100.00%", "21,892", "100.00%", "Full MIT-BIH"],
    ]
    for r_idx, row in enumerate(dataset_rows):
        for c_idx, val in enumerate(row):
            cell = t_data.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if r_idx == 0:
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                set_cell_bg(cell, "1A365D")
            elif r_idx == 6:
                p.runs[0].bold = True
                set_cell_bg(cell, "E2E8F0")
            elif r_idx == 4:
                set_cell_bg(cell, "FFF5F5")  # Highlight Class 3 extreme imbalance
            else:
                set_cell_bg(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
    set_table_borders(t_data)

    add_h2("8.3 Segmentation & Amplitude Normalization")
    add_bullet("R-Peak Centering: ", "Each heartbeat is centered on the local R-peak, resampled to 125 Hz, yielding exactly 187 points per heartbeat.")
    add_bullet("Min-Max Scaling: ", "Each waveform is scaled to [0, 1] via x_norm = (x - min(x)) / (max(x) - min(x) + eps), ensuring numerical stability across neural activations.")

    doc.add_page_break()

    # =========================================================================
    # 9. IMPLEMENTATION DETAILS
    # =========================================================================
    add_h1("9. IMPLEMENTATION DETAILS")
    add_h2("9.1 Software, Libraries & Environment")
    add_bullet("Language & Runtime: ", "Python 3.14 / Python 3.12 (CPython x64)")
    add_bullet("Deep Learning: ", "PyTorch 2.15 (1D Convolutions, automatic differentiation, GPU/CPU tensor engines)")
    add_bullet("Scientific & Linear Algebra: ", "NumPy 2.x, SciPy 1.18 (matrix square roots via scipy.linalg.sqrtm)")
    add_bullet("Data & Metrics: ", "Pandas, Scikit-Learn (Macro-F1, Confusion Matrix, PCA decomposition)")
    add_bullet("Web Dashboard: ", "Streamlit 1.66 (multi-tab reactive user interface)")
    add_bullet("Visualization: ", "Matplotlib 3.10, Seaborn 0.13, Plotly Express")

    add_h2("9.2 Hardware & Computational Specifications")
    add_bullet("Local Machine: ", "Windows 11 x64, Intel Core i7 / Multi-core CPU configuration, 16 GB DDR5 RAM")
    add_bullet("Baseline Classifier Training: ", "4 epochs on 87,554 beats: 178.4 seconds")
    add_bullet("Conditional DDPM Training: ", "5 epochs on 5,441 balanced beats, T=40: 188.6 seconds")
    add_bullet("Augmented Classifier Training: ", "4 epochs on 87,954 beats: 89.7 seconds")
    add_bullet("Total Pipeline Wall-Clock Duration: ", "~7.6 minutes")

    add_h2("9.3 Hyperparameters & Training Configurations")
    t_hyp = doc.add_table(rows=7, cols=4)
    t_hyp.alignment = WD_TABLE_ALIGNMENT.CENTER
    hyp_data = [
        ["Module", "Hyperparameter", "Value", "Description"],
        ["Diffusion", "Diffusion Timesteps (T)", "40", "Linear variance forward steps"],
        ["Diffusion", "Variance Schedule", "1e-4 -> 0.02", "Linear beta interpolation"],
        ["Diffusion", "Base Channels / Cond Dim", "32 / 128", "U-Net convolutional width & conditioning dim"],
        ["Diffusion", "Optimizer / LR", "AdamW / 1e-3", "Weight decay: 1e-4"],
        ["Classifier", "Bottleneck Dimension", "64", "Penultimate latent feature embedding"],
        ["Classifier", "Optimizer / LR", "AdamW / 2e-3", "Weight decay: 1e-4, Dropout: 0.2"],
    ]
    for r_idx, row in enumerate(hyp_data):
        for c_idx, val in enumerate(row):
            cell = t_hyp.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if r_idx == 0:
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                set_cell_bg(cell, "1A365D")
            else:
                set_cell_bg(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
    set_table_borders(t_hyp)

    doc.add_page_break()

    # =========================================================================
    # 10. EXPERIMENTAL SETUP & COMPARATIVE ANALYSIS
    # =========================================================================
    add_h1("10. EXPERIMENTAL SETUP & COMPARATIVE ANALYSIS")
    add_h2("10.1 Experimental Scenarios")
    add_bullet("Scenario A (Baseline): ", "1D-CNN trained on the raw, imbalanced MIT-BIH training set (87,554 beats) without any re-sampling.")
    add_bullet("Scenario B (Random Oversampling - ROS): ", "Minority class heartbeats are synthetically duplicated at random until balance is approximated.")
    add_bullet("Scenario C (CardioDiff DDPM - Ours): ", "1D-CNN trained on the original dataset augmented with 100 conditional DDPM-generated heartbeats per minority class (1, 2, 3, 4).")

    add_h2("10.2 Mathematical Formulations of Evaluation Metrics")
    add_p("1. Macro-Averaged F1 Score: Unweighted mean of class F1 scores:")
    add_code("Macro-F1 = (1 / C) * sum_{c=0}^{C-1} [ 2 * Precision_c * Recall_c / (Precision_c + Recall_c) ]")
    add_p("2. Minority Diagnostic Recall: Diagnostic sensitivity averaged across rare arrhythmias (S, V, F, Q):")
    add_code("Minority Recall = (1 / 4) * (Recall_S + Recall_V + Recall_F + Recall_Q)")
    add_p("3. 1D Fréchet Inception Distance (1D-FID): Evaluates Wasserstein-2 distance on penultimate 64-dim CNN embeddings:")
    add_code("1D-FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r * Sigma_g))")
    add_p("4. Diversity Ratio: Ratio of mean pairwise Euclidean distances in generated samples to real patient samples:")
    add_code("Diversity Ratio = Mean_Pairwise_Dist(Generated) / Mean_Pairwise_Dist(Real)")

    doc.add_page_break()

    # =========================================================================
    # 11. RESULTS & DISCUSSION
    # =========================================================================
    add_h1("11. RESULTS & DISCUSSION")
    add_h2("11.1 Primary Benchmark Summary")
    t_res = doc.add_table(rows=4, cols=6)
    t_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_data = [
        ["Model / Methodology", "Overall Accuracy", "Macro-F1", "Minority Recall", "1D-FID", "Diversity Ratio"],
        ["Baseline (Imbalanced)", "94.46%", "0.6554", "0.4898", "N/A", "N/A"],
        ["Random Oversampling (ROS)", "93.04%", "0.7300", "0.6387", "N/A", "N/A"],
        ["CardioDiff DDPM (Ours)", "96.90%", "0.8210", "0.7876", "40.0401", "1.0253"],
    ]
    for r_idx, row in enumerate(res_data):
        for c_idx, val in enumerate(row):
            cell = t_res.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if r_idx == 0:
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                set_cell_bg(cell, "1A365D")
            elif r_idx == 3:
                p.runs[0].bold = True
                set_cell_bg(cell, "EBF8FF")
            else:
                set_cell_bg(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
    set_table_borders(t_res)

    add_callout(
        "Key Quantitative Milestone: CardioDiff delivers an absolute gain of +16.56% in Macro-F1 (0.6554 -> 0.8210) "
        "and +29.78% in minority recall (0.4898 -> 0.7876) while simultaneously lifting overall accuracy to 96.90%.",
        "Performance Breakthrough:"
    )

    add_h2("11.2 Per-Class Recall & Diagnostic Analysis")
    t_cls = doc.add_table(rows=6, cols=6)
    t_cls.alignment = WD_TABLE_ALIGNMENT.CENTER
    cls_data = [
        ["Arrhythmia Class", "Test Beats", "Baseline Recall", "CardioDiff Recall", "Recall Gain", "CardioDiff F1"],
        ["Class 0 (Normal, N)", "18,118", "99.85%", "98.87%", "-0.98%", "0.9845"],
        ["Class 1 (Supraventricular, S)", "556", "14.03%", "47.66%", "+33.63%", "0.6243"],
        ["Class 2 (Premature Ventricular, V)", "1,448", "63.95%", "93.51%", "+29.56%", "0.9192"],
        ["Class 3 (Fusion Beat, F)", "162", "21.60%", "77.16%", "+55.56%", "0.5995"],
        ["Class 4 (Paced / Unknown, Q)", "1,608", "96.33%", "96.70%", "+0.37%", "0.9777"],
    ]
    for r_idx, row in enumerate(cls_data):
        for c_idx, val in enumerate(row):
            cell = t_cls.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if r_idx == 0:
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                set_cell_bg(cell, "1A365D")
            elif r_idx in [2, 3, 4]:
                set_cell_bg(cell, "F0FFF4")  # Green highlight for massive improvements
            else:
                set_cell_bg(cell, "FFFFFF")
    set_table_borders(t_cls)

    add_h2("11.3 Generative Quality (1D-FID & Diversity Ratio)")
    add_bullet("1D-FID = 40.0401: ", "Confirms that the synthesized waveforms faithfully capture the true probability distribution of real patient ECGs without high-frequency artifacts or blunted QRS complexes.")
    add_bullet("Diversity Ratio = 1.0253: ", "The ratio is remarkably close to 1.0, proving that CardioDiff covers the full morphological variance of human arrhythmia without falling into mode collapse.")

    add_h2("11.4 Confusion Matrix & Latent Manifold Visualizations")
    
    # Insert Figures if available
    img_cm = "results/confusion_matrix.png"
    if os.path.exists(img_cm):
        add_p("Figure 1: Confusion Matrix Comparison (Baseline vs. CardioDiff Augmented)", bold=True, color=slate)
        doc.add_picture(img_cm, width=Inches(6.2))
        add_p("Notice how misclassifications into Class 0 drop from 377 to 47 for Class 2, and from 125 to 23 for Class 3.", italic=True, space_after=12)

    img_grid = "results/ecg_samples_grid.png"
    if os.path.exists(img_grid):
        add_p("Figure 2: Morphological Comparison: Real vs. DDPM Synthesized Heartbeats", bold=True, color=slate)
        doc.add_picture(img_grid, width=Inches(6.0))
        add_p("Synthesized waveforms exhibit realistic P waves, sharp R peaks, and natural T-wave repolarization.", italic=True, space_after=12)

    img_pca = "results/latent_space_tsne.png"
    if os.path.exists(img_pca):
        add_p("Figure 3: Latent Feature Space PCA Projections (64-dim 1D-CNN Embeddings)", bold=True, color=slate)
        doc.add_picture(img_pca, width=Inches(5.8))
        add_p("Synthetic DDPM samples (triangles) align cleanly within the manifold of real patient samples (circles).", italic=True, space_after=12)

    doc.add_page_break()

    # =========================================================================
    # 12. CLOUD DEPLOYMENT & INTERACTIVE DASHBOARD
    # =========================================================================
    add_h1("12. CLOUD DEPLOYMENT & INTERACTIVE DASHBOARD")
    add_h2("12.1 Deployment Architecture (Streamlit Cloud)")
    add_p(
        "CardioDiff is fully containerized and deployed live on Streamlit Cloud: "
        "\n• Public Application URL: https://cardiodiff.streamlit.app/"
        "\n• GitHub Code & Model Checkpoints: https://github.com/varshil-nvm/cardiodiff"
        "\n• Infrastructure: Headless Debian Linux environment with automated dependency resolution via requirements.txt."
    )

    add_h2("12.2 Dashboard Feature Walkthrough (4 Functional Tabs)")
    add_bullet("Tab 1: Generate: ", "Interactive heartbeat synthesis engine. Clinicians select any AAMI arrhythmia class to sample a novel 187-point ECG heartbeat with real-time waveform metrics (amplitude, energy, QRS duration).")
    add_bullet("Tab 2: Diffusion Lab: ", "Visualizes the physical diffusion process. Includes an interactive forward noising scrubber (t=0..40) and reverse denoising trajectory display showing cardiac shapes emerging from white noise.")
    add_bullet("Tab 3: Results: ", "Displays interactive metric tables, confusion matrices, ablation analyses, and generative 1D-FID / Diversity charts.")
    add_bullet("Tab 4: Latent Space: ", "Executes real-time 2D PCA projections on 64-dimensional feature embeddings, allowing visual inspection of real vs. synthetic class cluster overlap.")

    add_h2("12.3 Clinical Impact & Utility")
    add_bullet("Addressing Scarcity: ", "Solves the fundamental data acquisition bottleneck for rare conditions like Fusion beats.")
    add_bullet("HIPAA/GDPR Compliance: ", "Synthetic beats contain zero Protected Health Information (PHI), enabling safe cross-institutional research collaboration.")

    doc.add_page_break()

    # =========================================================================
    # 13. CONCLUSION
    # =========================================================================
    add_h1("13. CONCLUSION")
    add_p(
        "This project successfully designed, implemented, and validated CardioDiff, a novel soft computing framework combining "
        "Conditional 1D Denoising Diffusion Probabilistic Models (DDPM) with deep 1D Convolutional Neural Networks for bio-signal synthesis: "
    )
    add_bullet("Overcame Severe Skew: ", "On 109,446 real MIT-BIH heartbeats, CardioDiff increased Macro-F1 from 0.6554 to 0.8210 (+16.56%) and minority recall from 0.4898 to 0.7876 (+29.78%).")
    add_bullet("Critical Arrhythmia Breakthrough: ", "Sensitivity for rare Fusion beats surged from 21.60% to 77.16% (+55.56%), while Ventricular ectopic detection reached 93.51% (+29.56%).")
    add_bullet("Superiority Over Heuristics: ", "With a 1D-FID of 40.0401 and Diversity Ratio of 1.0253, CardioDiff prevented the overfitting of Random Oversampling and the distortion of linear interpolation.")
    add_bullet("Live Clinical Tool: ", "The system is accessible globally as an interactive diagnostic platform at https://cardiodiff.streamlit.app/.")

    add_h1("14. FUTURE SCOPE")
    add_bullet("Multi-Lead 12-Lead Synthesis: ", "Extending the 1D U-Net to multi-channel tensors (12 leads) to capture cross-lead spatial correlations.")
    add_bullet("Classifier-Free Guidance (CFG): ", "Implementing guidance scale parameterization during reverse inference to tune the fidelity-diversity Pareto frontier.")
    add_bullet("Edge AI Deployment: ", "Quantizing PyTorch model weights to INT8 / ONNX for real-time inference on wearable smartwatches and Holter patches.")

    doc.add_page_break()

    # =========================================================================
    # 15. REFERENCES
    # =========================================================================
    add_h1("15. REFERENCES")
    refs = [
        "1. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45–50.",
        "2. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 33, 6840–6851.",
        "3. Sohl-Dickstein, J., Weiss, E., Khanh, N., & Sompolinsky, H. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. International Conference on Machine Learning (ICML), 2256–2265.",
        "4. Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). ECG heartbeat classification: A deep transferable representation. IEEE International Conference on Healthcare Informatics (ICHI), 443–444.",
        "5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.",
        "6. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.",
        "7. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in Neural Information Processing Systems (NeurIPS), 30, 6626–6637.",
        "8. ANSI/AAMI EC57:1998/(R)2008. Testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms. Association for the Advancement of Medical Instrumentation.",
        "9. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), 234–241.",
        "10. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980."
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(r)
        run.font.size = Pt(10)

    doc.add_page_break()

    # =========================================================================
    # 16. APPENDIX
    # =========================================================================
    add_h1("16. APPENDIX: CORE CODE SNIPPETS")
    add_h2("Appendix A: Conditional 1D Residual Block (PyTorch)")
    add_code(
        "class ConvBlock1D(nn.Module):\n"
        "    def __init__(self, in_channels: int, out_channels: int, cond_dim: int = 128):\n"
        "        super().__init__()\n"
        "        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)\n"
        "        self.bn1 = nn.BatchNorm1d(out_channels)\n"
        "        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)\n"
        "        self.bn2 = nn.BatchNorm1d(out_channels)\n"
        "        self.cond_proj = nn.Linear(cond_dim, out_channels)\n"
        "        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()\n\n"
        "    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:\n"
        "        h = F.relu(self.bn1(self.conv1(x)))\n"
        "        h = h + self.cond_proj(cond).unsqueeze(-1)\n"
        "        h = self.bn2(self.conv2(h))\n"
        "        return F.relu(h + self.shortcut(x))\n"
    )

    add_h2("Appendix B: 1D Fréchet Inception Distance (1D-FID) Calculation")
    add_code(
        "def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):\n"
        "    diff = mu1 - mu2\n"
        "    covmean = linalg.sqrtm(sigma1.dot(sigma2))\n"
        "    if isinstance(covmean, tuple): covmean = covmean[0]\n"
        "    if np.iscomplexobj(covmean): covmean = covmean.real\n"
        "    tr_covmean = np.trace(covmean)\n"
        "    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean\n"
        "    return float(np.real(fid))\n"
    )

    output_path = "CardioDiff_Project_Report.docx"
    doc.save(output_path)
    print(f"[Word] Successfully generated: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)

if __name__ == "__main__":
    build_document()
