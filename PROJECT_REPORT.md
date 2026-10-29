# ACADEMIC PROJECT REPORT

---

# CARDIODIFF: 1D DENOISING DIFFUSION PROBABILISTIC MODELS FOR ECG SIGNAL SYNTHESIS & IMBALANCED ARRHYTHMIA CLASSIFICATION

**Course:** Soft Computing  
**Academic Year:** 2025 – 2026  
**Department:** Department of Computer Science & Engineering  
**Institute:** Faculty of Engineering and Technology, Adani University, Ahmedabad, Gujarat, India  

---

### Project Undertaken By:

| Sr. No. | Student Name | Enrollment / Roll No. |
| :---: | :--- | :---: |
| 1 | **Shah Varshil** | `1AUA24BCS919` |
| 2 | **Siddhant Patel** | `1AUA24BCS917` |
| 3 | **Varmil Parikh** | `1AUA24BCS914` |

### Faculty Guide & Supervisor:
**Dr. Krupali Donda**  
*Assistant Professor, Department of Computer Science & Engineering*  
*Adani University, Ahmedabad*

---

### Project Links:
- **Live Streamlit Cloud Deployment:** [https://cardiodiff.streamlit.app/](https://cardiodiff.streamlit.app/)
- **GitHub Source Code & Model Weights:** [https://github.com/varshil-nvm/cardiodiff](https://github.com/varshil-nvm/cardiodiff)

---

\newpage

# TABLE OF CONTENTS

- [1. Title Page](#1-title-page)
- [2. Abstract](#2-abstract)
- [3. Introduction](#3-introduction)
  - [3.1 Background of the Problem](#31-background-of-the-problem)
  - [3.2 Motivation](#32-motivation)
  - [3.3 Relevance in Soft Computing and Biomedical AI](#33-relevance-in-soft-computing-and-biomedical-ai)
  - [3.4 Overview of the Proposed Solution](#34-overview-of-the-proposed-solution)
- [4. Problem Statement & Objectives](#4-problem-statement--objectives)
  - [4.1 Problem Statement](#41-problem-statement)
  - [4.2 Specific Objectives](#42-specific-objectives)
- [5. Literature Review / Existing Work](#5-literature-review--existing-work)
  - [5.1 Conventional Imbalanced Learning in Arrhythmia](#51-conventional-imbalanced-learning-in-arrhythmia)
  - [5.2 Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs)](#52-generative-adversarial-networks-gans-and-variational-autoencoders-vaes)
  - [5.3 Gaps and Limitations in Existing Paradigms](#53-gaps-and-limitations-in-existing-paradigms)
  - [5.4 Distinct Contributions of CardioDiff](#54-distinct-contributions-of-cardiodiff)
- [6. Proposed Methodology](#6-proposed-methodology)
  - [6.1 Overall End-to-End System Workflow](#61-overall-end-to-end-system-workflow)
  - [6.2 System Architecture Flowchart](#62-system-architecture-flowchart)
  - [6.3 1D-CNN Arrhythmia Classifier Architecture](#63-1d-cnn-arrhythmia-classifier-architecture)
  - [6.4 Class-Conditioned 1D DDPM Architecture](#64-class-conditioned-1d-ddpm-architecture)
- [7. Soft Computing Technique: Denoising Diffusion Probabilistic Models (DDPM)](#7-soft-computing-technique-denoising-diffusion-probabilistic-models-ddpm)
  - [7.1 Fundamental Concept of Diffusion in Continuous Feature Spaces](#71-fundamental-concept-of-diffusion-in-continuous-feature-spaces)
  - [7.2 Mathematical Foundations: Forward (Diffusion) and Reverse (Denoising) Processes](#72-mathematical-foundations-forward-diffusion-and-reverse-denoising-processes)
  - [7.3 Class Conditioning and Temporal Sinusoidal Embeddings](#73-class-conditioning-and-temporal-sinusoidal-embeddings)
  - [7.4 Algorithmic Implementations & Pseudocode](#74-algorithmic-implementations--pseudocode)
- [8. Dataset & Input Data Preprocessing](#8-dataset--input-data-preprocessing)
  - [8.1 Dataset Source & Clinical Description](#81-dataset-source--clinical-description)
  - [8.2 Class Distribution & Severe Imbalance Profile](#82-class-distribution--severe-imbalance-profile)
  - [8.3 Beat Segmentation and Normalization](#83-beat-segmentation-and-normalization)
- [9. Implementation Details](#9-implementation-details)
  - [9.1 Software, Libraries, and Execution Environment](#91-software-libraries-and-execution-environment)
  - [9.2 Hardware and Compute Specifications](#92-hardware-and-compute-specifications)
  - [9.3 Modular Directory & Codebase Architecture](#93-modular-directory--codebase-architecture)
  - [9.4 Hyperparameters & Training Configurations](#94-hyperparameters--training-configurations)
- [10. Experimental Setup & Comparative Analysis](#10-experimental-setup--comparative-analysis)
  - [10.1 Experimental Scenarios](#101-experimental-scenarios)
  - [10.2 Evaluation Metrics Formulations](#102-evaluation-metrics-formulations)
- [11. Results & Discussion](#11-results--discussion)
  - [11.1 Quantitative Classification Performance](#111-quantitative-classification-performance)
  - [11.2 Per-Class Recall & Precision Breakdown](#112-per-class-recall--precision-breakdown)
  - [11.3 Generative Quality: 1D-FID and Diversity Ratio](#113-generative-quality-1d-fid-and-diversity-ratio)
  - [11.4 Confusion Matrix & Latent Manifold Projection Analysis](#114-confusion-matrix--latent-manifold-projection-analysis)
- [12. Cloud Deployment & Interactive Dashboard](#12-cloud-deployment--interactive-dashboard)
  - [12.1 Deployment Architecture](#121-deployment-architecture)
  - [12.2 Dashboard Feature Walkthrough (4 Functional Tabs)](#122-dashboard-feature-walkthrough-4-functional-tabs)
  - [12.3 Clinical Utility and Practical Applicability](#123-clinical-utility-and-practical-applicability)
- [13. Conclusion](#13-conclusion)
- [14. Future Scope](#14-future-scope)
- [15. References](#15-references)
- [16. Appendix](#16-appendix)

---

\newpage

# 2. ABSTRACT

Cardiovascular diseases (CVDs) remain the leading cause of global mortality, demanding automated, real-time electrocardiogram (ECG) arrhythmia classification for patient monitoring. However, real-world clinical bio-signal datasets exhibit severe **class imbalance**, where normal heartbeats overpower critical, life-threatening arrhythmias (such as Premature Ventricular Contractions, Supraventricular Ectopic beats, and Ventricular Fusion beats). Machine learning classifiers trained on such unmitigated distributions exhibit drastic bias towards the majority class, sacrificing minority diagnostic recall. 

To overcome this fundamental challenge, this project develops **CardioDiff**, a soft computing framework combining **Conditional 1D Denoising Diffusion Probabilistic Models (DDPM)** with deep **1D Convolutional Neural Networks (1D-CNN)** for physiological heartbeat synthesis and augmented classification. CardioDiff models the non-linear, stochastic data distribution of 125 Hz ECG waveforms (187 points) across a parameterized Markov chain, learning the underlying manifold of underrepresented cardiac conditions. 

Trained and evaluated on **109,446 real clinical heartbeats** from the benchmark **PhysioNet / Kaggle MIT-BIH Arrhythmia Database** (87,554 training beats, 21,892 held-out test beats), CardioDiff achieves an overall accuracy of **96.90%**, a Macro-F1 score of **0.8210** (a **+16.56%** absolute improvement over the baseline 0.6554), and minority diagnostic recall of **0.7876** (a **+29.78%** increase over the baseline 0.4898). The synthesized waveforms attain a **1D Fréchet Inception Distance (1D-FID) of 40.0401** and an optimal **Diversity Ratio of 1.0253**, indicating high morphological fidelity without mode collapse. The end-to-end framework is containerized and deployed live on Streamlit Cloud ([https://cardiodiff.streamlit.app/](https://cardiodiff.streamlit.app/)).

---

# 3. INTRODUCTION

## 3.1 Background of the Problem
The electrocardiogram (ECG) represents the non-invasive recording of electrical potentials generated by cardiac muscle depolarization and repolarization throughout each cardiac cycle. A normal sinus heartbeat displays characteristic physiological deflections: the **P wave** (atrial depolarization), the **QRS complex** (ventricular depolarization), and the **T wave** (ventricular repolarization). Variations in the duration, amplitude, or morphological structure of these intervals serve as diagnostic indicators of cardiac arrhythmias. 

While automated machine learning algorithms have achieved impressive performance in controlled benchmark settings, real-world deployment is severely obstructed by extreme epidemiological class imbalance. Out of routine Holter recordings or ambulatory ECG feeds, over 80% to 90% of recorded heartbeats belong to normal rhythm, whereas malignant conditions—such as Ventricular Ectopic Beats (PVCs) or Fusion Beats—manifest infrequently.

## 3.2 Motivation
When standard connectionist models (e.g., deep neural networks) minimize empirical risk over severely skewed training sets, the loss gradient is dominated by normal heartbeats. Consequently, the network achieves an artificially high global accuracy (>94%) simply by predicting the majority class, while exhibiting disastrous false-negative rates on critical arrhythmias. In a medical diagnostic context, missing a malignant ventricular event can lead to undetected ventricular fibrillation and cardiac arrest. 

Traditional statistical oversampling techniques (such as Random Oversampling) merely duplicate identical signals, leading to acute overfitting on specific noise patterns. Classical heuristic interpolation methods (such as SMOTE) create linear blends between distinct cardiac beats that violate biophysical laws, yielding non-physiological waveforms with corrupted ST segments and distorted QRS peaks. Therefore, there is a compelling need for advanced generative soft computing paradigms that model the continuous, high-dimensional probability distribution of biological time-series data.

## 3.3 Relevance in Soft Computing and Biomedical AI
Soft computing addresses complex, non-linear, uncertain, and stochastic real-world problems through connectionist architectures, probabilistic reasoning, and evolutionary/generative learning. **Denoising Diffusion Probabilistic Models (DDPM)** represent a premier probabilistic soft computing paradigm. Rooted in non-equilibrium thermodynamics and stochastic calculus, diffusion models reformulate generative modeling as learning to reverse a gradual physical diffusion process that injects Gaussian uncertainty into signals. Coupled with **1D Convolutional Neural Networks**, this paradigm captures spatial-temporal bio-signal invariances and generates continuous, biologically valid synthetic samples.

## 3.4 Overview of the Proposed Solution
**CardioDiff** is an integrated end-to-end soft computing pipeline:
1. **Bio-Signal Pipeline:** Ingests and standardizes 187-sample single-lead ECG recordings from the MIT-BIH dataset into normalized amplitudes $\in [0, 1]$.
2. **Deep Baseline 1D-CNN:** Trained directly on raw imbalanced distributions to benchmark majority bias and establish an empirical performance lower bound.
3. **Class-Conditioned 1D DDPM Denoiser:** Features a specialized 1D U-Net backbone with skip connections, sinusoidal time-step embeddings, and learnable class conditioning vectors to model cardiac waveforms over $T=40$ diffusion steps.
4. **Physiological Minority Augmentation:** Synthesizes high-fidelity minority heartbeats to restore balance in the training distribution.
5. **Augmented 1D-CNN Classifier:** Trained on the expanded distribution, demonstrating massive recoveries in Macro-F1 and minority recall.
6. **Live Cloud Deployment:** Deployed as a multi-functional interactive web platform on Streamlit Cloud featuring real-time signal generation, forward/reverse diffusion trajectory inspection, latent manifold PCA exploration, and benchmark visualizations.

---

# 4. PROBLEM STATEMENT & OBJECTIVES

## 4.1 Problem Statement
Given a discrete 1D electrocardiogram signal $x \in \mathbb{R}^{187}$ representing a segmented cardiac cycle sampled at 125 Hz, and an associated clinical diagnosis $y \in \{0, 1, 2, 3, 4\}$ corresponding to the ANSI/AAMI EC57 standard:
- Class 0: Normal Sinus Rhythm (N)
- Class 1: Supraventricular Ectopic Beat (S)
- Class 2: Premature Ventricular Contraction (V)
- Class 3: Fusion of Ventricular and Normal Beat (F)
- Class 4: Unclassifiable / Paced Beat (Q)

The dataset exhibits extreme disproportion ($|C_0| \gg |C_4| > |C_2| \gg |C_1| \gg |C_3|$), where Fusion beats constitute less than 0.8% of the data. 

**The practical problem is:** How can a generative soft computing framework learn the true probability distribution $p(x|y)$ of scarce cardiac arrhythmias to synthesize realistic, diverse, electrophysiologically sound heartbeats, thereby eliminating classification bias and dramatically elevating minority diagnostic recall without degrading overall accuracy?

## 4.2 Specific Objectives
1. **Architect a Conditional 1D DDPM:** Design a 1D U-Net denoiser tailored for continuous single-lead time-series signals with temporal sinusoidal encodings and categorical class embedding projections.
2. **Implement a Penultimate Latent Feature Extractor:** Develop a hierarchical 3-block 1D-CNN classifier whose 64-dimensional penultimate bottleneck enables rigorous feature extraction, evaluation of 1D Fréchet Inception Distance (1D-FID), and manifold projection.
3. **Overcome Minority Class Blindness:** Elevate minority diagnostic recall (specifically for S, V, and F arrhythmias) by at least 25% over the unaugmented baseline on the test partition.
4. **Prevent Mode Collapse:** Maintain sample diversity with a Diversity Ratio near 1.0, avoiding the memorization pitfalls of Random Oversampling (ROS).
5. **Develop an Interactive Clinical Dashboard:** Deploy an interactive, web-based tool with real-time signal synthesis, step-by-step diffusion inspection, metrics analysis, and latent space visualization on Streamlit Cloud.

---

# 5. LITERATURE REVIEW / EXISTING WORK

## 5.1 Conventional Imbalanced Learning in Arrhythmia
Arrhythmia detection has traditionally utilized classical machine learning models (Support Vector Machines, Random Forests) and shallow Neural Networks operating on hand-crafted morphological features (R-peak amplitude, PR interval, QRS duration, QT interval). To counteract class imbalance, common data-level strategies have included:
- **Random Undersampling (RUS):** Discards majority class samples, discarding valuable epidemiological variability.
- **Random Oversampling (ROS):** Replicates minority samples verbatim, causing decision boundaries to tightly contract around repeated data points and causing severe overfitting.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Operates via $k$-nearest neighbor linear interpolation in raw feature space:
  $$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \sim U(0, 1)$$
  In non-linear physiological time series, linear combinations of two out-of-phase ECG beats produce flattened QRS peaks, artificial baseline wander, and non-physiological wave morphology.

## 5.2 Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs)
With the rise of deep generative models, researchers explored 1D-GANs (e.g., WaveGAN, BiWGAN) and 1D-VAEs for ECG synthesis:
- **VAEs:** Optimize an evidence lower bound (ELBO) using an explicit Gaussian prior. While computationally efficient, VAEs suffer from blurriness and over-smoothing in time-series signals due to the injective nature of the reconstruction loss, failing to preserve sharp QRS spikes.
- **GANs:** Employ an adversarial game between a generator and a discriminator. While capable of producing sharp waveforms, 1D-GANs are notorious for training instability, gradient vanishing, hyperparameter hypersensitivity, and **mode collapse** (where the generator produces only a single prototypical ventricular beat repeatedly).

## 5.3 Gaps and Limitations in Existing Paradigms

| Methodology | Morphological Fidelity | Mode Diversity | Training Stability | Biological Plausibility |
| :--- | :---: | :---: | :---: | :---: |
| **ROS** | Identical copies | Zero new diversity | High | N/A (Duplicate) |
| **SMOTE** | Low (Linear artifacts) | Moderate | High | Poor (Non-physiological) |
| **1D-VAE** | Moderate (Over-smoothed) | High | High | Moderate (Blunted peaks) |
| **1D-GAN** | High | Low (Mode collapse) | Poor (Adversarial divergence) | Moderate |
| **CardioDiff (DDPM)** | **High (Sharp QRS/T)** | **High (Diversity ~1.02)** | **Stable (MSE loss)** | **High (Electrophysiological)** |

## 5.4 Distinct Contributions of CardioDiff
CardioDiff eliminates the adversarial instability of GANs by formulating generation as a gradual stochastic denoising process trained with a stable mean squared error (MSE) objective. It replaces linear heuristic interpolation with class-conditioned iterative Langevin dynamics along the true non-linear probability density function $p(x|y)$, generating diverse, physiologically coherent ECG beats.

---

# 6. PROPOSED METHODOLOGY

## 6.1 Overall End-to-End System Workflow
The CardioDiff framework consists of four sequential stages:
1. **Data Ingestion & Formatting:** Loading 109,446 MIT-BIH beats, standardizing length to $L=187$ at 125 Hz, and scaling to $[0, 1]$.
2. **Baseline Evaluation:** Establishing baseline diagnostic metrics with an unaugmented 1D-CNN.
3. **Stochastic Diffusion Training & Generation:** Subsampling a balanced distribution across all 5 classes to train the conditional 1D DDPM U-Net denoiser, followed by conditional sampling of synthetic minority beats.
4. **Augmented Classifier Training & Validation:** Training a twin 1D-CNN on the augmented training set and benchmarking performance on the untouched 21,892 test samples.

## 6.2 System Architecture Flowchart

```mermaid
flowchart TD
    subgraph Data_Pipeline ["1. Data Ingestion & Preprocessing"]
        A["MIT-BIH Database<br/>(109,446 Total Heartbeats)"] --> B["Train Partition: 87,554 beats<br/>Test Partition: 21,892 beats"]
        B --> C["Min-Max Normalized<br/>(187 Time Samples, [0, 1])"]
    end

    subgraph Baseline_Stage ["2. Baseline Classifier"]
        C --> D["Train Baseline 1D-CNN<br/>(Imbalanced Data)"]
        D --> E["Baseline Checkpoint<br/>classifier_baseline.pt"]
    end

    subgraph Diffusion_Stage ["3. Conditional 1D DDPM"]
        C --> F["Balanced Subsampler<br/>(5,441 Multi-Class Beats)"]
        F --> G["Conditional 1D U-Net Denoiser<br/>(T=40 Timesteps, MSE Loss)"]
        G --> H["Diffusion Checkpoint<br/>diffusion_ecg.pt"]
        H --> I["Conditional Reverse Sampling<br/>(Minority Classes S, V, F, Q)"]
        I --> J["High-Fidelity Synthetic Beats<br/>(400 Validated Samples)"]
    end

    subgraph Augmented_Stage ["4. Augmented Classifier & Evaluation"]
        C --> K["Concatenate Real + Synthetic"]
        J --> K
        K --> L["Train Augmented 1D-CNN<br/>(87,954 Samples)"]
        L --> M["Augmented Checkpoint<br/>classifier_augmented.pt"]
        E --> N["Comparative Test Evaluation<br/>(21,892 Held-Out Test Beats)"]
        M --> N
        N --> O["1D-FID, Diversity Ratio,<br/>Macro-F1, Confusion Matrix"]
    end

    subgraph Cloud_Deployment ["5. Interactive Deployment"]
        M & H & O --> P["Streamlit Cloud Dashboard<br/>cardiodiff.streamlit.app"]
    end
```

## 6.3 1D-CNN Arrhythmia Classifier Architecture
The classifier serves dual roles: diagnosing heartbeats and acting as a feature extractor for 1D-FID calculations and manifold projections.
- **Input Layer:** $(B, 1, 187)$
- **Convolutional Block 1:** $\text{Conv1D}(1 \to 32, k=7, s=1, p=3) \to \text{BatchNorm1D}(32) \to \text{ReLU} \to \text{MaxPool1D}(2, s=2)$
- **Convolutional Block 2:** $\text{Conv1D}(32 \to 64, k=5, s=1, p=2) \to \text{BatchNorm1D}(64) \to \text{ReLU} \to \text{MaxPool1D}(2, s=2)$
- **Convolutional Block 3:** $\text{Conv1D}(64 \to 128, k=5, s=1, p=2) \to \text{BatchNorm1D}(128) \to \text{ReLU} \to \text{AdaptiveAvgPool1D}(1)$
- **Bottleneck Projection:** $\text{Flatten} \to \text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Dropout}(p=0.2)$
- **Classification Head:** $\text{Linear}(64 \to 5)$ predicting logits across $\{N, S, V, F, Q\}$.

```mermaid
graph LR
    In["Input Beat<br/>(1, 187)"] --> C1["Conv1D (32, k=7)<br/>BN + ReLU + MaxPool"]
    C1 --> C2["Conv1D (64, k=5)<br/>BN + ReLU + MaxPool"]
    C2 --> C3["Conv1D (128, k=5)<br/>BN + ReLU + AdaptivePool"]
    C3 --> FC1["Dense Bottleneck<br/>(64-dim Latent)"]
    FC1 --> FC2["Dense Output<br/>(5 Logits)"]
    FC1 -.-> FID["1D-FID & PCA<br/>Manifold Extractor"]
```

## 6.4 Class-Conditioned 1D DDPM Architecture
The generative core is a 1D U-Net denoiser parameterized by $\theta$.
- **Conditioning Projections:** 
  - Timestep $t \in [0, T-1]$ is mapped to a 128-dimensional vector via sinusoidal positional encoding.
  - Class label $y \in \{0, 1, 2, 3, 4\}$ is embedded into a 128-dimensional continuous vector.
  - The sum $c = \text{emb}(t) + \text{emb}(y)$ is passed through a 2-layer MLP and injected additively into each residual block.
- **Encoder Path:** 
  - Initial Conv ($1 \to 32$)
  - DownBlock 1: ConvBlock1D ($32 \to 64$) + MaxPool1D($2$)
  - DownBlock 2: ConvBlock1D ($64 \to 128$) + MaxPool1D($2$)
- **Bottleneck:** Two stacked residual ConvBlock1D layers ($128 \to 128$) with conditioned affine modulations.
- **Decoder Path:**
  - UpBlock 2: Bilinear Interpolation UpSampling $\to$ Skip Connection Concatenation $\to$ ConvBlock1D ($192 \to 64$)
  - UpBlock 1: UpSampling $\to$ Concatenation $\to$ ConvBlock1D ($96 \to 32$)
- **Output Projection:** Conv1D ($32 \to 1$) reconstructing predicted Gaussian noise $\epsilon_\theta(x_t, t, y)$.

---

# 7. SOFT COMPUTING TECHNIQUE: DENOISING DIFFUSION PROBABILISTIC MODELS (DDPM)

## 7.1 Fundamental Concept of Diffusion in Continuous Feature Spaces
In soft computing, generative modeling can be formulated as learning a vector field that points toward regions of high probability density in a data manifold. DDPM constructs a parameterized forward Markov chain that systematically adds Gaussian noise to an ECG signal until it degrades into pure isotropic white noise, alongside a reverse parameterized neural network that inverts this destruction step-by-step.

```mermaid
flowchart LR
    x0["x_0 (Real ECG)"] -- "q(x_1|x_0)" --> x1["x_1"]
    x1 -- "q(x_t|x_{t-1})" --> xt["x_t (Noisy)"]
    xt -- "q(x_T|x_{T-1})" --> xT["x_T ~ N(0, I)"]
    xT -. "p_theta(x_{T-1}|x_T, y)" .-> xt
    xt -. "p_theta(x_0|x_1, y)" .-> x0
```

## 7.2 Mathematical Foundations: Forward and Reverse Processes

### Forward Diffusion Process ($q$):
Given a real heartbeat $x_0 \sim q(x_0)$, the forward Markovian process adds Gaussian noise according to a predefined variance schedule $\beta_1, \beta_2, \dots, \beta_T$:
$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}\right)$$

A key property of Gaussian distributions allows sampling $x_t$ at any arbitrary timestep $t$ in closed form without simulating intermediate steps:
$$q(x_t | x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
Where:
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

Consequently, the perturbed signal is expressed analytically as:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### Reverse Denoising Process ($p_\theta$):
The true posterior $q(x_{t-1}|x_t, x_0)$ is intractable without knowing $x_0$. We approximate it using a deep neural network $p_\theta(x_{t-1} | x_t, y)$:
$$p_\theta(x_{t-1} | x_t, y) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t, y), \Sigma_\theta(x_t, t)\right)$$

Reparameterizing the mean $\mu_\theta$ in terms of the noise estimator $\epsilon_\theta(x_t, t, y)$ yields:
$$\mu_\theta(x_t, t, y) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, y) \right)$$

The posterior variance is computed analytically via:
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

### Training Objective:
The network parameters $\theta$ are optimized by minimizing the mean-squared error (MSE) between the true noise $\epsilon$ and predicted noise $\epsilon_\theta$:
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, y, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t, y\right) \right\|^2 \right]$$

## 7.3 Class Conditioning and Temporal Sinusoidal Embeddings
Temporal awareness is enforced through sinusoidal encodings of timestep $t$:
$$\text{PE}_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}_{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i/d}}\right)$$
The discrete arrhythmia class label $y \in \{0, 1, 2, 3, 4\}$ is mapped via a learnable dictionary $\mathbf{E} \in \mathbb{R}^{5 \times d}$ ($d=128$). The unified conditioning vector is projected into channel-wise scale and shift parameters across all 1D convolutional residual blocks.

## 7.4 Algorithmic Implementations & Pseudocode

```text
Algorithm 1: Conditional 1D DDPM Training for ECG Synthesis
Input: Training dataset D = {(x_i, y_i)}, timesteps T=40, beta schedule {beta_1..beta_T}
Output: Trained noise estimation network epsilon_theta
1: repeat
2:    Sample batch of real ECG beats (x_0, y) ~ D
3:    Sample random timesteps t ~ Uniform({1, ..., T})
4:    Sample standard Gaussian noise epsilon ~ N(0, I) of shape (B, 1, 187)
5:    Compute noisy signals:
         x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
6:    Predict noise using 1D U-Net:
         pred_epsilon = epsilon_theta(x_t, t, y)
7:    Calculate loss:
         Loss = Mean_Squared_Error(pred_epsilon, epsilon)
8:    Compute gradient nabla_theta(Loss) and update parameters via AdamW
9: until converged
```

```text
Algorithm 2: Class-Conditioned Reverse Sampling for ECG Augmentation
Input: Desired arrhythmia class y in {1, 2, 3, 4}, number of samples N, trained model epsilon_theta
Output: Synthetic physiological ECG signals x_0 of shape (N, 1, 187)
1: Initialize x_T ~ N(0, I) of shape (N, 1, 187)
2: for t = T down to 1 do
3:    Sample z ~ N(0, I) if t > 1, else z = 0
4:    Evaluate predicted noise:
         pred_eps = epsilon_theta(x_t, t, y)
5:    Compute predicted mean:
         mu = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * pred_eps)
6:    Compute posterior variance sigma_t = sqrt(tilde_beta_t)
7:    Update signal:
         x_{t-1} = mu + sigma_t * z
8: end for
9: Normalize output waveform amplitudes to [0, 1]
10: return x_0
```

---

# 8. DATASET & INPUT DATA PREPROCESSING

## 8.1 Dataset Source & Clinical Description
This project utilizes the benchmark **MIT-BIH Arrhythmia Database**, curated by Beth Israel Hospital and MIT, distributed through PhysioNet and Kaggle. The dataset contains 48 half-hour two-channel ambulatory ECG recordings obtained from 47 clinical subjects. Heartbeats are extracted, preprocessed, and categorized under the Association for the Advancement of Medical Instrumentation (**AAMI EC57**) standard.

## 8.2 Class Distribution & Severe Imbalance Profile
The combined dataset consists of **109,446 individual beats**, partitioned into a training set ($N_{\text{train}} = 87,554$) and an independent test set ($N_{\text{test}} = 21,892$).

| AAMI Class | Arrhythmia Category | Train Count | Train Proportion | Test Count | Test Proportion | Imbalance Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0 (N)** | Normal Sinus Rhythm | 72,471 | 82.77% | 18,118 | 82.76% | 1.00 (Majority) |
| **1 (S)** | Supraventricular Ectopic | 2,223 | 2.54% | 556 | 2.54% | 1 : 32.6 |
| **2 (V)** | Premature Ventricular Contraction | 5,788 | 6.61% | 1,448 | 6.61% | 1 : 12.5 |
| **3 (F)** | Fusion of Ventricular and Normal | 641 | 0.73% | 162 | 0.74% | **1 : 113.1** |
| **4 (Q)** | Unclassifiable / Paced Beat | 6,431 | 7.35% | 1,608 | 7.35% | 1 : 11.3 |
| **Total** | | **87,554** | **100.00%** | **21,892** | **100.00%** | |

The table exposes an extreme imbalance ratio of **1 : 113.1** between Normal beats and Fusion beats, confirming why traditional classifiers fail to learn robust representations for Class 3.

## 8.3 Beat Segmentation and Normalization
1. **R-Peak Centering:** Each heartbeat is segmented around its local R-peak, resampled to a fixed sampling frequency of **125 Hz**, yielding exactly **187 discrete time points** per heartbeat.
2. **Zero-Padding & Boundary Handling:** Trailing values post-T-wave are zero-padded to maintain a uniform input tensor shape $(B, 1, 187)$.
3. **Amplitude Normalization:** Each heartbeat is scaled to the interval $[0, 1]$ via min-max scaling:
   $$x_{\text{norm}}[n] = \frac{x[n] - \min(x)}{\max(x) - \min(x) + \epsilon}$$

---

# 9. IMPLEMENTATION DETAILS

## 9.1 Software, Libraries, and Execution Environment
- **Core Language:** Python 3.14 / 3.12
- **Deep Learning Framework:** PyTorch 2.15 (1D Convolutional operators, automatic differentiation, custom DDPM modules)
- **Numerical & Scientific Libraries:** NumPy, SciPy (Matrix square roots via `scipy.linalg.sqrtm` for 1D-FID calculation)
- **Data Engineering:** Pandas
- **Visualization:** Matplotlib, Seaborn, Plotly Express
- **Machine Learning Evaluation:** Scikit-Learn (Macro-F1, Confusion Matrix, PCA decomposition)
- **Interactive Framework:** Streamlit (Multi-tab application, state caching, responsive charts)
- **Version Control & CI/CD:** Git, GitHub API, Streamlit Cloud

## 9.2 Hardware and Compute Specifications
- **Local Development Environment:** Windows 11 x64, Intel Core i7 / Multi-core CPU configuration, 16 GB DDR5 RAM.
- **Compute Optimization:** Parallelized PyTorch multi-threading on CPU.
- **Training Durations:**
  - Baseline 1D-CNN (4 epochs on 87,554 beats): 178.4 seconds
  - Balanced Conditional DDPM (5 epochs on 5,441 beats, $T=40$): 188.6 seconds
  - Augmented 1D-CNN (4 epochs on 87,954 beats): 89.7 seconds
  - Total Training Cycle: ~7.6 minutes

## 9.3 Modular Directory & Codebase Architecture
```text
cardiodiff/
├── app.py                      # Multi-tab Streamlit dashboard
├── train_all.py                # Pipeline orchestrator (Baseline, DDPM, Augmentation)
├── evaluate.py                 # Evaluation engine (1D-FID, Diversity, Plots)
├── colab_train.md              # Google Colab GPU guide
├── REPORT.md                   # Concise Markdown summary
├── PROJECT_REPORT.md           # Full academic project report
├── requirements.txt            # Dependency manifest
├── models/
│   ├── ecg_data.py             # MIT-BIH dataset loader & Gaussian generator
│   ├── classifier.py           # 1D-CNN classifier & 64-dim feature extractor
│   ├── diffusion_1d.py         # Conditional 1D DDPM Denoiser (U-Net)
│   └── metrics.py              # Macro-F1, Minority Recall, 1D-FID, Diversity
├── weights/
│   ├── classifier_baseline.pt  # Baseline 1D-CNN weights (253 KB)
│   ├── diffusion_ecg.pt        # Trained 1D DDPM weights (2.9 MB)
│   └── classifier_augmented.pt # Augmented 1D-CNN weights (253 KB)
└── results/
    ├── metrics.csv             # Quantitative results table
    ├── metrics.json            # Classification report & confusion matrices
    ├── confusion_matrix.png    # Confusion matrix comparison
    ├── ecg_samples_grid.png    # Real vs. synthetic waveform comparisons
    ├── fid_diversity_comparison.png # Generative metric charts
    └── latent_space_tsne.png   # PCA latent manifold projection
```

## 9.4 Hyperparameters & Training Configurations

| Module | Hyperparameter | Configured Value | Description |
| :--- | :--- | :---: | :--- |
| **Diffusion** | Timesteps ($T$) | 40 | Linear variance diffusion steps |
| | Variance Schedule | $\beta_1=10^{-4} \to \beta_T=0.02$ | Standard linear schedule |
| | Base Channels | 32 | Initial convolutional channel width |
| | Conditioning Dim | 128 | Sinusoidal time & class embedding dim |
| | Optimizer / LR | AdamW / $10^{-3}$ | Weight decay: $10^{-4}$ |
| | Epochs / Batch Size | 5 / 128 | Trained on balanced 5,441-beat subset |
| **Classifier** | Hidden Bottleneck | 64 | Penultimate embedding dimension |
| | Dropout Rate | 0.2 | Applied to bottleneck dense layer |
| | Optimizer / LR | AdamW / $2 \times 10^{-3}$ | Weight decay: $10^{-4}$ |
| | Epochs / Batch Size | 4 / 128 | Full dataset batching |

---

# 10. EXPERIMENTAL SETUP & COMPARATIVE ANALYSIS

## 10.1 Experimental Scenarios
To assess the effectiveness of CardioDiff, three rigorous experimental scenarios were conducted on the official test set (21,892 heartbeats):
1. **Scenario A (Baseline Classifier):** 1D-CNN trained exclusively on the raw, imbalanced MIT-BIH training set (87,554 beats) without any re-sampling or generative augmentation.
2. **Scenario B (Random Oversampling - ROS):** Benchmark ablation where minority classes are synthetically replicated via uniform random duplication.
3. **Scenario C (CardioDiff DDPM Augmented - Ours):** 1D-CNN trained on the training set augmented with 100 conditional DDPM-generated heartbeats per minority class (1, 2, 3, 4).

## 10.2 Evaluation Metrics Formulations

### Macro-Averaged F1 Score:
$$\text{Macro-F1} = \frac{1}{C} \sum_{c=0}^{C-1} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### Minority Diagnostic Recall:
$$\text{Minority Recall} = \frac{1}{4} \sum_{c=1}^{4} \text{Recall}_c = \frac{1}{4} \left( R_S + R_V + R_F + R_Q \right)$$

### 1D Fréchet Inception Distance (1D-FID):
Let $\phi(x) \in \mathbb{R}^{64}$ denote the penultimate bottleneck representation extracted by the trained 1D-CNN. Real and generated embeddings follow continuous Gaussian distributions $\mathcal{N}(\mu_r, \Sigma_r)$ and $\mathcal{N}(\mu_g, \Sigma_g)$:
$$\text{1D-FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$
Lower 1D-FID scores correlate directly with superior electrophysiological realism.

### Diversity Ratio:
$$\text{Diversity Ratio} = \frac{\frac{1}{\binom{N_g}{2}} \sum_{i < j} \|g_i - g_j\|_2}{\frac{1}{\binom{N_r}{2}} \sum_{i < j} \|r_i - r_j\|_2}$$
A ratio near $1.0$ confirms generated samples exhibit the natural geometric dispersion of real heartbeats without collapsing into identical waveforms.

---

# 11. RESULTS & DISCUSSION

## 11.1 Quantitative Classification Performance
The quantitative outcomes on the 21,892 test heartbeats are summarized below:

| Model / Methodology | Overall Accuracy | Macro-F1 Score | Minority Recall | 1D-FID (Fidelity) | Diversity Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1D-CNN (Imbalanced)** | 94.46% | 0.6554 | 0.4898 | N/A | N/A |
| **Random Oversampling (ROS)** | 93.04% | 0.7300 | 0.6387 | N/A | N/A |
| **CardioDiff DDPM Augmented (Ours)** | **96.90%** | **0.8210** | **0.7876** | **40.0401** | **1.0253** |
| **Absolute Gain vs. Baseline** | **+2.44%** | **+16.56%** | **+29.78%** | *Strong Realism* | *Optimal Diversity* |

## 11.2 Per-Class Recall & Precision Breakdown

| Class | Diagnostic Label | Test Support | Baseline Recall | CardioDiff Recall | Recall Gain | CardioDiff F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | Normal Sinus Rhythm (N) | 18,118 | 99.85% | 98.87% | -0.98% | **0.9845** |
| **1** | Supraventricular Ectopic (S) | 556 | 14.03% | **47.66%** | **+33.63%** | **0.6243** |
| **2** | Premature Ventricular (V) | 1,448 | 63.95% | **93.51%** | **+29.56%** | **0.9192** |
| **3** | Fusion Beat (F) | 162 | 21.60% | **77.16%** | **+55.56%** | **0.5995** |
| **4** | Paced / Unknown Beat (Q) | 1,608 | 96.33% | **96.70%** | **+0.37%** | **0.9777** |

### Detailed Performance Analysis:
1. **Breakthrough on Rarest Arrhythmia (Class 3 - Fusion Beat):**
   With only 641 training beats and 162 test beats, the unaugmented baseline suffered a disastrous **21.60% recall** (missing nearly 80% of fusion beats). CardioDiff elevated Fusion recall to **77.16%**, a **+55.56% absolute improvement**, while maintaining a high precision of 0.490.
2. **Ventricular Contraction Protection (Class 2 - PVC):**
   Premature Ventricular Contractions are clinical precursors to ventricular tachycardia. Baseline recall of 63.95% is clinically unacceptable. CardioDiff boosted recall to **93.51%** (+29.56%), with an F1 score of **0.9192**.
3. **Supraventricular Recovery (Class 1 - S):**
   Baseline recall was a dismal 14.03%. CardioDiff more than tripled recall to **47.66%** (+33.63%), driving the F1 score from 0.2419 to **0.6243**.
4. **Preservation of Majority Specificity:**
   Normal heartbeat recall was preserved at **98.87%** (only a negligible 0.98% drop from 99.85%), ensuring the classifier does not trigger excessive false alarms.

## 11.3 Generative Quality: 1D-FID and Diversity Ratio
- **1D Fréchet Inception Distance (1D-FID = 40.0401):** The low Fréchet distance demonstrates that the 1D U-Net denoiser learns physiological priors (steep R-wave slopes, physiological ST elevation/depression, and smooth T-wave decays) rather than outputting high-frequency noise.
- **Diversity Ratio (1.0253):** A diversity ratio of exactly 1.0253 confirms that synthetic heartbeats span the natural clinical dispersion of real patient beats without mode collapse (which typically causes the ratio to collapse toward 0.0).

## 11.4 Confusion Matrix & Latent Manifold Projection Analysis
- **Confusion Matrix Dynamics:** In the baseline confusion matrix, massive numbers of minority beats (478 Class 1 beats, 377 Class 2 beats, and 125 Class 3 beats) were misclassified as Class 0 Normal. In the CardioDiff augmented confusion matrix, false negatives into Class 0 dropped dramatically (Class 2 misclassifications dropped from 377 to 47; Class 3 misclassifications dropped from 125 to 23).
- **Latent Space PCA Visualization:** Projecting the penultimate 64-dimensional feature embeddings onto the first two principal components shows distinct, well-separated clusters for each arrhythmia class. Synthetic DDPM samples lie directly along the boundaries of their respective real class clusters, validating that diffusion augmentation expands decision boundaries safely without creating out-of-distribution outliers.

---

# 12. CLOUD DEPLOYMENT & INTERACTIVE DASHBOARD

## 12.1 Deployment Architecture
CardioDiff is deployed as a cloud-native application on **Streamlit Cloud** linked to the GitHub repository:
- **Public URL:** [https://cardiodiff.streamlit.app/](https://cardiodiff.streamlit.app/)
- **Repository:** [https://github.com/varshil-nvm/cardiodiff](https://github.com/varshil-nvm/cardiodiff)
- **Deployment Platform:** Streamlit Cloud (Debian Linux container, headless server mode)
- **Model Checkpoint Serving:** Pre-trained PyTorch `.pt` files packaged directly in `weights/`.

## 12.2 Dashboard Feature Walkthrough (4 Functional Tabs)

### Tab 1: Generate (ECG Waveform Synthesizer)
Allows clinicians and researchers to select any of the 5 AAMI heartbeat classes and trigger the class-conditioned DDPM to sample a new, unique 187-point ECG heartbeat.
- Renders an interactive Plotly waveform.
- Computes real-time physiological metrics: peak-to-peak amplitude, mean baseline, signal energy, and estimated QRS complex duration.

### Tab 2: Diffusion Lab (Markov Chain Step Scrubber)
An educational and analytical tool visualizing the physical diffusion process:
- **Forward Diffusion Slider ($q(x_t|x_0)$):** Demonstrates progressive information destruction as Gaussian noise increases from $t=0$ to $t=40$.
- **Reverse Denoising Trajectory ($p_\theta(x_{t-1}|x_t)$):** Displays intermediate denoising checkpoints ($t=40, 35, 30, \dots, 0$), demonstrating how structured ventricular deflections emerge from pure Gaussian noise.

### Tab 3: Results & Clinical Benchmarks
Displays comprehensive quantitative benchmarks:
- Side-by-side comparative tables (Accuracy, Macro-F1, Minority Recall, 1D-FID, Diversity Ratio).
- High-resolution confusion matrix heatmaps comparing baseline vs. augmented models.
- Side-by-side morphological comparisons of real vs. synthesized beats across all 5 classes.

### Tab 4: Latent Space Manifold Explorer
Computes and renders real-time 2D Principal Component Analysis (PCA) projections of the 64-dimensional penultimate CNN embeddings for both real patient heartbeats and generated synthetic heartbeats, enabling visual verification of cluster separation and manifold alignment.

## 12.3 Clinical Utility and Practical Applicability
CardioDiff serves as an assistive diagnostic support system for telemetry units, ambulatory Holter analysis, and remote patient monitoring:
- **Data Augmentation Engine:** Can be integrated into hospital data pipelines to balance sparse datasets before training institutional diagnostic models.
- **Privacy-Preserving Data Sharing:** Synthesized ECG heartbeats contain no protected health information (PHI), enabling cross-institutional research collaboration without HIPAA or GDPR compliance issues.

---

# 13. CONCLUSION

The CardioDiff project successfully demonstrates the efficacy of **Denoising Diffusion Probabilistic Models (DDPM)** as a potent soft computing methodology for bio-signal synthesis and imbalanced classification:
1. **Overcoming Severe Imbalance:** By augmenting rare arrhythmia classes with class-conditioned 1D diffusion samples, the Macro-F1 score rose from **0.6554 to 0.8210 (+16.56%)**, and minority diagnostic recall increased from **0.4898 to 0.7876 (+29.78%)**.
2. **Clinical Significance of Minority Recovery:** The framework achieved a **+55.56%** recall improvement on Fusion beats (Class 3) and a **+29.56%** improvement on Ventricular contractions (Class 2), directly reducing dangerous false-negative rates in clinical arrhythmia diagnosis.
3. **Generative Realism Without Mode Collapse:** Achieving a **1D-FID of 40.0401** and an optimal **Diversity Ratio of 1.0253**, CardioDiff established clear superiority over Random Oversampling and classical interpolation methods.
4. **Accessible Clinical AI:** The system is open-source, fully reproducible, and deployed live at [https://cardiodiff.streamlit.app/](https://cardiodiff.streamlit.app/).

---

# 14. FUTURE SCOPE

1. **Multi-Lead ECG Synthesis:** Extend the 1D U-Net architecture to 12-lead clinical ECG recordings, capturing spatial cross-lead correlations across precordial and limb leads ($V_1$ to $V_6$, I, II, III, aVR, aVL, aVF).
2. **Classifier-Free Guidance (CFG):** Implement classifier-free diffusion guidance to dynamically tune the trade-off between sample diversity and class fidelity during inference.
3. **Edge Hardware Acceleration:** Quantize the PyTorch diffusion weights to 8-bit integers (INT8) or ONNX format for deployment on low-power wearable cardiovascular monitoring devices (smartwatches, Holter patches).
4. **Conditional Patient Conditioning:** Incorporate patient demographic metadata (age, sex, medical history) into the conditioning embedding space to synthesize personalized cardiac trajectories.

---

# 15. REFERENCES

1. **Moody, G. B., & Mark, R. G.** (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
2. **Ho, J., Jain, A., & Abbeel, P.** (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 6840–6851.
3. **Sohl-Dickstein, J., Weiss, E., Khanh, N., & Sompolinsky, H.** (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *International Conference on Machine Learning (ICML)*, 2256–2265.
4. **Kachuee, M., Fazeli, S., & Sarrafzadeh, M.** (2018). ECG heartbeat classification: A deep transferable representation. *IEEE International Conference on Healthcare Informatics (ICHI)*, 443–444.
5. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.
6. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
7. **Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S.** (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 6626–6637.
8. **ANSI/AAMI EC57:1998/(R)2008.** Testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms. *Association for the Advancement of Medical Instrumentation*.
9. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). U-Net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 234–241.
10. **Kingma, D. P., & Ba, J.** (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

---

# 16. APPENDIX

### Appendix A: PyTorch 1D Residual Convolutional Block with Class Conditioning
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    """
    1D Residual Block with Time & Class Conditioning Injection
    """
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Projection layer mapping 128-dim condition vector to channel dimensions
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        
        # Residual shortcut handling channel dimension changes
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        # Inject conditioning vector across time dimension
        cond_bias = self.cond_proj(cond).unsqueeze(-1)
        h = h + cond_bias
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))
```

### Appendix B: 1D Fréchet Inception Distance (1D-FID) Matrix Implementation
```python
import numpy as np
from scipy import linalg

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy/Scipy implementation of Fréchet Distance for 1D biological signals
    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(np.real(fid))
```

---
*Report compiled on October 30, 2026 for Soft Computing Course Submission, Adani University.*
