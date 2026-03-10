# Comparison of Public Motor‑Imagery (MI) EEG Datasets with C3, Cz, C4 Channels

## Key

- **Subject Count & Sessions** – approximate number of human subjects and whether multiple sessions are provided.
- **EEG Channels** – the channel montage or number of channels. The table explicitly notes when C3, Cz and C4 are present.
- **Sampling Rate** – nominal sampling frequency used during recording (some datasets are down-sampled in preprocessing).
- **MI Classes** – motor-imagery tasks available (e.g., left/right hands, feet, tongue).
- **Key Characteristics** – summary of why the dataset is useful (e.g., many subjects but few sessions, longitudinal data across days, high-density montage, tasks under distraction, etc.).

## Comparison Table

| Dataset | Subjects | Sessions per subject | Trials (per session) | Channels (C3, Cz, C4 present) | Sampling rate | MI classes | Characteristics |
|---|---|---|---|---|---|---|---|
| **PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)** | 109 volunteers | 14 runs recorded in a single visit | 1–2 min tasks; multiple trials per run | 64‑channel EEG using 10‑10 system — includes C3, Cz and C4 | 160 Hz | Fist (left/right/both), feet (both) – real and imagined movements | Large number of participants; short single‑session runs; cross‑subject deep‑learning and augmentation |
| **BCI Competition IV Dataset 2a / BNCI 2014‑001** | 9 subjects | 2 sessions | 6 runs of 48 trials per run (12 per class) | 22 EEG channels (Fz, FC3/FC1/FCz/FC4, C5/C3/C1/Cz/C2/C4/C6, CP3/CP1/CPz/CP2/CP4, P1/Pz/P2/POz) – includes C3, Cz and C4 | 250 Hz | Left hand, right hand, foot, tongue | Balanced classes and moderate sessions; benchmark for cross‑validation and domain adaptation |
| **BCI Competition IV Dataset 2b / BNCI 2014‑004** | 9 subjects | 2 sessions (screening & feedback) | 6 runs × 10 trials per run (approx. 120 trials) | 3 bipolar EEG channels located at C3, Cz and C4 | 250 Hz | Left vs right hand | Minimal dataset with three sensors; limited feedback but widely used |
| **High‑Gamma Dataset (HGD)** | 14 healthy subjects | 13 runs (training and test) | ~260 trials per run (four classes) | Recorded with 128 electrodes; analysis uses 44 motor‑cortex channels including C3, Cz and C4 | Recorded at 500 Hz and down‑sampled to 250 Hz | Left hand, right hand, feet and rest | High‑channel count; self‑annotations; high‑gamma focus |
| **OpenBMI Motor Imagery (Lee 2019)** | 54 subjects | 2 sessions | 100 trials per session (left vs right) | 62‑channel EEG (10‑05 system); channels include motor‑cortex electrodes such as C3, Cz, C4; 20 motor‑cortex channels often selected for analysis | 1 kHz (often down‑sampled to 250 Hz) | Left vs right hand | Large dataset; high‑density electrodes; useful for cross‑session and cross‑subject deep‑learning models |
| **High‑quality multi‑day MI dataset (WBCIC‑MI)** | 62 subjects | 2 or 3 sessions recorded on different days | 200 trials per session for 2‑class tasks; 300 trials for 3‑class tasks | 64‑channel wireless cap arranged by the 10‑20 system; recorded 59 EEG channels (including C3, Cz, C4) | 1 kHz, down‑sampled to 250 Hz | Left vs right hand; and foot for 3‑class paradigm | Large dataset with cross‑session recordings; benchmark for deep‑learning |
| **BNCI 2014‑002 (Laplacian derivations)** | 14 subjects | 1 session | 8 runs (3 training, 3 validation) × 20 trials per run | 15 electrodes arranged around C3, Cz and C4 (each centre electrode plus four neighbouring electrodes) | 512 Hz | Right‑hand vs feet (sustained MI for 5 s) | Focuses on Laplacian derivations; good for small‑sample training with validation |
| **GrosseWentrup 2009** | 10 subjects | Single session | 150 trials per condition (left vs right hand) | 128‑channel extended 10‑20 montage with Cz reference; includes C3, Cz, C4 | 500 Hz | Left vs right hand | High‑usage dataset for subject‑wise MI research |
| **Ofner 2017 (Upper‑Limb MI)** | 15 subjects | 2 sessions on different days | 10 runs per session, 42 trials per run (~60 trials per class) | 61 active EEG channels (10‑05 montage) including C3, Cz and C4 | 512 Hz | Six movement types (elbow flexion/extension, forearm pronation/supination, hand open/close) plus rest | Multiclass MI dataset widely used for CNN training |
| **Zhou 2016 (3‑class MI)** | 4 subjects | 3 sessions per subject | Each run has 75 trials (25 per class) | 14‑channel EEG (10‑20 montage) including C3, Cz, C4 | 250 Hz | Left hand, right hand, feet | Small three‑class dataset with multiple sessions; used for algorithm testing |
| **BCI Competition III Dataset IIIa** | 3 subjects | ≥6 runs per session | 40 trials per run (10 trials per each of 4 classes) | 60 EEG channels recorded with 64‑channel Neuroscan amplifier; includes C3, Cz and C4 | 250 Hz | Left hand, right hand, foot, tongue | Classic dataset with high‑quality montage; widely used across algorithms |
| **BCI Competition III Dataset IVa** | 5 subjects | Single session; training/test splits | 280 total trials split into training/test (subject‑dependent) | 118‑channel EEG (Berlin BCI) with 10‑20 montage; includes C3, Cz, C4 | 100 Hz (down‑sampled from 1000 Hz in some implementations) | Right hand vs foot (only two classes available to competitors) | Small dataset with few trials; used for transfer learning research |
| **BNCI 2014‑002 (Two‑Class MI – 002‑2014)** | 14 participants | 1 session with 8 runs (5 training, 3 validation) | 20 trials per run (right‑hand and foot) | 15 electrodes: centre electrodes at C3, Cz and C4 with four neighbouring electrodes each | 512 Hz | Right hand vs feet | Focused two‑class dataset for transfer learning |
| **Motor‑Imagery Under Distraction Dataset (TU Berlin)** | 16 participants | 1 session (calibration + 6 feedback runs) | 7 runs of 72 trials each | 63‑channel EEG (10‑20 system) recorded at 1 kHz; C3 and C4 used for Laplacian feedback | 1 kHz (down‑sampled to 100 Hz for online feedback) | Left vs right hand with various secondary tasks (clean, eyes‑closed, news, numbers, flicker, vibration) | Introduces distractions and secondary tasks; study of domain adaptation |
| **Cross‑session Variability MI Dataset (Ma et al., 2022)** | 25 subjects | 5 sessions recorded on different days (2–3 days apart) | 100 trials per session (left‑hand vs right‑hand) | ~64‑channel EEG (not explicitly stated, but C3 and C4 used in analysis) | 250 Hz (down‑sampled) | Left vs right hand | Large cross‑session dataset with 12,500 trials; supports transfer adaptation |
| **Large EEG Database with User Profiles (BrainConquest 2023)** | 87 participants | Single session | 6 runs per subject; each session has 240 trials (120 per class) | 27 active electrodes (Fz, FCz, Cz, CPz, C1, C3, C5, C2, C4, C6, etc.), referenced to left earlobe | 512 Hz | Left vs right hand | Large user‑profiled dataset; cross‑subject adaptation |
| **Multimodal Upper‑Extremity Dataset (Jeong 2020)** | 25 participants | 3 sessions over 3 days | 11 intuitive movement tasks (real and imagined) yielding 82,500 trials across all subjects | 60‑channel EEG (10‑20 system) plus 7 EMG and 4 EOG channels; includes C3, Cz and C4 | 1 kHz (often down‑sampled) | Real and imagined movements: hand grasp, wrist twist, arm reach, etc. | Rich multimodal dataset combining EEG, EMG and EOG; real and imagined tasks across sessions |

## Detailed Descriptions

### PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)

- **Source:** PhysioNet (public repository).
- **Participants:** 109 volunteers.
- **Sessions & Trials:** Each participant completed 14 runs including baseline, real and imaginary movements of left/right fists and feet. Runs last 1 or 2 minutes with multiple trials per run physionet.org.
- **EEG Montage & Channels:** Recorded with a 64‑channel EEG using the international 10‑10 system; includes C3, Cz and C4 physionet.org.
- **Sampling Rate:** 160 Hz physionet.org.
- **Motor‑Imagery Classes:** Imagined and executed movements of right fist and both feet (real and imagined).
- **Characteristics:** Large number of participants but recordings are short and single‑session. Widely used for cross‑subject deep learning and data augmentation due to diversity of subjects and tasks.

### BCI Competition IV Dataset 2a (BNCI 2014‑001)

- **Source:** BCI Competition IV / BNCI Horizon 2020.
- **Participants:** 9 subjects.
- **Sessions:** Each subject participated in two sessions recorded on different days. Each session comprises six runs with 48 trials each (12 per class) moabb.neurote….
- **EEG Channels:** 22 channels arranged at positions Fz, FC3, FC1, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2 and POz mdpi.com.
- **Sampling Rate:** 250 Hz moabb.neurote….
- **MI Classes:** Four classes – left hand, right hand, both feet, tongue.
- **Use in Literature:** Classic multi‑class MI datasets; numerous algorithms (CSP, FBCSP, CNNs, RNNs, Transformers) benchmark their performance on it. Balanced classes and moderate session count make it ideal for cross‑validation and domain adaptation studies.

### BCI Competition IV Dataset 2b (BNCI 2014‑004)

- **Source:** BCI Competition IV / BNCI Horizon 2020.
- **Participants:** 9 subjects.
- **Sessions & Trials:** Two sessions per subject (screening & feedback) comprising six runs × 10 trials per run (approx. 120 trials) pmc.ncbi.nlm….
- **EEG Channels:** Three bipolar EEG channels located at C3, Cz and C4 pmc.ncbi.nlm….
- **Sampling Rate:** 250 Hz pmc.ncbi.nlm….
- **MI Classes:** Left vs right hand.
- **Use in Literature:** Minimal dataset with three sensors; widely used in algorithms but limited feedback.

### High‑Gamma Dataset (HGD)

- **Source:** BCI Competition IV / BNCI Horizon 2020 (often referred to as the High‑Gamma Dataset).
- **Participants:** 14 healthy subjects.
- **Sessions:** 13 runs (training and test).
- **Trials:** Approximately 260 trials per run (four classes) nature.com.
- **EEG Montage & Channels:** Recorded with 128 electrodes; analysis uses 44 motor‑cortex channels including C3, Cz and C4 gist‑github.com.
- **Sampling Rate:** Recorded at 500 Hz and down‑sampled to 250 Hz nature.com.
- **MI Classes:** Left hand, right hand, feet and rest.
- **Characteristics:** High‑gamma dataset with high‑channel count; good self‑annotations included per subject.

### OpenBMI Motor Imagery (Lee 2019)

- **Source:** OpenBMI (Lee 2019).
- **Participants:** 54 subjects.
- **Sessions:** Two sessions.
- **Trials:** 100 trials per session (left vs right).
- **EEG Montage & Channels:** 62‑channel EEG (10‑05 system); channels include motor‑cortex electrodes such as C3, Cz, C4; twenty motor‑cortex channels often selected for analysis mdpi.com.
- **Sampling Rate:** 1 kHz (often down‑sampled to 250 Hz) mdpi.com.
- **MI Classes:** Left vs right hand.
- **Characteristics:** Large dataset with high‑density electrodes; training phase often includes cross‑session recordings and cross‑subject deep‑learning models.

### High‑quality multi‑day MI dataset (WBCIC‑MI)

- **Source:** WBCIC‑MI dataset.
- **Participants:** 62 subjects.
- **Sessions:** Two or three sessions recorded on different days.
- **Trials:** 200 trials per session for two‑class tasks; 300 trials for three‑class tasks pmc.ncbi.nlm….
- **EEG Montage & Channels:** 64‑channel wireless cap arranged by the 10‑20 system; recorded 59 EEG channels (including C3, Cz, C4) pmc.ncbi.nlm….
- **Sampling Rate:** 1 kHz, down‑sampled to 250 Hz pmc.ncbi.nlm….
- **MI Classes:** Left vs right hand; and foot for three‑class paradigm.
- **Characteristics:** Large dataset with cross‑session recordings across subjects; deep‑learning benchmarks.

### BNCI 2014‑002 (Laplacian derivations)

- **Source:** BNCI 2014‑002 (Laplacian derivations).
- **Participants:** 14 subjects.
- **Sessions:** One session.
- **Trials:** Eight runs (three training, three validation) × 20 trials per run.
- **EEG Montage & Channels:** Fifteen electrodes arranged around C3, Cz and C4 (each centre electrode plus four neighbouring electrodes) pmc.ncbi.nlm….
- **Sampling Rate:** 512 Hz pmc.ncbi.nlm….
- **MI Classes:** Right‑hand vs feet (sustained MI for five seconds).
- **Characteristics:** Comprehensive dataset focusing on Laplacian derivations; sensors good for learning small‑sample training with validation.

### GrosseWentrup 2009

- **Source:** GrosseWentrup 2009.
- **Participants:** 10 subjects.
- **Sessions:** Single session.
- **Trials:** 150 trials per condition (left vs right hand).
- **EEG Montage & Channels:** 128‑channel extended 10‑20 montage with Cz reference; includes C3, Cz, C4 moabb.neurote….
- **Sampling Rate:** 500 Hz moabb.neurote….
- **MI Classes:** Left vs right hand.
- **Characteristics:** High‑usage dataset used for subject‑wise MI research; many subjects training.

### Ofner 2017 (Upper‑Limb MI)

- **Source:** Ofner 2017.
- **Participants:** 15 subjects.
- **Sessions:** Two sessions on different days.
- **Trials:** Ten runs per session, 42 trials per run (~60 trials per class) moabb.neurote….
- **EEG Montage & Channels:** Sixty‑one active EEG channels (10‑05 montage) including C3, Cz and C4 moabb.neurote….
- **Sampling Rate:** 512 Hz moabb.neurote….
- **MI Classes:** Six movement types (elbow flexion/extension, forearm pronation/supination, hand open/close) plus rest.
- **Characteristics:** Multiclass MI dataset often used for training CNNs.

### Zhou 2016 (3‑class MI)

- **Source:** Zhou 2016.
- **Participants:** 4 subjects.
- **Sessions:** Three sessions per subject.
- **Trials:** Each run has 75 trials (25 per class).
- **EEG Montage & Channels:** Fourteen‑channel EEG (10‑20 montage) including C3, Cz, C4 moabb.neurote….
- **Sampling Rate:** 250 Hz moabb.neurote….
- **MI Classes:** Left hand, right hand, feet.
- **Characteristics:** Small dataset with three classes; multiple sessions per subject; testing across algorithms.

### BCI Competition III Dataset IIIa

- **Source:** BCI Competition III.
- **Participants:** 3 subjects.
- **Sessions:** Six or more runs per session.
- **Trials:** 40 trials per run (10 trials per each of four classes) doc.tu‑berlin….
- **EEG Montage & Channels:** Sixty EEG channels recorded with 64‑channel Neuroscan amplifier; includes C3, Cz and C4 doc.tu‑berlin….
- **Sampling Rate:** 250 Hz doc.tu‑berlin….
- **MI Classes:** Left hand, right hand, foot, tongue.
- **Characteristics:** Classic dataset with high‑quality montage; classification across multi‑algorithm research.

### BCI Competition III Dataset IVa

- **Source:** BCI Competition III.
- **Participants:** 5 subjects.
- **Sessions:** Single session; training/test splits.
- **Trials:** 280 total trials per subject split into training and test (subject‑dependent) doc.tu‑berlin….
- **EEG Montage & Channels:** One‑hundred‑eighteen‑channel EEG (Berlin BCI) with 10‑20 montage; includes C3, Cz, C4 github.com.
- **Sampling Rate:** 100 Hz (down‑sampled from 1000 Hz in some implementations) doc.tu‑berlin….
- **MI Classes:** Right hand vs foot (only two classes available to competitors).
- **Characteristics:** Small dataset with very few trials; used to transfer learning across subjects; research dataset.

### BNCI 2014‑002 (Two‑Class MI – 002‑2014)

- **Source:** BNCI 2014‑002.
- **Participants:** 14 participants.
- **Sessions:** One session with eight runs (five training, three validation).
- **Trials:** Twenty trials per run (right‑hand and foot) pmc.ncbi.nlm….
- **EEG Montage & Channels:** Fifteen electrodes: centre electrodes at C3, Cz and C4 with four neighbouring electrodes each pmc.ncbi.nlm….
- **Sampling Rate:** 512 Hz pmc.ncbi.nlm….
- **MI Classes:** Right hand vs feet.
- **Characteristics:** Focused dataset of electrode montages used for transfer learning adaptation.

### Motor‑Imagery Under Distraction Dataset (TU Berlin)

- **Source:** TU Berlin motor‑imagery under distraction dataset.
- **Participants:** 16 participants.
- **Sessions:** One session (calibration + six feedback runs).
- **Trials:** Seven runs of 72 trials each frontiersin.org.
- **EEG Montage & Channels:** Sixty‑three‑channel EEG (10‑20 system) recorded at 1 kHz; C3 and C4 used for Laplacian feedback frontiersin.org.
- **Sampling Rate:** 1 kHz (down‑sampled to 100 Hz for online feedback) frontiersin.org.
- **MI Classes:** Left vs right hand with various secondary tasks (clean, eyes‑closed, news, numbers, flicker, vibration) frontiersin.org.
- **Characteristics:** Introduces distraction and secondary tasks; study of MI domain adaptation.

### Cross‑session Variability MI Dataset (Ma et al., 2022)

- **Source:** Ma et al., 2022.
- **Participants:** 25 subjects.
- **Sessions:** Five sessions recorded on different days (two–three days apart).
- **Trials:** 100 trials per session (left‑hand vs right‑hand) nature.com.
- **EEG Montage & Channels:** Approximately 64‑channel EEG (not explicitly stated, but C3 and C4 used in analysis) nature.com.
- **Sampling Rate:** 250 Hz (down‑sampled) nature.com.
- **MI Classes:** Left vs right hand.
- **Characteristics:** Multiclass cross‑session research dataset with 12,500 trials; transfer adaptation.

### Large EEG Database with User Profiles (BrainConquest 2023)

- **Source:** BrainConquest 2023.
- **Participants:** 87 participants.
- **Sessions:** Single session.
- **Trials:** Six runs per subject; each session has 240 trials (120 per class) nature.com.
- **EEG Montage & Channels:** Twenty‑seven active electrodes (Fz, FCz, Cz, CPz, C1, C3, C5, C2, C4, C6, etc.), referenced to left earlobe nature.com.
- **Sampling Rate:** 512 Hz nature.com.
- **MI Classes:** Left vs right hand.
- **Characteristics:** Large user‑profiled dataset focusing on questions of MI; very cross‑subject and adaptation demo.

### Multimodal Upper‑Extremity Dataset (Jeong 2020)

- **Source:** Jeong 2020 multimodal upper‑extremity dataset.
- **Participants:** 25 participants.
- **Sessions:** Three sessions over three days.
- **Trials:** Eleven intuitive movement tasks (real and imagined movements) yielding 82,500 trials across all subjects oup.silverchair‑c….
- **EEG Montage & Channels:** Sixty‑channel EEG (10‑20 system) plus seven EMG and four EOG channels; includes C3, Cz and C4 oup.silverchair‑c….
- **Sampling Rate:** 1 kHz (often down‑sampled) oup.silverchair‑c….
- **MI Classes:** Real and imagined movements: hand grasp, wrist twist, arm reach, etc.
- **Characteristics:** Rich real‑world dataset combining EEG, EMG and EOG with repeated sessions; CNNs and both real and imagined movement tasks.