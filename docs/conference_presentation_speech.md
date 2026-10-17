# ICFCR 2026 — Paper 105
# FINAL LIVE CONFERENCE DELIVERY SPEECH (v2)

**Paper:** Non-Invasive Leakage Severity Estimation in Water Distribution Networks  
using Wavelet Scattering Transforms: A Comparative Sensor Study  
**Presenter:** Govind Jee · **ICFCR 2026, MESCE Kuttippuram**

---

> [!IMPORTANT]
> PPTX and PDF are FROZEN. Frozen v9 PPT = presentation structure. Accepted paper PID105.pdf = scientific authority. Do not modify the PPTX or PDF.

---

## Slide 1 — Title Slide

**SAY**

Good morning. I'm Govind Jee, presenting our paper on non-invasive leakage severity estimation using Wavelet Scattering Transforms — joint work with Shiju Kumar P S and Raj Kishor at SRM Institute of Science and Technology. The central question: can an external pipe-wall sensor tell us not just whether a leak is present, but how severe the flow event is?

**POINT TO** — Title. Make eye contact with the audience.

**EMPHASIZE** — "Not just whether — but how severe."

**TRANSITION** — Let me start with why that distinction matters operationally.

---

## Slide 2 — Why Leakage Severity Matters

**SAY**

The left side opens with the water loss problem: Non-Revenue Water can consume up to half of municipal supply. Below that, the fundamental limitation of existing systems — binary leak detection answers only one question, shown in the callout: *"Is there a leak?"* Nothing more.

The amber operational gap box names the real need: *"Which leak deserves priority? What is the severity?"* When multiple alarms trigger, a binary answer gives no basis for triage.

The right panel states the research question: can WST features from non-invasive pipe-wall vibration distinguish three hydraulic severity states without fluid contact? The three severity state boxes: **No Demand** — the hydraulic control state; **Low Flow at 0.18 L/s** — lower-severity, maintenance priority; **High Flow at 0.47 L/s** — higher-severity, maintenance priority.

At the bottom: **Binary Alarm → Severity Estimation → Maintenance Prioritization**.

**POINT TO** — Left panel top to bottom. Amber box. Right panel research question. Three severity blocks. Bottom strip.

**EMPHASIZE** — Amber gap: "Which leak deserves priority?" The three states: ND / 0.18 / 0.47 L/s.

**TRANSITION** — Let me position this study within the existing research landscape.

---

## Slide 3 — Research Gap, Prior Work & Contribution

**SAY**

The top banner maps four tasks: Binary Detection, Leak-Type Classification, Regression Sizing, and the green cell — **Severity Estimation: THIS STUDY**.

The comparison table shows representative prior work per task. Binary detection gives presence or absence only. Leak-type classification identifies crack, gasket, or orifice — but type is not severity. Regression-based approaches can provide continuous flow estimates, but the cited approach involved substantial calibration. The highlighted green row is ours: lightweight WST plus RBF-SVM, non-invasive pipe vibration, three-state triage, no model needed. The slide notes cross-study comparisons must be interpreted cautiously — datasets and conditions differ.

Key contribution statement: *"Comparative evaluation of three sensor modalities for hydraulic flow-severity classification using WST plus SVM under benchmark conditions."*

Three contribution cards: cross-domain sensor comparison under identical conditions; WST plus RBF-SVM achieving 99.0% ± 1.2% on the accelerometer and 93.7% ± 1.9% on pressure; and evidence that external vibration shows strong discriminative performance for flow severity under the benchmark conditions.

**POINT TO** — Top banner; pause on green cell. Table highlighted row. Contribution cards.

**EMPHASIZE** — Green row: "THIS STUDY." 99.0% ± 1.2% on Contribution 2 card.

**TRANSITION** — Now — where does the data come from?

---

## Slide 4 — Benchmark Dataset & Provenance

**SAY**

The navy citation banner identifies the source: **Aghashahi, Sela and Banks, 2022**, Mendeley Data, DOI 10.17632/tbrnp6vrnj.1. I want to be explicit: the study uses measurements from the published experimental testbed described in the referenced literature. We did not construct the physical testbed. Our contribution is the WST–SVM framework applied to the selected benchmark data.

The three-column layout shows the selection. Left: the full benchmark covers two topologies, five leak conditions, four hydraulic states, three sensor modalities. Middle: our selection — branched topology, orifice leak, states ND, 0.18, and 0.47 L/s, all three sensors. Right: the resulting research scope.

The attribution footer captures this — public benchmark, our WST–SVM analysis.

**POINT TO** — Citation banner. Sweep left → middle → right. Attribution footer.

**EMPHASIZE** — "We did not construct the testbed." Branched / Orifice / ND–0.18–0.47 / three sensors.

**TRANSITION** — Now let me explain what makes the three sensors physically distinct.

---

## Slide 5 — Study Design & Sensor Modalities

**SAY**

The context strip at the top establishes the shared conditions: 25 mm HDPE branched network, 30-second recordings, the same three hydraulic states. Every sensor observes the same pipe under the same conditions — that is what makes the comparison meaningful.

Three panels, each a different physical sensing domain. **Accelerometer** — clamped externally. Pipe-wall structural vibration. 51.2 kHz. PCB 333B50. Invasive indicator: **No — no pipe penetration.**

**Dynamic Pressure** — inside the pipe. Fluid pressure transients. 51.2 kHz. PCB 102B16. Invasive: **Yes.**

**Hydrophone** — submerged in fluid. Acoustic emission in the water column. 8 kHz. Aquarian H2a. Invasive: **Yes.**

Three physically distinct signal paths observing the same event. Which domain best discriminates flow severity? That is the comparative question.

**POINT TO** — Context strip. Each panel header and invasive indicator bar.

**EMPHASIZE** — Same event — three distinct signal paths. Non-invasive NO / YES / YES. Rates: 51.2 / 51.2 / 8 kHz.

**TRANSITION** — Let me now trace the complete processing pipeline.

---

## Slide 6 — WST–SVM Framework Pipeline

**SAY**

The slide subtitle says: *"Seven stages transforming 30-second benchmark recordings into hydraulic severity decisions."* Let me trace them.

**Stage 1 — Raw Signal**: we begin with the 30-second benchmark recordings from the three modalities.

**Stage 2 — Preprocessing**: the slide shows the conditioning operations, which are applied independently to each 1-second segment — mean-centering for DC removal, Symlet-3 level-4 denoising, and MAD-based soft thresholding.

**Stage 3 — 1-Second Segmentation**: 30 non-overlapping windows per recording. Totals: 510 accelerometer / 541 pressure / 900 hydrophone segments.

**Stage 4 — WST Extraction**: Complex Morlet wavelet, J = 9, Q = 1. Orders: 1st and 2nd; 0th excluded.

**Stage 5 — Feature Vector**: low-pass pooling with log compression.

**Stage 6 — RBF-SVM**: kernel RBF, C = 100, γ = 1. StandardScaler on training partition only within each fold. 5-fold stratified CV. Output: macro-F1 and accuracy ± SD.

**Stage 7 — Severity Class**: ND, Low at 0.18 L/s, or High at 0.47 L/s.

The bottom note: **empirical device-level profiling was not performed.**

**POINT TO** — Trace all seven stages. Pause at Stage 3 (counts) and Stage 6 (SVM params). Bottom note.

**EMPHASIZE** — "Seven stages." C = 100, γ = 1, fold-isolated. "Empirical profiling not performed."

**TRANSITION** — Let me go deeper on segmentation and WST.

---

## Slide 7 — Signal Preprocessing & Segmentation

**SAY**

The left panel shows the windowing step. A 30-second recording — the blue bar — sliced into 30 non-overlapping 1-second windows, shown as the colored rectangles. Each segment is conditioned independently prior to WST feature extraction.

Four conditioning steps per segment: ① mean-centering; ② Symlet-3 level-4 wavelet denoising; ③ MAD-based noise estimation; ④ universal soft thresholding and reconstruction.

The right side shows the resulting dataset: **510 accelerometer segments** — test split 102. **541 dynamic pressure segments** — test split 109. **900 hydrophone segments** — test split 180.

The green CV rigor note: StandardScaler on training partition only within each fold. And: *"Recording-level grouped CV is not specified in the paper."*

**POINT TO** — 30s bar → window rectangles. Steps ①–④. Segment count cards. CV rigor note.

**EMPHASIZE** — 510 / 541 / 900; test splits 102 / 109 / 180. "Recording-level grouped CV not specified."

**TRANSITION** — Now — why Wavelet Scattering Transform?

---

## Slide 8 — Why Wavelet Scattering Transform?

**SAY**

The left side presents two panels. First — the representation challenge. Turbulent flow produces transient, non-stationary vibrations. Fourier-based methods rely on stationarity assumptions that may not hold for transient turbulence. Simple summary statistics — the slide states this explicitly — do not capture the multi-scale temporal structure WST targets. WST uses **fixed analytical filters with no learned transform weights**.

Second — the WST configuration: Complex Morlet wavelets, J = 9 dyadic scales, Q = 1, 1st and 2nd scattering orders included, 0th order excluded, log compression to stabilize dynamic range.

The right panel shows the seven-step operation flow: ① conditioned 1-second segment; ② wavelet convolution with ψλ₁; ③ modulus — strips phase, retains envelope; ④ second-order cascade; ⑤ time averaging for shift invariance; ⑥ log compression; ⑦ compact feature vector to RBF-SVM.

**POINT TO** — Left: challenge bullets then configuration items. Right: trace ①→⑦ with hand.

**EMPHASIZE** — "Fixed filters — no learned weights." J = 9, Q = 1. 1st + 2nd orders; 0th excluded.

**TRANSITION** — With features defined, let me explain the evaluation framework.

---

## Slide 9 — Classification & Evaluation Protocol

**SAY**

Two halves — the distinction is critical.

**Left — Table III: SVM Grid Search on the accelerometer only.** Linear, RBF with γ = scale, and Polynomial kernels tested at C = 1, 10, and 100. The amber box highlights the peak: **99.2% — RBF, C = 100, γ = scale.** The slide's italic note: *"99.2% reflects peak grid-search tuning on the accelerometer partition, distinct from the standardized cross-sensor evaluation."* This is model selection — not the headline result.

**Right — Final Cross-Sensor Evaluation Protocol.** Parameters fixed across all three modalities: **RBF, C = 100, γ = 1** — to evaluate physical sensor discriminability, not per-modality tuning. 5-fold stratified cross-validation. StandardScaler on training partition only within each fold. **Primary headline: mean accuracy ± SD across the five folds.** Representative test splits — 102, 109, 180 — support per-class metrics and confusion matrices.

The footer states: *"The 99.2% result is an exploratory grid-search peak on the accelerometer alone."*

**POINT TO** — Left: table; tap 99.2%. Say "model selection." Shift right: parameters, CV protocol, footer.

**EMPHASIZE** — "99.2% is model selection — not the headline." C = 100, γ = 1, fixed, 5-fold CV, mean ± SD.

**TRANSITION** — Now the headline results.

---

## Slide 10 — Comparative Sensor Performance

**SAY**

The bar chart shows 5-fold stratified CV mean accuracy for each modality with error bars representing standard deviation. The metric cards confirm: **Accelerometer: 99.0% ± 1.2%** — Macro F1 = 0.9905. **Dynamic Pressure: 93.7% ± 1.9%.** **Hydrophone: 42.0% ± 1.7%.**

The ± values are standard deviations across folds — fold-to-fold consistency, not a confidence interval.

The amber context box: *"Under identical benchmark conditions, non-invasive pipe-wall vibration achieved the highest mean accuracy among evaluated modalities for three-state flow severity estimation."*

**POINT TO** — Bar chart and error bars. Three metric cards. Amber context box.

**EMPHASIZE** — 99.0 ± 1.2%, 93.7 ± 1.9%, 42.0 ± 1.7%. "Highest mean accuracy — benchmark conditions."

**TRANSITION** — Let me give you the full class-level evidence for the accelerometer.

---

## Slide 11 — Accelerometer: Class-Level Evidence Chain

**SAY**

The headline box: **99.0% ± 1.2% — 5-fold stratified CV mean. Macro F1 = 0.9905.**

The table — Table IV in the paper, n = 102 — gives per-class precision, recall, F1, and support. No Demand: P = 1.0, R = 1.0, F1 = 1.0, N = 32 — perfect. Low Flow: P = 1.0, R = 0.971, F1 = 0.986, N = 35. High Flow: P = 0.972, R = 1.0, F1 = 0.986, N = 35. Macro average F1 = 0.9905. In plain terms: **101 of 102 segments correctly classified. One misclassification — Low Flow predicted as High Flow.**

The LDA plot on the left shows three well-separated color-coded clusters in WST feature space. **LDA is a feature-space visualization tool — not the final classifier.** The RBF-SVM is the classifier. The LDA plot provides supporting visual evidence of feature separability, consistent with the CV performance.

The slide's note: representative test split for class-level error analysis; headline accuracy is the 5-fold CV mean.

**POINT TO** — Right: headline box → table rows → error diagnostic. Left: LDA clusters.

**EMPHASIZE** — 101/102. Macro F1 = 0.9905. LDA = visualization only, not the classifier.

**TRANSITION** — The same analysis for the invasive sensors reveals a different pattern.

---

## Slide 12 — Invasive Sensors: Evidence & Error Analysis

**SAY**

**Dynamic Pressure — 93.7% ± 1.9%.** Confusion matrix, n = 109: No Demand — all 35 correct. Low Flow — 35 of 37 correct. High Flow — 34 of 37 correct. That gives **exactly 5 errors**: 1 Low classified as ND, 1 Low as High, 1 High as ND, and 2 High as Low. Macro F1 = 0.954. Error diagnostic: *"Strong No Demand recognition; residual errors concentrated around the Low/High boundary."*

**Hydrophone — 42.0% ± 1.7%.** Confusion matrix, n = 180: broad confusion across all three classes — no diagonal dominance. Macro F1 = 0.437. The diagnostic attributes this to *"acoustic saturation/clipping under high-flow turbulence, reducing discriminative spectral structure"* — this is the paper's interpretation.

The footer: *"Headline accuracy is the 5-fold CV mean; confusion matrices are from representative held-out splits."*

**POINT TO** — Left matrix: ND diagonal then off-diagonal errors. Right matrix: broad off-diagonal pattern.

**EMPHASIZE** — Exactly 5 pressure errors at the Low/High boundary. Hydro: broad confusion, Macro F1 = 0.437.

**TRANSITION** — Why do these sensors produce such different results from the same event?

---

## Slide 13 — Why Do the Sensors Behave Differently?

**SAY**

Three column panels explain the signal-path differences, followed by a synthesis table.

**Accelerometer (99.0%):** turbulence generates structural vibration in the pipe wall. The paper interprets different flow magnitudes as producing structurally distinct vibration signatures that WST captures as separable features.

**Dynamic Pressure (93.7%):** turbulence creates pressure fluctuations shaped by network boundary conditions — wave reflections, pipe geometry — which the paper associates with Low/High boundary overlap. That is consistent with the 5 observed errors.

**Hydrophone (42.0%):** acoustic emission in the fluid column. The paper interprets the broad confusion as acoustic saturation or clipping at high-flow turbulence, reducing discriminative spectral content.

These are the paper's physical interpretations — not independently validated mechanisms.

The synthesis table organizes all three sensors by CV accuracy, signal domain, feature separability, error pattern, and interpretation.

**POINT TO** — Three column panels in turn. Synthesis table.

**EMPHASIZE** — "Same event — three signal paths — three outcomes." "Paper's interpretations."

**TRANSITION** — Let me address what this study establishes and where its scope ends.

---

## Slide 14 — Practical Meaning, Limitations & Future Work

**SAY**

The left panel: the framework acts as a **discrete severity proxy** — enabling triage, not continuous flow metering. The slide notes theoretical edge feasibility — WST is O(N log N), fixed-filter, lightweight — but states explicitly: **empirical device-level profiling was not performed.**

The right panel — four scientific limitations: **controlled single-testbed scope** — 25 mm HDPE, branched, orifice leak only; **discrete severity levels** — no continuous flow estimation; **laboratory environment** — no field noise or real network conditions; **empirical hardware profiling** not measured.

The scope table maps six dimensions from current to future: single testbed → real networks; discrete states → continuous regression; branched HDPE → other materials and topologies; current WST+SVM → optimized representations; theoretical edge feasibility → empirical MCU benchmarking.

**POINT TO** — Left panel items. Right panel four limitations. Scope table current vs. future columns.

**EMPHASIZE** — "Discrete severity proxy — not continuous flow metering." "Empirical device profiling not performed."

**TRANSITION** — Let me close with the key takeaways.

---

## Slide 15 — Conclusion & Key Takeaways

**SAY**

The three metric boxes summarize the primary results: **99.0% ± 1.2%** Accelerometer. **93.7% ± 1.9%** Dynamic Pressure. **42.0% ± 1.7%** Hydrophone. All five-fold stratified CV means.

The finding strip: non-invasive pipe-wall vibration achieved the highest mean accuracy among evaluated modalities under benchmark conditions — demonstrating that external structural vibration carries discriminative information about hydraulic severity without pipe penetration.

Four bullets close the narrative: NRW loss motivates the work; WST J = 9, Q = 1 plus RBF-SVM C = 100, γ = 1 is the method; the comparative result is 99.0% non-invasive versus 93.7% and 42.0% invasive; the boundary — controlled benchmark baseline, field network validation is the critical next step.

Thank you. Happy to take your questions.

**POINT TO** — Three metric boxes. Navy finding strip. Four bullets.

**EMPHASIZE** — All three results. "Highest mean accuracy — benchmark conditions." "Field validation is next."

---

# WORD COUNT (programmatically verified)

| Slide | Title | Words |
|---|---|---|
| 1 | Title Slide | 59 |
| 2 | Why Leakage Severity Matters | 137 |
| 3 | Research Gap, Prior Work & Contribution | 157 |
| 4 | Benchmark Dataset & Provenance | 109 |
| 5 | Study Design & Sensor Modalities | 118 |
| 6 | WST–SVM Framework Pipeline | 167 |
| 7 | Signal Preprocessing & Segmentation | 110 |
| 8 | Why Wavelet Scattering Transform? | 132 |
| 9 | Classification & Evaluation Protocol | 156 |
| 10 | Comparative Sensor Performance | 79 |
| 11 | Accelerometer: Class-Level Evidence Chain | 164 |
| 12 | Invasive Sensors: Evidence & Error Analysis | 131 |
| 13 | Why Do the Sensors Behave Differently? | 129 |
| 14 | Practical Meaning, Limitations & Future Work | 122 |
| 15 | Conclusion & Key Takeaways | 118 |
| **TOTAL** | | **1,888 words** |

> [!NOTE]
> **Timing estimates (SAY content only):**
>
> | Rate | Time |
> |---|---|
> | 120 wpm | 15.7 min |
> | 130 wpm | **14.5 min** |
> | 140 wpm | 13.5 min |
>
> At 130 wpm, the SAY content is approximately 14.5 minutes before additional pauses and transitions. A real stopwatch rehearsal is therefore required to confirm the final delivery time.


---

# 15-LINE MEMORIZATION CUE SHEET

```
S01  "Can an external pipe-wall sensor say HOW SEVERE — not just whether?"

S02  NRW = up to half / binary: "Is there a leak?" only
     Amber gap: WHICH leak? WHAT severity?
     Three states: ND / 0.18 L/s / 0.47 L/s → Alarm → Severity → Maintenance Priority

S03  4-task banner: Binary → Type → Regression → THIS STUDY (severity, green cell)
     Highlighted row = us / Contribution: 3 sensors + WST + SVM + benchmark
     Cards: 99.0% ± 1.2% accel / 93.7% ± 1.9% pressure / "strong discriminative performance"

S04  Public benchmark: Aghashahi et al., 2022, Mendeley Data
     WE DID NOT BUILD THE TESTBED / our contribution: WST–SVM analysis on selected data
     Selection: Branched / Orifice / ND–0.18–0.47 / 3 sensors

S05  Same event — 3 physical domains: vibration / pressure / acoustic
     Accel: non-invasive, 51.2 kHz, PCB 333B50
     Pressure: invasive, 51.2 kHz, PCB 102B16 / Hydro: invasive, 8 kHz, Aquarian H2a

S06  SEVEN STAGES: Raw → Preprocessing(per-seg) → Segmentation → WST → Feature → RBF-SVM → Severity
     30s divided into 1s windows / each segment independently conditioned
     C=100 / γ=1 / fold-isolated scaler / 5-fold CV / "empirical profiling NOT performed"

S07  30s → 30×1s non-overlapping / steps ①mean-centre ②Symlet-3 L4 ③MAD ④soft-threshold
     Totals: 510/541/900 | Test splits: 102/109/180
     "Recording-level grouped CV not specified in the paper"

S08  WHY: non-stationary turbulence / stationarity assumption may not hold / simple stats miss multi-scale
     CONFIG: Morlet / J=9 / Q=1 / 1st+2nd / 0th excluded / fixed filters / no learned weights
     7-step flow: Conv → Modulus → 2nd-order → Time-avg → Log → Feature vector → SVM

S09  LEFT = TABLE III / 99.2% = accel only / γ=scale / MODEL SELECTION ONLY
     RIGHT = final: RBF / C=100 / γ=1 FIXED / 5-fold CV / mean±SD = PRIMARY HEADLINE
     99.2% ≠ 99.0% — different purposes / splits 102/109/180 = class-level diagnostics only

S10  99.0±1.2% | 93.7±1.9% | 42.0±1.7% — 5-fold CV means / ± = SD across folds
     Macro F1 = 0.9905 on accel card
     "Highest mean accuracy among evaluated modalities — benchmark conditions"

S11  PRIMARY: 99.0±1.2% / Table IV, n=102 / 101/102 correct / 1 error: Low→High
     Macro F1 = 0.9905 / LDA = feature-space VISUALIZATION ONLY — SVM is the classifier
     CV headline ≠ representative split

S12  Pressure 93.7±1.9%, n=109 / EXACTLY 5 ERRORS: 1L→ND / 1L→HF / 1HF→ND / 2HF→Low
     Macro F1 = 0.954 / Low/High boundary errors
     Hydro 42.0±1.7%, n=180 / broad confusion / Macro F1=0.437
     "Paper attributes to acoustic saturation/clipping"

S13  3 signal paths: pipe-wall vibration / fluid pressure+boundary effects / acoustic saturation
     Accel → distinct WST patterns / Pressure → Low/High overlap / Hydro → broad confusion
     Paper's interpretations — NOT independently validated / synthesis table: 6 columns

S14  Discrete severity PROXY — NOT continuous flow metering
     4 limitations: HDPE/branched/orifice/lab / discrete only / lab env / no device profiling
     Scope table: current → real networks / discrete → regression

S15  99.0±1.2% | 93.7±1.9% | 42.0±1.7%
     "Highest mean accuracy — benchmark conditions"
     "Controlled baseline — field validation is the critical next step"
     Thank you
```

---

*Final live delivery speech v2 — PPT-synchronized to frozen v9 + paper PID105.pdf.*  
*Do not modify the PPTX or PDF.*
