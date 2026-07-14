# Interaction Evaluation Metrics

This document describes the metrics computed by the scripts in this directory. It focuses on the implemented behavior: what data enters each metric, the exact quantity that is calculated, how results are aggregated, which direction is better, and what the metric does **not** establish.

The evaluation has four complementary parts:

| Evaluation | Script | Main question | Preferred direction |
|---|---|---|---|
| Contact and collision | [`01_evaluate_physical_plausibility.py`](01_evaluate_physical_plausibility.py) | Does the intended body part approach the annotated contact region, and does the human avoid penetrating the scene? | Contact distances ↓, NCS ↑, penetration depths ↓ |
| Pose diversity | [`02_evaluate_diversity.py`](02_evaluate_diversity.py) | How varied are the SMPL-X solutions across the evaluated set? | Entropy ↑, cluster size ↑ |
| CLIP semantics | [`03_evaluate_semantics.py`](03_evaluate_semantics.py) | Do rendered views semantically match the text instruction? | CLIP score ↑ |
| VLM evaluation | [`04_evaluate_vlm.py`](04_evaluate_vlm.py) | Does a multimodal judge see the correct object, action, contact, and physical arrangement? | All 1–5 scores ↑ |

These metrics are not interchangeable. In particular, a low contact distance does not guarantee a correct action, a high CLIP score does not guarantee valid contact, and a high non-collision score does not guarantee that the intended contact exists.

## 1. Geometric contact and collision metrics

The physical-plausibility evaluator reports two groups of metrics:

1. **Contact distances**, evaluated separately for every usable interaction edge in the Scene Interaction Graph (SIG), such as `human.left_hand -> table`.
2. **Non-collision metrics**, evaluated once for the complete human and the visible contact-scene crop.

### 1.1 How an intended contact pair is constructed

For each SIG edge connecting a human body part to the target object or floor, the evaluator constructs:

- a moving human region: the corresponding SMPL-X contact segment, such as `left_hand_contact` or `right_foot_contact`;
- a fixed scene region: the part of the scene mesh selected by that body part's annotated 2D ground-truth contact mask.

The 2D mask is projected onto the visible scene mesh. Depth-discontinuous or remote mask components are filtered to reduce accidental selection of background surfaces. The retained mesh faces are sampled uniformly on their surfaces, using 2,048 samples per interaction edge. By default, the selected face set is not expanded (`contact_region_expand_rings = 0`). Bilateral contacts can be swapped when the initial human pose indicates that the opposite left/right assignment improves the spatial correspondence by at least 2 cm.

For each evaluated human vertex \(h_i\) in a contact segment, the code finds its nearest sampled point in the associated scene contact region \(S\):

\[
d_i = \min_{s \in S} \lVert h_i-s \rVert_2.
\]

The distance is an unsigned Euclidean distance in metres. Because it is computed from **every human contact-segment vertex to the scene region**, it is a directed, segment-to-region nearest-neighbour measurement rather than a symmetric Chamfer distance.

### 1.2 `min_distance_m`

\[
d_{\min}=\min_i d_i.
\]

This is the smallest distance between any vertex in the evaluated human body segment and its annotated scene contact region.

- **Range:** \([0,\infty)\) metres.
- **Better:** lower.
- **Meaning:** values near zero show that at least one point of the intended body segment reaches the contact region.
- **Limitation:** a single touching vertex can produce a very small value even if most of the body part is far away, incorrectly oriented, or penetrating the object. It is therefore evidence of local proximity, not sufficient evidence of complete or anatomically plausible contact.

### 1.3 `max_distance_m`

\[
d_{\max}=\max_i d_i.
\]

This is the largest nearest-region distance among all vertices in the evaluated contact segment.

- **Range:** \([0,\infty)\) metres.
- **Better:** usually lower.
- **Meaning:** it measures the worst-covered point of the entire body segment. A large value means that some part of the designated segment remains far from the annotated region.
- **Limitation:** it is sensitive to segment extent and outliers. For example, not every vertex of a hand surface is expected to touch an object during a valid grip, so this is a strict coverage statistic rather than a literal maximum contact gap requirement.

### 1.4 `mean_distance_m`

For a contact segment containing \(M\) vertices:

\[
d_{\mathrm{mean}}=\frac{1}{M}\sum_{i=1}^{M}d_i.
\]

This is the average nearest-region distance over the complete contact segment.

- **Range:** \([0,\infty)\) metres.
- **Better:** lower.
- **Meaning:** it summarizes how closely the body segment as a whole lies to the intended scene contact region.
- **Interpretation:** unlike `min_distance_m`, it cannot be minimized by only one close vertex. Unlike `max_distance_m`, it is less dominated by one outlying vertex.
- **Limitation:** it remains unsigned. A vertex just inside an object and a vertex the same distance outside it contribute equally. Use it together with NCS and the penetration-depth metrics.

### 1.5 `ncs` — non-collision score

The evaluator samples the visible contact-scene crop surface (700,000 points by default) and queries each scene point against the signed distance field of the evaluated volumetric SMPL-X human. A scene point is classified as penetrating when its human SDF value is negative.

For \(N\) sampled scene points and \(N_{\mathrm{inside}}\) points inside the human volume:

\[
\mathrm{NCS}=\frac{N-N_{\mathrm{inside}}}{N}
=1-\frac{N_{\mathrm{inside}}}{N}.
\]

- **Range:** \([0,1]\).
- **Better:** higher; 1 means no sampled scene points were inside the human volume.
- **Meaning:** the fraction of sampled scene-surface points that do not collide with the human.
- **Example:** `ncs = 0.999` means 99.9% of sampled scene points are outside the human and 0.1% are classified as penetrating.

Important interpretation details:

- NCS is a **sample fraction**, not a collision volume or penetrated-body fraction.
- Sampling covers the scene mesh retained by the contact-camera crop, not necessarily the entire reconstructed scene.
- Large scene regions can make NCS appear very close to 1 even when a small but meaningful local penetration exists. Penetration depths and visual/VLM inspection complement it.
- Results depend on the reconstructed mesh, SMPL-X SDF accuracy, crop, and surface sampling density.

### 1.6 `mean_penetration_m`

For the subset \(P\) of sampled scene points with negative human SDF, the penetration depth is \(p_j=-\operatorname{SDF}(x_j)>0\). The metric is:

\[
p_{\mathrm{mean}}=
\begin{cases}
\frac{1}{|P|}\sum_{j\in P}p_j, & |P|>0,\\
0, & |P|=0.
\end{cases}
\]

- **Range:** \([0,\infty)\) metres.
- **Better:** lower.
- **Meaning:** the average penetration depth **conditioned on a sampled point being penetrating**.
- **Important:** non-penetrating points are excluded from this average. Consequently, this metric describes penetration severity, while NCS describes penetration prevalence. Two samples can have identical mean penetration but very different NCS values.

### 1.7 `max_penetration_m`

\[
p_{\max}=
\begin{cases}
\max_{j\in P}p_j, & |P|>0,\\
0, & |P|=0.
\end{cases}
\]

- **Range:** \([0,\infty)\) metres.
- **Better:** lower.
- **Meaning:** the deepest detected penetration among sampled scene points.
- **Limitation:** it is sensitive to individual samples and SDF/reconstruction errors, but it can reveal severe localized collisions that barely affect NCS.

### 1.8 Per-interaction aggregation

An interaction can contain multiple SIG contact edges. The top-level physical-plausibility row aggregates them as follows:

\[
\texttt{mean\_min\_contact\_distance\_m}
=\frac{1}{E}\sum_{e=1}^{E}d_{\min}^{(e)},
\]

\[
\texttt{mean\_max\_contact\_distance\_m}
=\frac{1}{E}\sum_{e=1}^{E}d_{\max}^{(e)},
\]

\[
\texttt{mean\_contact\_distance\_m}
=\frac{1}{E}\sum_{e=1}^{E}d_{\mathrm{mean}}^{(e)},
\]

where \(E=\texttt{num\_edges}\). Every valid contact edge receives equal weight, regardless of the number of vertices in its body segment or the area of its scene region.

The interaction's `ncs`, `mean_penetration_m`, and `max_penetration_m` are already whole-human/scene-crop statistics. They are repeated on every per-edge CSV row for convenience; they are **not recomputed for each edge**.

### 1.9 Dataset-level aggregation

The `__mean__` row in `physical_plausibility.csv` is an unweighted arithmetic mean of the corresponding per-interaction values. Thus, every interaction receives equal weight. The `num_edges` value in that row is instead the total number of edges across all interactions.

In particular, dataset-level `max_penetration_m` is the **mean of the per-interaction maxima**, not the single worst penetration over the dataset. In the JSON aggregate this is named `mean_max_penetration_m`, which makes that distinction explicit.

## 2. Pose diversity metrics

The diversity evaluator describes the distribution of poses across the complete evaluated set. These are dataset-level metrics; they do not score whether any single interaction is correct.

### 2.1 Feature vector

For an optimized interaction, the evaluator concatenates the following SMPL-X parameters:

\[
x=[\texttt{transl},\ \texttt{global\_orient},\ \texttt{body\_pose},\ \texttt{betas},\ \texttt{scale}].
\]

For `output_init`, it uses the first-frame `transl`, `global_orient`, `body_pose`, and `betas`; no scale value is appended.

By default, every feature dimension is standardized over the evaluated dataset:

\[
z_{n,d}=\frac{x_{n,d}-\mu_d}{\max(\sigma_d,10^{-8})}.
\]

The `--no_standardize` option disables this step. Standardization prevents dimensions with larger numeric scales from automatically dominating Euclidean distance, but it also makes the resulting distance values dataset-dependent and unitless.

The standardized vectors are clustered using deterministic k-means with Euclidean distance. The requested default is 15 clusters, but the effective number is

\[
K=\min(15,N),
\]

where \(N\) is the number of evaluated interactions. Initial centers are selected by deterministic farthest-point initialization, beginning with the first feature vector.

### 2.2 `entropy`

If cluster \(k\) contains a fraction \(p_k=n_k/N\) of samples, the evaluator computes natural-log Shannon entropy:

\[
H=-\sum_{k:p_k>0}p_k\ln p_k.
\]

- **Range:** \([0,\ln K]\) for a fixed effective cluster count \(K\).
- **Better for diversity:** higher.
- **Meaning:** high entropy indicates that samples are distributed more evenly across clusters; low entropy indicates concentration in relatively few pose clusters.
- **Maximum:** \(\ln K\), reached when cluster occupancy is uniform (as closely as integer sample counts permit).
- **Not normalized:** values from runs with different \(K\) are not directly comparable without accounting for the different maximum \(\ln K\). A normalized variant, not currently output by the script, would be \(H/\ln K\) for \(K>1\).

Entropy captures **between-cluster occupancy**, not geometric separation or variation within a cluster. It is also influenced by the chosen feature representation, standardization, cluster count, and evaluated sample set.

### 2.3 `cluster_size`

After k-means, every sample \(z_n\) has an assigned center \(c_{a(n)}\). The reported cluster size is the mean Euclidean distance from samples to their assigned centers:

\[
C=\frac{1}{N}\sum_{n=1}^{N}\lVert z_n-c_{a(n)}\rVert_2.
\]

- **Range:** \([0,\infty)\).
- **Better for within-cluster variation:** higher.
- **Units:** unitless when standardization is enabled; mixed raw parameter units with `--no_standardize`.
- **Meaning:** a larger value means greater average spread within the discovered clusters. A value of zero means every sample coincides with its assigned center, which necessarily occurs for singleton clusters.

Despite its name, this is not the number of members in a cluster, its radius, diameter, or k-means squared inertia. It is the dataset-wide mean of ordinary, non-squared distances to assigned centroids. The actual member counts are stored separately as `cluster_counts`, and each sample's distance is stored as `distance_to_cluster_center`.

A high value indicates variation but is not automatically desirable without qualification: it can also reflect noisy or implausible poses. Diversity should therefore be reported alongside semantic and physical-quality metrics.

### 2.4 Supporting diversity fields

- `num_samples`: number of interactions included in clustering.
- `num_clusters`: effective \(K\), after limiting the requested count to `num_samples`.
- `standardized`: whether per-dimension z-scoring was applied.
- `cluster_counts`: number of samples assigned to each cluster.
- `cluster_id`: zero-based assigned cluster for an interaction; the numeric ID has no semantic ordering.
- `distance_to_cluster_center`: that interaction's Euclidean distance to its assigned center in the feature space used for clustering.

## 3. CLIP semantic consistency

The CLIP evaluator compares the original natural-language interaction instruction with multiple rendered views of the static human-scene interaction. The default model is `openai/clip-vit-base-patch32`.

### 3.1 Per-view `clip_score`

For render \(I_v\) and instruction text \(T\), CLIP produces an image embedding \(f_I(I_v)\) and text embedding \(f_T(T)\). The code L2-normalizes both embeddings and takes their dot product:

\[
s_v=
\frac{f_I(I_v)}{\lVert f_I(I_v)\rVert_2}^{\!\top}
\frac{f_T(T)}{\lVert f_T(T)\rVert_2}.
\]

This is cosine similarity.

- **Theoretical range:** \([-1,1]\).
- **Better:** higher.
- **Meaning:** higher values indicate stronger semantic alignment between a particular rendered view and the instruction in CLIP's embedding space.
- **Not a probability:** a score such as 0.30 does not mean 30% correctness.
- **Not CLIP's scaled logit:** the implementation does not apply the learned logit scale or a softmax over alternative texts. It is raw normalized cosine similarity.

### 3.2 Per-interaction `clip_score`

If an interaction has \(V\) renders, its score is the arithmetic mean of the view scores:

\[
s_{\mathrm{interaction}}=\frac{1}{V}\sum_{v=1}^{V}s_v.
\]

All selected views have equal weight. Individual values and render paths are retained in the per-interaction `metrics.json`; `num_renders` records \(V\).

### 3.3 Dataset-level `mean_clip_score`

For \(N\) interactions:

\[
s_{\mathrm{dataset}}=\frac{1}{N}\sum_{n=1}^{N}s_{\mathrm{interaction},n}.
\]

This is an unweighted mean of interaction means. Therefore, every interaction has equal influence even if render counts differ. The `num_renders` value in the CSV `__mean__` row is the total number of renders, not the denominator of a render-weighted global score.

### 3.4 What CLIP score does and does not measure

CLIP is useful for coarse text-image semantic consistency: whether the image appears to contain the requested human action and relevant scene content. It does not explicitly reason about 3D geometry, signed penetration, exact body-part correspondence, or support. It can reward an image that contains the right object and action cues despite incorrect contact. Scores are model-, prompt-, rendering-, and viewpoint-dependent, so comparisons are most meaningful when those are held constant.

## 4. VLM rubric scores

The VLM evaluator sends the interaction instruction and all selected rendered views to either the configured Gemini model or a Qwen model served through Ollama. The prompt in [`prompt_eval_interactions.md`](prompt_eval_interactions.md) asks the model to inspect all views jointly and assign four integer scores.

The shared scale is:

| Score | Rubric meaning |
|---:|---|
| 1 | Incorrect, missing, or impossible to judge |
| 2 | Mostly incorrect, very unclear, or physically invalid |
| 3 | Partially correct, but ambiguous, incomplete, awkward, or weakly supported |
| 4 | Mostly correct with minor contact, support, or pose issues |
| 5 | Clearly correct, physically plausible, with required contacts well satisfied |

These are ordinal rubric scores. Treating their differences as equal intervals when computing a mean is a convenient summary convention, not a guarantee that the perceptual difference between 1 and 2 equals that between 4 and 5.

### 4.1 `target_object_score`

This is the VLM response for **Target Object Correctness**: whether the human interacts with the object or scene element named in the instruction and whether it is the main support/contact object.

- **Range:** intended to be an integer from 1 to 5.
- **Better:** higher.
- **Prompt cap:** at most 2 if the human does not interact with the named target object.
- **Focus:** object identity and actual use, rather than pose quality in isolation.

### 4.2 `human_action_score`

This is the VLM response for **Human Action Correctness**: whether the pose expresses the requested action.

- **Range:** intended to be an integer from 1 to 5.
- **Better:** higher.
- **Prompt cap:** at most 3 if the action is recognizable but required contacts are missing or wrong.
- **Focus:** recognizable action and pose configuration. A broadly action-like but physically awkward pose should receive partial rather than full credit.

### 4.3 `contact_score`

This is the VLM response for **Contact and Spatial Relation**: whether instruction-relevant body parts plausibly contact, grip, rest on, press, or are appropriately close to the correct object region.

- **Range:** intended to be an integer from 1 to 5.
- **Better:** higher.
- **Prompt rules:** visual overlap must not be mistaken for contact; hidden or ambiguous contact and clearly floating support cap the score at 3; severe target-object penetration caps it at 2; a score of 5 requires all instruction-specified contacts to be clearly plausible.
- **Focus:** semantic and anatomical plausibility of contact, which complements the unsigned geometric distance metrics.

### 4.4 `physical_plausibility_score`

This is the VLM response for **Physical Plausibility**: whether the body-object arrangement has plausible support, balance, placement, and collision behavior.

- **Range:** intended to be an integer from 1 to 5.
- **Better:** higher.
- **Prompt rules:** floating support caps the score at 3; severe target-object or nearby-solid penetration caps it at 2; torso/pelvis or multi-limb penetration should receive 1 or 2; any visible floating, major penetration, impossible balance, or nonsensical support rules out a 5.
- **Focus:** visually apparent whole-interaction feasibility, not render aesthetics or reconstruction completeness.

### 4.5 Per-interaction `mean_score`

The evaluator extracts the four available criterion scores and computes their arithmetic mean:

\[
\texttt{mean\_score}=\frac{1}{|A|}\sum_{c\in A}s_c,
\]

where \(A\) is the set of successfully parsed criteria. Normally \(|A|=4\). However, the implementation averages the remaining valid criteria if a criterion is absent or cannot be parsed. A non-null mean therefore does not by itself prove that all four component scores were present; inspect the component columns or JSON when completeness matters.

The parser converts any numeric model output to an integer with Python's `int(...)`. It does not currently enforce the 1–5 range. The prompt requests valid integer values, so out-of-range or fractional provider responses should be treated as malformed even if they are mechanically written to the CSV.

### 4.6 Dataset-level VLM aggregation

The `__mean__` row in `vlm.csv` is computed independently for every column as the arithmetic mean of the non-missing per-interaction values in that column. This has two consequences:

1. each interaction has equal weight for a given column; and
2. if values are missing, different columns can be averaged over different subsets.

The dataset `mean_score` is the mean of the already-computed per-interaction `mean_score` values. It is not recomputed directly from the four dataset-level criterion means, although the two are equal when every interaction contains all four criteria.

### 4.7 Sources of VLM variability

VLM metrics depend on the provider, exact model version, prompt, selected views, image resizing, temperature, and sampling seed. The current defaults include temperature 0.9 and seed 12345. Even with a seed, remote models and serving stacks may not be perfectly reproducible. For controlled comparisons, keep the provider, model, prompt template, renders, and generation settings fixed, and retain `metrics.json`, which stores the model identity, raw response, reasons, best-view IDs, and failure modes.

## 5. Reading the metrics together

A robust interpretation uses all metric families rather than selecting one headline number:

| Observation | Likely interpretation |
|---|---|
| Low contact distance, high NCS | Intended body region is close and little collision is detected; semantic correctness still needs CLIP/VLM confirmation. |
| Low contact distance, low NCS or high penetration | The body reaches the object but may do so by intersecting it. |
| High NCS, high contact distance | Collision-free but likely floating or missing the intended contact. |
| High CLIP score, poor geometric metrics | The render contains the right semantic cues, but detailed 3D contact may be wrong. |
| High VLM scores, poor SDF collision metrics | The collision may be hidden by viewpoints or too subtle for visual judgment; inspect geometry and reconstruction/SDF quality. |
| High entropy and cluster size, poor quality scores | The set is varied, but diversity may include invalid or semantically incorrect poses. |

No universal numeric threshold is hard-coded for accepting an interaction. Appropriate thresholds should be chosen on a validation set and kept fixed across compared methods.

## 6. Output files and aggregation hierarchy

For each output mode (`output`, `output_round1`, or `output_init`), results follow this structure:

```text
<output_mode>/
├── physical_plausibility.csv       # one row per interaction plus __mean__
├── physical_plausibility.json      # interaction summaries and aggregate
├── diversity.csv                   # one dataset-level row
├── diversity.json                  # metrics, counts, and assignments
├── semantics.csv                   # one row per interaction plus __mean__
├── semantics.json                  # interaction summaries and aggregate
├── vlm.csv                         # one row per completed interaction plus __mean__
└── interaction_XX/
    ├── physical_plausibility/
    │   ├── metrics.csv             # one row per contact edge
    │   └── metrics.json
    ├── semantics/
    │   ├── metrics.csv             # interaction-level CLIP mean
    │   └── metrics.json            # includes per-view CLIP scores
    └── vlm/
        ├── metrics.csv             # four criteria and interaction mean
        └── metrics.json            # full VLM response and metadata
```

The hierarchy matters: contact distances begin at the **edge** level, CLIP begins at the **render-view** level, VLM judges all views jointly at the **interaction** level, and diversity is defined over the **entire evaluated set**.
