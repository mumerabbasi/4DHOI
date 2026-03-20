# Future Directions for 4DHOI

## Thesis Positioning

4DHOI already does something meaningful and unusual for a two-month build:

- it turns free-form text into an explicit Part Affordance Graph,
- uses that graph to drive first-frame and video generation,
- reconstructs human and object geometry from the generated video,
- aligns everything into one scene frame,
- and refines object motion with part-aware, PAG-aware contact losses.

That is already a real thesis nucleus.

The current version is strongest as a systems paper or thesis prototype around explicit HOI structure. It is weaker as a polished top-venue paper because:

- the pipeline is fragmented across many error-prone stages,
- the final optimization still keeps the human sequence fixed,
- object dynamics are rigid and mostly kinematic,
- the method is not benchmarked with publication-grade metrics,
- and the repo does not yet contribute a learned interaction prior or a new large-scale evaluation protocol.

So the right question is not "is there a thesis here?" The answer is yes. The right question is what kind of top-venue paper this thesis should become.

My recommendation is to position 4DHOI as:

> an explicit, part-aware, language-conditioned 4D human-object interaction framework that bridges generation and reconstruction, then extend it into either a stronger HOI model paper or a benchmark-plus-method paper.

## What 4DHOI Already Contributes

Before comparing against outside work, it helps to name your current contributions clearly.

### Current strengths that are already research-relevant

- An explicit semantic interaction intermediate, the PAG, that survives through the whole stack instead of disappearing after prompt construction.
- A practical bridge from text-conditioned video generation to 4D geometry recovery.
- A part-aware object segmentation pipeline that maps language-level part names onto mesh triangles.
- A contact-aware refinement objective that distinguishes continuous vs intermittent contact and relatively static vs drifting contact.
- A modular, inspectable implementation where every stage exports intermediate artifacts.

### Current gaps relative to publication-grade work

- No end-to-end interaction prior learned across the whole pipeline.
- No strong mechanism to repair errors propagated from video generation into geometry.
- No human-pose correction in the final default optimizer.
- No articulated, deformable, or nonrigid object modeling.
- No benchmark-ready evaluation protocol for text-to-4D HOI fidelity.

## Comparison Matrix

The table below uses official primary sources available as of March 20, 2026. One venue label needs a correction from your requested list:

- `HUMOTO` is not a main-track ICCV paper; the official page lists it under the ICCV 2025 Wild3D workshop.
- The closest recent papers I found are concentrated in CVPR, ICCV, and ICLR; I did not find a 2025 SIGGRAPH / TOG paper that matches this problem as directly as the papers listed below.

| Work | Venue / status | Main setting | Output / framing | Where it is stronger than 4DHOI | Where 4DHOI is stronger or more explicit |
| --- | --- | --- | --- | --- | --- |
| [AvatarGO](https://openreview.net/forum?id=Trf0R8eoGF) | ICLR 2025 Poster | Text-to-4D HOI generation | Generative 4D avatars with object interaction | More directly a 4D HOI generator; stronger learned generative prior for interaction animation | 4DHOI has a more explicit modular geometry stack, explicit PAG semantics, explicit mesh-part labeling, and easier debugging of failure sources |
| [HSI-GPT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_HSI-GPT_A_General-Purpose_Large_Scene-Motion-Language_Model_for_Human_Scene_Interaction_CVPR_2025_paper.html) | CVPR 2025 | Human-scene interaction modeling | Tokenized multimodal instruction-following model | Much stronger learned interaction prior; supports multiple HSI tasks in one model | 4DHOI is more directly object-part explicit and currently closer to a text-to-4D asset pipeline than a sequence model |
| [InterMimic](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_InterMimic_Towards_Universal_Whole-Body_Control_for_Physics-Based_Human-Object_Interactions_CVPR_2025_paper.html) | CVPR 2025 | Physics-based HOI control | Whole-body control policy for physical interaction | Stronger physics realism and interaction control; better contact plausibility | 4DHOI is broader as a generation-to-geometry pipeline and has explicit part-affordance structure |
| [HOI-TG](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_End-to-End_HOI_Reconstruction_Transformer_with_Graph-based_Encoding_CVPR_2025_paper.html) | CVPR 2025 | HOI reconstruction | End-to-end human-object reconstruction transformer | Stronger end-to-end reconstruction learning; better balance of global mesh recovery and local contact reconstruction | 4DHOI handles text-conditioned generation and explicit part semantics, which HOI-TG does not target |
| [MonST3R](https://openreview.net/forum?id=lJpqxFgWCM) | ICLR 2025 Spotlight | Dynamic scene geometry from video | Geometry-first feed-forward dynamic reconstruction | Stronger direct video-to-geometry philosophy; far less stage fragmentation | 4DHOI is more HOI-specific, with contact semantics, object parts, and interaction-structured reasoning |
| [Uni4D](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_Uni4D_Unifying_Visual_Foundation_Models_for_4D_Modeling_from_a_CVPR_2025_paper.html) | CVPR 2025 | Single-video 4D modeling | Unified 4D modeling from video | Stronger foundation-model-style 4D lifting from video itself | 4DHOI has richer explicit interaction semantics and more direct language grounding |
| [Shape of Motion](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.html) | ICCV 2025 | 4D reconstruction from a single video | Reconstruction-first dynamic scene modeling | Stronger single-video 4D reconstruction formulation and likely better temporal coherence in pure reconstruction settings | 4DHOI explicitly models human-object contact semantics and can begin from text prompts rather than only observed video |
| [Hand-held Object Reconstruction from RGB Video with Dynamic Interaction](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Hand-held_Object_Reconstruction_from_RGB_Video_with_Dynamic_Interaction_CVPR_2025_paper.html) | CVPR 2025 | Dynamic object reconstruction under hand interaction | Reconstruction-first, object-centric | Stronger object pose and geometry reasoning under hand interaction; tighter reconstruction objective for manipulated rigid objects | 4DHOI covers a broader full-body HOI pipeline and not only handheld object reconstruction |
| [HOIGen-1M](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_HOIGen-1M_A_Large-scale_Dataset_for_Human-Object_Interaction_Video_Generation_CVPR_2025_paper.html) | CVPR 2025 | Dataset / benchmark | Large-scale HOI video generation dataset | Stronger data scale and evaluation leverage; useful for training and benchmarking | 4DHOI contributes explicit 4D geometry, not only video generation |
| [CORE4D](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CORE4D_A_4D_Human-Object-Human_Interaction_Dataset_for_Collaborative_Object_REarrangement_CVPR_2025_paper.html) | CVPR 2025 | Dataset for collaborative 4D interaction | Multi-human, object-centric 4D interaction data | Stronger real/synthetic dataset contribution and collaborative interaction coverage | 4DHOI is a method stack rather than a data paper and is more directly language-conditioned |
| [HUMOTO](https://openreview.net/forum?id=kUZ4wWEzTV) | ICCV 2025 Wild3D workshop | Mocap HOI dataset | 4D capture dataset for multi-object interaction | Stronger motion-capture-quality HOI data for training and evaluation | 4DHOI already provides a working end-to-end system and can use such data as future supervision |
| [FIction](https://openaccess.thecvf.com/content/CVPR2025/html/Ashutosh_FIction_4D_Future_Interaction_Prediction_from_Video_CVPR_2025_paper.html) | CVPR 2025 | Future interaction prediction | Predictive 4D interaction reasoning from video | Stronger anticipation and future-interaction reasoning | 4DHOI currently focuses on generation and reconstruction, not forecasting, but its explicit PAG and object trajectories make forecasting a natural extension |

## Axis-by-Axis Comparison

### Problem setting

- AvatarGO is the closest text-to-4D HOI generation paper.
- HSI-GPT is broader human-scene interaction modeling rather than a direct 4D mesh pipeline.
- HOI-TG, MonST3R, Uni4D, Shape of Motion, and the hand-held reconstruction paper are reconstruction-first rather than text-first.
- InterMimic is a control-and-physics paper rather than a generation or reconstruction pipeline.
- HOIGen-1M, CORE4D, HUMOTO, and FIction are dataset / benchmark / prediction papers rather than direct end-to-end method baselines.

### Input modality

- 4DHOI starts from text and then generates a video that becomes its reconstruction input.
- AvatarGO is also text-conditioned, which is why it is the closest generative comparison.
- Most of the reconstruction papers start from observed video, not a generated video.
- InterMimic starts from a control setting with physics, not a language-first pipeline.

### Output representation

- 4DHOI outputs explicit aligned human meshes, object meshes, object-part triangle labels, object trajectories, and refined scene sequences.
- AvatarGO targets generated 4D interaction animation directly.
- HOI-TG, MonST3R, Uni4D, and Shape of Motion care more about direct dynamic 4D recovery than about preserving explicit semantic intermediates.
- HSI-GPT is more model- and token-centric than mesh-pipeline-centric.

### Generative vs reconstructive vs control-oriented framing

- 4DHOI is hybrid: generative upstream, reconstructive in the middle, and geometric optimization downstream.
- AvatarGO is more purely generative.
- HOI-TG, MonST3R, Uni4D, Shape of Motion, and the hand-held reconstruction paper are primarily reconstructive.
- InterMimic is primarily control-oriented and physics-based.
- FIction is predictive rather than reconstructive or directly generative.

### Contact / affordance modeling

- 4DHOI is unusually explicit here because it uses the PAG all the way through the stack.
- AvatarGO and HSI-GPT have stronger learned interaction priors, but their interaction structure is less exposed as a reusable geometric object than your PAG.
- HOI-TG gives stronger end-to-end HOI reconstruction but not the same explicit language-to-part graph interface.
- InterMimic is stronger on physical contact realism than your current geometric contact losses.

### Physical plausibility

- InterMimic and PhysFlow are clearly ahead of 4DHOI on physical realism.
- 4DHOI currently enforces non-penetration and contact consistency but is still mainly kinematic.
- This is one of the clearest places where your method can improve into a publishable contribution.

### Object-part reasoning

- 4DHOI is stronger than many neighboring methods because object parts are explicit in both language and mesh space.
- That is a real differentiator relative to MonST3R, Uni4D, and Shape of Motion, which are stronger on unified 4D reconstruction but not centered on explicit part-level HOI semantics.
- It is also a strength relative to benchmark papers like HOIGen-1M and CORE4D, which provide data leverage but not your specific semantic mechanism.

### Temporal consistency

- Reconstruction-first methods like Shape of Motion and MonST3R expose a cleaner route to temporal consistency because they avoid several of your intermediate stage boundaries.
- 4DHOI currently restores temporal consistency later, through object tracking and PAG-guided refinement.
- That works, but it means temporal coherence is recovered rather than built in from the start.

### Training-data requirements

- 4DHOI is strong as a low-training systems thesis because it reuses external models instead of requiring a massive bespoke training corpus.
- AvatarGO, HSI-GPT, InterMimic, and reconstruction-first learned papers gain power from stronger learned priors, which usually means heavier training assumptions.
- HOIGen-1M, CORE4D, and HUMOTO suggest the data you would need if you decide to move from a modular stack to a learned top-venue model.

### Evaluation style

- 4DHOI currently evaluates itself mostly through intermediate artifacts, overlays, diagnostics, and qualitative end results.
- HOIGen-1M, CORE4D, HUMOTO, and FIction show that the field is moving toward standardized data and task-specific metrics.
- A strong publication version of 4DHOI should have benchmark-ready metrics for contact, part correctness, temporal drift, and geometry consistency.

## Detailed Reading of the Landscape

### 1. Generative HOI papers

The most directly comparable generative paper in your list is AvatarGO. It is important because it proves that top venues are interested in text-to-4D HOI generation itself, not only reconstruction. Its strengths are directness and learned generative coherence. Its weakness, relative to your thesis, is that it is less explicit as a systems decomposition.

This creates an opportunity:

- If you push 4DHOI toward a stronger learned interaction prior while keeping your explicit semantic structure, you can aim for a "more controllable and more inspectable than pure generative baselines" story.

HSI-GPT is different. It is broader than your current repo and less directly tied to 4D geometry output, but it matters because it shows that HSI is moving toward general-purpose multimodal models. The lesson is not "turn 4DHOI into a chatbot." The lesson is that your semantic interaction representation should become more model-like and reusable, not only a static JSON.

### 2. Reconstruction-first papers

MonST3R, Uni4D, Shape of Motion, HOI-TG, and the hand-held object reconstruction paper all attack the stage-fragmentation problem from the opposite direction: start from video, recover dynamic structure more directly, and reduce heuristic glue between stages.

These papers expose your biggest current weakness:

- 4DHOI is semantically explicit, but geometrically over-fragmented.

That weakness is also your biggest future-paper opportunity:

- replace several brittle intermediate stages with a more unified video-to-4D lifting module while preserving explicit interaction semantics.

### 3. Physics / control papers

InterMimic and PhysFlow matter even though they are not one-to-one baselines.

InterMimic shows that physically grounded human-object interaction is now credible enough to publish at the top tier if the control problem is framed well.

PhysFlow shows that physics-based dynamic reasoning is increasingly being paired with foundation models and generative video supervision.

The key lesson is that publication-level HOI work is moving beyond contact as geometry only. It is moving toward:

- dynamics,
- material behavior,
- stability,
- control,
- and predictive realism.

Your current contact losses are a good start, but they are not yet enough to claim physically plausible interaction.

### 4. Dataset and benchmark papers

HOIGen-1M, CORE4D, and HUMOTO matter because they define where evaluation is going.

Right now, 4DHOI has many intermediate artifacts but no benchmark-quality claim. These data papers suggest a strong route to publication:

- either train better with them,
- evaluate on them,
- or release a benchmark specific to text-to-4D HOI that those datasets do not yet cover.

FIction strengthens this point from another angle: evaluation of interaction should include temporal future consistency, not only per-frame geometry or video aesthetics.

## The Best Future Directions

Below are the five directions I would prioritize if the goal is a top-venue paper rather than only a finished thesis.

## Direction 1: Collapse the pipeline into a stronger video-to-4D HOI lifting module

### Claim

Build a new core model that lifts generated or real monocular HOI video into aligned 4D human-object geometry more directly, replacing several brittle hand-offs.

### Why this matters

This direction answers the biggest weakness exposed by MonST3R, Uni4D, Shape of Motion, and HOI-TG:

- too many stage boundaries,
- too many opportunities for errors to compound,
- too little shared reasoning across geometry, motion, and contact.

### Repo subsystems to modify

- `Estimate_Depth/`
- `Estimate_Human_Motion/`
- `Align_Meshes/`
- `Track_Object_Mesh/`
- `Track_Human_Object_Mesh/`

### New method components

- a shared video encoder that predicts object motion, coarse geometry, and contact cues jointly,
- a learned interaction token stream derived from the PAG,
- differentiable lifting of object and human states into a shared scene frame,
- refinement initialized from learned predictions rather than fully from modular hand-offs.

### Concrete implementation target

Keep the current modular pipeline as a teacher / bootstrap source, but add a learned HOI lifting model that predicts:

- per-frame object pose,
- object-part contact confidence,
- coarse scene-aligned human/object geometry,
- and a correction prior for refinement.

### Experiments and ablations

- Compare learned initialization vs current modular initialization.
- Remove the PAG tokens and measure contact degradation.
- Compare generated-video input vs real-video input.
- Evaluate temporal stability, contact precision, and final refinement convergence.

### Directly answers

- MonST3R
- Uni4D
- Shape of Motion
- HOI-TG

### Main risks

- Hard to train without large paired HOI data.
- Risk of losing the interpretability that currently makes 4DHOI distinctive.

## Direction 2: Make the generation stage interaction-aware instead of only prompt-aware

### Claim

Turn PAG-conditioned generation into a real HOI generation method where the generated video is trained or adapted to satisfy contact, part, and motion semantics, not just text aesthetics.

### Why this matters

Right now the generation stack is strong enough for prototyping, but it is not yet a publishable HOI generation model on its own. AvatarGO and HOIGen-1M show that direct HOI generation is a publishable space.

### Repo subsystems to modify

- `Generate_PAG/`
- `Generate_Video/`
- `Segment_Video/`

### New method components

- PAG-conditioned control tokens for the video generator,
- interaction-critic scoring based on mask consistency, part visibility, and contact plausibility,
- self-filtering or preference optimization over generated video candidates,
- optional synthetic supervision from the current downstream geometry stack.

### Concrete implementation target

Generate multiple videos per prompt, automatically score them by:

- contact consistency with the PAG,
- object-part visibility and segmentation quality,
- static-camera fidelity,
- downstream reconstruction consistency.

Then train or fine-tune a smaller interaction adapter so generation improves on those dimensions directly.

### Experiments and ablations

- Prompt-only vs PAG-conditioned generation.
- With and without automatic interaction critic.
- Human evaluation on HOI faithfulness vs aesthetics.
- Downstream reconstruction success as a generation metric.

### Directly answers

- AvatarGO
- HOIGen-1M
- HSI-GPT
- 4Real-Video

### Main risks

- Easy to improve visual quality without improving actual interaction faithfulness.
- Requires careful metrics; otherwise the story will sound qualitative.

## Direction 3: Upgrade geometric contact losses into physics-aware interaction realism

### Claim

Move from kinematic consistency to physically plausible interaction by modeling contact persistence, support, friction, and simple object dynamics.

### Why this matters

This is the cleanest path from "good engineering thesis" to "top-tier HOI paper." InterMimic and PhysFlow show that physical plausibility is no longer optional when interaction is the research focus.

### Repo subsystems to modify

- `Track_Human_Object_Mesh/`
- `Track_Human_Object_Joint/`
- `Segment_Object_Mesh/`

### New method components

- contact state variables instead of only fixed edge semantics,
- simple rigid-body or quasi-static physical consistency terms,
- support constraints and gravity-aware reasoning,
- friction-aware sliding vs sticking behavior,
- optional object articulation constraints where relevant.

### Concrete implementation target

Add a second-stage optimizer after current refinement that reasons over:

- contact activation,
- support surfaces,
- relative velocity at contact,
- penetration-free motion under simple dynamics.

Even a quasi-static physics layer would already be a large step up.

### Experiments and ablations

- Geometric contact only vs physics-aware contact.
- Sliding interactions vs static grasping interactions.
- Penetration, drift, and support-violation metrics.
- User study on realism of manipulated object behavior.

### Directly answers

- InterMimic
- PhysFlow
- FIction

### Main risks

- Physics assumptions may be too strong for noisy generated videos.
- The method could become brittle without better uncertainty handling.

## Direction 4: Generalize beyond rigid single-object reasoning to articulated, bimanual, and nonrigid HOI

### Claim

Extend 4DHOI from rigid-object interaction to richer object structure: articulated tools, deformable objects, bimanual manipulation, and multi-object sequences.

### Why this matters

This is where your current PAG and part-label pipeline become especially valuable. Many end-to-end methods do not have a clean interface for compositional object structure. You already do.

### Repo subsystems to modify

- `Generate_PAG/`
- `Segment_Object_Mesh/`
- `Track_Object_Mesh/`
- `Track_Human_Object_Mesh/`

### New method components

- articulation graph per object,
- joint-axis or hinge-state estimation,
- part-state tracking instead of whole-object-only `SE(3)`,
- support for multiple simultaneous contacts from both hands and environment,
- optional deformable-object latent state for cloth-like or soft objects.

### Concrete implementation target

Start with articulated rigid objects because they are tractable:

- scissors,
- umbrella opening/closing,
- drawers,
- suitcase handles,
- exercise machines,
- folding tools.

Represent them as:

- object-level root transform,
- plus part-level articulation parameters,
- plus PAG edges attached to specific moving parts.

### Experiments and ablations

- Rigid-only vs articulated tracking.
- Whole-object contact vs part-specific contact.
- Single-hand vs bimanual interactions.
- Performance on synthetic articulated benchmarks and carefully chosen generated cases.

### Directly answers

- Hand-held Object Reconstruction from RGB Video with Dynamic Interaction
- CORE4D
- HUMOTO

### Main risks

- Labeling and tracking articulated parts is significantly harder than static rigid parts.
- Failure cases become harder to diagnose without new visualization tools.

## Direction 5: Turn 4DHOI into a benchmark paper for text-to-4D HOI quality

### Claim

Define the evaluation problem that the field is currently missing: how to measure text-to-4D HOI faithfulness, contact correctness, part consistency, temporal stability, and physical plausibility.

### Why this matters

This may be the most realistic top-venue route if you want a strong contribution without retraining a huge model from scratch.

HOIGen-1M, CORE4D, HUMOTO, and FIction all suggest that evaluation and data are central. Your repo already exports exactly the kind of intermediate artifacts that a benchmark needs.

### Repo subsystems to modify

- `Generate_PAG/`
- `Segment_Video/`
- `Track_Object_Mesh/`
- `Track_Human_Object_Mesh/`
- new evaluation package under a dedicated directory

### New method components

- an HOI quality suite with automatic metrics for:
  - text-to-PAG consistency,
  - PAG-to-video consistency,
  - video-to-geometry consistency,
  - part-contact correctness,
  - temporal drift,
  - penetration and support violations,
  - camera-lock fidelity,
  - reconstruction confidence / uncertainty.

### Concrete implementation target

Release a benchmark with:

- canonical prompt templates,
- PAG annotations,
- success/failure taxonomies,
- quantitative evaluation scripts,
- and a set of strong baselines including your own system.

This could be especially strong if paired with:

- generated videos,
- selected real HOI videos,
- and a small curated expert-rating set.

### Experiments and ablations

- Correlate automatic metrics with human judgments.
- Compare prompt-only generation baselines vs 4DHOI.
- Compare with and without PAG conditioning.
- Compare modular vs unified future versions of the system.

### Directly answers

- HOIGen-1M
- CORE4D
- HUMOTO
- FIction

### Main risks

- Benchmark papers need unusually clean definitions and metric validation.
- If the benchmark is too tied to one pipeline, reviewers may see it as self-serving.

## What I Would Publish First

If the goal is the strongest paper, not the fastest paper, I would aim for one of these two targets.

### Best method-paper target

Build a stronger unified HOI lifting-and-refinement model:

- keep the PAG,
- keep explicit object-part semantics,
- reduce stage fragmentation,
- add physics-aware contact,
- and evaluate against reconstruction and HOI-generation baselines.

This is the most ambitious and the most "top venue" version.

### Best thesis-to-paper target with highest leverage

Publish 4DHOI as:

- a part-aware text-to-4D HOI framework,
- plus a benchmark/evaluation suite for text-to-4D HOI fidelity.

This is less glamorous than a huge learned model, but it is very defensible and well aligned with what the repo already does unusually well.

## Suggested Contribution Statement

If I had to compress your future paper direction into one sentence, I would aim for:

> We introduce an explicit part-aware framework for text-conditioned 4D human-object interaction that unifies semantic affordance structure, monocular geometric lifting, and contact-aware temporal refinement, together with a benchmark for evaluating HOI faithfulness in 4D.

That statement fits your current repo, and it leaves room for the strongest future improvements.

## Sources

- [AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation](https://openreview.net/forum?id=Trf0R8eoGF), ICLR 2025 Poster.
- [MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion](https://openreview.net/forum?id=lJpqxFgWCM), ICLR 2025 Spotlight.
- [HSI-GPT: A General-Purpose Large Scene-Motion-Language Model for Human Scene Interaction](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_HSI-GPT_A_General-Purpose_Large_Scene-Motion-Language_Model_for_Human_Scene_Interaction_CVPR_2025_paper.html), CVPR 2025.
- [InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_InterMimic_Towards_Universal_Whole-Body_Control_for_Physics-Based_Human-Object_Interactions_CVPR_2025_paper.html), CVPR 2025.
- [End-to-End HOI Reconstruction Transformer with Graph-based Encoding](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_End-to-End_HOI_Reconstruction_Transformer_with_Graph-based_Encoding_CVPR_2025_paper.html), CVPR 2025.
- [Hand-held Object Reconstruction from RGB Video with Dynamic Interaction](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Hand-held_Object_Reconstruction_from_RGB_Video_with_Dynamic_Interaction_CVPR_2025_paper.html), CVPR 2025.
- [Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_Uni4D_Unifying_Visual_Foundation_Models_for_4D_Modeling_from_a_CVPR_2025_paper.html), CVPR 2025.
- [Shape of Motion: 4D Reconstruction from a Single Video](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.html), ICCV 2025.
- [HOIGen-1M: A Large-scale Dataset for Human-Object Interaction Video Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_HOIGen-1M_A_Large-scale_Dataset_for_Human-Object_Interaction_Video_Generation_CVPR_2025_paper.html), CVPR 2025.
- [CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CORE4D_A_4D_Human-Object-Human_Interaction_Dataset_for_Collaborative_Object_REarrangement_CVPR_2025_paper.html), CVPR 2025.
- [HUMOTO: A 4D Dataset of Mocap Human Object Interactions](https://openreview.net/forum?id=kUZ4wWEzTV), ICCV 2025 Wild3D workshop.
- [FIction: 4D Future Interaction Prediction from Video](https://openaccess.thecvf.com/content/CVPR2025/html/Ashutosh_FIction_4D_Future_Interaction_Prediction_from_Video_CVPR_2025_paper.html), CVPR 2025.
- [4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_4Real-Video_Learning_Generalizable_Photo-Realistic_4D_Video_Diffusion_CVPR_2025_paper.html), CVPR 2025.
- [Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Unleashing_the_Potential_of_Multi-modal_Foundation_Models_and_Video_Diffusion_CVPR_2025_paper.html), CVPR 2025. This paper introduces PhysFlow.
