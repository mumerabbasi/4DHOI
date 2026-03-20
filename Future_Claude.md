# Future Directions for 4DHOI

This document compares 4DHOI against recent methods from top venues (CVPR, SIGGRAPH, ICLR, NeurIPS, ECCV, ICCV, 3DV) and identifies concrete research directions for extending the work into a top-venue publication.

---

## Where 4DHOI Stands in the Landscape

### What 4DHOI Does Uniquely Well

4DHOI occupies a distinctive position in the literature. Its key strengths:

1. **Zero-shot generalization**: Unlike CHOIS (ECCV 2024), HOI-Diff (CVPR 2024), HOI-Dyn (NeurIPS 2025), OMOMO (CVPR 2024), and InterAct (CVPR 2025), 4DHOI requires no training data. It competes with InterDreamer (NeurIPS 2024), InteractAnything (CVPR 2025), and ZeroHSI (3DV 2026) in the zero-shot regime.

2. **Part-level affordance reasoning via PAG**: No other zero-shot method has an explicit graph encoding which body parts contact which object parts with temporal attributes (continuous/static). InterDreamer uses LLM for semantics but lacks part-level graph structure. InteractAnything uses LLM + diffusion for affordance but generates static poses only.

3. **Full pipeline from text to 4D**: Most methods do either generation OR reconstruction. 4DHOI does both: text -> video -> 3D reconstruction -> joint refinement. ArtHOI (2025) shares this philosophy but focuses on articulated objects.

4. **PAG-guided physics-aware refinement**: The joint refinement stage uses PAG edges for contact supervision, SDF for penetration, and object states for motion smoothness -- a principled combination that no single competitor matches.

### Where 4DHOI Falls Short

Comparing against the state of the art reveals clear gaps:

| Limitation | Competitors That Solve It |
|------------|--------------------------|
| Rigid objects only | ArtHOI (articulated objects), ARCTIC (articulated manipulation) |
| No physics simulation | InterMimic (CVPR 2025 Highlight), InterDiff (ICCV 2023), PhysPT (CVPR 2024) |
| Coarse hand model (SMPL) | HOLD (CVPR 2024), BIGS (CVPR 2025), OpenHOI (NeurIPS 2025 Oral), ARCTIC |
| No learned interaction priors | HOI-Dyn (NeurIPS 2025), SyncDiff (ICCV 2025), DAViD (ICCV 2025) |
| Sequential error accumulation | CARI4D (CVPR 2026, NVIDIA) uses learned render-and-compare |
| Single viewpoint reconstruction | CAT4D (CVPR 2025) generates multi-view video |
| No formal evaluation benchmark | InterAct (CVPR 2025) defines 6 benchmark tasks; BEHAVE/HOI4D provide ground truth |

---

## Concrete Future Directions

Below are research directions ordered by estimated impact and feasibility. Each direction includes the specific technical approach, which papers to position against, and what the contribution claim would be.

---

### Direction 1: Articulated Object Support

**Impact: Very High | Directly addresses a primary limitation**

#### The Gap
4DHOI assumes all objects are rigid with SE(3) per-frame transforms. Real-world interactions involve articulated objects: opening a laptop, pulling a drawer, using scissors, opening a fridge door.

#### Closest Competitor
**ArtHOI (2025, under review)**: First zero-shot framework for articulated HOI via 4D reconstruction from video priors. Uses flow-based part segmentation and decoupled reconstruction. However, ArtHOI does NOT have PAG -- it lacks explicit contact-level reasoning.

#### Concrete Technical Approach
1. **Extend PAG with articulation edges**: Add a new edge type connecting object parts to each other with a joint type attribute (`hinge`, `prismatic`, `ball`). The LLM can infer joint types from object semantics (e.g., laptop hinge between "laptop, screen" and "laptop, keyboard").

2. **Per-part SE(3) tracking**: Instead of one SE(3) per object, track one SE(3) per object part. Parts share a parent transform (the object's global pose) with local articulation transforms.

3. **Joint limit constraints**: Add loss terms that enforce articulation constraints -- hinge joints have 1-DOF rotation, prismatic joints have 1-DOF translation.

4. **PAG-guided articulation**: The `is_rel_static` attribute between two object parts (object-to-object edges) naturally describes whether parts move relative to each other.

#### Contribution Claim
"First zero-shot 4D HOI method with both part-level contact reasoning AND articulated object support, unifying PAG-guided contact constraints with learned articulation estimation."

#### Positioning
- vs. ArtHOI: 4DHOI would have PAG contact constraints that ArtHOI lacks
- vs. ARCTIC: 4DHOI would be zero-shot while ARCTIC requires MoCap data
- vs. rigid 4DHOI: Direct extension solving the #1 limitation

---

### Direction 2: Physics-Informed Refinement via Differentiable Simulation

**Impact: Very High | Addresses physical plausibility gap**

#### The Gap
4DHOI uses SDF penetration as a proxy for physics. Real interactions involve forces, friction, gravity, momentum, and mass. A person lifting a heavy box moves differently than lifting an empty box. Current 4DHOI output can have physically implausible motions (sliding feet, floating objects, penetrating geometry).

#### Key Competitors
- **InterMimic (CVPR 2025 Highlight)**: Physics-based RL policy for HOI. Can take kinematic output and make it physically valid. But requires simulation setup and is not differentiable end-to-end.
- **InterDiff (ICCV 2023)**: Physics-informed diffusion. Integrates physics corrections into denoising. Requires training data.
- **PhysPT (CVPR 2024)**: Physics-aware transformer for human dynamics. Self-supervised. No explicit object modeling.
- **PA-HOI (ACM MM 2025)**: Dataset showing how object physical properties affect motion.

#### Concrete Technical Approach
1. **Replace SDF penetration with differentiable contact simulation**: Use a differentiable physics engine (e.g., DiffTaichi, Warp, or Brax) to simulate contact forces between human and object meshes. Back-propagate through the simulation to refine poses.

2. **Gravity and support constraints**: Add gravity as a force on objects. Objects not in hand contact must be supported (resting on surfaces, hanging, etc.). This is already partially in PAG via `is_translational` -- extend it with physical reasoning.

3. **Contact force consistency**: When PAG says "hand grasps handle," the contact force must be sufficient to support the object's weight against gravity. This creates a force-balance constraint.

4. **Friction cone constraints**: At contact points, enforce Coulomb friction -- tangential forces must lie within the friction cone.

#### Contribution Claim
"First zero-shot HOI method that combines semantic contact graphs (PAG) with differentiable physics simulation, producing interactions that satisfy both semantic plausibility (correct contacts) and physical plausibility (force balance, friction, gravity)."

#### Positioning
- vs. InterMimic: Differentiable and end-to-end, no RL training required
- vs. InterDiff: Zero-shot, no training data
- vs. current 4DHOI: Replaces heuristic SDF loss with principled physics

---

### Direction 3: Dexterous Hand Integration

**Impact: High | Critical for manipulation realism**

#### The Gap
4DHOI uses SMPL (6890 vertices) which has very coarse hands. Real manipulation requires finger-level contact: grasping a cup handle, pressing buttons, turning keys. The current PAG treats "left hand" as a single body part node.

#### Key Competitors
- **HOLD (CVPR 2024 Highlight)**: Category-agnostic hand-object reconstruction from video. Neural implicit representation. No full-body.
- **BIGS (CVPR 2025)**: Bimanual category-agnostic reconstruction via 3D Gaussians. Hands only.
- **OpenHOI (NeurIPS 2025 Oral)**: Open-world hand-object synthesis using 3D MLLM for affordance grounding. Dexterous but hands only.
- **ContactGen (ICCV 2023)**: Three-component contact maps (location, part, direction) for grasp generation.
- **ARCTIC (CVPR 2023)**: Bimanual dexterous manipulation dataset with articulated objects.
- **SMPLer-X / SMPLest-X (NeurIPS 2023 / TPAMI 2025)**: Foundation models for expressive body + hand + face estimation.

#### Concrete Technical Approach
1. **Switch from SMPL to SMPL-X with full hand articulation**: GVHMR already predicts SMPL-X parameters. Instead of collapsing to SMPL via the smplx2smpl matrix, retain the full SMPL-X mesh (~10475 vertices) including articulated fingers.

2. **Extend PAG with finger-level nodes**: Add finger parts (thumb, index, middle, ring, pinky) as body part nodes. The LLM can reason about which fingers contact which object parts.

3. **Hand-specific refinement**: After the global joint refinement, run a second refinement stage focusing on hand-object contact with higher-resolution constraints. Use ContactGen-style contact maps as additional supervision.

4. **Grasp taxonomy integration**: Use grasp taxonomy (power grasp, precision grasp, pinch, etc.) as a prior. The LLM can predict grasp type from interaction description, and this constrains finger configurations.

#### Contribution Claim
"First zero-shot full-body HOI method with finger-level dexterity, combining PAG part-level contact reasoning with SMPL-X hand articulation for manipulation-accurate interactions."

#### Positioning
- vs. HOLD/BIGS/OpenHOI: Full-body (not just hands) AND zero-shot
- vs. current 4DHOI: Finger-level contact instead of coarse hand blobs
- vs. ContactGen: Dynamic 4D interactions, not just static grasps

---

### Direction 4: Learned PAG Refinement via Vision-Language Feedback

**Impact: High | Makes the system self-correcting**

#### The Gap
PAG is generated by an LLM from text only. The LLM has no visual feedback -- it cannot see whether the generated video actually matches the intended interaction. Errors in PAG propagate through the entire pipeline. The multi-sample voting helps but does not fix systematic LLM biases.

#### Key Competitors
- **InteractAnything (CVPR 2025)**: Uses LLM-distilled human feedback for detailed optimization. But static poses only.
- **DAViD (ICCV 2025)**: Learns dynamic affordance from generated videos. Closes the loop between generation and 3D.
- **EgoChoir (NeurIPS 2024)**: Estimates 3D interaction regions from visual features.
- **DECO (ICCV 2023)**: Dense vertex-level contact estimation from images.

#### Concrete Technical Approach
1. **Visual PAG verification**: After generating the video, use a VLM (GPT-4V, Qwen-VL) to analyze the generated video and verify/correct the PAG. Ask: "Does the left hand maintain continuous contact with the iron handle throughout the video?" Update `is_continuous` based on visual evidence.

2. **Contact detection from video**: Use DECO-style contact estimation on the generated video frames to produce dense contact maps. Compare with PAG edges. If PAG says contact but vision says no contact (or vice versa), flag for correction.

3. **Iterative PAG-Video loop**: Generate PAG -> generate video -> verify PAG against video -> regenerate corrected PAG -> regenerate video. This closes the loop and self-corrects.

4. **Learned PAG prior**: Fine-tune a small model on (text, video, verified_PAG) triples collected from the pipeline. Over time, this model can predict better PAGs than the zero-shot LLM.

#### Contribution Claim
"A self-correcting zero-shot HOI pipeline where vision-language feedback refines the Part Affordance Graph using visual evidence from generated videos, closing the semantic-visual loop."

---

### Direction 5: Multi-View Reconstruction via Diffusion-Based View Synthesis

**Impact: High | Solves depth ambiguity**

#### The Gap
4DHOI generates and reconstructs from a single camera view. This creates fundamental depth ambiguity -- the Z dimension is poorly constrained. Depth Anything 3 helps but monocular depth has inherent limitations.

#### Key Competitors
- **CAT4D (CVPR 2025)**: Transforms monocular video into multi-view video using a multi-view video diffusion model, enabling robust 4D Gaussian reconstruction.
- **4D Gaussian Splatting (CVPR 2024)**: Multi-view 4D reconstruction.
- **Shape of Motion (ICCV 2025)**: SE(3) motion bases for 4D reconstruction from monocular video.

#### Concrete Technical Approach
1. **Multi-view video generation**: After generating the frontal video, use a view-conditioned video diffusion model (like CAT4D's) to synthesize side and back views of the same interaction.

2. **Multi-view consistent reconstruction**: Replace single-view depth + Chamfer alignment with multi-view reconstruction. Project object meshes into all views and optimize alignment jointly.

3. **View-dependent PAG verification**: Different viewpoints reveal different contacts. Verify PAG edges using multi-view evidence (a contact visible from the side may be occluded frontally).

4. **Gaussian Splatting as intermediate representation**: Replace or augment PLY meshes with 3D Gaussian Splatting for the reconstruction phase. 4DGS naturally handles multi-view consistency and real-time rendering.

#### Contribution Claim
"Multi-view zero-shot 4D HOI by combining PAG-guided video generation with diffusion-based novel view synthesis, eliminating the single-view depth ambiguity bottleneck."

---

### Direction 6: Temporal PAG with Phase-Aware Interaction Modeling

**Impact: High | Novel representation contribution**

#### The Gap
Current PAG is essentially static -- it describes the overall interaction but does not model temporal phases. Real interactions have distinct phases: approach, pre-grasp, grasp, manipulation, release, retract. Contact attributes change across phases (not just binary on/off).

#### Key Competitors
- **HOI-Dyn (NeurIPS 2025)**: Models driver-responder dynamics but requires training data.
- **SyncDiff (ICCV 2025)**: Frequency decomposition for high/low-frequency motion components.
- **ManipNet (SIGGRAPH 2021)**: Hierarchical approach with approach/grasp/movement phases.

#### Concrete Technical Approach
1. **Temporal PAG (T-PAG)**: Extend PAG with temporal phase annotations. Each edge gets a `phases` attribute listing which interaction phases it's active in. Example:
   ```json
   {
     "nodes": ["cup, handle", "person 1, right hand"],
     "phases": [
       {"name": "approach", "is_continuous": false, "is_rel_static": false},
       {"name": "grasp", "is_continuous": true, "is_rel_static": true},
       {"name": "lift", "is_continuous": true, "is_rel_static": true},
       {"name": "drink", "is_continuous": true, "is_rel_static": false},
       {"name": "place", "is_continuous": true, "is_rel_static": true},
       {"name": "release", "is_continuous": false, "is_rel_static": false}
     ]
   }
   ```

2. **Phase detection from video**: Use a VLM to detect phase boundaries in the generated video (frame ranges for each phase). This naturally emerges from video understanding.

3. **Phase-aware loss scheduling**: Apply different contact constraint weights per phase. During "approach," contact drift loss should be relaxed. During "grasp," it should be strict.

4. **Phase-conditioned smoothness**: Different phases have different motion characteristics. "Approach" is smooth and slow; "manipulation" may have fast rotations.

#### Contribution Claim
"Temporal Part Affordance Graphs (T-PAG) that model interaction phases with time-varying contact semantics, enabling phase-aware optimization that respects the temporal structure of human-object interactions."

---

### Direction 7: Evaluation Framework and Benchmarking

**Impact: Very High for publishability | Currently missing**

#### The Gap
4DHOI currently has no formal quantitative evaluation. For a top-venue publication, rigorous evaluation against baselines on established benchmarks is essential.

#### Key Benchmarks
- **BEHAVE (CVPR 2022)**: Full-body HOI with SMPL + object meshes. Used by InterDreamer.
- **HOI4D (CVPR 2022)**: Egocentric HOI with 2.4M frames, 16 categories.
- **GRAB (ECCV 2020)**: Whole-body grasping with SMPL-X.
- **HOI-M3 (CVPR 2024 Highlight)**: Multi-person multi-object with dense MoCap.
- **InterAct (CVPR 2025)**: 6 standardized benchmark tasks for HOI generation.

#### Key Metrics to Implement
1. **Contact precision/recall**: Compare predicted contact regions with ground truth contact maps (BEHAVE, GRAB provide these).
2. **Penetration volume**: Measure intersection volume between human and object meshes. Compare against InterDiff, InterMimic.
3. **FID/KID on rendered frames**: Measure visual quality of reconstructed interactions.
4. **MPJPE (Mean Per-Joint Position Error)**: For human pose accuracy against MoCap ground truth.
5. **Object pose error**: ATE (Absolute Trajectory Error) for object 6DoF tracking.
6. **Physical plausibility metrics**: Ground contact consistency, balance (center of mass over support polygon), smoothness.
7. **Interaction quality**: CLIP-based text-interaction alignment score.
8. **User studies**: A/B preference tests comparing against InterDreamer, InteractAnything.

#### Key Baselines to Compare Against
1. **InterDreamer (NeurIPS 2024)**: Most directly comparable zero-shot method
2. **InteractAnything (CVPR 2025)**: Zero-shot with LLM feedback (but static)
3. **CHOIS (ECCV 2024)**: Trained diffusion baseline
4. **HOI-Diff (CVPR 2024)**: Trained diffusion with affordance
5. **ZeroHSI (3DV 2026)**: Zero-shot via video generation
6. **CARI4D (CVPR 2026)**: Category-agnostic 4D HOI reconstruction (reconstruction-only baseline)

#### Contribution Claim
This is not a separate paper contribution but is essential for any submission. The evaluation should demonstrate that PAG-guided refinement produces quantitatively better contact accuracy and physical plausibility than methods without explicit contact graphs.

---

### Direction 8: End-to-End Differentiable Pipeline

**Impact: Medium-High | Architectural contribution**

#### The Gap
4DHOI's 17-stage pipeline runs sequentially with no gradient flow between stages. Errors in early stages (bad segmentation, wrong depth) propagate without correction. CARI4D (CVPR 2026) uses learned render-and-compare for joint refinement.

#### Concrete Technical Approach
1. **Differentiable rendering in the loop**: Replace the discrete Chamfer loss with a differentiable renderer that renders the reconstructed scene and compares against the generated video via photometric loss.

2. **Joint depth-pose optimization**: Instead of estimating depth separately and then aligning, jointly optimize depth and mesh alignment via differentiable rendering.

3. **Learned render-and-compare**: Train a small network that takes (rendered image, target image) pairs and predicts pose corrections. This replaces the iterative optimization with a fast forward pass. CARI4D demonstrates this approach.

4. **Gradient flow from refinement to tracking**: Allow the joint refinement stage to provide gradients to the object tracking stage, enabling end-to-end optimization of the tracking + refinement pipeline.

---

### Direction 9: Dynamic Gaussian Splatting Representation

**Impact: Medium-High | Modern representation**

#### The Gap
4DHOI uses explicit PLY meshes for both humans and objects. 3D Gaussian Splatting has emerged as the dominant representation for real-time rendering and view synthesis (4DGS at CVPR 2024, SplatFlow at CVPR 2025).

#### Key Competitors
- **4D Gaussian Splatting (CVPR 2024)**: Real-time dynamic scene rendering with deformable Gaussians.
- **SplatFlow (CVPR 2025)**: Self-supervised 4DGS from optical flow.
- **BIGS (CVPR 2025)**: 3D Gaussians for bimanual hand-object interaction.
- **DreamGaussian4D**: Efficient 4D generation with Gaussians.

#### Concrete Technical Approach
1. **Hybrid mesh-Gaussian representation**: Keep SMPL meshes for humans (needed for part segmentation and contact reasoning) but represent objects as 3D Gaussians. This gives photorealistic object rendering while maintaining structured human contact points.

2. **PAG-guided Gaussian segmentation**: Use PAG part nodes to assign Gaussians to semantic object parts. This enables part-level contact constraints on Gaussians.

3. **Deformable Gaussians for non-rigid objects**: For soft objects (cushions, clothing), Gaussians can deform naturally without the rigid SE(3) assumption.

4. **Real-time interactive visualization**: 3DGS enables real-time rendering of the reconstructed 4D interaction, which is valuable for applications (VR/AR, gaming, training simulation).

---

### Direction 10: Scaling to Complex Multi-Person Multi-Object Scenes

**Impact: Medium | Broadens applicability**

#### The Gap
4DHOI handles single-human + single/few-objects. Real-world interactions involve multiple people interacting with multiple objects simultaneously (e.g., two people carrying a table, a family cooking together).

#### Key Competitors
- **HOI-M3 (CVPR 2024 Highlight)**: Multi-person multi-object dataset with dense MoCap.
- **SyncDiff (ICCV 2025)**: Synchronized multi-body interaction synthesis.
- **InterAct (CVPR 2025)**: Unified framework handling diverse HOI configurations.

#### Concrete Technical Approach
1. **PAG naturally scales**: PAG already supports multiple persons and objects. "person 1, left hand" and "person 2, right hand" can both have edges to the same object part. This is already partially implemented.

2. **Person-person interaction edges**: Extend PAG with person-to-person edges (e.g., handshake, high-five, passing objects). The LLM can reason about these.

3. **Multi-agent video generation**: Use video generation models that handle multiple characters. Guide with per-person PAG subgraphs.

4. **Scalable optimization**: For N humans and M objects, the joint refinement has O(N*M) contact pairs. Use PAG sparsity to prune non-interacting pairs and optimize only relevant contacts.

---

## Detailed Comparison Table: 4DHOI vs. Recent Methods

| Method | Venue | Zero-Shot | Full-Body | 4D | Part Contact | Articulated Obj | Physics | PAG/Graph |
|--------|-------|-----------|-----------|----|----|----|----|-----|
| **4DHOI (ours)** | -- | Yes | Yes | Yes | PAG | No | SDF penetration | Yes |
| InterDreamer | NeurIPS 2024 | Yes | Yes | Yes | No | No | World model | No |
| InteractAnything | CVPR 2025 | Yes | Yes | No (static) | Affordance parsing | No | Optimization | No |
| ZeroHSI | 3DV 2026 | Yes | Yes | Yes | No | No | No | No |
| ArtHOI | 2025 (review) | Yes | Yes | Yes | Flow-based | Yes | No | No |
| CARI4D | CVPR 2026 | Recon only | Yes | Yes | Contact constraints | No | Contact physics | No |
| CHOIS | ECCV 2024 | No (trained) | Yes | Yes | No | No | No | No |
| HOI-Diff | CVPR 2024 | No (trained) | Yes | Yes | Learned affordance | No | No | No |
| HOI-Dyn | NeurIPS 2025 | No (trained) | Yes | Yes | No | No | Dynamics model | No |
| InterMimic | CVPR 2025 | No (RL) | Yes | Yes | Implicit | No | Full simulation | No |
| SyncDiff | ICCV 2025 | No (trained) | Yes | Yes | Graphical model | No | No | Yes (learned) |
| OpenHOI | NeurIPS 2025 | Yes | Hands only | Yes | 3D MLLM affordance | No | Refinement | No |
| HOLD | CVPR 2024 | Yes | Hands only | Yes (video) | Implicit | No | No | No |
| BIGS | CVPR 2025 | Yes | Hands only | Yes | Gaussian contact | No | No | No |
| DAViD | ICCV 2025 | Yes | Yes | Yes | Dynamic affordance | No | No | No |
| CONTHO | CVPR 2024 | No (trained) | Yes | No (image) | Contact transformer | No | No | No |
| CAT4D | CVPR 2025 | Recon only | No | Yes | No | No | No | No |
| Shape of Motion | ICCV 2025 | Recon only | No | Yes | No | No | No | No |
| ProciGen-HDM | CVPR 2024 | No (synthetic) | Yes | No (image) | Implicit | No | No | No |
| InterAct | CVPR 2025 | No (trained) | Yes | Yes | Optimization | No | Penetration fix | No |

---

## Recommended Research Strategy

### For Maximum Impact: Combine Directions 1 + 3 + 7

**Title idea**: "4DHOI++: Zero-Shot 4D Human-Object Interaction with Articulated Objects, Dexterous Hands, and Part Affordance Guidance"

This combination would:
- Extend PAG to handle articulated objects (Direction 1) -- new representation
- Add SMPL-X finger-level contact (Direction 3) -- finer manipulation
- Provide rigorous evaluation on BEHAVE, GRAB, HOI4D (Direction 7) -- publishable results

**Why this combination works**: It addresses the two biggest limitations (rigid objects, coarse hands) while adding the evaluation needed for publication. The PAG extension is a clean, elegant contribution that builds on the existing framework.

### For Strongest Novelty: Combine Directions 2 + 6

**Title idea**: "Temporal Part Affordance Graphs with Differentiable Physics for Zero-Shot 4D Human-Object Interaction"

This combination would:
- Introduce Temporal PAG (T-PAG) with interaction phases (Direction 6) -- novel representation
- Use differentiable physics simulation for refinement (Direction 2) -- principled physics

**Why this combination works**: T-PAG is a genuinely novel representation that no competitor has. Differentiable physics makes the refinement principled. Together, they make a strong "new representation + new optimization" story.

### For Fastest Path to Publication: Direction 7 + one of {1, 3, 4}

If time is a concern, the fastest path is to:
1. Implement formal evaluation (Direction 7) -- essential regardless
2. Pick ONE extension (articulated objects OR dexterous hands OR visual PAG verification)
3. Show quantitative improvements on established benchmarks

---

## Key Papers to Read in Full

These are the papers most relevant to 4DHOI's positioning:

1. **InterDreamer** (NeurIPS 2024) -- most directly comparable zero-shot method
2. **InteractAnything** (CVPR 2025) -- closest in LLM-guided affordance approach
3. **ArtHOI** (2025) -- shares the video-to-4D pipeline, extends to articulated objects
4. **CARI4D** (CVPR 2026, NVIDIA) -- state-of-the-art 4D HOI reconstruction
5. **InterMimic** (CVPR 2025 Highlight) -- physics-based refinement of kinematic HOI
6. **ZeroHSI** (3DV 2026) -- zero-shot 4D via video generation (human-scene)
7. **HOI-Dyn** (NeurIPS 2025) -- interaction dynamics modeling
8. **OpenHOI** (NeurIPS 2025 Oral) -- 3D MLLM for affordance grounding
9. **SyncDiff** (ICCV 2025) -- synchronized multi-body interaction
10. **DAViD** (ICCV 2025) -- dynamic affordance from video diffusion models
11. **InterAct** (CVPR 2025) -- largest HOI benchmark and data consolidation
12. **CONTHO** (CVPR 2024) -- contact-based joint reconstruction transformer
