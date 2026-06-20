# Agentic Verification Loop for 4DHSI

## Summary

Build a paper-style evaluation and verification system for `4DHSI` that improves robustness by sampling multiple stochastic candidates, scoring them with deterministic geometry metrics plus targeted VLM judgments, and selecting or repairing the best interaction. The first milestone is a thesis-defensible prototype, not a fully autonomous production agent.

Use a moderate budget per interaction:

- 4-8 human inpaint candidates.
- 3-4 contact-overlay candidates for the best human candidates.
- 2-3 final optimization variants.
- At most 2 repair rounds.

The core principle is that VLMs should judge narrow evidence packets, not one broad render. Each question gets specific crops, renders, and metrics for contact, penetration, support/floating, pose plausibility, and scene preservation.

## Key Changes

### Candidate Sampling

Add candidate-aware outputs to the stochastic stages:

- Human inpainting writes candidates under `02_Generate_Human_Frame/output/<interaction>/candidates/human_<id>/`.
- Contact estimation writes candidates under `03_Estimate_Contact/output/<interaction>/candidates/human_<id>/contact_<id>/`.
- Static optimization writes variants under `05_Optimize_Static_Scene/output/<interaction>/candidates/<candidate_id>/opt_<id>/`.

Each candidate should have a `candidate_manifest.json` containing source interaction name, upstream candidate ids, model/prompt/settings, artifact paths, seed or sample index, and status.

### Deterministic Geometry Evaluation

Add `07_Evaluate_Static_Scene` to read optimized candidates and produce:

- `metrics.json`
- `evidence/`
- `vlm_judgments.json`
- `verification_summary.json`

Compute deterministic metrics before VLM judging:

- Per-contact edge distance.
- Required support/floating distance for feet or hips.
- Penetration from SMPL-X SDF scene-point query stats.
- Self-intersection loss.
- Pose drift from GVHMR.
- Height plausibility.
- Contact mask projection quality.

Use hard gates to reject obvious failures before VLM judging: missing contact, severe penetration, empty contact projection, missing mesh, or missing metrics.

### Targeted VLM Evidence Packets

Generate specific evidence for each remaining candidate:

- Full original-camera render or available debug render.
- Contact evidence per SIG edge.
- Penetration evidence from SDF debug artifacts.
- Support/floating evidence around feet, hips, floor, or object support.
- Pose evidence from available mesh/debug views.
- Scene preservation evidence from the original scene, inpainted frame, and target mask overlay.

Use separate structured VLM prompts:

- `contact_judge`
- `penetration_judge`
- `support_judge`
- `pose_judge`
- `interaction_judge`

Each judge returns:

```json
{
  "score_0_to_5": 0,
  "pass": false,
  "failure_tags": ["missing_contact"],
  "reason": "short concrete reason"
}
```

Final candidate score should combine deterministic metrics and VLM scores, with deterministic hard failures overriding a high VLM score.

### Agentic Selection and Repair Loop

Use this staged loop:

1. Generate 4-8 human inpaint candidates.
2. Score human candidates for scene preservation, full-body visibility, target-object contact plausibility, and anatomy.
3. Keep the top 2 human candidates.
4. For each kept human candidate, generate 3-4 contact-overlay candidates.
5. Score contact masks with color extraction quality, projected scene-face stability, and targeted VLM contact checks.
6. Keep the top 2 contact candidates overall.
7. Run 2-3 optimization variants per kept contact candidate with different seeds and weights.
8. Verify final meshes using deterministic metrics plus targeted VLM judges.
9. Select the best passing candidate.
10. If no candidate passes, repair based on failure tags:
    - `bad_human_pose`: resample human frame.
    - `bad_contact_mask`: resample contact overlay.
    - `floating`: rerun optimization with stronger floor/support/contact weights.
    - `penetration`: rerun optimization with stronger scene-intersection weight.
    - `wrong_target`: rerun target selection or SIG target prompt.

Limit repair to 2 rounds so experiments remain reproducible.

## Evaluation Plan

Run a paper-style study over a fixed interaction set:

- Start with the existing 7 interactions.
- Expand to 15-20 interactions covering lifting, sitting, lying, leaning, hand contact, foot contact, and hips/support contact.
- Keep scenes and prompts fixed for all methods.

Compare:

- Baseline: current single-sample 4DHSI pipeline.
- Sampling only: multiple candidates with simple VLM image selection.
- Geometry verification: deterministic metrics and hard gates only.
- Full agentic loop: sampling, targeted VLM evidence, geometry metrics, and repair.

Report:

- Success rate by manual inspection.
- Mean contact distance per required edge.
- Penetration count / min SDF / inside-point ratio.
- Floating/support failure rate.
- VLM score agreement with manual labels.
- Cost/runtime per successful interaction.
- Ablation: broad single-render VLM judge vs targeted evidence packets.

Manual labels should use a compact rubric: correct target, required contacts, no severe penetration, no floating, plausible pose, acceptable scene preservation.

## Assumptions

- Scope is `4DHSI` only; `4DHOI` is reference material.
- Primary milestone is paper-style evaluation, not full production automation.
- Moderate sampling budget is acceptable.
- Existing Gemini/Ollama/OpenAI-compatible model usage remains allowed.
- The first implementation should reuse current artifacts from `03_Estimate_Contact` and `05_Optimize_Static_Scene` rather than replacing the optimizer.
- The verification system should be reproducible: every candidate, prompt, score, and rejection reason must be saved to disk.
