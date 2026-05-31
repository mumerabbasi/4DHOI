from __future__ import annotations

import argparse
import base64
import csv
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCANNET_ROOT = PROJECT_DIR.parent / "Scannet++" / "data"
DEFAULT_SMPL_SEG_JSON = PROJECT_DIR / "05_Estimate_Human_Pose" / "assets" / "smplx_vert_segmentation.json"
CONTACT_PROMPT = SCRIPT_DIR / "system_prompt_contact.md"
POSE_PROMPT = SCRIPT_DIR / "system_prompt_pose.md"
PENETRATION_PROMPT = SCRIPT_DIR / "system_prompt_penetration.md"


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_label(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").replace("-", " ").split())


def slugify(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def default_paths(interaction_name: str) -> dict[str, Path]:
    return {
        "optimizer_output_root": PROJECT_DIR /
        "06_Optimize_Static_Scene" /
        "output" /
        interaction_name,
        "sig_json": PROJECT_DIR /
        "01_Generate_SIG" /
        "output" /
        interaction_name /
        "scene_interaction_graph.json",
        "input_scene_json": PROJECT_DIR /
        "01_Generate_SIG" /
        "input_prompts" /
        interaction_name /
        "input_scene.json",
        "outdir": SCRIPT_DIR /
        "output" /
        interaction_name,
        "contact_spec": PROJECT_DIR /
        "04_Estimate_Contact" /
        "output" /
        interaction_name /
        "contact_spec.json",
        "contact_canvas_image": PROJECT_DIR /
        "04_Estimate_Contact" /
        "output" /
        interaction_name /
        "prompt" /
        "target_scene_crop.png",
    }


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).resolve() if raw_path else default_path.resolve()


def rank_failure_tags(tags: list[str]) -> list[str]:
    priority = [
        "missing_contact",
        "severe_penetration",
        "implausible_pose",
        "wrong_interaction",
        "wrong_target",
        "no_decision",
    ]
    normalized = [slugify(tag) for tag in tags if tag]
    return sorted(
        set(normalized),
        key=lambda tag: (priority.index(tag) if tag in priority else 999, tag),
    )


def read_final_loss_summary(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {"available": False, "path": str(csv_path)}
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        return {"available": False, "path": str(csv_path), "error": "empty CSV"}

    parsed: dict[str, Any] = {"available": True, "path": str(csv_path)}
    for key, value in rows[-1].items():
        if value == "":
            parsed[key] = value
            continue
        try:
            parsed[key] = float(value) if "." in value or "e" in value.lower() else int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def load_optimized_params(params_path: Path) -> dict[str, Any]:
    if not params_path.exists():
        return {"available": False, "path": str(params_path)}

    try:
        if zipfile.is_zipfile(params_path):
            with zipfile.ZipFile(params_path) as archive:
                data_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
                payload = pickle.loads(archive.read(data_name))
            if isinstance(payload, dict):
                return {
                    "available": True,
                    "path": str(params_path),
                    "height_m": payload.get("height_m"),
                    "scale": payload.get("scale"),
                    "canonical_height_unscaled_m": payload.get("canonical_height_unscaled_m"),
                    "loader": "zip_pickle",
                }
    except Exception:
        pass

    try:
        import torch
    except Exception as error:
        return {
            "available": False,
            "path": str(params_path),
            "error": f"Could not import torch to read optimized params: {error}",
        }

    try:
        payload = torch.load(params_path, map_location="cpu", weights_only=False)
    except Exception as error:
        return {
            "available": False,
            "path": str(params_path),
            "error": f"Could not read optimized params: {error}",
        }
    if not isinstance(payload, dict):
        return {"available": False, "path": str(params_path), "error": "optimized params payload is not a dict"}
    return {
        "available": True,
        "path": str(params_path),
        "height_m": payload.get("height_m"),
        "scale": payload.get("scale"),
        "canonical_height_unscaled_m": payload.get("canonical_height_unscaled_m"),
        "loader": "torch",
    }


def collect_contact_metrics(alignment_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    final_edges = (
        alignment_summary.get("human", {})
        .get("final_frame_0", {})
        .get("interaction_edges", [])
    )
    if not isinstance(final_edges, list):
        raise ValueError("alignment_summary human.final_frame_0.interaction_edges must be a list")

    threshold_m = float(args.contact_threshold_m)
    edges: list[dict[str, Any]] = []
    distances: list[float] = []
    failure_tags: list[str] = []

    for index, edge in enumerate(final_edges):
        if not isinstance(edge, dict):
            continue
        distance = edge.get("nocontact_distance_m")
        distance_m = float(distance) if distance is not None else None
        passed = distance_m is not None and distance_m <= threshold_m
        if not passed:
            failure_tags.append("missing_contact")
        if distance_m is not None:
            distances.append(distance_m)
        edges.append(
            {
                "index": int(index),
                "moving_part_name": edge.get("moving_part_name"),
                "moving_segment_id": edge.get("moving_segment_id"),
                "fixed_part_name": edge.get("fixed_part_name"),
                "fixed_entity_name": edge.get("fixed_entity_name"),
                "fixed_point_count": edge.get("fixed_point_count"),
                "moving_vertex_count": edge.get("moving_vertex_count"),
                "reduction": edge.get("reduction"),
                "nocontact_raw": edge.get("nocontact_raw"),
                "nocontact_distance_m": distance_m,
                "threshold_m": threshold_m,
                "pass": bool(passed),
            }
        )

    return {
        "pass": bool(edges) and all(bool(edge["pass"]) for edge in edges),
        "failure_tags": rank_failure_tags(failure_tags),
        "edge_count": int(len(edges)),
        "edges": edges,
        "mean_distance_m": float(sum(distances) / len(distances)) if distances else None,
        "max_distance_m": float(max(distances)) if distances else None,
    }


def collect_penetration_metrics(alignment_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    final_debug = (
        alignment_summary.get("human", {})
        .get("optimization", {})
        .get("scene_intersect_debug", {})
        .get("final", {})
    )
    scene_points = final_debug.get("scene_points", {}) if isinstance(final_debug, dict) else {}
    min_sdf = scene_points.get("min_sdf_m")
    inside_points = scene_points.get("num_inside_points")
    min_sdf_f = float(min_sdf) if min_sdf is not None else None
    inside_i = int(inside_points) if inside_points is not None else None

    severe_by_sdf = min_sdf_f is not None and min_sdf_f < float(args.severe_penetration_min_sdf_m)
    severe_by_count = inside_i is not None and inside_i > int(args.severe_penetration_inside_points)
    severe = bool(severe_by_sdf or severe_by_count)

    return {
        "pass": not severe,
        "failure_tags": ["severe_penetration"] if severe else [],
        "severe_penetration": severe,
        "severe_by_min_sdf": bool(severe_by_sdf),
        "severe_by_inside_points": bool(severe_by_count),
        "thresholds": {
            "min_sdf_m": float(args.severe_penetration_min_sdf_m),
            "inside_points": int(args.severe_penetration_inside_points),
        },
        "scene_points": scene_points,
        "sdf_grid": final_debug.get("sdf_grid", {}) if isinstance(final_debug, dict) else {},
    }


def collect_metrics(
    sig_payload: dict[str, Any],
    alignment_summary: dict[str, Any],
    optimizer_root: Path,
    sig_json_path: Path,
    input_scene_json_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    contact = collect_contact_metrics(alignment_summary, args)
    penetration = collect_penetration_metrics(alignment_summary, args)
    optimization = alignment_summary.get("human", {}).get("optimization", {})
    optimizer_config = alignment_summary.get("optimizer", {})
    target = alignment_summary.get("target_object", {})
    if not isinstance(target, dict):
        target = {}

    failure_tags: list[str] = []
    failure_tags.extend(contact.get("failure_tags", []))
    failure_tags.extend(penetration.get("failure_tags", []))
    ranked_tags = rank_failure_tags(failure_tags)

    return {
        "interaction_name": alignment_summary.get("interaction_name", args.interaction_name),
        "scene_id": alignment_summary.get("scene_id"),
        "target_object": target,
        "interaction": sig_payload.get("interaction", ""),
        "source_paths": {
            "sig_json": str(sig_json_path),
            "input_scene_json": str(input_scene_json_path),
            "optimizer_output_root": str(optimizer_root),
            "alignment_summary": str(optimizer_root / "alignment_summary.json"),
            "world_mesh": str(optimizer_root / "meshes" / "frame_0000_world.ply"),
        },
        "thresholds": {
            "contact_threshold_m": float(args.contact_threshold_m),
            "severe_penetration_min_sdf_m": float(args.severe_penetration_min_sdf_m),
            "severe_penetration_inside_points": int(args.severe_penetration_inside_points),
        },
        "contact": contact,
        "penetration": penetration,
        "optimization": {
            "final_total_loss": optimization.get("final_total_loss"),
            "final_iter": optimization.get("final_iter"),
            "stage_iters": optimization.get("stage_iters"),
            "scene_intersect_stats": optimization.get("scene_intersect_stats"),
            "final_loss_summary": read_final_loss_summary(optimizer_root / "debug" / "csv" / "final_loss_summary.csv"),
        },
        "height": {
            "optimized_params": load_optimized_params(optimizer_root / "debug" / "params" / "optimized_frame_0000.pt"),
            "height_prior": optimizer_config.get("height_prior"),
        },
        "deterministic": {
            "pass": not bool(ranked_tags),
            "failure_tags": ranked_tags,
        },
    }


def build_verification_summary(metrics: dict[str, Any], vlm_judgments: dict[str, Any] | None = None) -> dict[str, Any]:
    failure_tags = list(metrics.get("deterministic", {}).get("failure_tags", []))
    vlm_enabled = bool(vlm_judgments and vlm_judgments.get("enabled"))
    vlm_no_decision = False

    if vlm_enabled and vlm_judgments:
        for judgment in vlm_judgments.get("contact_edges", []):
            verdict = judgment.get("pass")
            if verdict is False:
                failure_tags.append("missing_contact")
            elif verdict is None:
                vlm_no_decision = True
                failure_tags.append("no_decision")

        pose = vlm_judgments.get("pose")
        if isinstance(pose, dict):
            verdict = pose.get("pass")
            if verdict is False:
                failure_tags.append("implausible_pose")
            elif verdict is None:
                vlm_no_decision = True
                failure_tags.append("no_decision")

        penetration = vlm_judgments.get("penetration")
        if isinstance(penetration, dict):
            verdict = penetration.get("pass")
            if verdict is False:
                failure_tags.append("severe_penetration")
            elif verdict is None:
                vlm_no_decision = True
                failure_tags.append("no_decision")

    ranked_tags = rank_failure_tags(failure_tags)
    if metrics.get("deterministic", {}).get("failure_tags"):
        status = "fail"
    elif vlm_enabled and any(tag != "no_decision" for tag in ranked_tags):
        status = "fail"
    elif vlm_enabled and vlm_no_decision:
        status = "no_decision"
    else:
        status = "pass"

    return {
        "interaction_name": metrics.get("interaction_name"),
        "scene_id": metrics.get("scene_id"),
        "target_object": metrics.get("target_object"),
        "status": status,
        "failure_tags": ranked_tags,
        "deterministic_pass": metrics.get("deterministic", {}).get("pass"),
        "vlm_enabled": vlm_enabled,
        "contact_pass": metrics.get("contact", {}).get("pass"),
        "penetration_pass": metrics.get("penetration", {}).get("pass"),
        "max_contact_distance_m": metrics.get("contact", {}).get("max_distance_m"),
        "mean_contact_distance_m": metrics.get("contact", {}).get("mean_distance_m"),
        "severe_penetration": metrics.get("penetration", {}).get("severe_penetration"),
        "final_total_loss": metrics.get("optimization", {}).get("final_total_loss"),
    }


def validate_required_inputs(optimizer_root: Path, sig_json_path: Path, input_scene_json_path: Path) -> None:
    required = [
        sig_json_path,
        input_scene_json_path,
        optimizer_root / "alignment_summary.json",
        optimizer_root / "meshes" / "frame_0000_world.ply",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required evaluation input(s): " + ", ".join(missing))


def read_prompt(path: Path, fallback: str) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start: end + 1])
    if not isinstance(payload, dict):
        raise ValueError("VLM response JSON must be an object")
    return payload


def normalize_judgment(payload: dict[str, Any], edge_index: int | None = None) -> dict[str, Any]:
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"pass", "fail", "no_decision"}:
        raise ValueError("VLM JSON decision must be pass, fail, or no_decision")
    passed = {"pass": True, "fail": False, "no_decision": None}[decision]
    result = {
        "decision": decision,
        "pass": passed,
        "reason": str(payload.get("reason", ""))[:1000],
    }
    if edge_index is not None:
        result = {"edge_index": int(edge_index), **result}
    return result


def vlm_request(
    client: Any,
    model: str,
    system_prompt: str,
    task_text: str,
    image_paths: list[Path],
    timeout: float,
    thinking_effort: str,
) -> str:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": task_text,
        }
    ]
    for path in image_paths:
        if path.exists():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
                }
            )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0,
        timeout=timeout,
        extra_body={"reasoning_effort": thinking_effort},
    )
    return response.choices[0].message.content or ""


def judge_with_vlm(
    judge_name: str,
    client: Any,
    model: str,
    system_prompt: str,
    task_text: str,
    image_paths: list[Path],
    args: argparse.Namespace,
    edge_index: int | None = None,
) -> dict[str, Any]:
    for attempt in range(int(args.retries) + 1):
        try:
            suffix = f" edge={edge_index}" if edge_index is not None else ""
            log("vlm", f"start judge={judge_name}{suffix} attempt={attempt + 1}")
            text = vlm_request(
                client,
                args.model,
                system_prompt,
                task_text,
                image_paths,
                args.timeout,
                args.vlm_thinking_effort,
            )
            payload = extract_json_object(text)
            judgment = normalize_judgment(payload, edge_index=edge_index)
            log("vlm", f"done judge={judge_name}{suffix} decision={judgment['decision']}")
            return judgment
        except Exception as error:
            log("warn", f"judge={judge_name} failed attempt={attempt + 1}: {error}")
    return normalize_judgment(
        {
            "decision": "no_decision",
            "reason": f"VLM judge failed after retries: {judge_name}",
        },
        edge_index=edge_index,
    )


def contact_task_text(edge: dict[str, Any], interaction: str) -> str:
    return "\n".join(
        [
            f"Interaction: {interaction}",
            f"Body part: {edge.get('body_part')}",
            f"Target: {edge.get('target')}",
        ]
    )


def pose_task_text(pose: dict[str, Any], interaction: str) -> str:
    return "\n".join(
        [
            f"Interaction: {interaction}",
        ]
    )


def penetration_task_text(penetration: dict[str, Any], interaction: str) -> str:
    return "\n".join(
        [
            f"Interaction: {interaction}",
        ]
    )


def resolve_standard_paths(args: argparse.Namespace) -> dict[str, Path]:
    defaults = default_paths(args.interaction_name)
    return {
        "optimizer_root": resolve_path(args.optimizer_output_root, defaults["optimizer_output_root"]),
        "sig_json": resolve_path(args.sig_json, defaults["sig_json"]),
        "input_scene_json": resolve_path(args.input_scene_json, defaults["input_scene_json"]),
        "outdir": resolve_path(args.outdir, defaults["outdir"]),
    }


def load_metrics_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    paths = resolve_standard_paths(args)
    optimizer_root = paths["optimizer_root"]
    sig_json_path = paths["sig_json"]
    input_scene_json_path = paths["input_scene_json"]
    validate_required_inputs(optimizer_root, sig_json_path, input_scene_json_path)
    log("load", f"interaction={args.interaction_name} outdir={paths['outdir']}")
    log("load", f"SIG: {sig_json_path}")
    log("load", f"input scene: {input_scene_json_path}")
    log("load", f"optimizer summary: {optimizer_root / 'alignment_summary.json'}")
    sig_payload = load_json(sig_json_path)
    input_scene = load_json(input_scene_json_path)
    alignment_summary = load_json(optimizer_root / "alignment_summary.json")
    metrics = collect_metrics(
        sig_payload=sig_payload,
        alignment_summary=alignment_summary,
        optimizer_root=optimizer_root,
        sig_json_path=sig_json_path,
        input_scene_json_path=input_scene_json_path,
        args=args,
    )
    return metrics, input_scene, optimizer_root, paths["outdir"]


def slim_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    contact_edges = []
    for edge in metrics.get("contact", {}).get("edges", []):
        contact_edges.append(
            {
                "edge_index": edge.get("index"),
                "body_part": edge.get("moving_part_name"),
                "target": edge.get("fixed_entity_name") or edge.get("fixed_part_name"),
                "distance_m": edge.get("nocontact_distance_m"),
                "threshold_m": edge.get("threshold_m"),
                "pass": edge.get("pass"),
            }
        )
    penetration = metrics.get("penetration", {})
    scene_points = penetration.get("scene_points", {})
    optimization = metrics.get("optimization", {})
    return {
        "interaction_name": metrics.get("interaction_name"),
        "scene_id": metrics.get("scene_id"),
        "interaction": metrics.get("interaction"),
        "target_object": metrics.get("target_object"),
        "contact": {
            "pass": metrics.get("contact", {}).get("pass"),
            "edge_count": metrics.get("contact", {}).get("edge_count"),
            "mean_distance_m": metrics.get("contact", {}).get("mean_distance_m"),
            "max_distance_m": metrics.get("contact", {}).get("max_distance_m"),
            "edges": contact_edges,
        },
        "penetration": {
            "pass": penetration.get("pass"),
            "severe_penetration": penetration.get("severe_penetration"),
            "num_inside_points": scene_points.get("num_inside_points"),
            "min_sdf_m": scene_points.get("min_sdf_m"),
            "thresholds": penetration.get("thresholds"),
        },
        "deterministic": metrics.get("deterministic"),
        "final_total_loss": optimization.get("final_total_loss"),
    }


def contact_edge_slug(edge: dict[str, Any]) -> str:
    part_name = edge.get("moving_part_name") or edge.get("body_part") or "part"
    target = edge.get("fixed_entity_name") or edge.get("fixed_part_name") or edge.get("target") or "target"
    return f"edge_{int(edge.get('index', edge.get('edge_index', 0))):02d}_{slugify(part_name)}_to_{slugify(target)}"


def collect_view_sequence(pattern: str, count: int) -> list[Path]:
    return [Path(pattern.format(index=index)) for index in range(count)]


def require_rendered_evidence(metrics: dict[str, Any], outdir: Path) -> dict[str, Any]:
    evidence_dir = outdir / "evidence"
    render_summary_path = evidence_dir / "render_summary.json"
    if not render_summary_path.exists():
        raise FileNotFoundError(
            "Rendered views were not found. Run "
            f"01_render_views.py --interaction_name {metrics.get('interaction_name')} first."
        )
    render_summary = load_json(render_summary_path)
    global_count = int(render_summary.get("global_view_count", 0))
    global_images = collect_view_sequence(str(evidence_dir / "global" / "view_{index:02d}.png"), global_count)
    missing = [path for path in global_images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing rendered global view: {missing[0]}")

    summary_edges = {
        int(edge.get("edge_index", index)): edge
        for index, edge in enumerate(render_summary.get("contact_edges", []))
    }
    contact_edges = []
    for edge in metrics.get("contact", {}).get("edges", []):
        edge_index = int(edge["index"])
        summary_edge = summary_edges.get(edge_index, {})
        part_name = edge.get("moving_part_name")
        target = edge.get("fixed_entity_name") or edge.get("fixed_part_name")
        view_count = int(summary_edge.get("view_count", 0))
        edge_dir = evidence_dir / "contact" / contact_edge_slug(edge)
        views = []
        for view_index in range(view_count):
            context = edge_dir / f"view_{view_index:02d}_context.png"
            local = edge_dir / f"view_{view_index:02d}_local_contact.png"
            if not context.exists() or not local.exists():
                raise FileNotFoundError(
                    "Missing contact rendered view. Run "
                    f"01_render_views.py --interaction_name {metrics.get('interaction_name')} first. "
                    f"Missing: {context if not context.exists() else local}"
                )
            views.append({"images": {"context": str(context), "local_contact": str(local)}})
        contact_edges.append(
            {
                "edge_index": edge_index,
                "body_part": part_name,
                "target": target,
                "view_count": view_count,
                "views": views,
            }
        )

    return {
        "render_summary": render_summary,
        "contact_edges": contact_edges,
        "pose": {
            "interaction": metrics.get("interaction"),
            "view_count": global_count,
            "images": {"views": [str(path) for path in global_images]},
        },
        "penetration": {
            "view_count": global_count,
            "images": {"views": [str(path) for path in global_images]},
        },
    }


def slim_judgment(judgment: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": judgment.get("decision"),
        "pass": judgment.get("pass"),
        "reason": judgment.get("reason"),
    }


def run_ollama_vlm_judgments(
    rendered_evidence: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.skip_vlm:
        log("vlm", "skipped because --skip-vlm was set")
        return {
            "enabled": False,
            "provider": "ollama",
            "model": None,
            "contact_edges": [],
            "pose": {},
            "penetration": {},
        }

    try:
        from openai import OpenAI
    except Exception as error:
        raise RuntimeError(f"Could not import OpenAI client for VLM judging: {error}") from error

    client = OpenAI(base_url=args.ollama_host, api_key=args.ollama_api_key, max_retries=0)
    log("vlm", f"enabled provider=ollama model={args.model} retries={args.retries}")
    contact_prompt = read_prompt(CONTACT_PROMPT, "Judge contact and return strict JSON.")
    pose_prompt = read_prompt(POSE_PROMPT, "Judge pose and return strict JSON.")
    penetration_prompt = read_prompt(PENETRATION_PROMPT, "Judge penetration and return strict JSON.")

    pose = rendered_evidence.get("pose", {})
    interaction = str(pose.get("interaction", ""))
    contact_judgments = []
    for edge in rendered_evidence.get("contact_edges", []):
        edge_index = int(edge.get("edge_index", len(contact_judgments)))
        image_paths: list[Path] = []
        for view in edge.get("views", []):
            images = view.get("images", {})
            for key in ["context", "local_contact"]:
                if images.get(key):
                    image_paths.append(Path(images[key]))
        if not image_paths:
            judgment = {
                "decision": "no_decision",
                "pass": None,
                "reason": "No selected contact images were available for VLM judging.",
            }
        else:
            judgment = judge_with_vlm(
                "contact",
                client,
                args.model,
                contact_prompt,
                contact_task_text(edge, interaction),
                image_paths,
                args,
                edge_index=edge_index,
            )
        contact_judgments.append(
            {
                "edge_index": edge_index,
                "body_part": edge.get("body_part"),
                "target": edge.get("target"),
                **slim_judgment(judgment),
            }
        )

    pose_images = [Path(path) for path in pose.get("images", {}).get("views", []) if path]
    pose_judgment = judge_with_vlm(
        "pose",
        client,
        args.model,
        pose_prompt,
        pose_task_text(pose, interaction),
        pose_images,
        args,
    )
    penetration = rendered_evidence.get("penetration", {})
    penetration_images = [Path(path) for path in penetration.get("images", {}).get("views", []) if path]
    penetration_judgment = judge_with_vlm(
        "penetration",
        client,
        args.model,
        penetration_prompt,
        penetration_task_text(penetration, interaction),
        penetration_images,
        args,
    )
    return {
        "enabled": True,
        "provider": "ollama",
        "model": args.model,
        "contact_edges": contact_judgments,
        "pose": slim_judgment(pose_judgment),
        "penetration": slim_judgment(penetration_judgment),
    }


def build_human_verification_summary(
    metrics: dict[str, Any],
    vlm_judgments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = build_verification_summary(metrics, vlm_judgments)
    if vlm_judgments:
        summary["vlm_provider"] = vlm_judgments.get("provider")
        summary["vlm_model"] = vlm_judgments.get("model")
        summary["vlm_decisions"] = {
            "contact_edges": [
                {
                    "edge_index": edge.get("edge_index"),
                    "body_part": edge.get("body_part"),
                    "target": edge.get("target"),
                    "decision": edge.get("decision"),
                    "reason": edge.get("reason"),
                }
                for edge in vlm_judgments.get("contact_edges", [])
            ],
            "pose": slim_judgment(vlm_judgments.get("pose", {})),
            "penetration": slim_judgment(vlm_judgments.get("penetration", {})),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a rendered static scene with deterministic metrics and Ollama/Qwen VLM.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--optimizer-output-root", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--input-scene-json", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--contact-threshold-m", type=float, default=0.05)
    parser.add_argument("--severe-penetration-min-sdf-m", type=float, default=-0.03)
    parser.add_argument("--severe-penetration-inside-points", type=int, default=1000)
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--ollama-api-key", default="ollama")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--vlm-thinking-effort", default="medium", choices=["low", "medium", "high"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, _input_scene, _optimizer_root, outdir = load_metrics_context(args)
    outdir.mkdir(parents=True, exist_ok=True)
    rendered_evidence = require_rendered_evidence(metrics, outdir)
    save_json(outdir / "metrics.json", slim_metrics(metrics))
    log(
        "metrics",
        f"contact_pass={metrics['contact']['pass']} penetration_pass={metrics['penetration']['pass']} "
        f"edges={metrics['contact']['edge_count']} deterministic_pass={metrics['deterministic']['pass']}",
    )
    vlm_judgments = run_ollama_vlm_judgments(rendered_evidence, args)
    save_json(outdir / "vlm_judgments.json", vlm_judgments)
    summary = build_human_verification_summary(metrics, vlm_judgments)
    save_json(outdir / "verification_summary.json", summary)
    log("summary", f"status={summary['status']} failure_tags={summary['failure_tags']}")
    log("summary", f"wrote metrics: {outdir / 'metrics.json'}")
    log("summary", f"wrote VLM judgments: {outdir / 'vlm_judgments.json'}")
    log("summary", f"wrote verification summary: {outdir / 'verification_summary.json'}")


def run_cli() -> None:
    try:
        main()
    except Exception as error:
        raise SystemExit(f"[error] {error}") from None


if __name__ == "__main__":
    run_cli()
