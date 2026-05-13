from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    load_binary_mask,
    load_json,
    read_api_key,
    resize_cover_center_crop,
    resolve_scene_image_path,
    resolve_target_mask_path,
    run_gemini_image_edit,
    save_target_mask_overlay,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run Gemini human inpaint from a saved prompt.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--api-key-file",
        default=str(project_dir / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--model", default="gemini-2.5-flash-image")
    parser.add_argument("--overwrite-inpainted", action="store_true")
    parser.set_defaults(script_dir=script_dir, project_dir=project_dir)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    script_dir: Path = args.script_dir
    project_dir: Path = args.project_dir

    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else project_dir / "01_Generate_SIG" / "input_prompts" / args.interaction_name
    )
    output_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / "output" / args.interaction_name
    )
    selection_json_path = (
        Path(args.selection_json).resolve()
        if args.selection_json
        else project_dir
        / "02_Select_Target_Instance"
        / "output"
        / args.interaction_name
        / "target_selection.json"
    )
    scannet_root = (
        Path(args.scannet_root).resolve()
        if args.scannet_root
        else project_dir.parent / "Scannet++" / "data"
    )
    prompt_path = (
        Path(args.prompt).resolve()
        if args.prompt
        else output_root / "prompt.md"
    )

    from PIL import Image

    input_payload = load_json(input_dir / "input_scene.json")
    selection_payload = load_json(selection_json_path)
    target_mask_path = resolve_target_mask_path(
        selection_json_path,
        selection_payload,
    )
    scene_image_path = resolve_scene_image_path(
        scannet_root,
        input_payload["scene_context"],
    )
    scene_image = Image.open(scene_image_path).convert("RGB")
    target_mask_image = Image.open(target_mask_path).convert("RGB")
    target_mask = load_binary_mask(target_mask_path)

    inpainted_path = output_root / "inpainted_frame.png"
    inpainted_resized_path = output_root / "inpainted_frame_resized.png"
    target_mask_overlay_path = output_root / "target_mask_overlay.png"
    output_root.mkdir(parents=True, exist_ok=True)

    if inpainted_path.exists() and not args.overwrite_inpainted:
        inpainted = Image.open(inpainted_path).convert("RGB")
        print(
            "Skipping Gemini call; found existing inpainted frame: "
            f"{inpainted_path}"
        )
    else:
        if inpainted_path.exists():
            print(f"Overwriting existing inpainted frame: {inpainted_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                "Run 01_build_prompt.py first."
            )
        prompt = prompt_path.read_text(encoding="utf-8")
        api_key = read_api_key(Path(args.api_key_file).resolve())
        inpainted = run_gemini_image_edit(
            api_key=api_key,
            model=args.model,
            prompt=prompt,
            scene_image=scene_image,
            target_mask_image=target_mask_image,
        )
        inpainted.save(inpainted_path)

    inpainted_resized = resize_cover_center_crop(inpainted, scene_image.size)
    inpainted_resized.save(inpainted_resized_path)
    save_target_mask_overlay(
        inpainted_resized,
        target_mask,
        target_mask_overlay_path,
    )

    print(f"Wrote inpainted frame: {inpainted_path}")
    print(f"Wrote resized inpainted frame: {inpainted_resized_path}")
    print(f"Wrote target mask overlay: {target_mask_overlay_path}")
    return inpainted_path


if __name__ == "__main__":
    main()
