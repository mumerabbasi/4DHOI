from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import build_prompt, load_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Build the human-frame Gemini prompt.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--scene-image", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.set_defaults(script_dir=script_dir, project_dir=project_dir)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    script_dir: Path = args.script_dir
    project_dir: Path = args.project_dir

    output_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / "output" / args.interaction_name
    )
    sig_json_path = (
        Path(args.sig_json).resolve()
        if args.sig_json
        else project_dir
        / "01_Generate_SIG"
        / "output"
        / args.interaction_name
        / "sig.json"
    )
    if args.scene_image:
        scene_image_path = Path(args.scene_image).resolve()
    else:
        scene_image_path = sig_json_path.parent / "scene_image.png"
    system_prompt_path = (
        Path(args.system_prompt).resolve()
        if args.system_prompt
        else script_dir / "system_prompt_human_inpaint.md"
    )
    prompt_path = output_root / "prompt" / "prompt.md"
    prompt_dir = prompt_path.parent
    prompt_scene_image_path = prompt_dir / "scene_image.png"

    sig_payload = load_json(sig_json_path)
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    prompt = build_prompt(system_prompt, sig_payload["interaction"])

    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    shutil.copy2(scene_image_path, prompt_scene_image_path)

    print(f"Wrote human-frame prompt: {prompt_path}")
    print(f"Wrote scene image: {prompt_scene_image_path}")
    return prompt_path


if __name__ == "__main__":
    main()
