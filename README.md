# 4DHSI Static Scene Pipeline

This project is now focused on static human-scene interaction.

Pipeline:

0. `05_Estimate_Human_Pose/00_create_smplx_contact_regions.py` (run once)
1. `01_Generate_SIG/01_generate_sig.py`
2. `02_Select_Target_Instance/01_select_target_instance.py` (SAM3 target mask)
3. `03_Generate_Human_Frame/01_build_prompt.py`
4. `03_Generate_Human_Frame/02_generate_human_frame.py`
5. `04_Estimate_Contact/01_build_prompt.py` (prompt + contact crops)
6. `04_Estimate_Contact/02_estimate_contact.py`
7. `05_Estimate_Human_Pose/01_estimate_static_pose.py`
8. `06_Optimize_Static_Scene/01_optimize_static_scene.py`

Example order:

```bash
conda run -n gvhmr python 05_Estimate_Human_Pose/00_create_smplx_contact_regions.py
conda run -n 4dhsi python 01_Generate_SIG/01_generate_sig.py --video_name video_01
conda run -n 4dhsi python 02_Select_Target_Instance/01_select_target_instance.py --video_name video_01
conda run -n 4dhsi python 03_Generate_Human_Frame/01_build_prompt.py --video_name video_01
conda run -n 4dhsi python 03_Generate_Human_Frame/02_generate_human_frame.py --video_name video_01
conda run -n 4dhsi python 04_Estimate_Contact/01_build_prompt.py --video_name video_01
conda run -n 4dhsi python 04_Estimate_Contact/02_estimate_contact.py --video_name video_01
conda run -n gvhmr python 05_Estimate_Human_Pose/01_estimate_static_pose.py --video_name video_01
conda run -n 4dhsi python 06_Optimize_Static_Scene/01_optimize_static_scene.py --video_name video_01
```

The input file for each scene is:

```text
01_Generate_SIG/input_prompts/<video_name>/input_scene.json
```

Gemini image generation reads the API key from:

```text
.secrets/gemini_api_key
```

Keep that file local only. The `.secrets/` directory is ignored by git.

Manual Gemini contact workflow:

```text
04_Estimate_Contact/output/<video_name>/prompt/prompt.md
04_Estimate_Contact/output/<video_name>/prompt/reference_inpainted_crop.png
04_Estimate_Contact/output/<video_name>/prompt/target_scene_crop.png
```

Upload those three files to Gemini online, save the generated contact overlay as:

```text
04_Estimate_Contact/output/<video_name>/contact_overlay.png
```

Then run `04_Estimate_Contact/02_estimate_contact.py` to normalize the overlay and extract contact masks.
