You generate a Scene Interaction Graph (SIG) for a static human-scene interaction.

Produce a compact, consistent graph containing the target object, human body
parts, physical interaction edges, and a detailed static interaction description.

Input:
- A scene image from the selected camera view.
- A JSON request:

{
  "interaction": "short interaction description"
}

Output only valid JSON. Do not include markdown or commentary.

Required output schema:

{
  "target_object": {
    "label": "short target object name"
  },
  "human_part_nodes": [
    "person 1, left hand",
    "person 1, right hand"
  ],
  "scene_nodes": [
    "target_object",
    "floor"
  ],
  "interaction_edges": [
    {
      "human_part": "left hand",
      "scene_element": "target_object",
      "notes": "brief reason for this contact"
    }
  ],
  "interaction": "Detailed static interaction description under 150 words."
}

Rules:

1. Use the scene image and interaction text together. The interaction text tells you the intended action; the image tells you the visible scene layout and object context.
2. Choose exactly one target object.
3. `target_object.label` must be a short noun phrase naming the target object.
4. When the interaction requires contact with multiple closely associated scene objects that function together as one support or interaction target, name the target object as a short composite noun phrase. For example, if the body rests on a bed and the head rests on a pillow, use "bed with pillows" rather than only "bed". Keep the target as one object node; do not create separate pillow, cushion, blanket, or object-part nodes.
5. Use human part nodes only from this vocabulary:
   - left hand
   - right hand
   - left arm
   - right arm
   - left shoulder
   - right shoulder
   - left leg
   - right leg
   - left foot
   - right foot
   - head
   - hips
6. Include a human part node when that part is physically important for contact or pose.
7. `scene_nodes` can include only:
   - target_object
   - floor
8. `interaction_edges` represent direct physical contact only.
9. Use `scene_element: "target_object"` for contacts with the selected target object.
10. Use `scene_element: "floor"` only when the human part should touch the floor in the static pose.
11. Do not force feet to touch the floor. If the person is lying, sitting with feet off the ground, hanging, jumping, climbing, or otherwise unsupported by the floor, omit floor contact.
12. Do not create object part nodes. Do not create 3D part nodes.
13. Keep interaction edges focused on physically necessary contacts in the static scene.
14. Every listed interaction edge is an active static contact by definition.
15. The output `interaction` must be a detailed static description under 150 words. It should describe the human pose, target object, physical contacts, clothing, shoes, hair, and facial expression. It should not describe video motion.
16. The scene is a clean, spacious indoor area with white walls and a wooden floor unless the input image clearly shows otherwise.

Few-shot examples:

The real request includes the scene image plus the JSON request. The examples below
show only the JSON part to keep the prompt compact.

Example 1 input JSON:

{
  "interaction": "a person lifting the brown small wooden table with both hands"
}

Example 1 output:

{
  "target_object": {
    "label": "brown small wooden table"
  },
  "human_part_nodes": [
    "person 1, left hand",
    "person 1, right hand",
    "person 1, left foot",
    "person 1, right foot"
  ],
  "scene_nodes": [
    "target_object",
    "floor"
  ],
  "interaction_edges": [
    {
      "human_part": "left hand",
      "scene_element": "target_object",
      "notes": "the left hand grips the table while preparing to lift"
    },
    {
      "human_part": "right hand",
      "scene_element": "target_object",
      "notes": "the right hand grips the opposite side of the table"
    },
    {
      "human_part": "left foot",
      "scene_element": "floor",
      "notes": "the left foot supports the standing pose on the floor"
    },
    {
      "human_part": "right foot",
      "scene_element": "floor",
      "notes": "the right foot supports the standing pose on the floor"
    }
  ],
  "interaction": "A person stands beside the brown small wooden table and prepares to lift it with both hands. The left and right hands firmly grasp opposite sides of the table while the table remains fixed in its original position. Both feet are planted on the wooden floor in a balanced stance, with slightly bent knees and a forward-leaning torso. The person has short black hair, a neutral focused facial expression, a gray shirt, blue jeans, and white sneakers. The scene remains a clean spacious indoor room with white walls and a wooden floor."
}

Example 2 input JSON:

{
  "interaction": "a person lifting the white trash bin with both hands"
}

Example 2 output:

{
  "target_object": {
    "label": "white trash bin"
  },
  "human_part_nodes": [
    "person 1, left hand",
    "person 1, right hand",
    "person 1, left foot",
    "person 1, right foot"
  ],
  "scene_nodes": [
    "target_object",
    "floor"
  ],
  "interaction_edges": [
    {
      "human_part": "left hand",
      "scene_element": "target_object",
      "notes": "the left hand holds one side of the bin"
    },
    {
      "human_part": "right hand",
      "scene_element": "target_object",
      "notes": "the right hand holds the other side of the bin"
    },
    {
      "human_part": "left foot",
      "scene_element": "floor",
      "notes": "the left foot anchors the stance"
    },
    {
      "human_part": "right foot",
      "scene_element": "floor",
      "notes": "the right foot anchors the stance"
    }
  ],
  "interaction": "A person stands close to the white trash bin and prepares to lift it using both hands. The left and right hands grasp the sides or rim of the bin while the bin stays in its original scene position. Both feet remain on the wooden floor in a stable stance, and the torso bends slightly toward the bin. The person has short black hair, a neutral focused facial expression, a gray shirt, blue jeans, and white sneakers. The surrounding room stays clean and spacious with white walls and a wooden floor."
}

Example 3 input JSON:

{
  "interaction": "a person lying on bed with his head on the pillow"
}

Example 3 output:

{
  "target_object": {
    "label": "bed with pillows"
  },
  "human_part_nodes": [
    "person 1, left leg",
    "person 1, right leg",
    "person 1, head",
    "person 1, hips"
  ],
  "scene_nodes": [
    "target_object"
  ],
  "interaction_edges": [
    {
      "human_part": "head",
      "scene_element": "target_object",
      "notes": "the head rests on the pillow as part of the bed setup"
    },
    {
      "human_part": "hips",
      "scene_element": "target_object",
      "notes": "the hips are supported by the mattress"
    },
    {
      "human_part": "left leg",
      "scene_element": "target_object",
      "notes": "the left leg rests on the bed rather than the floor"
    },
    {
      "human_part": "right leg",
      "scene_element": "target_object",
      "notes": "the right leg rests on the bed rather than the floor"
    }
  ],
  "interaction": "A person lies on the bed with the hips and legs supported by the mattress while the head rests on a pillow. The bed and pillows function together as the single target object for this static sleeping pose, so there is no separate pillow node. Both legs remain on the bed, and no foot-floor contact is needed. The person has short black hair, a relaxed sleepy facial expression, a gray shirt, comfortable pants, and socks. The room remains a clean indoor bedroom with white walls and a wooden floor."
}
