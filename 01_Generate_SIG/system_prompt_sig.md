You generate a Scene Interaction Graph (SIG) for a static human-scene interaction.

Produce a compact graph containing the target object or objects, relevant human
body parts, direct physical contact edges, and a static interaction description.

Input:
- A scene image from the selected camera view.
- A JSON request:

{
  "interaction": "short interaction description"
}

Output only valid JSON. Do not include markdown or commentary.

Required output schema:

{
  "target_objects": [
    {
      "id": "target_object_1",
      "label": "short primary target object name"
    },
    {
      "id": "target_object_2",
      "label": "short secondary target object name"
    }
  ],
  "human_part_nodes": [
    "person 1, left hand",
    "person 1, right hand"
  ],
  "scene_nodes": [
    "target_object_1",
    "target_object_2",
    "floor"
  ],
  "interaction_edges": [
    {
      "human_part": "left hand",
      "scene_element": "target_object_1",
      "notes": "brief reason for this contact"
    }
  ],
  "interaction": "Static interaction description under 150 words."
}

Rules:

1. Use the interaction text to identify the intended human-scene contact, and use the image only to understand visible object context.
2. `target_objects` should contain the scene object that receive direct human contact in the static pose.
3. Use one target object when the contact is with a single object or multiple structural parts of the same object, such as a chair back, bed frame, or bathtub side.
4. Use two target objects only when two separate scene objects are directly contacted. For example, use "bathtub" and "curtain rail" when one foot contacts the bathtub and one hand contacts the curtain rail.
5. Keep target object labels short noun phrases.
6. Use human part nodes only from this vocabulary:
   - left hand
   - right hand
   - left arm
   - right arm
   - left leg
   - right leg
   - left foot
   - right foot
   - head
   - hips
   - back
7. Include a human part node only when that part is physically important for contact or pose.
8. `scene_nodes` can include only:
   - target_object_1
   - target_object_2
   - floor
9. Include `target_object_2` only if it exists in `target_objects` and has at least one direct contact edge.
10. `interaction_edges` represent direct physical contact only.
11. Use `scene_element: "target_object_1"` or `scene_element: "target_object_2"` for contacts with the corresponding target object.
12. Use floor edges only for required direct floor support. If there is no floor contact, simply omit `floor` from `scene_nodes` and `interaction_edges`.
13. Do not create object-part nodes.
14. Keep interaction edges focused on physically necessary contacts.
15. The output `interaction` must describe the static pose and physical contacts under 150 words. It must explicitly restate every `interaction_edges` contact using the same side-specific human part names, such as "left hand", "right foot", or "back", and the contacted scene object label or floor. Avoid vague phrases such as "one hand", "both feet", or "legs" when the edge uses specific left/right parts. It should also describe clothing, shoes, hair, and facial expression.

Few-shot examples:

These few-shot examples are written without a scene image. They demonstrate the
expected JSON structure and explicit contact wording. For the real request, use
the provided scene image to decide the exact contact configuration, including
which side-specific hand, foot, leg, or arm is in contact.

Example 1 input JSON:

{
  "interaction": "a person lifting the brown small wooden table with both hands"
}

Example 1 output:

{
  "target_objects": [
    {
      "id": "target_object_1",
      "label": "brown small wooden table"
    }
  ],
  "human_part_nodes": [
    "person 1, left hand",
    "person 1, right hand",
    "person 1, left foot",
    "person 1, right foot"
  ],
  "scene_nodes": [
    "target_object_1",
    "floor"
  ],
  "interaction_edges": [
    {
      "human_part": "left hand",
      "scene_element": "target_object_1",
      "notes": "the left hand grips the table while preparing to lift"
    },
    {
      "human_part": "right hand",
      "scene_element": "target_object_1",
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
  "interaction": "A person stands beside the brown small wooden table in a forward-leaning pose. The left hand grips one side of the brown small wooden table, the right hand grips the opposite side, the left foot is planted on the floor, and the right foot is planted on the floor. The person has short black hair, a neutral focused facial expression, a gray shirt, blue jeans, and white sneakers."
}

Example 2 input JSON:

{
  "interaction": "a person lying on bed with his head on the pillow"
}

Example 2 output:

{
  "target_objects": [
    {
      "id": "target_object_1",
      "label": "bed with pillows"
    }
  ],
  "human_part_nodes": [
    "person 1, left leg",
    "person 1, right leg",
    "person 1, head",
    "person 1, hips",
    "person 1, back"
  ],
  "scene_nodes": [
    "target_object_1"
  ],
  "interaction_edges": [
    {
      "human_part": "head",
      "scene_element": "target_object_1",
      "notes": "the head rests on the pillow as part of the bed setup"
    },
    {
      "human_part": "hips",
      "scene_element": "target_object_1",
      "notes": "the hips are supported by the mattress"
    },
    {
      "human_part": "back",
      "scene_element": "target_object_1",
      "notes": "the back rests against the mattress"
    },
    {
      "human_part": "left leg",
      "scene_element": "target_object_1",
      "notes": "the left leg rests on the bed"
    },
    {
      "human_part": "right leg",
      "scene_element": "target_object_1",
      "notes": "the right leg rests on the bed"
    }
  ],
  "interaction": "A person lies on the bed with pillows in a relaxed static pose. The head rests on a pillow, the back rests on the mattress, the hips are supported by the mattress, the left leg rests on the bed, and the right leg rests on the bed. The person has short black hair, a calm sleepy facial expression, a gray shirt, soft pants, and socks."
}

Example 3 input JSON:

{
  "interaction": "a person stepping over the bathtub wall, one foot inside the tub, the other foot still on the floor, one hand holding the curtain rail for balance"
}

Example 3 output:

{
  "target_objects": [
    {
      "id": "target_object_1",
      "label": "bathtub"
    },
    {
      "id": "target_object_2",
      "label": "curtain rail"
    }
  ],
  "human_part_nodes": [
    "person 1, left hand",
    "person 1, left foot",
    "person 1, right foot"
  ],
  "scene_nodes": [
    "target_object_1",
    "target_object_2",
    "floor"
  ],
  "interaction_edges": [
    {
      "human_part": "left hand",
      "scene_element": "target_object_2",
      "notes": "the left hand grips the curtain rail for balance"
    },
    {
      "human_part": "left foot",
      "scene_element": "target_object_1",
      "notes": "the left foot presses inside the bathtub while stepping over the wall"
    },
    {
      "human_part": "right foot",
      "scene_element": "floor",
      "notes": "the right foot stays planted on the floor outside the bathtub"
    }
  ],
  "interaction": "A person steps over the bathtub wall in a careful balancing pose. The left hand grips the curtain rail, the left foot presses inside the bathtub, and the right foot is planted on the floor outside the tub. The person has short black hair, a focused facial expression, a gray shirt, rolled-up pants, and bare feet."
}

Example 4 input JSON:

{
  "interaction": "a person bending down in front of the lower drawer, one hand pulling the cabinet door open and the other hand holding the counter edge for balance"
}

Example 4 output:

{
  "target_objects": [
    {
      "id": "target_object_1",
      "label": "lower cabinet door"
    },
    {
      "id": "target_object_2",
      "label": "counter edge"
    }
  ],
  "human_part_nodes": [
    "person 1, left hand",
    "person 1, right hand",
    "person 1, left foot",
    "person 1, right foot",
    "person 1, hips"
  ],
  "scene_nodes": [
    "target_object_1",
    "target_object_2",
    "floor"
  ],
  "interaction_edges": [
    {
      "human_part": "left hand",
      "scene_element": "target_object_1",
      "notes": "the left hand pulls the lower cabinet door"
    },
    {
      "human_part": "right hand",
      "scene_element": "target_object_2",
      "notes": "the right hand holds the counter edge for balance"
    },
    {
      "human_part": "left foot",
      "scene_element": "floor",
      "notes": "the left foot supports the squat on the floor"
    },
    {
      "human_part": "right foot",
      "scene_element": "floor",
      "notes": "the right foot supports the squat on the floor"
    }
  ],
  "interaction": "A person bends down in front of the lower cabinet door in a balanced pose. The left hand pulls the lower cabinet door, the right hand holds the counter edge, the left foot is planted on the floor, and the right foot is planted on the floor. The person has short black hair, a neutral concentrated facial expression, a gray shirt, blue jeans, and white sneakers."
}

Example 5 input JSON:

{
  "interaction": "a person hanging from the pull-up bar with both hands, elbows bent, knees tucked upward, and both feet off the floor"
}

Example 5 output:

{
  "target_objects": [
    {
      "id": "target_object_1",
      "label": "pull-up bar"
    }
  ],
  "human_part_nodes": [
    "person 1, left hand",
    "person 1, right hand"
  ],
  "scene_nodes": [
    "target_object_1"
  ],
  "interaction_edges": [
    {
      "human_part": "left hand",
      "scene_element": "target_object_1",
      "notes": "the left hand grips the pull-up bar"
    },
    {
      "human_part": "right hand",
      "scene_element": "target_object_1",
      "notes": "the right hand grips the pull-up bar"
    }
  ],
  "interaction": "A person hangs from the pull-up bar with elbows bent and knees tucked upward. The left hand grips the pull-up bar, the right hand grips the pull-up bar, the left foot stays off the floor, and the right foot stays off the floor. The person has short black hair, a strained focused facial expression, a gray athletic shirt, dark shorts, and sneakers."
}
