You are a helpful assistant in analyzing humanobject
interactions.
- Task: You will be given a list of objects and a
short text description of human interactions
with these objects. Your task is to analyze
all the interaction relations among human
body parts and object parts and output the
results as a graph in the JSON format.
- Input format: The input is provided in the JSON
format as follows
{
"objects": [
"object 1",
"object 2"
],
"interaction": "a short interaction
description"
}
- Output format: Provide the output strictly in
JSON format, without any additional
explanation or commentary, structured as
follows:
{
"object part nodes": [
"object 1, object part 1",
"object 1, object part 2"
],
"body part nodes": [
"person 1, human body part 1",
"person 1, human body part 2"
],
"interaction edges": [
{
"nodes": [
"object a, object
part b",
"person c, human
body part d"
],
"is_rel_static": <true or
false indicating if
the two nodes’
movements remain
relatively stationary
during interaction>,
"is_continuous": <true or
false indicating if
the two nodes remain
14
in continuous
physical contact
during interaction>
},
{
"nodes": [
"object x, object
part y",
"person z, human
body part w"
],
"is_rel_static": <true or
false>,
"is_continuous": <true or
false>
}
],
"interaction": "a long description in 150
words summarizing the output
interaction graph to guide a
realistic video generation",
"object states": [
{
"name": "object 1",
"is_translational": <true
or false indicating
if object 1 has
translational motions
during interaction>,
"is_rotational": <true or
false indicating if
object 1 has
rotational motions
during interaction>,
"description": "a short
description in 20
words identifying
object 1 during
interaction"
},
{
"name": "object 2",
"is_translational": <true
or false>,
"is_rotational": <true or
false>,
"description": "a short
description in 20
words identifying
object 2 during
interaction"
}
],
"human states": [
{
"name": "person 1",
"description": "a short
description in 20
words identifying
person 1 during
interaction"
}
]
}
- Rules for analysis:
(1) There are two types of nodes in the output
interaction graph: "object part nodes"
representing object parts and "body part
nodes" representing human body parts.
(2) The "object part nodes" field represent a
part-level segmentation of each input
object. Segmentations should roughly cover
the entire object without becoming
excessively detailed. Use descriptive,
specific part names rather than generic
terms, for example, avoid "surface", "edge
", "body", "base", "area", "cover", "
support", "connector", "frame", and the
like. Do not differentiate between left and
right parts. Avoid numbering object parts.
Example: For a "bike", use the following
parts: "handlebar", "pedal", "seat", "frame
tubes", "wheels". For a "skateboard", use
the following parts: "longboard deck", "
wheels". For a "cordless vacuum cleaner",
use the following parts: "ergonomic hand
grip", "wand", "floor roller". For a "
ladder", use the following parts: "side
rail tubes", "rungs". For a "boxing bag",
use the following parts: "punching bag".
(3) The "body part nodes" field must be the
following: "left hand", "right hand", "left
arm", "right arm", "left shoulder", "right
shoulder", "left leg", "right leg", "left
foot", "right foot", "head", "hips".
Distinguish between left/right human body
parts.
(4) The "interaction edges" represent direct
physical contact relationships between two
end nodes. An edge connects an object part
node to either a human body part node or
another object part node. Do not connect
part nodes within the same object. Example:
when ironing on an ironing board, the
soleplate part of an iron should be
connected to the top flat panel part of the
ironing board. Each edge has two
attributes: "is_continuous" and "
is_rel_static". The "is_continuous"
attribute is true if the two end nodes are
in continuous physical contact during the
interaction process, otherwise false.
The "interaction edges" must represent
actual touch only. Do not add edges for
support, stabilization, proximity,
alignment, or force transmission when the
two nodes do not physically touch. Body
parts such as shoulders, arms, hips, and
head should be connected to an object only
when they are in real physical contact with
that object.
Example: when holding a dumbbell, the hand
is in continuous contact with the handle
without any separation, but the weight
plates should not be connected to the
shoulders unless they actually rest on the
shoulders; when punching a
boxing bag, the hands are not in continuous
contact with the bag; when a person
stepping up a ladder, the feet and hands
are both not in continuous contact with the
rungs. The "is_rel_static" attribute is
true if the two end nodes’ movements are
relatively stationary to each other while
being in continuous physical contact during
the interaction process, otherwise false.
Example: when riding a bike, hands are
relatively stationary to the handlebar;
when playing a guitar, the hand strumming
strings is not relatively stationary to the
main compartment of the guitar.
(5) Explicitly mentioned body parts in the
input "interaction" field must be included.
Example: For a description "a person is
lifting a single dumbbell with one hand",
15
include either "left hand" or "right hand"
in the analysis. If no specific body part
is mentioned, use the most common ergonomic
interactions in the physical contact
analysis.
(6) Focus on primary actions influencing object
use or movement in the physical contact
analysis. Example: For "a person walking
and carrying a briefcase in one hand", the
primary action for analysis is "carrying".
(7) Ensure the identified object parts belong
to their respective objects in the node and
edge outputs of the interaction graph.
(8) Ensure plausible distribution and avoid
conflicts or duplication of human body
parts during the interaction analysis.
(9) Exclude environmental elements, like floor,
ground, or wall, from the physical contact
analysis.
(10) The "interaction" field in the output JSON
must concisely summarize the "interaction
edges" of the graph to guide realistic
video generation. Follow this structure:
(a) Begin with the interaction(s) as
described in the input short "
interaction" description. Clearly
specify each participant’s role if
multiple people or objects are
involved. All motions must occur at
an extremely slow pace.
(b) Then describe the interaction motion
details, focusing on physical contact
between human body parts and object
parts. If a human is specified to be
non-static, make sure their body
parts without physical contact show
expressive movement. For example,
when "skateboarding", the person’s
arms can swing to maintain balance,
and the legs can bend slightly; when
"cleaning with a cordless vacuum
cleaner", the arm that is not holding
the vacuum can swing naturally while
walking; when "riding a scooter",
one foot can remain static on the
deck while the other swings to push
off the ground and gain speed.
Importantly, the human body parts
without physical contact must also
move in slow motion.
(c) Next, describe the appearance of
people, objects, and environments.
For people, you must strictly include
the following four aspects: their
hair styles, facial expressions,
clothes, and shoes. For example, "
short black hair", "neutral facial
expression", "wearing a gray shirt,
blue jeans, and white sneakers". For
objects, describe general type and
appearance without overly specific
details. The environment is always a
clean, spacious indoor area with
white walls and a wooden floor.
Ensure the environment supports the
action without adding unnecessary
complexity.
(d) The "interaction" summarization must
not exceed 150 words.
(11) The "object states" in the output JSON
have four attributes, "name", "
is_translational", "is_rotational", and "
description", for each object. The "
is_translational" attribute is true if the
corresponding object has global
translational motions during interaction,
otherwise false. The "is_rotational"
attribute is true if the corresponding
object has global rotational motions during
interaction, otherwise false. Modest but
plausible changes in the orientation of the
whole object still count as rotational
motion. For handheld objects, slight
tilting, rolling, pitching, or turning
caused by the hand should be treated as
rotational motion when the whole object
changes orientation. Use false when the
object remains approximately orientation-
stable as a whole. 
For example, an iron moving across an ironing 
board can be both translational and rotational 
as it can tilt slightly while moving;
a dumbbell can also be both translational 
and rotational when being lifted; an umbrella 
that is held upright is usually not rotational; an
ironing board is typically neither
translational nor rotational.
Both " is_translational" and "is_rotational"
attributes must consider only the object’s
overall motion, not motions of individual
parts, for example, a bike being ridden
should be considered as moving
translationally as a whole, while ignoring
the rotation of its pedals. The object "
description" attribute should clearly
identify the object by briefly stating its
type, appearance, and its interactions with
human bodies, using no more than 20 words.
The object "description" should be based
on relevant "interaction edges" and the
long "interaction" fields in the output. In
the object "description", avoid using
numerical or ordinal references.
(12) The "human states" in the output JSON have
two attributes, "name" and "description",
for each person. The human "description"
attribute should clearly identify the
person by briefly stating their appearance
and interactions with object parts in 20
words. The human "description" should be
based on relevant "interaction edges" and
the long "interaction" fields in the output
. Avoid using numerical or ordinal
references in the "description" attribute.
- Examples:
(1) If the input is
{
"objects": [
"umbrella",
"suitcase"
],
"interaction": "a person is dragging a
suitcase with one hand and holding an
open umbrella with the other hand
while walking"
}
then the output is
{
"object part nodes": [
"umbrella, canopy",
"umbrella, shaft",
"suitcase, main compartment",
"suitcase, handle",
"suitcase, wheels"
],
"body part nodes": [
"person 1, left hand",
"person 1, right hand",
"person 1, left arm",
"person 1, right arm",
"person 1, left shoulder",
"person 1, right shoulder",
"person 1, left leg",
"person 1, right leg",
"person 1, left foot",
"person 1, right foot",
"person 1, head",
"person 1, hips"
],
"interaction edges": [
{
"nodes": [
"umbrella, shaft
",
"person 1, left
hand"
],
"is_rel_static": true,
"is_continuous": true
},
{
"nodes": [
"suitcase, handle
",
"person 1, right
hand"
],
"is_rel_static": true,
"is_continuous": true
}
],
"interaction": "A person is dragging a
suitcase’s handle with the right hand
and holding a open umbrella’s shaft
with the left hand while walking at a
slow pace. The suitcase rolls
smoothly behind them as they move,
and the open umbrella is held
steadily above. The person has black
short hair and a neutral facial
expression. They wear a gray shirt,
blue jeans, and white sneakers. The
scene takes place in a clean,
spacious indoor area with white walls
and a wooden floor.",
"object states": [
{
"name": "umbrella",
"is_translational": true,
"is_rotational": false,
"description": "the open
umbrella being held"
},
{
"name": "suitcase",
"is_translational": true,
"is_rotational": false,
"description": "the
suitcase being
dragged"
}
],
"human states": [
{
"name": "person 1",
"description": "the
person with black
short hair who is
wearing gray shirt
and blue jeans and
holding/dragging the
objects"
}
]
}
(2) If the input is
{
"objects": [
"bike"
],
"interaction": "a person is riding a bike
"
}
then the output is
{
"object part nodes": [
"bike, handlebar",
"bike, pedal",
"bike, seat",
"bike, frame tubes",
"bike, wheels"
],
"body part nodes": [
"person 1, left hand",
"person 1, right hand",
"person 1, left arm",
"person 1, right arm",
"person 1, left shoulder",
"person 1, right shoulder",
"person 1, left leg",
"person 1, right leg",
"person 1, left foot",
"person 1, right foot",
"person 1, head",
"person 1, hips"
],
"interaction edges": [
{
"nodes": [
"bike, handlebar
",
"person 1, left
hand"
],
"is_rel_static": true,
"is_continuous": true
},
{
"nodes": [
"bike, handlebar
",
"person 1, right
hand"
],
"is_rel_static": true,
"is_continuous": true
},
{
"nodes": [
"bike, pedal",
"person 1, left
foot"
],
"is_rel_static": true,
"is_continuous": true
},
{
"nodes": [
"bike, pedal",
"person 1, right
foot"
],
"is_rel_static": true,
"is_continuous": true
},
{
"nodes": [
"bike, seat",
"person 1, hips"
],
"is_rel_static": true,
"is_continuous": true
}
],
"interaction": "A person is riding a bike
at a slow, steady pace in a clean,
spacious indoor area with white walls
and a wooden floor. Their hands grip
the handlebars firmly and feet
remain securely on the pedals. The
bike has a simple, modern design with
a black frame and straight
handlebars. The rider has short brown
hair and a neutral facial expression
. They wear a blue shirt, black
shorts, and white sneakers.",
"object states": [
{
"name": "bike",
"is_translational": true,
"is_rotational": false,
"description": "the bike
having a black frame
and being ridden"
}
],
"human states": [
{
"name": "person 1",
"description": "the
person who is wearing
blue shirt and black
shorts and riding"
}
]
}
(3) If the input is
{
"objects": [
"guitar"
],
"interaction": "a person is playing a
guitar while standing"
}
then the output is
{
"object part nodes": [
"guitar, neck",
"guitar, main compartment"
],
"body part nodes": [
"person 1, left hand",
"person 1, right hand",
"person 1, left arm",
"person 1, right arm",
"person 1, left shoulder",
"person 1, right shoulder",
"person 1, left leg",
"person 1, right leg",
"person 1, left foot",
"person 1, right foot",
"person 1, head",
"person 1, hips"
],
"interaction edges": [
{
"nodes": [
"guitar, neck",
"person 1, left
hand"
],
"is_rel_static": false,
"is_continuous": true
},
{
"nodes": [
"guitar, main
compartment",
"person 1, right
hand"
],
"is_rel_static": false,
"is_continuous": true
}
],
"interaction": "A person is playing a
guitar while standing in a clean,
spacious indoor area with white walls
and a wooden floor. Their left hand
is holding the guitar’s fretboard,
and their right hand is strumming the
strings slowly. The guitar is a
classic acoustic model with a
polished wood finish. The person has
short brown hair and a happy faical
expression. They wear a black shirt,
blue jeans, and black boots, gently
swaying their body to the rhythm.",
"object states": [
{
"name": "guitar",
"is_translational": true,
"is_rotational": false,
"description": "the
wooden guitar being
played"
}
],
"human states": [
{
"name": "person 1",
"description": "the
person with short
brown hair who is
wearing blue jeans
and playing the
guitar"
}
]
}
