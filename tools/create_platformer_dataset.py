"""
TRMC Dataset Creator: 2D Platformer Axioms
Generates synthetic contrastive pairs for game mechanics, controls, and strategies.
"""

import json
import os
from pathlib import Path

def generate_platformer_axioms():
    axioms = [
        {
            "category": "Physics",
            "text": "Momentum in 2D platformers requires a variable acceleration curve where initial movement is slow but builds to a maximum velocity (V-max), ensuring the player feels weight and inertia.",
            "negative_text": "Physics in 2D platformers should use binary speed, where the character instantly snaps to maximum velocity and stops instantly, removing all sense of weight or momentum."
        },
        {
            "category": "Controls",
            "text": "Coyote Time is a vital mechanic that allows a player to jump for a few frames after leaving a platform edge, compensating for human reaction time and visual lag.",
            "negative_text": "To ensure difficulty, a game must immediately disable the jump action the exact pixel a character's hitbox stops overlapping with a platform collider."
        },
        {
            "category": "Action",
            "text": "Variable Jump Height is implemented by scaling the upward impulse based on how long the jump button is held, allowing for precision in platforming navigation.",
            "negative_text": "Jump height must always be a fixed constant regardless of button pressure to ensure predictable physics at the cost of player agency."
        },
        {
            "category": "Strategy",
            "text": "Sequence Breaking in Metroidvanias is often achieved through high-level techniques like wall-jump resets or damage-boosting, allowing players to bypass intended narrative gates.",
            "negative_text": "Proper game design requires hard invisible walls that prevent a player from entering an area until a specific inventory flag is set, regardless of player skill."
        },
        {
            "category": "Mechanics",
            "text": "Input Buffering allows a player to press an action button shortly before the character is able to perform it, such as jumping just before landing, ensuring the game feels responsive.",
            "negative_text": "An input should be discarded if the character is in a state that cannot perform it, forcing the player to wait for the exact frame of animation completion."
        },
        {
            "category": "Spatial-ARC",
            "text": "In a grid-based environment, an object defined by a set of contiguous blue pixels must maintain its shape (geometry) when moved, unless it collides with a boundary color like red.",
            "negative_text": "Objects in a grid are individual pixels; moving an object should result in a random scatter of pixels across the canvas regardless of initial shape."
        },
        {
            "category": "Strategy-Pathfinding",
            "text": "Optimal pathing in 2D space involves calculating the shortest distance to a goal while accounting for gravity constants and jump arcs to ensure the 'reachability' of a platform.",
            "negative_text": "Pathfinding should ignore verticality; the agent should walk directly toward the X-coordinate of the goal and only jump if it hits a wall for more than 2 seconds."
        },
        {
            "category": "Physics-Inertia",
            "text": "Air Control is the ability to adjust a character's horizontal trajectory while falling; this breaks realistic physics to prioritize player agency and mid-air correction.",
            "negative_text": "Once a character leaves the ground, their horizontal velocity must be locked to the value at the moment of takeoff, mimicking Newtonian physics exactly."
        },
        {
            "category": "ARC-Logic",
            "text": "Symmetry detection is a core reasoning task; if a pattern appears on the left side of a divider, the recursive core must predict its mirrored counterpart on the right.",
            "negative_text": "Pattern recognition is randomized; every quadrant of a grid should be treated as an isolated event with no logical connection to its neighbors."
        }
    ]
    
    output_path = Path("/home/daniel/TRMC/platformer_axioms.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for axiom in axioms:
            f.write(json.dumps(axiom) + "\n")
            
    print(f"✅ TRMC Platformer Dataset Created: {output_path}")
    print(f"📊 Encoded {len(axioms)} fundamental 2D axioms into Contrastive format.")

if __name__ == "__main__":
    generate_platformer_axioms()