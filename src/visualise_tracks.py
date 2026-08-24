import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise tracks on video")
    parser.add_argument("--input", type=Path, required=True, help="Path to source video")
    parser.add_argument("--tracks", type=Path, required=True, help="Path to tracks.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path to output annotated video")
    return parser.parse_args()

# Uses track ID as a seed to randomly generate a consistent hue.
# Colour converted from HSV space to RGB
def track_colour(track_id):
    rng = random.Random(track_id)
    hue = rng.random()
    saturation = 0.75
    value = 0.90
    
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    
    return (
        round(r * 255),
        round(g * 255),
        round(b * 255),
    )

def main():
    args = parse_args()

if __name__ == "__main__":
    main()