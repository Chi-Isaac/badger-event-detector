import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise tracks on video")
    parser.add_argument("--input", type=Path, required=True, help="Path to source video")
    parser.add_argument("--tracks", type=Path, required=True, help="Path to tracks.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path to output annotated video")
    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == "__main__":
    main()