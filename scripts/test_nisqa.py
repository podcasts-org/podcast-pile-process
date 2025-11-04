#!/usr/bin/env python3
"""
Test script for NISQA quality assessment
Usage: python scripts/test_nisqa.py <audio_file>
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import librosa

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from podcastpile.nisqa import NISQAPredictor


def main():
    parser = argparse.ArgumentParser(description='Test NISQA quality assessment on an audio file')
    parser.add_argument('audio_file', help='Path to audio file (WAV, MP3, etc.)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (default: auto)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for prediction (default: 8)')
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"Loading audio file: {audio_path}")
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Sample rate: {sr} Hz")
    print()

    # Initialize NISQA
    print("Loading NISQA model...")
    import torch
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nisqa = NISQAPredictor(device=device, dim=True)
    print("✓ NISQA model loaded")
    print()

    # Predict
    print("Running NISQA assessment...")
    predictions = nisqa.predict_arrays(
        audio_arrays=[audio],
        sample_rate=16000,
        batch_size=args.batch_size
    )

    # Display results
    print("=" * 50)
    print("NISQA Quality Assessment Results")
    print("=" * 50)
    print(f"MOS (Mean Opinion Score): {predictions['mos'][0]:.3f}")
    print(f"Noisiness:                {predictions['noisiness'][0]:.3f}")
    print(f"Discontinuity:            {predictions['discontinuity'][0]:.3f}")
    print(f"Coloration:               {predictions['coloration'][0]:.3f}")
    print(f"Loudness:                 {predictions['loudness'][0]:.3f}")
    print("=" * 50)
    print()

    # Quality interpretation
    mos = predictions['mos'][0]
    if mos >= 4.0:
        quality = "High Quality"
    elif mos >= 3.0:
        quality = "Medium Quality"
    else:
        quality = "Low Quality"

    print(f"Overall Quality: {quality} (MOS: {mos:.3f})")
    print()
    print("Note: MOS scale is 1-5, where 5 is excellent and 1 is bad")


if __name__ == "__main__":
    main()
