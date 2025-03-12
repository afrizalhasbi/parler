#!/bin/bash
# Check if directory argument is provided
rm -rf samples_cs
python tcs.py
dir="samples_cs"
# Remove any trailing slash from dir and append .wav to create output file name.
out="${dir%/}.wav"

# Use find to list .wav files (only in the given directory, not subdirectories),
# then sort them using version sort (-V), which handles numerical parts correctly.
ffmpeg -f concat -safe 0 -i <(
  find "$dir" -maxdepth 1 -type f -name '*.wav' | sort -V | while IFS= read -r file; do
    echo "file '$PWD/$file'"
  done
) -c copy "$out"

echo "Saved audio in samples_cs and aggregated in samples_cs.mp3"
