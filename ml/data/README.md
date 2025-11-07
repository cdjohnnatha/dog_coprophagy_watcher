# Dataset Processing Script

This script processes video datasets for machine learning model training, specifically designed for dog behavior analysis (coprophagy detection).

## Usage

Run the script interactively:

```bash
./process_dataset.sh
```

The script will prompt you for:
- **Input directory** (default: `./ellie_dataset`): Source directory containing video files organized by class
- **Output directory** (default: `./processed_dataset`): Where processed videos and metadata will be saved

## Prerequisites

- **FFmpeg**: Required for video processing
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

## Expected Input Structure

The input directory should contain subdirectories for each class:

```
ellie_dataset/
├── poop/          # Videos of defecation
├── coprophagy/    # Videos of coprophagy (eating feces)
└── not_poop/      # Videos of other activities
```

## Supported Video Formats

- MP4 (.mp4, .MP4)
- MOV (.mov, .MOV)
- M4V (.m4v, .M4V)

## Processing Features

### Video Normalization
- **Frame rate**: Converts all videos to 8 FPS
- **Resolution**: Scales to 640x360 (16:9 aspect ratio)
- **Codec**: H.264 with fast encoding preset
- **Audio**: Removed (silent output)

### Filters Applied
- `fps=8`: Normalizes frame rate to 8 FPS
- `scale=640:-2`: Scales to fixed width 640px, proportional height (multiple of 2)
- `pad=640:360`: Adds padding to maintain 16:9 aspect ratio (centered)

### Metadata Extraction
Generates a CSV file (`metadata_out.csv`) containing:
- `filename`: Processed filename with class prefix
- `label`: Class label (poop, coprophagy, not_poop)
- `width`: Video width in pixels
- `height`: Video height in pixels
- `fps`: Frame rate
- `duration_sec`: Duration in seconds
- `orig_path`: Original file path

## Output Structure

```
processed_dataset/
├── poop/              # Processed videos for defecation class
├── coprophagy/        # Processed videos for coprophagy class
├── not_poop/          # Processed videos for other activities
└── metadata_out.csv   # Metadata for all processed videos
```

## Example Usage

```bash
# Make script executable (first time only)
chmod +x process_dataset.sh

# Run the script
./process_dataset.sh

# Follow prompts:
# Input directory (default: ./ellie_dataset): ./my_videos
# Output directory (default: ./processed_dataset): ./processed
```

## Error Handling

The script includes robust error handling:
- Skips missing class directories with warnings
- Continues processing even if individual files fail
- Uses `set -euo pipefail` for strict error checking

## Performance Notes

- Processing time depends on video length and complexity
- Uses FFmpeg with GPU acceleration when available
- Generates one output file per input file
- Maintains class-based organization for ML training
