#!/usr/bin/env bash
set -euo pipefail

# Interactive script to process video dataset
# Prompts for input and output directories with defaults

# Default directories
DEFAULT_INPUT_DIR="./ellie_dataset"
DEFAULT_OUTPUT_DIR="./processed_dataset"

# Interactive prompts for directories
read -p "Input directory (default: ${DEFAULT_INPUT_DIR}): " IN_DIR
IN_DIR="${IN_DIR:-${DEFAULT_INPUT_DIR}}"

read -p "Output directory (default: ${DEFAULT_OUTPUT_DIR}): " OUT_DIR
OUT_DIR="${OUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

CLASSES=("poop" "coprophagy" "not_poop")
EXTS=("mp4" "mov" "m4v" "MP4" "MOV" "M4V")

mkdir -p "${OUT_DIR}"
# CSV of metadata for processed material
META_CSV="${OUT_DIR}/metadata_out.csv"
echo "filename,label,width,height,fps,duration_sec,orig_path" > "${META_CSV}"

# Function: processes 1 video file
process_one() {
  local in_file="$1"
  local label="$2"
  local rel_dir="$3"

  local out_subdir="${OUT_DIR}/${label}"
  mkdir -p "${out_subdir}"

  # Normalized base name (ideally without spaces/accents; here we just keep the basename)
  local base="$(basename "${in_file}")"
  local name_noext="${base%.*}"
  local out_file="${out_subdir}/${name_noext}.mp4"

  echo "→ Converting: ${in_file}  →  ${out_file}"

  # Filters:
  # - fps=8           → normalizes frame rate
  # - scale=640:-2    → scales to fixed width 640, proportional height (multiple of 2)
  # - pad=640:360     → padding to fixed 16:9 aspect ratio (centered)
  ffmpeg -hide_banner -loglevel error -y \
    -i "${in_file}" \
    -vf "fps=8,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2" \
    -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
    -r 8 -vsync cfr -movflags +faststart \
    -an \
    "${out_file}"

  # Post-processing metadata extraction
  local w h fps dur
  w="$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "${out_file}")" || w=""
  h="$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "${out_file}")" || h=""
  fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "${out_file}" | awk -F/ '{ if ($2==0 || $2=="") print $1; else printf "%.3f", $1/$2 }')" || fps=""
  dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${out_file}" | awk '{ printf "%.3f", $1 }')" || dur=""

  echo "${label}/${name_noext}.mp4,${label},${w},${h},${fps},${dur},${in_file}" >> "${META_CSV}"
}

# Loop through classes and extensions
for label in "${CLASSES[@]}"; do
  in_class_dir="${IN_DIR}/${label}"
  [ -d "${in_class_dir}" ] || { echo "(!) Folder not found: ${in_class_dir} — skipping"; continue; }

  # Find files by extension (basic case-insensitive)
  shopt -s nullglob
  for ext in "${EXTS[@]}"; do
    for f in "${in_class_dir}"/*.${ext}; do
      process_one "${f}" "${label}" "${label}"
    done
  done
done

echo
echo "✅ Completed!"
echo "→ Output: ${OUT_DIR}"
echo "→ Metadata: ${META_CSV}"