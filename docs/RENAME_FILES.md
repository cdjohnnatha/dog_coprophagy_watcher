# File Renaming Script Documentation

## Overview

The `rename_files.py` script is a utility tool that renames all files in a specified folder with a consistent prefix and zero-padded numbering scheme.

## Usage

```bash
python scripts/rename_files.py <folder_path> <file_name_prefix>
```

### Arguments

- `folder_path`: Path to the folder containing files to rename (required)
- `file_name_prefix`: Prefix to use for renamed files (required)

### Examples

```bash
# Rename all files in a dataset folder
python scripts/rename_files.py /path/to/dataset dog_behavior

# Rename images for training
python scripts/rename_files.py images/train training_sample
```

## Behavior

### File Naming Convention

Files are renamed using the format: `{prefix}_{number}{extension}`

- **Prefix**: The user-specified prefix
- **Number**: Zero-padded sequential number starting from 001
- **Extension**: Original file extension is preserved

### Example Output

Given files: `photo1.jpg`, `data.csv`, `image.png`

Command: `python rename_files.py ./folder sample`

Result:
- `photo1.jpg` → `sample_001.jpg`
- `data.csv` → `sample_002.csv`
- `image.png` → `sample_003.png`

### Processing Details

1. **File Discovery**: Scans the target folder for all files (ignores subdirectories)
2. **Sorting**: Files are sorted alphabetically for consistent numbering
3. **Padding**: Automatically determines appropriate zero-padding (minimum 3 digits, scales for larger file counts)
4. **Extension Preservation**: Original file extensions are maintained
5. **Error Handling**: Gracefully handles permission errors and missing files

### Error Handling

The script includes robust error handling for:
- Non-existent directories
- Invalid paths
- Permission issues during renaming
- Empty directories

## Dependencies

- Python 3.x
- Standard library modules only (no external dependencies)

## Location

- **File**: `scripts/rename_files.py`
- **Executable**: Yes (chmod +x to make directly executable)

## Use Cases

- **Dataset Preparation**: Standardizing file names for machine learning datasets
- **Batch Processing**: Renaming large collections of files consistently
- **Organization**: Creating ordered file sequences for processing pipelines
- **Backup Management**: Creating timestamped or sequenced backup files
