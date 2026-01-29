"""
Snapshot Setup Script for Reasoning Poisoning Experiments.

This script creates the folder infrastructure for manual data poisoning experiments.
It creates 11 phase directories, each representing a different attack strategy,
and populates each with a copy of the clean baseline data.

Attack Phases:
    00_baseline       - Control group with unmodified data
    01_single_bot     - Single fake positive review injected
    02_bot_army       - Multiple coordinated fake reviews
    03_negative_smear - Negative misinformation about competitors
    04_recency        - Fake "recent" information to override older data
    05_fake_authority - Fabricated authority sources (fake news, officials)
    06_attribute      - False attribute claims (certifications, awards)
    07_strike_logic   - Logical constraints ("only X is available due to strike")
    08_mild_safety    - Mild safety warnings about competitors
    09_severe_safety  - Severe safety warnings (danger, avoid at all costs)
    10_paradox        - Contradictory information to test reasoning

Usage:
    python setup_snapshots.py

    After running, manually edit files in each phase folder to create
    the "poisoned" dataset for that experimental condition.

Output:
    experiments_snapshots/
    ├── 00_baseline/          (clean copy)
    ├── 01_single_bot/        (clean copy - edit to add single fake review)
    ├── 02_bot_army/          (clean copy - edit to add multiple fakes)
    ├── ...
    └── 10_paradox/           (clean copy - edit to add contradictions)
"""

import os
import shutil
from typing import List

# --- CONFIGURATION ---

# Source directory containing clean scraped data
SOURCE_DATA_DIR: str = "mock_internet/clean"

# Master directory for all experiment snapshots
SNAPSHOTS_DIR: str = "experiments_snapshots"

# Phase directories representing different attack strategies
PHASE_DIRECTORIES: List[str] = [
    "00_baseline",
    "01_single_bot",
    "02_bot_army",
    "03_negative_smear",
    "04_recency",
    "05_fake_authority",
    "06_attribute",
    "07_strike_logic",
    "08_mild_safety",
    "09_severe_safety",
    "10_paradox",
]

# Phase descriptions for documentation
PHASE_DESCRIPTIONS: dict = {
    "00_baseline": "Control group - unmodified clean data",
    "01_single_bot": "Single fake positive review/recommendation",
    "02_bot_army": "Multiple coordinated fake reviews (astroturfing)",
    "03_negative_smear": "Negative misinformation about competitors",
    "04_recency": "Fake 'recent' data to override older legitimate info",
    "05_fake_authority": "Fabricated authority (fake news articles, officials)",
    "06_attribute": "False attribute claims (fake certifications, awards)",
    "07_strike_logic": "Logical constraints (e.g., 'only X available due to strike')",
    "08_mild_safety": "Mild safety warnings about competitors",
    "09_severe_safety": "Severe safety warnings (danger alerts, avoid advisories)",
    "10_paradox": "Contradictory information to test reasoning consistency",
}


def validate_source_directory() -> bool:
    """
    Validate that the source data directory exists and contains files.

    Returns:
        True if valid, False otherwise.
    """
    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"ERROR: Source directory '{SOURCE_DATA_DIR}' not found.")
        print("Please ensure you have scraped data before running setup.")
        return False

    files = [f for f in os.listdir(SOURCE_DATA_DIR) if f.endswith(".txt")]
    if not files:
        print(f"ERROR: No .txt files found in '{SOURCE_DATA_DIR}'.")
        return False

    print(f"[OK] Found {len(files)} source files in '{SOURCE_DATA_DIR}'")
    return True


def create_snapshot_directory(phase_name: str, source_files: List[str]) -> int:
    """
    Create a single phase snapshot directory and copy all source files.

    Args:
        phase_name: Name of the phase directory (e.g., "00_baseline").
        source_files: List of filenames to copy from source directory.

    Returns:
        Number of files copied.
    """
    phase_path = os.path.join(SNAPSHOTS_DIR, phase_name)
    os.makedirs(phase_path, exist_ok=True)

    copied = 0
    for filename in source_files:
        src = os.path.join(SOURCE_DATA_DIR, filename)
        dst = os.path.join(phase_path, filename)
        shutil.copy2(src, dst)
        copied += 1

    return copied


def create_phase_readme(phase_name: str) -> None:
    """
    Create a README file in each phase directory with instructions.

    Args:
        phase_name: Name of the phase directory.
    """
    phase_path = os.path.join(SNAPSHOTS_DIR, phase_name)
    readme_path = os.path.join(phase_path, "PHASE_README.md")

    description = PHASE_DESCRIPTIONS.get(phase_name, "No description available")

    content = f"""# Phase: {phase_name}

## Description
{description}

## Instructions
1. This folder contains a copy of the clean baseline data.
2. **Edit the existing files** or **add new files** to create the "poisoned" dataset.
3. When you run `run_pipeline.py`, this folder's data will be used for this phase.

## Poisoning Guidelines
- Keep file format consistent (SOURCE_URL header, then content)
- For new files, use naming pattern: `poison_{{description}}.txt`
- Document your changes in the CHANGES.md file (create if needed)

## Example Edits
- Add fake reviews praising a specific service
- Insert warnings or safety concerns
- Modify dates to appear more recent
- Add fabricated authority quotes

---
*This file is auto-generated. Edit at will.*
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def setup_snapshots() -> None:
    """
    Main function to create the complete snapshot infrastructure.

    Creates the master directory, all phase subdirectories, copies
    clean data to each, and adds README files with instructions.
    """
    print("=" * 60)
    print("REASONING POISONING - SNAPSHOT SETUP")
    print("=" * 60)

    # Validate source data exists
    if not validate_source_directory():
        return

    # Get list of source files
    source_files = [f for f in os.listdir(SOURCE_DATA_DIR) if f.endswith(".txt")]

    # Create master snapshots directory
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    print(f"\n[CREATE] Master directory: {SNAPSHOTS_DIR}/")

    # Create each phase directory
    print(f"\n[SETUP] Creating {len(PHASE_DIRECTORIES)} phase directories...\n")

    for phase_name in PHASE_DIRECTORIES:
        description = PHASE_DESCRIPTIONS.get(phase_name, "")
        print(f"  {phase_name}/")
        print(f"    └─ {description}")

        # Copy all source files
        copied = create_snapshot_directory(phase_name, source_files)
        print(f"    └─ Copied {copied} files")

        # Create phase README
        create_phase_readme(phase_name)

    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f"""
Next Steps:
1. Navigate to: {SNAPSHOTS_DIR}/
2. Edit files in each phase folder to create poisoned datasets
3. Start with 00_baseline (keep clean) as your control
4. Run: python run_pipeline.py

Phase folders ready for manual editing:
""")

    for phase_name in PHASE_DIRECTORIES:
        print(f"  - {SNAPSHOTS_DIR}/{phase_name}/")


if __name__ == "__main__":
    setup_snapshots()
