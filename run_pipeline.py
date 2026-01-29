"""
Pipeline Orchestrator for Reasoning Poisoning Experiments.

This script orchestrates the full experiment pipeline by iterating through
all phase directories in experiments_snapshots/, running the experiment
on each phase, and collecting results in a centralized logs directory.

The orchestrator ensures:
    1. Clean database state between phases (no data leakage)
    2. Consistent experiment parameters across all phases
    3. Organized output with phase-specific naming
    4. Progress tracking and error handling

Workflow:
    1. Validate that snapshot directories exist
    2. For each phase (00_baseline through 10_paradox):
       a. Log phase start
       b. Reset the vector database
       c. Build new database from phase data
       d. Run all models on all queries
       e. Save results to logs/results_{phase_name}.csv
    3. Generate summary report

Usage:
    # Run all phases
    python run_pipeline.py

    # Run specific phases only
    python run_pipeline.py --phases 00_baseline 05_fake_authority

    # Run setup first (create snapshot folders)
    python run_pipeline.py --setup

Prerequisites:
    - Run setup_snapshots.py first (or use --setup flag)
    - Manually poison data in each phase folder
    - Ollama running with required models
"""

import os
import sys
import time
import shutil
import argparse
from typing import List, Optional
from datetime import datetime

# Import the experiment runner
from experiment import run_experiment, MODELS_TO_TEST

# --- CONFIGURATION ---

# Directory containing all phase snapshots
SNAPSHOTS_DIR: str = "experiments_snapshots"

# Directory for output logs
LOGS_DIR: str = "logs"

# Shared vector database path (reset between phases)
DB_PATH: str = "vector_db_active"

# Queries file
QUERIES_FILE: str = "queries.txt"

# Expected phase directories (in order)
EXPECTED_PHASES: List[str] = [
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


def print_banner(text: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = 70
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def validate_snapshots_exist() -> bool:
    """
    Validate that the snapshots directory exists with phase folders.

    Returns:
        True if valid, False if setup is needed.
    """
    if not os.path.exists(SNAPSHOTS_DIR):
        return False

    existing = [
        d for d in os.listdir(SNAPSHOTS_DIR)
        if os.path.isdir(os.path.join(SNAPSHOTS_DIR, d))
    ]

    if not existing:
        return False

    return True


def get_available_phases() -> List[str]:
    """
    Get list of available phase directories, sorted.

    Returns:
        Sorted list of phase directory names.
    """
    if not os.path.exists(SNAPSHOTS_DIR):
        return []

    phases = [
        d for d in os.listdir(SNAPSHOTS_DIR)
        if os.path.isdir(os.path.join(SNAPSHOTS_DIR, d))
        and not d.startswith(".")
    ]

    return sorted(phases)


def check_phase_has_data(phase_name: str) -> bool:
    """
    Check if a phase directory contains data files.

    Args:
        phase_name: Name of the phase directory.

    Returns:
        True if the phase has .txt files.
    """
    phase_path = os.path.join(SNAPSHOTS_DIR, phase_name)
    
    if not os.path.exists(phase_path):
        return False

    files = [f for f in os.listdir(phase_path) if f.endswith(".txt")]
    return len(files) > 0


def run_setup() -> None:
    """
    Run the snapshot setup script to create folder infrastructure.
    """
    print_banner("RUNNING SNAPSHOT SETUP")

    # Import and run setup
    from setup_snapshots import setup_snapshots
    setup_snapshots()


def reset_database() -> None:
    """
    Delete the shared vector database to ensure clean state.
    """
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"[RESET] Cleared database: {DB_PATH}")
    else:
        print(f"[RESET] No existing database to clear")


def run_single_phase(
    phase_name: str,
    phase_index: int,
    total_phases: int
) -> bool:
    """
    Run the experiment for a single phase.

    Args:
        phase_name: Name of the phase directory.
        phase_index: Current phase number (1-indexed).
        total_phases: Total number of phases to run.

    Returns:
        True if successful, False if error occurred.
    """
    print_banner(f"PHASE {phase_index}/{total_phases}: {phase_name}", "═")

    # Paths
    data_source = os.path.join(SNAPSHOTS_DIR, phase_name)
    output_file = os.path.join(LOGS_DIR, f"results_{phase_name}.csv")

    # Check phase has data
    if not check_phase_has_data(phase_name):
        print(f"[SKIP] No data files found in {phase_name}")
        return False

    # Count files
    files = [f for f in os.listdir(data_source) if f.endswith(".txt")]
    print(f"[INFO] Data source: {data_source}")
    print(f"[INFO] Files in phase: {len(files)}")
    print(f"[INFO] Output: {output_file}")

    # Run experiment
    try:
        start_time = time.time()

        results = run_experiment(
            data_source=data_source,
            queries_file=QUERIES_FILE,
            output_file=output_file,
            db_path=DB_PATH,
            phase_name=phase_name,
            reset_db=True  # Always reset between phases
        )

        duration = time.time() - start_time
        print(f"\n[SUCCESS] Phase '{phase_name}' completed in {duration:.1f}s")
        print(f"          Results: {len(results)} rows saved to {output_file}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Phase '{phase_name}' failed: {e}")
        return False


def generate_summary_report(
    phases_run: List[str],
    phases_failed: List[str],
    start_time: datetime,
    end_time: datetime
) -> None:
    """
    Generate a summary report of the pipeline run.

    Args:
        phases_run: List of successfully completed phases.
        phases_failed: List of failed phases.
        start_time: When the pipeline started.
        end_time: When the pipeline finished.
    """
    summary_file = os.path.join(LOGS_DIR, "pipeline_summary.txt")

    duration = end_time - start_time

    content = f"""
================================================================================
REASONING POISONING EXPERIMENT - PIPELINE SUMMARY
================================================================================

Run Started:  {start_time.strftime("%Y-%m-%d %H:%M:%S")}
Run Finished: {end_time.strftime("%Y-%m-%d %H:%M:%S")}
Total Duration: {duration}

Models Tested:
{chr(10).join(f"  - {m}" for m in MODELS_TO_TEST)}

Phases Completed Successfully ({len(phases_run)}):
{chr(10).join(f"  ✓ {p}" for p in phases_run) if phases_run else "  None"}

Phases Failed ({len(phases_failed)}):
{chr(10).join(f"  ✗ {p}" for p in phases_failed) if phases_failed else "  None"}

Output Files:
{chr(10).join(f"  - logs/results_{p}.csv" for p in phases_run)}

================================================================================
"""

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(content)
    print(f"Summary saved to: {summary_file}")


def run_pipeline(phases_to_run: Optional[List[str]] = None) -> None:
    """
    Main pipeline orchestrator function.

    Runs the experiment across all (or specified) phase directories,
    ensuring clean database state between each phase.

    Args:
        phases_to_run: Optional list of specific phases to run.
                       If None, runs all available phases.
    """
    print_banner("REASONING POISONING - PIPELINE ORCHESTRATOR", "█")

    # Validate setup
    if not validate_snapshots_exist():
        print("\n[ERROR] Snapshot directories not found!")
        print("        Run 'python run_pipeline.py --setup' first, then poison your data.")
        return

    # Get available phases
    available_phases = get_available_phases()
    print(f"\n[INFO] Found {len(available_phases)} phase directories")

    # Determine which phases to run
    if phases_to_run:
        # Filter to only requested phases that exist
        phases = [p for p in phases_to_run if p in available_phases]
        if not phases:
            print(f"[ERROR] None of the requested phases exist: {phases_to_run}")
            return
        print(f"[INFO] Running {len(phases)} requested phases")
    else:
        phases = available_phases
        print(f"[INFO] Running all {len(phases)} phases")

    # Create logs directory
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"[INFO] Results will be saved to: {LOGS_DIR}/")

    # Print phase list
    print("\n[PHASES]")
    for i, phase in enumerate(phases):
        has_data = "✓" if check_phase_has_data(phase) else "✗"
        print(f"  {i+1:2}. [{has_data}] {phase}")

    # Confirm
    print("\n" + "-" * 70)
    print("Starting pipeline in 3 seconds... (Ctrl+C to cancel)")
    print("-" * 70)
    time.sleep(3)

    # Track results
    phases_run: List[str] = []
    phases_failed: List[str] = []
    start_time = datetime.now()

    # Run each phase
    for i, phase_name in enumerate(phases):
        success = run_single_phase(phase_name, i + 1, len(phases))

        if success:
            phases_run.append(phase_name)
        else:
            phases_failed.append(phase_name)

        # Brief pause between phases
        if i < len(phases) - 1:
            print("\n[PAUSE] 2 seconds before next phase...")
            time.sleep(2)

    # Generate summary
    end_time = datetime.now()
    generate_summary_report(phases_run, phases_failed, start_time, end_time)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Orchestrate reasoning poisoning experiments across all phases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run setup to create snapshot folders
  python run_pipeline.py --setup

  # Run all phases
  python run_pipeline.py

  # Run specific phases only
  python run_pipeline.py --phases 00_baseline 05_fake_authority 09_severe_safety

  # List available phases
  python run_pipeline.py --list
        """
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run setup to create snapshot folder infrastructure"
    )

    parser.add_argument(
        "--phases",
        nargs="+",
        help="Specific phases to run (default: all phases)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available phases and exit"
    )

    return parser.parse_args()


# --- MAIN ENTRY POINT ---

if __name__ == "__main__":
    args = parse_args()

    if args.setup:
        run_setup()
    elif args.list:
        print_banner("AVAILABLE PHASES")
        phases = get_available_phases()
        if phases:
            for i, phase in enumerate(phases):
                has_data = "✓" if check_phase_has_data(phase) else "✗"
                print(f"  {i+1:2}. [{has_data}] {phase}")
        else:
            print("  No phases found. Run --setup first.")
    else:
        run_pipeline(phases_to_run=args.phases)
