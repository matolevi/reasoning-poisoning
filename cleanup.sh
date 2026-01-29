#!/bin/bash
# Cleanup script - removes all runtime-generated files and folders

echo "🧹 Cleaning up runtime files..."

# Remove vector databases
rm -rf vector_db_active simple_vector_db_clean
echo "  ✓ Removed vector databases"

# Remove experiment snapshots (user edits)
rm -rf experiments_snapshots
echo "  ✓ Removed experiment snapshots"

# Remove logs
rm -rf logs
echo "  ✓ Removed logs"

# Remove old tournament results file
rm -f tournament_results_batch.txt
echo "  ✓ Removed old result files"

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "  ✓ Removed Python cache"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "To start fresh:"
echo "  1. git checkout d4321b0 -- mock_internet/clean/  # Restore data"
echo "  2. python run_pipeline.py --setup                # Setup folders"
echo "  3. Edit experiments_snapshots/ folders           # Add poison"
echo "  4. python run_pipeline.py --phases 00_baseline   # Run experiments"
