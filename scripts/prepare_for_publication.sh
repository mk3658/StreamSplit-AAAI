#!/bin/bash
# prepare_for_publication.sh
# Script to prepare StreamSplit codebase for AAAI 2026 publication

set -e  # Exit on error

echo "================================================"
echo "StreamSplit - AAAI 2026 Publication Preparation"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Clean Python artifacts
echo -e "${YELLOW}[1/7] Cleaning Python artifacts...${NC}"
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.py[cod]" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Python artifacts cleaned${NC}"
echo ""

# Step 2: Clean macOS system files
echo -e "${YELLOW}[2/7] Cleaning macOS system files...${NC}"
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "._*" -delete 2>/dev/null || true
echo -e "${GREEN}✓ macOS files cleaned${NC}"
echo ""

# Step 3: Remove internal development files
echo -e "${YELLOW}[3/7] Removing internal development files...${NC}"
rm -f ITERATION_CHECKLIST.md
rm -f IMPLEMENTATION_CHECKLIST.md
rm -f IMPLEMENTATION_SUMMARY.md
rm -f notes.txt TODO.md
echo -e "${GREEN}✓ Internal files removed${NC}"
echo ""

# Step 4: Check for large files
echo -e "${YELLOW}[4/7] Checking for large files...${NC}"
echo "Files larger than 10MB:"
find . -type f -size +10M ! -path "./venv/*" ! -path "./.git/*" -exec ls -lh {} \; | awk '{print $9, $5}'
echo ""
echo -e "${YELLOW}NOTE: Move large checkpoint files to external hosting${NC}"
echo ""

# Step 5: Verify critical files exist
echo -e "${YELLOW}[5/7] Verifying critical files...${NC}"
critical_files=(
    "README.md"
    "LICENSE"
    "requirements.txt"
    "setup.py"
    "train.py"
    "train_rl.py"
    "configs/streamsplit.yaml"
    "edge/rl_splitting.py"
)

all_exist=true
for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file (MISSING!)${NC}"
        all_exist=false
    fi
done

if [ "$all_exist" = true ]; then
    echo -e "${GREEN}✓ All critical files present${NC}"
else
    echo -e "${RED}⚠ Some critical files are missing!${NC}"
fi
echo ""

# Step 6: Count files to be published
echo -e "${YELLOW}[6/7] Counting files to publish...${NC}"
echo "Python files: $(find . -name "*.py" ! -path "./venv/*" ! -path "./.git/*" | wc -l)"
echo "Config files: $(find configs -name "*.yaml" 2>/dev/null | wc -l)"
echo "Markdown docs: $(find . -name "*.md" ! -path "./venv/*" ! -path "./.git/*" | wc -l)"
echo ""

# Step 7: Generate publication summary
echo -e "${YELLOW}[7/7] Generating publication summary...${NC}"

cat > PUBLICATION_SUMMARY.txt << 'EOF'
StreamSplit - AAAI 2026 Publication Summary
============================================

Files Ready for Publication:
----------------------------

Core Implementation (18 files):
  edge/__init__.py
  edge/audio_processing.py
  edge/contrastive_learning.py
  edge/memory_bank.py
  edge/resource_monitor.py
  edge/rl_splitting.py ← KEY CONTRIBUTION
  server/__init__.py
  server/aggregation.py
  server/hybrid_loss.py
  models/__init__.py
  models/mobilenet_v3.py
  datasets/__init__.py
  datasets/audioset.py
  datasets/edge_audio.py
  utils/__init__.py
  utils/logger.py
  utils/metrics.py
  utils/visualization.py

Training Scripts (4 files):
  train.py
  train_rl.py
  demo.py
  demo_rl.py

Configuration (1 file):
  configs/streamsplit.yaml

Documentation (8 files):
  README.md
  LICENSE
  CITATION.md
  CONTRIBUTING.md
  QUICKSTART.md
  SYSTEM_OVERVIEW.md
  docs/DATASET_IMPLEMENTATION.md
  docs/RL_MODULE_SUMMARY.md

Setup (3 files):
  requirements.txt
  requirements-edge.txt
  setup.py

Testing (1 file):
  tests/test_datasets.py

Files to Host Externally:
-------------------------
  checkpoints/*.pth → Zenodo/Hugging Face
  data/audioset/audio/ → Provide download scripts
  logs/ → Exclude

Next Steps:
-----------
1. Upload checkpoints to Zenodo or Hugging Face
2. Update README with download links
3. Add paper arXiv link (when available)
4. Create GitHub release tag
5. Submit to AAAI 2026!

Repository Size (excluding checkpoints/data):
---------------------------------------------
Estimated: 2-5 MB (easy to clone)

Publication Checklist:
---------------------
[ ] Remove debug print statements
[ ] Update README with final results
[ ] Add BibTeX citation
[ ] Upload model weights externally
[ ] Pin dependency versions
[ ] Test fresh installation
[ ] Create release tag (v1.0-aaai2026)

Contact:
--------
Add your email for questions after publication

EOF

echo -e "${GREEN}✓ Summary generated: PUBLICATION_SUMMARY.txt${NC}"
echo ""

# Final recommendations
echo "================================================"
echo -e "${GREEN}Preparation Complete!${NC}"
echo "================================================"
echo ""
echo "Next Steps:"
echo "1. Review PUBLICATION_SUMMARY.txt"
echo "2. Upload checkpoints to external hosting"
echo "3. Update README with download links"
echo "4. Run: git status (to see what will be committed)"
echo "5. Test installation: pip install -e ."
echo ""
echo -e "${YELLOW}Files to upload externally:${NC}"
if [ -d "checkpoints" ]; then
    echo "  checkpoints/ ($(du -sh checkpoints 2>/dev/null | cut -f1))"
fi
if [ -d "data" ]; then
    echo "  data/ ($(du -sh data 2>/dev/null | cut -f1))"
fi
echo ""
echo -e "${GREEN}Good luck with AAAI 2026! 🚀${NC}"
