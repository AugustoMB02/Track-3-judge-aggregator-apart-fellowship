#!/bin/bash
# Track 3: Full Dataset Pruning Experiments - Command Reference
# Run this on the Gan branch with .venv environment

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Track 3: Pruning Experiments - Full Dataset${NC}"
echo -e "${BLUE}============================================${NC}\n"

# Step 1: Setup
echo -e "${YELLOW}Step 1: Activate Environment${NC}"
echo "Command: source .venv/bin/activate"
source .venv/bin/activate

# Step 2: Prepare data
echo -e "\n${YELLOW}Step 2: Prepare Data${NC}"
echo "Command: python3 prepare_data_for_track3.py"
python3 prepare_data_for_track3.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Data preparation failed${NC}"
    exit 1
fi

# Step 3: Create output directory
echo -e "\n${YELLOW}Step 3: Create Output Directories${NC}"
mkdir -p "results/track3_full_dataset"
mkdir -p "results/track3_full_dataset/helpsteer2_prune_half"
mkdir -p "results/track3_full_dataset/ultrafeedback_prune_half"
echo "✓ Output directories ready"

# Step 4: Run HelpSteer2 Pruning
echo -e "\n${YELLOW}Step 4: Run HelpSteer2 Pruning Experiment${NC}"
echo "This will take approximately 30-60 minutes..."
python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
    --config config/selection_experiment_prune_half.yaml \
    --data datasets/helpsteer2_track3_full_dataset.pkl \
    --output results/track3_full_dataset/helpsteer2_prune_half \
    --max-iterations 20 \
    --min-judges 15

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}HelpSteer2 experiment encountered issues (check logs)${NC}"
else
    HELPSTEER2_RUN_DIR=$(ls -td results/track3_full_dataset/helpsteer2_prune_half_* 2>/dev/null | head -n 1)
    echo -e "${GREEN}✓ HelpSteer2 experiment complete${NC}"
fi

# Step 5: Run UltraFeedback Pruning
echo -e "\n${YELLOW}Step 5: Run UltraFeedback Pruning Experiment${NC}"
echo "This will take approximately 15-30 minutes..."
python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
    --config config/selection_experiment_prune_half_ultrafeedback.yaml \
    --data datasets/ultrafeedback_track3_full_dataset.pkl \
    --output results/track3_full_dataset/ultrafeedback_prune_half \
    --max-iterations 40 \
    --min-judges 27

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}UltraFeedback experiment encountered issues (check logs)${NC}"
else
    ULTRAFEEDBACK_RUN_DIR=$(ls -td results/track3_full_dataset/ultrafeedback_prune_half_* 2>/dev/null | head -n 1)
    echo -e "${GREEN}✓ UltraFeedback experiment complete${NC}"
fi

# Step 6: Generate Visualizations
echo -e "\n${YELLOW}Step 6: Generate Visualizations${NC}"

if [ -n "${HELPSTEER2_RUN_DIR}" ]; then
    mkdir -p "${HELPSTEER2_RUN_DIR}/visualizations"
    python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
        --run-dir "${HELPSTEER2_RUN_DIR}" \
        --output "${HELPSTEER2_RUN_DIR}/visualizations"
fi

if [ -n "${ULTRAFEEDBACK_RUN_DIR}" ]; then
    mkdir -p "${ULTRAFEEDBACK_RUN_DIR}/visualizations"
    python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
        --run-dir "${ULTRAFEEDBACK_RUN_DIR}" \
        --output "${ULTRAFEEDBACK_RUN_DIR}/visualizations"
fi

echo -e "${GREEN}✓ Visualizations generated${NC}"

# Step 7: Copy to final location
echo -e "\n${YELLOW}Step 7: Copy Results to Final Location${NC}"
mkdir -p "Results track 3 full dataset"
cp -r results/track3_full_dataset/* "Results track 3 full dataset/" 2>/dev/null || true
echo -e "${GREEN}✓ Results copied to 'Results track 3 full dataset/'${NC}"

# Step 8: Summary
echo -e "\n${BLUE}============================================${NC}"
echo -e "${GREEN}✓ All experiments complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "\nResults location:"
echo -e "  ${YELLOW}results/track3_full_dataset/${NC}"
echo -e "  ${YELLOW}Results track 3 full dataset/${NC}"
echo -e "\nGenerated files:"
echo -e "  - Iteration results (JSON)"
echo -e "  - Metrics plots (PNG)"
echo -e "  - Configuration files"
echo -e "\nNext steps:"
echo -e "  1. Review visualizations in 'Results track 3 full dataset/'"
echo -e "  2. Compare HelpSteer2 vs UltraFeedback results"
echo -e "  3. Analyze selected judge sets"
echo -e "  4. Generate summary report"
echo -e "\n${BLUE}============================================${NC}\n"
