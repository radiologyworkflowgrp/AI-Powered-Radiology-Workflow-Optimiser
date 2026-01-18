#!/bin/bash
# ML Integration Verification Script

echo "============================================================"
echo "ML Integration Verification"
echo "============================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
OVERALL_STATUS=0

# 1. Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅${NC} $PYTHON_VERSION"
else
    echo -e "${RED}❌${NC} Python 3 not found"
    OVERALL_STATUS=1
fi
echo ""

# 2. Check Python dependencies
echo "2. Checking Python dependencies..."
if [ -f "prioritization-ml/venv/bin/python3" ]; then
    prioritization-ml/venv/bin/python3 -c "import torch, pika, pandas, numpy, fastapi, uvicorn, joblib, prometheus_client" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅${NC} All Python dependencies installed (in venv)"
        TORCH_VERSION=$(prioritization-ml/venv/bin/python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        DEVICE=$(prioritization-ml/venv/bin/python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null)
        echo "   PyTorch: $TORCH_VERSION (Device: $DEVICE)"
    else
        echo -e "${RED}❌${NC} Missing Python dependencies in venv"
        OVERALL_STATUS=1
    fi
else
    echo -e "${RED}❌${NC} Python venv not found"
    OVERALL_STATUS=1
fi
echo ""

# 3. Check model files
echo "3. Checking ML model files..."
if [ -f "prioritization-ml/models/priority_model.pt" ] && \
   [ -f "prioritization-ml/models/vocab.json" ] && \
   [ -f "prioritization-ml/models/model_config.json" ]; then
    echo -e "${GREEN}✅${NC} Model files present"
    MODEL_SIZE=$(du -h prioritization-ml/models/priority_model.pt | cut -f1)
    echo "   Model size: $MODEL_SIZE"
else
    echo -e "${RED}❌${NC} Model files missing"
    OVERALL_STATUS=1
fi
echo ""

# 4. Test model loading
echo "4. Testing model loading..."
prioritization-ml/venv/bin/python3 -c "import sys; sys.path.insert(0, 'prioritization-ml'); from ml_priority_system_pytorch import load_artifacts; load_artifacts('prioritization-ml/models')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅${NC} Model loads successfully"
else
    echo -e "${RED}❌${NC} Model loading failed"
    OVERALL_STATUS=1
fi
echo ""

# ... RabbitMQ and Redis checks remain same ...

# 8. Verify file paths
echo "8. Verifying integration files..."
FILES=(
    "ml-services/modelRunner.js"
    "test-ml-integration.js"
    "prioritization-ml/ml_priority_system_pytorch.py"
    "prioritization-ml/requirements.txt"
    "ML_INTEGRATION_GUIDE.md"
)

ALL_FILES_PRESENT=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅${NC} $file"
    else
        echo -e "${RED}❌${NC} $file (missing)"
        ALL_FILES_PRESENT=false
        OVERALL_STATUS=1
    fi
done
echo ""

# Summary
echo "============================================================"
echo "Verification Summary"
echo "============================================================"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Start main backend:    npm run dev"
    echo "  2. Start ML services:     npm run dev:ml-models"
    echo "  3. Start priority worker: npm run start:ml-priority"
    echo "  4. Run integration test:  node test-ml-integration.js"
else
    echo -e "${RED}❌ Some checks failed. Please fix the issues above.${NC}"
fi
echo "============================================================"

exit $OVERALL_STATUS
