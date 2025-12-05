#!/bin/bash

echo "🚀 AI-Powered Radiology Workflow Optimizer - Installation Script"
echo "================================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}❌ Please do not run this script as root${NC}"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

echo "📋 Checking prerequisites..."
echo ""

# Check Node.js
if command_exists node; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js is installed: $NODE_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  Node.js is not installed${NC}"
    echo "Installing Node.js using nvm..."
    
    # Install nvm
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    
    # Load nvm
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    # Install Node.js 18
    nvm install 18
    nvm use 18
    
    print_status $? "Node.js installation"
fi

# Check npm
if command_exists npm; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✅ npm is installed: $NPM_VERSION${NC}"
else
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Python is installed: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python is not installed${NC}"
    echo "Please install Python 3: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Check MongoDB
if command_exists mongod; then
    echo -e "${GREEN}✅ MongoDB is installed${NC}"
else
    echo -e "${YELLOW}⚠️  MongoDB is not installed${NC}"
    echo "Install with: sudo apt install mongodb-org"
fi

# Check MySQL
if command_exists mysql; then
    echo -e "${GREEN}✅ MySQL is installed${NC}"
else
    echo -e "${YELLOW}⚠️  MySQL is not installed${NC}"
    echo "Install with: sudo apt install mysql-server"
fi

# Check Redis
if command_exists redis-cli; then
    echo -e "${GREEN}✅ Redis is installed${NC}"
else
    echo -e "${YELLOW}⚠️  Redis is not installed${NC}"
    echo "Install with: sudo apt install redis-server"
fi

# Check RabbitMQ
if command_exists rabbitmqctl; then
    echo -e "${GREEN}✅ RabbitMQ is installed${NC}"
else
    echo -e "${YELLOW}⚠️  RabbitMQ is not installed${NC}"
    echo "Install with: sudo apt install rabbitmq-server"
fi

echo ""
echo "📦 Installing project dependencies..."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install backend dependencies
echo "Installing backend dependencies..."
cd "$SCRIPT_DIR/integrated-backend"
npm install
print_status $? "Backend dependencies installation"

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd "$SCRIPT_DIR/RadiologyFrontend"
npm install
print_status $? "Frontend dependencies installation"

# Setup Python virtual environment
echo ""
echo "Setting up Python ML environment..."
cd "$SCRIPT_DIR/integrated-backend/prioritization-ml"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status $? "Python virtual environment creation"
fi

source venv/bin/activate
pip install -r requirements.txt
print_status $? "Python dependencies installation"
deactivate

echo ""
echo "================================================================"
echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "📝 Next steps:"
echo "1. Setup databases:"
echo "   cd integrated-backend"
echo "   ./setup-mysql.sh"
echo "   ./start-rabbitmq.sh"
echo ""
echo "2. Start backend:"
echo "   cd integrated-backend"
echo "   npm run dev"
echo ""
echo "3. Start frontend (in new terminal):"
echo "   cd RadiologyFrontend"
echo "   npm run dev"
echo ""
echo "4. Access the application:"
echo "   Frontend: http://localhost:8080"
echo "   Backend: http://localhost:3002"
echo ""
echo "📚 For detailed instructions, see SETUP_GUIDE.md"
echo "================================================================"
