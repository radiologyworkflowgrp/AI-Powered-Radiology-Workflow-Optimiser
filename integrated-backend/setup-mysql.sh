#!/bin/bash

# MySQL Setup Script for DICOM Integration

echo "🔧 Setting up MySQL for DICOM Integration..."
echo ""

# Check if MySQL is installed
if ! command -v mysql &> /dev/null; then
    echo "❌ MySQL is not installed!"
    echo ""
    echo "To install MySQL on Ubuntu/Pop!_OS:"
    echo "  sudo apt update"
    echo "  sudo apt install mysql-server"
    echo ""
    exit 1
fi

echo "Creating MySQL user and database..."

# Create user and grant privileges
sudo mysql -e "
CREATE USER IF NOT EXISTS 'appuser'@'localhost' IDENTIFIED BY 'AppUser123!';
CREATE DATABASE IF NOT EXISTS radiology_hospital;
GRANT ALL PRIVILEGES ON radiology_hospital.* TO 'appuser'@'localhost';
FLUSH PRIVILEGES;
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ MySQL user 'appuser' created successfully"
    echo "✅ Database 'radiology_hospital' created"
    echo "✅ Privileges granted"
    echo ""
    echo "📝 Credentials:"
    echo "   User: appuser"
    echo "   Password: AppUser123!"
    echo "   Database: radiology_hospital"
    echo ""
    echo "✅ MySQL is ready for DICOM integration!"
    echo ""
    echo "Next step: Restart your backend server (type 'rs' in nodemon)"
else
    echo "❌ Failed to set up MySQL"
    echo ""
    echo "Try running manually:"
    echo "  sudo mysql"
    echo "  CREATE USER 'appuser'@'localhost' IDENTIFIED BY 'AppUser123!';"
    echo "  GRANT ALL PRIVILEGES ON radiology_hospital.* TO 'appuser'@'localhost';"
    echo "  FLUSH PRIVILEGES;"
fi
