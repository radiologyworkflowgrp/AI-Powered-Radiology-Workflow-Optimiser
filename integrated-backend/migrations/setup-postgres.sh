#!/bin/bash

# PostgreSQL Setup Script for Radiology Workflow System

echo "🔧 Setting up PostgreSQL for Radiology Workflow System..."
echo ""

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed!"
    echo ""
    echo "To install PostgreSQL on Ubuntu/Pop!_OS:"
    echo "  sudo apt update"
    echo "  sudo apt install postgresql postgresql-contrib"
    echo ""
    echo "After installation, start PostgreSQL:"
    echo "  sudo systemctl start postgresql"
    echo "  sudo systemctl enable postgresql"
    echo ""
    exit 1
fi

echo "✅ PostgreSQL is installed"
echo ""

# Check if PostgreSQL is running
if ! sudo systemctl is-active --quiet postgresql; then
    echo "⚠️  PostgreSQL is not running. Starting it now..."
    sudo systemctl start postgresql
    sleep 2
fi

echo "✅ PostgreSQL is running"
echo ""

# Create user and database
echo "Creating PostgreSQL user and database..."

sudo -u postgres psql << EOF
-- Create user if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'radiology_user') THEN
        CREATE USER radiology_user WITH PASSWORD 'SecurePassword123!';
    END IF;
END
\$\$;

-- Create database if not exists
SELECT 'CREATE DATABASE radiology_hospital OWNER radiology_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'radiology_hospital')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE radiology_hospital TO radiology_user;

-- Connect to the database and grant schema privileges
\c radiology_hospital
GRANT ALL ON SCHEMA public TO radiology_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO radiology_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO radiology_user;

EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ PostgreSQL user 'radiology_user' created successfully"
    echo "✅ Database 'radiology_hospital' created"
    echo "✅ Privileges granted"
    echo ""
    echo "📝 Credentials:"
    echo "   Host: localhost"
    echo "   Port: 5432"
    echo "   User: radiology_user"
    echo "   Password: SecurePassword123!"
    echo "   Database: radiology_hospital"
    echo ""
    echo "✅ PostgreSQL is ready for the radiology workflow system!"
    echo ""
    echo "Next steps:"
    echo "  1. Update your .env file with PostgreSQL credentials"
    echo "  2. Install dependencies: npm install pg sequelize"
    echo "  3. Run the migration scripts to transfer data"
else
    echo ""
    echo "❌ Failed to setup PostgreSQL"
    echo "Please check the error messages above"
    exit 1
fi
