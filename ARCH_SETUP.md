# Quick Setup Guide - Arch Linux

## MongoDB Installation (In Progress)

The MongoDB installation is currently downloading. Once complete:

1. **Start MongoDB**:
   ```bash
   sudo systemctl start mongodb
   sudo systemctl enable mongodb  # Auto-start on boot
   ```

2. **Check Status**:
   ```bash
   sudo systemctl status mongodb
   ```

## Alternative: If installation fails

If the AUR installation has issues, you can use MongoDB in Docker:

```bash
# Pull MongoDB image
docker pull mongo:latest

# Run MongoDB container
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -v ~/mongodb_data:/data/db \
  mongo:latest

# Check if running
docker ps | grep mongodb
```

Then update `.env`:
```env
MONGODB_URI=mongodb://localhost:27017/radiology_hospital
```

## Start the Backend Server

Once MongoDB is running:

```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/integrated-backend

# Install dependencies (if not done)
npm install

# Start server
npm start
```

## Testing Credentials Auto-Generation

1. **Login as admin** (use existing admin credentials)
2. **Create a patient without email/password**:
   ```bash
   curl -X POST http://localhost:3002/api/patients \
     -H "Content-Type: application/json" \
     -H "Cookie: authToken=YOUR_ADMIN_TOKEN" \
     -d '{
       "name": "Test Patient",
       "age": 30,
       "gender": "Male"
     }'
   ```

3. **Check response** - should include auto-generated credentials!

## Troubleshooting

### MongoDB Connection Issues
```bash
# Check if MongoDB is running
sudo systemctl status mongodb

# Check logs
sudo journalctl -u mongodb -n 50

# Test connection
mongosh --eval "db.version()"
```

### PostgreSQL Issues
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Test connection
PGPASSWORD='SecurePassword123!' psql -h localhost -U radiology_user -d radiology_hospital -c "SELECT 1"
```

### Server Won't Start
```bash
# Check for port conflicts
sudo lsof -i :3002

# Check Node.js version
node --version  # Should be v14 or higher
```
