# Project Start Instructions

## 1. Start System Services
First, ensure all database services are running:

```bash
sudo systemctl start mongodb postgresql redis rabbitmq
```

**Check status:**
```bash
sudo systemctl status mongodb postgresql redis rabbitmq
```
*All services should clearly show `Active: active (running)`.*

## 2. Start Backend Server
Run the Node.js backend (runs on port 3002):

```bash
cd integrated-backend
npm start
```

## 3. Start Frontend Application
In a **new terminal**, start the React frontend (runs on port 8080):

```bash
cd RadiologyFrontend
npm run dev
```

## Access the Application
- **Frontend**: http://localhost:8080 (or the URL shown in terminal)
- **Backend API**: http://localhost:3002

## Troubleshooting
If `npm start` fails with connection errors:

1. **RabbitMQ Error**: Make sure you have the fix in `.env` (`RABBITMQ_URL=amqp://127.0.0.1:5672`).
2. **MongoDB Error**: Ensure `sudo systemctl start mongodb` was successful.
3. **Database Reset**: If you need to reset the database, delete the PostgreSQL tables or drop the Mongo database.
