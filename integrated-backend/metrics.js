const promClient = require('prom-client');

// Create a Registry to register the metrics
const register = new promClient.Registry();

// Add default metrics (CPU, memory, etc.)
promClient.collectDefaultMetrics({ register });

// Custom metrics
const httpRequestDuration = new promClient.Histogram({
    name: 'http_request_duration_seconds',
    help: 'Duration of HTTP requests in seconds',
    labelNames: ['method', 'route', 'status_code'],
    registers: [register]
});

const httpRequestTotal = new promClient.Counter({
    name: 'http_requests_total',
    help: 'Total number of HTTP requests',
    labelNames: ['method', 'route', 'status_code'],
    registers: [register]
});

const databaseConnectionStatus = new promClient.Gauge({
    name: 'database_connection_status',
    help: 'Database connection status (1 = connected, 0 = disconnected)',
    labelNames: ['database'],
    registers: [register]
});

const queueSize = new promClient.Gauge({
    name: 'queue_size',
    help: 'Number of messages in queue',
    labelNames: ['queue_name'],
    registers: [register]
});

// Middleware to track HTTP metrics
const middleware = () => {
    return (req, res, next) => {
        const start = Date.now();

        res.on('finish', () => {
            const duration = (Date.now() - start) / 1000; // Convert to seconds
            const route = req.route ? req.route.path : req.path;

            httpRequestDuration.observe(
                { method: req.method, route, status_code: res.statusCode },
                duration
            );

            httpRequestTotal.inc({
                method: req.method,
                route,
                status_code: res.statusCode
            });
        });

        next();
    };
};

// Helper functions
const setDatabaseConnection = (database, isConnected) => {
    databaseConnectionStatus.set({ database }, isConnected ? 1 : 0);
};

const setQueueSize = (queueName, size) => {
    queueSize.set({ queue_name: queueName }, size);
};

// Metrics endpoint handler
const metricsHandler = async (req, res) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
};

module.exports = {
    register,
    middleware,
    setDatabaseConnection,
    setQueueSize,
    metricsHandler,
    httpRequestDuration,
    httpRequestTotal,
    databaseConnectionStatus,
    queueSize
};
