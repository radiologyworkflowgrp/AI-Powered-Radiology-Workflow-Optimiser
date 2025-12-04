const redis = require("redis");

// Create Redis client
const redisClient = redis.createClient({
    socket: {
        host: "localhost",
        port: 6379,
    },
});

// Handle Redis connection events
redisClient.on("connect", () => {
    console.log("Redis client connected");
});

redisClient.on("ready", () => {
    console.log("Redis client ready");
});

redisClient.on("error", (err) => {
    console.error("Redis client error:", err);
});

redisClient.on("end", () => {
    console.log("Redis client disconnected");
});

// Connect to Redis
(async () => {
    try {
        await redisClient.connect();
    } catch (error) {
        console.error("Failed to connect to Redis:", error);
    }
})();

module.exports = redisClient;
