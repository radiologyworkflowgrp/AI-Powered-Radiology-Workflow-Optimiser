const jwt = require('jsonwebtoken');
const redisClient = require('../redisClient');

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';

/**
 * Authentication middleware
 * Verifies JWT token from cookie or Authorization header
 * Checks session in Redis
 */
const authMiddleware = async (req, res, next) => {
    try {
        // Get token from cookie or Authorization header
        let token = req.cookies?.authToken;

        if (!token && req.headers.authorization) {
            const authHeader = req.headers.authorization;
            if (authHeader.startsWith('Bearer ')) {
                token = authHeader.substring(7);
            }
        }

        if (!token) {
            return res.status(401).json({ message: 'No authentication token provided' });
        }

        // Verify JWT token
        const decoded = jwt.verify(token, JWT_SECRET);

        // Check if session exists in Redis
        const sessionData = await redisClient.get(`session:${decoded.userId}`);

        if (!sessionData) {
            return res.status(401).json({ message: 'Session expired or invalid' });
        }

        // Attach user data to request
        req.user = JSON.parse(sessionData);
        req.user.userId = decoded.userId; // Ensure userId is set

        next();
    } catch (error) {
        if (error.name === 'JsonWebTokenError') {
            return res.status(401).json({ message: 'Invalid token' });
        }
        if (error.name === 'TokenExpiredError') {
            return res.status(401).json({ message: 'Token expired' });
        }

        console.error('Auth middleware error:', error);
        return res.status(500).json({ message: 'Authentication error', error: error.message });
    }
};

module.exports = authMiddleware;
