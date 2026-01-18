const jwt = require('jsonwebtoken');
const Patient = require('../mongoSchemas/Patient');
const mongoose = require('mongoose');

/**
 * Middleware to extract user information from JWT token
 * Attaches user info to req.user
 */
async function extractUserInfo(req, res, next) {
    try {
        const token = req.cookies?.authToken || req.headers.authorization?.replace('Bearer ', '');

        if (!token) {
            req.user = null;
            return next();
        }

        const decoded = jwt.verify(token, process.env.JWT_SECRET || "your-secret-key-change-in-production");
        req.user = {
            userId: decoded.userId,
            role: decoded.role,
            email: decoded.email
        };

        console.log('✓ User authenticated:', req.user.role, req.user.userId);
        next();
    } catch (error) {
        console.log('⚠️  Token verification failed:', error.message);
        req.user = null;
        next();
    }
}

/**
 * Get list of patient IDs that the current user can access
 * @param {object} user - User object from req.user
 * @returns {array|null} Array of patient IDs or null (for admin/no restriction)
 */
async function getAllowedPatientIds(user) {
    if (!user) {
        // No user logged in - return null (no restriction for backwards compatibility)
        return null;
    }

    if (user.role === 'admin') {
        // Admin can see all patients
        return null;
    }

    if (user.role === 'doctor') {
        // Doctor can only see assigned patients
        const assignedPatients = await Patient.find({ 'assignedDoctor.id': user.userId });
        const patientIds = assignedPatients.map(p => p._id.toString());
        console.log(`✓ Doctor ${user.userId} has access to ${patientIds.length} patients`);
        return patientIds;
    }

    // Default: no restriction
    return null;
}

/**
 * Middleware to check if user can access a specific patient
 * Requires extractUserInfo to be called first
 * Checks req.params.patientId or req.query.patientId
 */
async function checkPatientAccess(req, res, next) {
    try {
        const patientId = req.params.patientId || req.query.patientId;

        if (!patientId) {
            return next(); // No patient ID to check
        }

        const allowedIds = await getAllowedPatientIds(req.user);

        if (allowedIds === null) {
            // Admin or no restriction
            return next();
        }

        if (allowedIds.includes(patientId)) {
            return next();
        }

        return res.status(403).json({
            message: 'Access denied: You do not have permission to access this patient data',
            error: 'FORBIDDEN'
        });
    } catch (error) {
        console.error('Error checking patient access:', error);
        return res.status(500).json({
            message: 'Error checking access permissions',
            error: error.message
        });
    }
}

/**
 * Middleware to require authentication
 * Returns 401 if no valid token
 */
function requireAuth(req, res, next) {
    if (!req.user) {
        return res.status(401).json({
            message: 'Authentication required',
            error: 'UNAUTHORIZED'
        });
    }
    next();
}

/**
 * Middleware to require admin role
 */
function requireAdmin(req, res, next) {
    if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({
            message: 'Admin access required',
            error: 'FORBIDDEN'
        });
    }
    next();
}

module.exports = {
    extractUserInfo,
    getAllowedPatientIds,
    checkPatientAccess,
    requireAuth,
    requireAdmin
};
