const express = require('express');
const router = express.Router();
const User = require('../mongoSchemas/User');
const authMiddleware = require('../middleware/authMiddleware');

// GET /api/users - Get users by email (for admin to view credentials)
router.get('/', async (req, res) => {
    try {
        const { email } = req.query;

        if (!email) {
            return res.status(400).json({ message: "Email query parameter is required" });
        }

        // Find user by email
        // In a real app, this should be restricted to admins only
        const user = await User.findOne({ email });

        if (!user) {
            return res.json([]);
        }

        res.json([user]);
    } catch (error) {
        console.error("Error fetching user:", error);
        res.status(500).json({ message: "Server error" });
    }
});

module.exports = router;
