const express = require("express");
const jwt = require("jsonwebtoken");
const Patient = require("../models/Patient");
const Doctor = require("../models/Doctor");
const Admin = require("../models/Admin");
const redisClient = require("../redisClient");
const authMiddleware = require("../middleware/authMiddleware");
const LoginActivity = require("../models/LoginActivity");

const router = express.Router();

const JWT_SECRET = process.env.JWT_SECRET || "your-secret-key-change-in-production";
const JWT_EXPIRES_IN = "7d"; // 7 days
const REDIS_SESSION_EXPIRY = 7 * 24 * 60 * 60; // 7 days in seconds

// Helper function to get the correct model based on role
const getModelByRole = (role) => {
    switch (role) {
        case "patient":
            return Patient;
        case "doctor":
            return Doctor;
        case "admin":
            return Admin;
        default:
            return Patient;
    }
};

// Register new user
router.post("/register", async (req, res) => {
    try {
        const { email, password, name, role, specialty } = req.body;

        // Validate input
        if (!email || !password || !name) {
            return res.status(400).json({ message: "Please provide email, password, and name" });
        }

        const userRole = role || "patient";
        const Model = getModelByRole(userRole);

        // Check if user already exists in the specific collection
        const existingUser = await Model.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: `${userRole} already exists with this email` });
        }

        // Create user data based on role
        const userData = {
            email,
            password,
            name,
        };

        // Add role-specific fields
        if (userRole === "doctor") {
            userData.specialty = specialty || "General Medicine";
        }

        // Create new user in the appropriate collection
        const newUser = new Model(userData);
        await newUser.save();

        // Generate JWT token for auto-login after registration
        const token = jwt.sign(
            {
                userId: newUser._id,
                email: newUser.email,
                role: userRole,
                profileCompleted: newUser.profileCompleted || false
            },
            JWT_SECRET,
            { expiresIn: JWT_EXPIRES_IN }
        );

        // Store session in Redis
        const sessionData = {
            userId: newUser._id,
            email: newUser.email,
            name: newUser.name,
            role: userRole,
            profileCompleted: newUser.profileCompleted || false,
        };

        await redisClient.setEx(
            `session:${newUser._id}`,
            REDIS_SESSION_EXPIRY,
            JSON.stringify(sessionData)
        );

        // Set HTTP-only cookie
        res.cookie("authToken", token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === "production",
            sameSite: "strict",
            maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
        });

        res.status(201).json({
            message: `${userRole.charAt(0).toUpperCase() + userRole.slice(1)} registered successfully`,
            user: {
                id: newUser._id,
                email: newUser.email,
                name: newUser.name,
                role: userRole,
                profileCompleted: newUser.profileCompleted || false,
            },
        });
    } catch (error) {
        console.error("Registration error:", error);
        res.status(500).json({ message: "Server error during registration", error: error.message });
    }
});

// Login user
router.post("/login", async (req, res) => {
    try {
        const { email, password, role } = req.body;

        // Validate input
        if (!email || !password) {
            return res.status(400).json({ message: "Please provide email and password" });
        }

        const userRole = role || "patient";
        const Model = getModelByRole(userRole);

        // Find user in the specific collection
        const user = await Model.findOne({ email });
        if (!user) {
            return res.status(401).json({ message: "Invalid credentials" });
        }

        // Check password
        const isMatch = await user.comparePassword(password);
        if (!isMatch) {
            return res.status(401).json({ message: "Invalid credentials" });
        }

        // Generate JWT token
        const token = jwt.sign(
            {
                userId: user._id,
                email: user.email,
                role: userRole,
                profileCompleted: user.profileCompleted || false
            },
            JWT_SECRET,
            { expiresIn: JWT_EXPIRES_IN }
        );

        // Store session in Redis
        const sessionData = {
            userId: user._id,
            email: user.email,
            name: user.name,
            role: userRole,
            profileCompleted: user.profileCompleted || false,
        };

        await redisClient.setEx(
            `session:${user._id}`,
            REDIS_SESSION_EXPIRY,
            JSON.stringify(sessionData)
        );

        // Set HTTP-only cookie
        res.cookie("authToken", token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === "production",
            sameSite: "strict",
            maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
        });

        await LoginActivity.create({
            userId: user._id,
            role: userRole,
            email: user.email,
            ipAddress: req.ip,
            userAgent: req.get("User-Agent") || "",
        });

        res.json({
            message: "Login successful",
            user: {
                id: user._id,
                email: user.email,
                name: user.name,
                role: userRole,
                profileCompleted: user.profileCompleted || false,
            },
        });
    } catch (error) {
        console.error("Login error:", error);
        res.status(500).json({ message: "Server error during login", error: error.message });
    }
});

// Logout user
router.post("/logout", authMiddleware, async (req, res) => {
    try {
        // Delete session from Redis
        await redisClient.del(`session:${req.user.userId}`);

        // Clear cookie
        res.clearCookie("authToken");

        res.json({ message: "Logout successful" });
    } catch (error) {
        console.error("Logout error:", error);
        res.status(500).json({ message: "Server error during logout", error: error.message });
    }
});

// Verify session (for auto-login)
router.get("/verify", authMiddleware, (req, res) => {
    res.json({
        message: "User is authenticated",
        user: req.user,
    });
});

module.exports = router;
