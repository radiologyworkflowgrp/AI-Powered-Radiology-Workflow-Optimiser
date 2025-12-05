const express = require("express");
const jwt = require("jsonwebtoken");
const Patient = require("../mongoSchemas/Patient");
const Doctor = require("../mongoSchemas/Doctor");
const Admin = require("../mongoSchemas/Admin");
const redisClient = require("../redisClient");
const authMiddleware = require("../middleware/authMiddleware");
const LoginActivity = require("../mongoSchemas/LoginActivity");
const User = require("../mongoSchemas/User");

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

        // Check if user already exists in User collection
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: `${userRole} already exists with this email` });
        }

        let referenceId = null;
        let profileCreated = false;

        // Create profile in specific collection (using MongoDB models for now as per imports)
        // Note: The plan mentions Postgres for Patient/Doctor, but imports currently point to Mongo schemas.
        // We will stick to the existing imports (Mongo) for consistency with current code structure 
        // unless I change imports to Postgres models. 
        // HOWEVER, the task was "keep only login details in mongodb and rest of the data in postgresql".
        // The imports in authRoutes currently are:
        // const Patient = require("../mongoSchemas/Patient");
        // const Doctor = require("../mongoSchemas/Doctor");

        // I need to use the Postgres models if I want to store data in Postgres.
        // But the imports are Mongo. Let's check imports again.
        // Lines 3-5 imply Mongo schemas.
        // I should switch to Postgres models here if the goal is Postgres for data.

        // Let's use the Postgres models available via require('../models') if they exist.
        // I will assume for this step I am modifying the logic to standard separation.

        // Re-importing Postgres models dynamically inside here or changing top imports would be better.
        // For now, let's just implement the User creation part clearly.

        const { Patient, Doctor, Admin } = require('../models');

        if (userRole === 'patient') {
            const newPatient = await Patient.create({
                name,
                email, // Optional reference
                // password removed
            });
            referenceId = newPatient.id;
            profileCreated = true;
        } else if (userRole === 'doctor') {
            const newDoctor = await Doctor.create({
                name,
                email,
                specialty: specialty || "General Medicine"
            });
            referenceId = newDoctor.id;
            profileCreated = true;
        } else if (userRole === 'admin') {
            const newAdmin = await Admin.create({
                name,
                email
            });
            referenceId = newAdmin.id;
            profileCreated = true;
        }

        if (!profileCreated) {
            return res.status(400).json({ message: "Invalid role" });
        }

        // Create User for Auth
        const newUser = new User({
            email,
            password, // User model handles hashing
            role: userRole,
            referenceId: referenceId.toString()
        });

        await newUser.save();

        // Generate JWT token
        const token = jwt.sign(
            {
                userId: newUser._id, // Auth ID
                referenceId: referenceId, // Profile ID
                email: newUser.email,
                role: userRole,
                profileCompleted: false
            },
            JWT_SECRET,
            { expiresIn: JWT_EXPIRES_IN }
        );

        // Store session in Redis
        const sessionData = {
            userId: newUser._id,
            referenceId: referenceId,
            email: newUser.email,
            name: name,
            role: userRole,
            profileCompleted: false,
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
                referenceId: referenceId,
                email: newUser.email,
                name: name,
                role: userRole,
                profileCompleted: false,
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

        // Find user in the User collection (MongoDB)
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(401).json({ message: "Invalid credentials" });
        }

        // Check password
        const isMatch = await user.comparePassword(password);
        if (!isMatch) {
            return res.status(401).json({ message: "Invalid credentials" });
        }

        // Fetch profile details from Postgres based on role and referenceId
        const { Patient, Doctor, Admin } = require('../models');
        let profile = null;
        let profileName = "User";

        try {
            if (user.role === 'patient') {
                profile = await Patient.findByPk(user.referenceId);
            } else if (user.role === 'doctor') {
                profile = await Doctor.findByPk(user.referenceId);
            } else if (user.role === 'admin') {
                profile = await Admin.findByPk(user.referenceId);
            }

            if (profile) {
                profileName = profile.name;
            }
        } catch (dbError) {
            console.warn(`Failed to fetch profile for user ${user.email}:`, dbError);
        }

        // Generate JWT token
        const token = jwt.sign(
            {
                userId: user._id, // Auth ID
                referenceId: user.referenceId, // Profile ID
                email: user.email,
                role: user.role,
                profileCompleted: profile ? (profile.profileCompleted || false) : false
            },
            JWT_SECRET,
            { expiresIn: JWT_EXPIRES_IN }
        );

        // Store session in Redis
        const sessionData = {
            userId: user._id,
            referenceId: user.referenceId,
            email: user.email,
            name: profileName,
            role: user.role,
            profileCompleted: profile ? (profile.profileCompleted || false) : false,
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
            userId: user._id.toString(),
            role: user.role,
            email: user.email,
            ipAddress: req.ip,
            userAgent: req.get("User-Agent") || "",
        });

        res.json({
            message: "Login successful",
            user: {
                id: user._id,
                referenceId: user.referenceId,
                email: user.email,
                name: profileName,
                role: user.role,
                profileCompleted: profile ? (profile.profileCompleted || false) : false,
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
