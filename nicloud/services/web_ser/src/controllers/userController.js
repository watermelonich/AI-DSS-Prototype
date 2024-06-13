// services/web_ser/src/controllers/userController.js
const User = require('../models/userModel');
const logger = require('../utils/logger');

// Create a new user
exports.createUser = async (req, res) => {
    try {
        const user = new User(req.body);
        await user.save();
        logger.info(`User created: ${user}`);
        res.status(201).json(user);
    } catch (error) {
        logger.error(`Error creating user: ${error}`);
        res.status(400).json({ error: error.message });
    }
};

// Get all users
exports.getUsers = async (req, res) => {
    try {
        const users = await User.find();
        res.status(200).json(users);
    } catch (error) {
        logger.error(`Error fetching users: ${error}`);
        res.status(500).json({ error: error.message });
    }
};
