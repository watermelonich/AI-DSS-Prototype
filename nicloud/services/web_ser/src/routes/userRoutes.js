// services/web_ser/src/routes/userRoutes.js
const express = require('express');
const { createUser, getUsers } = require('../controllers/userController');

const router = express.Router();

router.post('/users', createUser);
router.get('/users', getUsers);

module.exports = router;
