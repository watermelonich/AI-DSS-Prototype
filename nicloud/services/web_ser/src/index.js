// services/web_ser/src/index.js
const express = require('express');
const mongoose = require('mongoose');
const userRoutes = require('./routes/userRoutes');
const logger = require('./utils/logger');

const app = express();
const PORT = process.env.PORT || 3000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/web_ser';

app.use(express.json());

// Connect to MongoDB
mongoose.connect(MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
}).then(() => {
    logger.info('Connected to MongoDB');
}).catch(error => {
    logger.error(`Error connecting to MongoDB: ${error}`);
});

// Set up routes
app.use('/api', userRoutes);

app.listen(PORT, () => {
    logger.info(`Server running on port ${PORT}`);
});
