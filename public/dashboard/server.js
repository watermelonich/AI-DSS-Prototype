// const express = require('express');
// const app = express();
// const port = 5002;

// app.get('/', (req, res) => {
//   res.send('Hello, world!');
// });

// app.listen(port, () => {
//   console.log(`Server is running on http://127.0.0.1:${port}`);
// });

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 5002;

// Create a storage engine that specifies the destination directory
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'storage');
        fs.mkdirSync(uploadDir, { recursive: true });
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    },
});

const upload = multer({ storage: storage });

app.use(express.static(path.join(__dirname, 'public')));

// Serve the HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/upload', upload.array('files'), (req, res) => {
    res.json({ message: 'Files uploaded successfully!' });
});

app.listen(port, () => {
    console.log(`Server is running on http://127.0.0.1:${port}`);
});
