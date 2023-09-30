const express = require('express');
const app = express();

app.use(express.static('public/dashboard/index.html')); // Serve files from the 'public' directory

app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});
