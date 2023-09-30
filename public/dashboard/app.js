const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:3000/public/dashboard/index.html');

  const cardData = await page.evaluate(() => {
    const extractedData = [];
    const cards = document.querySelectorAll('.card');

    cards.forEach(card => {
      const title = card.querySelector('.card-title').textContent;
      const identifier = card.querySelector('small').textContent.replace('Identifier: ', '');
      const files = Array.from(card.querySelectorAll('.list-group-item')).map(item => item.textContent);

      extractedData.push({ title, identifier, files });
    });

    return extractedData;
  });

   // Get the full path to the desired file
  const filePath = path.join(__dirname,'..','..', 'models', 'datasets', 'database.json');

  // Write data to the specified file
  fs.writeFileSync(filePath, JSON.stringify(cardData, null, 2));

  await browser.close();
})();
