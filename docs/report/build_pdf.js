// report.html を、HTML のレイアウトのまま A4 の PDF にする（Chromium の印刷機能を使う）。
//   node docs/report/build_pdf.js [入力 HTML] [出力 PDF]
// 既定は同じディレクトリの report.html → report.pdf。Playwright（npm の playwright）と Chromium が必要。
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const src = path.resolve(process.argv[2] || path.join(__dirname, 'report.html'));
  const out = path.resolve(process.argv[3] || src.replace(/\.html?$/, '.pdf'));
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + src, { waitUntil: 'networkidle' });
  await page.emulateMedia({ media: 'print' });
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({
    path: out,
    format: 'A4',
    printBackground: true,
    preferCSSPageSize: true, // 余白は report.html の @page に従う
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate:
      '<div style="width:100%;font-size:8px;color:#898781;text-align:center;">' +
      '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
  });
  await browser.close();
  console.log('wrote', out);
})();
