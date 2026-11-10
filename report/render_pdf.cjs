// Render the report HTML to an A4 PDF with page numbers:
//   node report/render_pdf.cjs report/report.html report/Phase2_Report.pdf
const { chromium } = require("playwright");
const path = require("path");

(async () => {
  const [input, output] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.resolve(input), { waitUntil: "networkidle" });
  await page.pdf({
    path: output,
    format: "A4",
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate: `<div style="width:100%; font-family:Carlito, sans-serif; font-size:8px; color:#52514e; padding:0 18mm;
      display:flex; justify-content:space-between;"><span>Dhruv Sandilya (231CS122) · IoT Mini Project · Phase 2 Report</span>
      <span class="pageNumber"></span></div>`,
    margin: { top: "18mm", bottom: "18mm", left: "0mm", right: "0mm" },
  });
  await browser.close();
  console.log("wrote", output);
})();
