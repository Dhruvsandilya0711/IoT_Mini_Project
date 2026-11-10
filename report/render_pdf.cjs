// Render the report HTML to an A4 PDF:
//   node report/render_pdf.cjs report/report.html report/Mini_Project_Report.pdf
//
// The cover is printed without the running header and footer, and the contents page gets the page number of
// every heading, looked up in a first render with pdftotext. The cover and the body are then joined with
// pdfunite. Both tools come with poppler-utils; without them the PDF is still written, with the header and
// footer on every page and no page numbers in the contents.
const { chromium } = require("playwright");
const { execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const SIDE = "padding:0 20mm; box-sizing:border-box; width:100%; font-family:Carlito, Calibri, sans-serif; " +
  "font-size:7.5pt; color:#5A6575; display:flex; justify-content:space-between; -webkit-print-color-adjust:exact;";
const HEADER = `<div style="${SIDE}"><div style="width:100%; display:flex; justify-content:space-between;
  border-bottom:0.5pt solid #D5DCE4; padding-bottom:3pt;"><span>Addressing Class Imbalance in CICIoT2023-Based
  Intrusion Detection</span><span>Mini Project Report</span></div></div>`;
const FOOTER = `<div style="${SIDE}"><span>Dhruv Sandilya &middot; 231CS122 &middot; NITK Surathkal</span>
  <span class="pageNumber"></span></div>`;
const PRINT = { format: "A4", printBackground: true, preferCSSPageSize: true };
const WITH_HF = { ...PRINT, displayHeaderFooter: true, headerTemplate: HEADER, footerTemplate: FOOTER };

function available(tool) {
  try {
    execFileSync(tool, ["-v"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

const norm = (s) => s.normalize("NFKC").replace(/\s+/g, " ").trim();

async function fillContents(page, tmpPdf) {
  // A placeholder in every page-number slot keeps the layout of pass 1 and tells contents lines from headings.
  await page.$$eval(".toc li .pg", (els) => els.forEach((e) => { e.textContent = "99"; }));
  await page.pdf({ ...WITH_HF, path: tmpPdf });
  const text = execFileSync("pdftotext", ["-layout", tmpPdf, "-"], { encoding: "utf8", maxBuffer: 1 << 26 });
  const pages = text.split("\f").map((p) => p.split("\n").map(norm));
  const tocPage = pages.findIndex((lines) => lines.includes("Contents"));
  const targets = await page.$$eval(".toc li", (els) => els.map((e) => e.dataset.find));
  const numbers = targets.map((find) => {
    const want = norm(find).slice(0, 40);
    for (let i = tocPage + 1; i < pages.length; i++) {
      if (pages[i].some((line) => line.startsWith(want) && !line.endsWith(" 99"))) return String(i + 1);
    }
    return "";
  });
  const missing = targets.filter((_, i) => !numbers[i]);
  if (missing.length) console.warn("no page found for:", missing.join(" | "));
  await page.$$eval(".toc li .pg", (els, nums) => els.forEach((e, i) => { e.textContent = nums[i]; }), numbers);
}

(async () => {
  const [input, output] = process.argv.slice(2);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "report-"));
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.resolve(input), { waitUntil: "networkidle" });

  if (available("pdftotext")) await fillContents(page, path.join(tmp, "pass1.pdf"));
  if (available("pdfunite")) {
    const cover = path.join(tmp, "cover.pdf");
    const body = path.join(tmp, "body.pdf");
    await page.pdf({ ...PRINT, path: cover, pageRanges: "1" });
    await page.pdf({ ...WITH_HF, path: body, pageRanges: "2-9999" });
    execFileSync("pdfunite", [cover, body, output]);
  } else {
    await page.pdf({ ...WITH_HF, path: output });
  }
  await browser.close();
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log("wrote", output);
})();
