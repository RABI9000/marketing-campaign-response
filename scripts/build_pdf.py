"""
Build the one-page summary PDF (the assignment's "short summary, max 1 page").

It covers Approach, Key Insights, Model Performance, and Business
Recommendations, with a clickable link to the live app and the repo. The full
analysis lives in the notebook. Rendered with headless Chromium so the links
stay clickable.

    python scripts/build_pdf.py

If the live app URL changes, edit APP_URL below and re-run.
"""

import os

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Links (clickable in the PDF) ----------------------------------------- #
APP_URL = "https://marketing-campaign-response-ajrjqsucchbawv2ykmcklt.streamlit.app"
GITHUB_URL = "https://github.com/RABI9000/marketing-campaign-response"
NOTEBOOK_URL = GITHUB_URL + "/blob/main/notebook/marketing_campaign_analysis.ipynb"

OUT_PDF = os.path.join(ROOT, "reports", "Marketing_Campaign_Summary.pdf")

HTML = r"""
<!doctype html>
<html><head><meta charset="utf-8"><style>
  @page { size: A4; margin: 11mm 12mm; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color: #2b3440; font-size: 9.4pt; line-height: 1.4; margin: 0; }
  h1,h2,h3 { color: #1F3A5F; line-height: 1.15; margin: 0; }
  p { margin: 4px 0; }
  a { color: #2E6F9E; text-decoration: none; }
  b { color: #1F3A5F; }
  .accent { color: #E8743B; }

  .head { background: linear-gradient(120deg,#1F3A5F 0%,#2E6F9E 58%,#3FA7A0 100%);
          color:#fff; border-radius:11px; padding:14px 18px; }
  .head h1 { color:#fff; font-size:17pt; letter-spacing:-.3px; }
  .head .sub { color:#e6edf4; font-size:9.6pt; margin-top:2px; }
  .btns { margin-top:9px; }
  .appline { margin-top:9px; font-size:9.3pt; color:#fff; }
  .appline b { color:#fff; }
  .appline a { color:#fff; text-decoration:underline; word-break:break-all; }

  .kpis { display:flex; gap:7px; margin:9px 0; }
  .kpi { flex:1; border:1px solid #e9edf2; border-left:4px solid #2E6F9E; border-radius:8px;
         padding:6px 9px; }
  .kpi .v { font-size:13pt; font-weight:800; color:#1F3A5F; }
  .kpi .l { font-size:7.4pt; color:#6b7785; font-weight:600; text-transform:uppercase; letter-spacing:.3px; }

  .oneline { background:#fbfcfd; border:1px solid #e9edf2; border-radius:8px; padding:7px 11px; margin-bottom:9px; }

  .cols { display:flex; gap:13px; }
  .col { flex:1; }
  h3 { font-size:10.4pt; border-left:4px solid #E8743B; padding-left:8px; margin:9px 0 5px 0; }
  ul { margin:3px 0; padding-left:16px; } li { margin:2px 0; }

  table { width:100%; border-collapse:collapse; margin:4px 0; font-size:8.7pt; }
  th,td { border:1px solid #e3e8ee; padding:3px 6px; text-align:left; }
  th { background:#1F3A5F; color:#fff; font-weight:600; }
  tr.win td { background:#eaf6ec; font-weight:700; }

  .note { font-size:8.7pt; color:#445; margin-top:4px; }
  .foot { margin-top:11px; border-top:1px solid #e3e8ee; padding-top:7px; font-size:8.4pt; color:#6b7785; }
</style></head><body>

<div class="head">
  <h1>Marketing Campaign Response &mdash; Summary</h1>
  <div class="sub">Predicting who responds to a campaign, and turning it into a targeting and budget plan.</div>
  <div class="appline">&#9658;&nbsp; <b>Live interactive app:</b> <a href="__APP_URL__">__APP_URL__</a></div>
</div>

<div class="kpis">
  <div class="kpi"><div class="v">2,237</div><div class="l">Customers</div></div>
  <div class="kpi"><div class="v">14.9%</div><div class="l">Baseline response</div></div>
  <div class="kpi"><div class="v">XGBoost</div><div class="l">Best model</div></div>
  <div class="kpi"><div class="v">0.90</div><div class="l">ROC-AUC</div></div>
  <div class="kpi"><div class="v">~3&times;</div><div class="l">Warm-list lift</div></div>
</div>

<div class="oneline"><b>Bottom line.</b> A retailer's last campaign reached everyone, but only ~15%
responded, so most of the spend was wasted. The model scores each customer's likelihood to respond
(ROC-AUC &asymp; 0.90). Aimed at the right people, the same budget earns far more.</div>

<div class="cols">
  <div class="col">
    <h3>Approach</h3>
    <p>Cleaned 2,237 customer records (filled 24 missing incomes, removed a few typo rows, tidied messy
    labels), then built 10 behaviour features: spending, recency, engagement, past-campaign history, income
    tiers. Trained and compared three models inside leak-free scikit-learn pipelines with 5-fold
    cross-validation. Because only ~15% respond, models are judged on ranking quality (ROC-AUC) and recall,
    not plain accuracy, and each is told to weight the rare "yes" cases.</p>

    <h3>Key insights</h3>
    <ul>
      <li><b>Past responders are gold:</b> people who accepted a past campaign respond at <b>41%</b> vs
      <b>8%</b> (~3&times;).</li>
      <li><b>Higher income, higher response:</b> <b>28%</b> vs <b>10%</b>; high income + a past acceptance
      reaches <b>47%</b>.</li>
      <li><b>Recent buyers are warm:</b> responders last bought ~35 days ago vs ~52 for everyone else.</li>
      <li><b>Households without kids respond more:</b> <b>27%</b> vs <b>10%</b> with children.</li>
      <li><b>Catalog buyers over-respond</b> (4.2 vs 2.4 purchases); website <i>visits</i> don't predict
      response at all.</li>
    </ul>
  </div>

  <div class="col">
    <h3>Model performance</h3>
    <table>
      <tr><th>Model</th><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th><th>AUC</th></tr>
      <tr class="win"><td>XGBoost &#9733;</td><td>0.89</td><td>0.63</td><td>0.61</td><td>0.62</td><td>0.90</td></tr>
      <tr><td>Random Forest</td><td>0.89</td><td>0.72</td><td>0.42</td><td>0.53</td><td>0.89</td></tr>
      <tr><td>Logistic Reg.</td><td>0.81</td><td>0.42</td><td>0.73</td><td>0.53</td><td>0.88</td></tr>
    </table>
    <p class="note">XGBoost wins on ranking (ROC-AUC) and F1. Since a missed responder (~$11) costs more
    than a wasted contact (~$3), we lean toward recall and set the contact cutoff where expected profit is
    highest, not at a textbook 0.5.</p>

    <h3>Business recommendations</h3>
    <ul>
      <li><b>Contact first:</b> rank everyone by predicted probability; start with the ~460 prior
      responders (~3&times; the base rate, free to find).</li>
      <li><b>Budget:</b> concentrate on the top 2&ndash;3 probability deciles; stop funding the unresponsive
      bottom half.</li>
      <li><b>Personalise:</b> premium offers for high-income responders, recency-triggered sends for warm
      buyers, sustained catalog investment.</li>
      <li><b>Prove it:</b> track profit per contact against a random-targeting control so the lift is
      provable in dollars.</li>
    </ul>
  </div>
</div>

<div class="foot">
  <b>Code &amp; notebook:</b> <a href="__GITHUB_URL__">github.com/RABI9000/marketing-campaign-response</a>
  &nbsp;&middot;&nbsp; <b>Full analysis:</b> <a href="__NOTEBOOK_URL__">Jupyter notebook</a><br>
  Built with Python, pandas, scikit-learn, XGBoost, and Streamlit.
</div>

</body></html>
"""

html = (HTML.replace("__APP_URL__", APP_URL)
            .replace("__GITHUB_URL__", GITHUB_URL)
            .replace("__NOTEBOOK_URL__", NOTEBOOK_URL))

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.set_content(html, wait_until="networkidle")
    page.pdf(path=OUT_PDF, format="A4", print_background=True,
             margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    browser.close()

print(f"Wrote {OUT_PDF} ({os.path.getsize(OUT_PDF)//1024} KB)")
