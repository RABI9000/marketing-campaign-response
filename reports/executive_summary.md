# Executive Summary — Predicting Marketing Campaign Response
*Prepared for the Marketing Director · one page*

## The situation
Our last campaign was sent to the entire customer base, but only **~15% responded**. That means most
of the budget was spent contacting people who were never going to buy. We built a model that scores
**every customer's likelihood to respond**, so we can target the campaign instead of broadcasting it.

## Approach
Cleaned **2,237** customer records → engineered **10 behavioural features** (total spend, recency,
engagement, prior-campaign history, income tier) → trained and compared **three models** with
leakage-free pipelines and 5-fold cross-validation → translated the winner into a concrete targeting
and budget plan.

## Key insights (the money findings)
| Finding | Evidence | What it means |
|---|---|---|
| **Past responders are gold** | Prior acceptors respond at **41%** vs **8%** (≈3×) | A high-yield list we already own |
| **Affluence pays** | High-income respond **28%** vs **10%**; high-income + prior accept → **47%** | Premium customers deserve premium offers |
| **Recency wins** | Responders bought **~35 days** ago vs **~52** | Trigger campaigns off recent purchases |
| **Childless households respond more** | **27%** vs **10%** with children | Cheap, always-available targeting filter |
| **Channel reveals intent** | Catalog buyers over-respond; web *visits* don't predict response | Invest in catalog; ignore raw web traffic |

## Best model
**XGBoost** — **ROC-AUC ≈ 0.90**, F1 ≈ 0.62, cross-validated and stable (Logistic Regression and
Random Forest were close but weaker on ranking/balance). It ranks customers well enough that a
model-driven contact list dramatically out-converts random contacting, and its probability output lets
us set the contact threshold to **maximise profit** for any given budget.

## What leadership should do next
1. **Re-target the ~460 prior responders immediately** — highest yield, zero new identification cost.
2. **Concentrate budget on the top 2–3 probability deciles**; stop funding the unresponsive bottom half.
3. **Personalise by segment** — premium offers for high-income responders, recency-triggered sends for
   warm buyers, sustained catalog investment.
4. **Prove it against a random-targeting control** so the lift is measured in dollars, not asserted.

## Expected business impact
Same budget, pointed at customers who respond ~3× more often → **materially more accepted offers per
dollar**, less wasted contact spend, and a **repeatable, measurable targeting engine** instead of an
annual guess. The model doesn't just predict response — it tells marketing **where the next dollar earns
the most.**

---
*Full methodology in `notebook/marketing_campaign_analysis.ipynb`; interactive version in the Streamlit app.*
