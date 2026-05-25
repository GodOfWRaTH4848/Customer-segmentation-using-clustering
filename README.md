# Customer Segmentation using KMeans Clustering

## Overview

Retail businesses generate large volumes of transactional data, but identifying meaningful customer groups from that data is a non-trivial challenge.

This project applies K-Means clustering on customer behavioural and transactional features to segment customers into three actionable business groups: **High-Value**, **Growth**, and **At-Risk**.

The pipeline covers the full workflow from raw data to business recommendations, including feature engineering, PCA-based dimensionality reduction, and cluster validation using the elbow method and silhouette scoring.

## Dataset

- **Source**: Synthetic retail dataset (transactions + customer demographics)
- **Tables**: `transactions.csv` (purchases) and `customers.csv` (demographics)
- **Size**: ~3,800 unique customers across ~20,000+ transactions
- **Features used**: Transaction recency, purchase frequency, monetary value (log-transformed), customer tenure, store location, payment method, age group

## Methods

- **Preprocessing**: Null handling, dtype correction, PII removal, dataset merging
- **Feature Engineering**: Revenue calculation (post-discount × quantity), log transformation, age group binning, payment and location grouping, RFM metrics, `since_signup_days`
- **Dimensionality Reduction**: PCA applied to correlated RFM features (3 components, high variance retained)
- **Algorithm**: KMeans clustering on PCA components
- **Evaluation**: Elbow method (inertia) + silhouette score → optimal k=3
- **Visualisation**: Cluster heatmap, violin plots, scatter plot, age group bar charts

## Results

### KMeans Cluster Profiles

| Cluster | Recency (days) | Frequency | Monetary (log) | Tenure (days) | Label |
|---|---|---|---|---|---|
| 0 | 502 | 4.3 | 7.77 | 1246 | At-Risk |
| 1 | 106 | 9.3 | 8.86 | 1327 | High-Value |
| 2 | 107 | 3.0 | 7.25 | 542 | Growth |

### Key Observations

- **Recency and frequency** were the primary differentiators between segments. Monetary value showed less separation, suggesting spend per transaction is relatively consistent across customer types.
- **High-Value** customers have the lowest recency, highest frequency, and longest tenure — deeply engaged, long-term users.
- **Growth** customers are newer (avg. 542 days tenure) but recently active — strong candidates for loyalty incentives to convert them into High-Value customers.
- **At-Risk** customers have been registered the longest but haven't purchased in over 500 days on average — the highest priority for reactivation campaigns.

### Business Recommendations

| Segment | Action |
|---|---|
| **High-Value** | VIP loyalty rewards, early access, premium support |
| **Growth** | Targeted upsell, bundle offers, loyalty onboarding |
| **At-Risk** | Reactivation campaigns, win-back discounts, satisfaction surveys |

## How to Run

1. Clone the repo or download the notebook (`.ipynb`)
2. Open in Google Colab or Jupyter Notebook
3. Run cells sequentially — data loads directly from the GitHub repo URLs in the notebook

## Skills Demonstrated

- Data preprocessing and cleaning
- Feature engineering (RFM metrics, log transformation, categorical grouping)
- Dimensionality reduction (PCA)
- Unsupervised ML — KMeans clustering
- Cluster validation (elbow method, silhouette score)
- Data visualisation (heatmaps, violin plots, scatter plots)
- Business insight generation from ML outputs
