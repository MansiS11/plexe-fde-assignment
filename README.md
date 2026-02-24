# ML for Marketplace Optimization — Plexe Take Home Assignment

# Architecture Diagram:
Dataset → Feature Pipeline → Model → SHAP → FastAPI → Docker → User

## Business Context
A fast-growing online marketplace is experiencing declining margins due to customer dissatisfaction, seller complaints, and inconsistent delivery performance.
The operations team requested guidance on where machine learning could create the highest business impact.
Rather than starting with a predefined prediction task, this project identifies high-value ML opportunities through data exploration and operational analysis.

## Selected ML Problem
Predict whether an order will result in a **bad customer experience** (≤2 star review).
### Why this matters
Bad reviews correlate with:
- refunds and operational costs
- customer churn
- seller disputes
- reduced marketplace trust
Early prediction enables proactive intervention.

## Dataset
Brazilian E-Commerce Public Dataset (Olist)
~100K real marketplace orders including:
- orders and payments
- delivery timelines
- seller and customer information
- customer reviews

## Approach
1. Exploratory Data Analysis to identify operational drivers
2. Feature engineering focused on delivery performance
3. Model experimentation
4. Explainable prediction API deployment

## Key Insights
- Delivery delays strongly increase probability of poor reviews
- Long shipping duration is the strongest dissatisfaction driver
- Freight cost contributes to negative customer perception

## Model
Model: XGBoost Classifier
Why XGBoost?
- Strong performance on tabular marketplace data
- Handles nonlinear relationships
- Interpretable via SHAP explanations

## Evaluation
Metrics:
- ROC-AUC (ranking quality)
- Recall (catch unhappy customers early)
Error analysis shows reduced performance for low-history sellers (cold start problem).

## Running the API
### Local Run
uvicorn src.serve:app --reload
Open:
http://localhost:8000/docs
# Example Request
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"price":120,"freight_value":20,"delivery_days":6,"is_delayed":1}'

## Docker Deployment
Build image
docker build -t plexe-ml .
Run container
docker run -p 8000:8000 plexe-ml

## Limitations
- Limited behavioral history for new sellers
- Static offline training
- No real-time feature updates

## Future Work
- Real-time monitoring pipeline
- Seller risk scoring dashboard
- Continuous model retraining
- A/B testing marketplace interventions
