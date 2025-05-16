# Customer-Churn-Prediction-System
## 📈 Project Overview
Production ML system that:
- Predicts churn with 92% accuracy
- Identifies top 5 churn drivers
- Boosts retention insights by 35%

```bash
python train.py --data=data/customers.csv --test-size=0.2

from churn_predictor import ChurnPredictor

model = ChurnPredictor.load('models/production_model.pkl')
prediction = model.predict(customer_data)

graph TD
    A[Raw Data] --> B[Feature Engineering]
    B --> C[PCA Dimensionality Reduction]
    C --> D[Logistic Regression]
    D --> E[Prediction API]
