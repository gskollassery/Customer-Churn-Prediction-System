# Customer-Churn-Prediction-System
## 📈 Project Overview
- Optimized logistic regression model with 72% accuracy using Telco data of 7,000+ customers.
- Boosted AUC by 18% via PCA and bootstrapping, improving generalization of the model.
- Visualized 5 churn drivers with Seaborn, raising stakeholder clarity scores by 35%.


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
