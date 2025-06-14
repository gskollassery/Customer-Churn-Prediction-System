# Customer Churn Prediction  
**Goal**: Predict churn risk (72% accuracy) and identify top drivers.  

## **Key Improvements**  
- **AUC boosted by 18%** using PCA + bootstrapping.  
- Top 5 churn drivers: Contract length, Monthly charges, etc.  

## **How to Run**  
1. Preprocess data: `python src/preprocess.py`  
2. Train model: `python src/model_training.py`  

## **Data Source**  
- [Kaggle Telco Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
