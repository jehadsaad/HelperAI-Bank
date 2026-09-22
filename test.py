from src.prediction import predict


# Example customer data
features = [
    20000,  # LIMIT_BAL
    2,      # SEX
    2,      # EDUCATION
    1,      # MARRIAGE
    24,     # AGE
    2,      # PAY_0
    2,      # PAY_2
    -1,     # PAY_3
    -1,     # PAY_4
    -2,     # PAY_5
    -2,     # PAY_6
    3913,   # BILL_AMT1
    3102,   # BILL_AMT2
    689,    # BILL_AMT3
    0,      # BILL_AMT4
    0,      # BILL_AMT5
    0,      # BILL_AMT6
    0,      # PAY_AMT1
    689,    # PAY_AMT2
    0,      # PAY_AMT3
    0,      # PAY_AMT4
    0,      # PAY_AMT5
    0       # PAY_AMT6
]


result = predict(features)

print("Prediction Result:")
print(result)