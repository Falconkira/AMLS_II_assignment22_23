import os
from A.A import data_preprocessing
from A.A import createEFF
from A.A import trainEFF
from A.A import evaluate


# ======================================================================================================================
# Data preprocessing
os.listdir("Datasets") 
data_train, data_val, data_test, temp = data_preprocessing()
# ======================================================================================================================

# Build and train model object.
model_A = createEFF()
# Train model based on the training set
acc_A_train = trainEFF(model_A, data_train, data_val) 
# Test model based on the test set.
acc_A_test = evaluate(model_A, temp, data_test)   

# ======================================================================================================================
# Print out results
print('TA:{},{}'.format(acc_A_train, acc_A_test))
