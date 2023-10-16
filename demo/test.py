import pandas as pd
import scikit_posthocs as sp

# Sample data (replace with your data)
data = {
    'Decision Tree': [1, 0.998, 0.999],
    'Log regression': [0.999, 0.996, 0.998],
    'MLP': [0.871, 0.999, 0.929],
    'SVM': [1, 0.950, 0.974],
}

# Create a DataFrame
df = pd.DataFrame(data, index=['Accuracy', 'Recall', 'F1 Score'])
print("Original Dataframe:")
print(df)

# Perform the Scott-Knott effect size test for a chosen metric
test_result = sp.posthoc_nemenyi_friedman(df,)
print("Scott-Knott test result:")
print(test_result)


print("Ranked Models based on performances metrics:")
print(ranked_models)
