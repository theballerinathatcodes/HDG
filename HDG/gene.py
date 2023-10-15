import numpy as np
import statsmodels.api as sm

# Simulated genotype data (0 = reference, 1 = variant)
genotype = np.random.randint(0, 2, size=100)

# Simulated disease status (0 = healthy, 1 = affected)
disease_status = np.random.randint(0, 2, size=100)

# Perform logistic regression for genetic association
genotype = sm.add_constant(genotype)  # Add an intercept term
model = sm.Logit(disease_status, genotype)
result = model.fit()

# Print the results
print(result.summary())
