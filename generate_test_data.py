import pandas as pd
import numpy as np

# Create synthetic data
n_samples = 500
np.random.seed(42)  # For reproducibility

data = {
    'education_level': np.random.choice(['None', 'Primary', 'Secondary', 'Graduate', 'Postgraduate'], n_samples),
    'exposure_to_messaging': np.random.choice(['High', 'Low'], n_samples),
    'motivation_level': np.random.choice(['High', 'Medium', 'Low'], n_samples),
    'knows_vaccination_location': np.random.choice(['Yes', 'No'], n_samples),
    'region': np.random.choice(['Nasarawa', 'Abuja', 'Adamawa'], n_samples)
}

# Create vaccination status based on these factors
vaccination_prob = []
for i in range(n_samples):
    prob = 0.5
    # Education effect (inverse)
    edu_level = {'None': 0.3, 'Primary': 0.2, 'Secondary': 0.1, 'Graduate': -0.1, 'Postgraduate': -0.2}
    prob += edu_level[data['education_level'][i]]
    
    # Exposure effect
    prob += 0.3 if data['exposure_to_messaging'][i] == 'High' else -0.1
    
    # Motivation effect
    mot_level = {'High': 0.2, 'Medium': 0.0, 'Low': -0.2}
    prob += mot_level[data['motivation_level'][i]]
    
    # Ability effect
    prob += 0.25 if data['knows_vaccination_location'][i] == 'Yes' else -0.15
    
    # Region effect
    reg_effect = {'Nasarawa': 0.15, 'Abuja': 0.0, 'Adamawa': -0.15}
    prob += reg_effect[data['region'][i]]
    
    # Add some random noise
    prob += np.random.normal(0, 0.1)
    
    # Ensure probability is between 0 and 1
    prob = max(0, min(1, prob))
    vaccination_prob.append(prob)

# Generate vaccination status based on probabilities
data['vaccination_status'] = np.random.binomial(1, vaccination_prob)

# Convert education level to numeric for model training
education_mapping = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Graduate': 3, 'Postgraduate': 4}
data['education_numeric'] = [education_mapping[level] for level in data['education_level']]
data['exposure_numeric'] = [1 if exp == 'High' else 0 for exp in data['exposure_to_messaging']]
data['knows_location_numeric'] = [1 if knows == 'Yes' else 0 for knows in data['knows_vaccination_location']]
motivation_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
data['motivation_numeric'] = [motivation_mapping[level] for level in data['motivation_level']]

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv('test_hpv_data.csv', index=False)
print("Test data saved to 'test_hpv_data.csv'")