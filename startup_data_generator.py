import os
import pandas as pd
import numpy as np
from datetime import datetime

def ensure_data_exists():
    """
    Check if data exists in default locations, and if not, 
    create a simple synthetic dataset to get started.
    
    Returns:
        str: Path to the data file
    """
    # Define default paths to check
    default_data_paths = [
        os.path.join('data', 'hpv_survey_data.csv'),
        os.path.join('data', 'hpv_survey_synthetic_data.csv'),
        os.path.join('temp', 'hpv_survey_data.csv'),
    ]
    
    # Check if any of the default paths exist
    for path in default_data_paths:
        if os.path.exists(path):
            print(f"Found existing data at: {path}")
            return path
    
    # No data found, generate synthetic data
    print("No data found, generating synthetic data...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Generate a simple dataset
    n = 500  # Number of records
    
    # Create base dataframe with random values
    df = pd.DataFrame({
        'education': np.random.choice(
            ['No formal education', 'Primary School Certificate', 'SSCE/GCE', 
             'OND', 'HND/BSc', 'Postgraduate degree'], 
            size=n
        ),
        'gender': np.random.choice(['Woman', 'Man'], size=n),
        'age': np.random.choice(['18-29', '30-39', '40 and older'], size=n),
        'State': np.random.choice(['Abuja', 'Adamawa', 'Nasarawa'], size=n),
        'agegirl': np.random.choice(['9-11', '12-14', '15-17'], size=n),
        'important': np.random.choice(
            ['Strongly agree', 'Agree', 'No Opinion / Don\'t know', 
             'Disagree', 'Strongly disagree'], 
            size=n
        ),
        'knowwhere': np.random.choice(
            ['Strongly agree', 'Agree', 'No Opinion / Don\'t know', 
             'Disagree', 'Strongly disagree'], 
            size=n
        ),
        'difficult': np.random.choice(
            ['Strongly agree', 'Agree', 'No Opinion / Don\'t know', 
             'Disagree', 'Strongly disagree'], 
            size=n
        ),
        'seenads': np.random.choice(['Yes', 'No'], size=n),
        'heardhpvvax': np.random.choice(['Yes', 'No'], size=n)
    })
    
    # Add often column (depends on seenads)
    df['often'] = np.where(
        df['seenads'] == 'Yes',
        np.random.choice(['Only once', '2-3 times', 'More than 3 times'], size=n),
        ''
    )
    
    # Add agegiven column
    df['agegiven'] = np.random.choice(
        ['9-14', '15-19', '20-24', '25+', 'Don\'t know/Can\'t say'], 
        size=n
    )
    
    # Add hpvax column with correlations to other variables
    df['hpvax_tmp'] = 0.0
    
    # Adjust probability based on education (inverse relationship)
    education_scores = {
        'No formal education': 0.8,
        'Primary School Certificate': 0.7, 
        'SSCE/GCE': 0.55,
        'OND': 0.45,
        'HND/BSc': 0.4,
        'Postgraduate degree': 0.35
    }
    for edu, score in education_scores.items():
        df.loc[df['education'] == edu, 'hpvax_tmp'] += score
    
    # Adjust for seenads (exposure)
    df.loc[df['seenads'] == 'Yes', 'hpvax_tmp'] += 0.3
    df.loc[df['seenads'] == 'No', 'hpvax_tmp'] -= 0.3
    
    # Adjust for ability (knowwhere)
    ability_scores = {
        'Strongly agree': 0.3,
        'Agree': 0.2,
        'No Opinion / Don\'t know': 0,
        'Disagree': -0.2,
        'Strongly disagree': -0.3
    }
    for resp, score in ability_scores.items():
        df.loc[df['knowwhere'] == resp, 'hpvax_tmp'] += score
    
    # Add some randomness
    df['hpvax_tmp'] += np.random.normal(0, 0.1, size=n)
    
    # Convert to probability and then to categorical
    df['hpvax_prob'] = 1 / (1 + np.exp(-df['hpvax_tmp']))
    df['hpvax'] = np.where(
        df['hpvax_prob'] > np.random.random(size=n),
        'Yes', 
        np.where(np.random.random(size=n) > 0.95, 'I don\'t know', 'No')
    )
    
    # Add likely column based on important and other factors
    df['hpvlikely'] = np.random.choice(
        ['Very likely', 'Somewhat likely', 'No opinion, don\'t know', 
         'Somewhat unlikely', 'Very unlikely'], 
        size=n
    )
    
    # Drop the temporary columns
    df = df.drop(['hpvax_tmp', 'hpvax_prob'], axis=1)
    
    # Add metadata
    df['metadata'] = f"Simple synthetic data generated on {datetime.now().strftime('%Y-%m-%d')}"
    
    # Save the file
    output_path = os.path.join('data', 'hpv_survey_synthetic_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic data saved to: {output_path}")
    
    return output_path