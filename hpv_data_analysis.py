import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from survey_metadata import SURVEY_QUESTIONS, SURVEY_OPTIONS, COMB_MAPPING

class HPVVaccinationChatbot:
    def __init__(self, data_path=None):
        """
        Initialize the HPV Vaccination chatbot with the Nigerian research data
        
        Parameters:
        data_path (str): Path to the CSV/Excel file containing the survey data
        """
        self.data = None
        self.model = None
        self.key_findings = {
            "education": "Education showed a negative association with vaccination rates. Caregivers with no formal education had higher rates (87.5%) compared to those with postgraduate education (39.9%).",
            "exposure": "Exposure to HPV vaccine messaging was strongly associated with vaccination (70.4% vs 19.1%). Less educated caregivers reported higher exposure to messaging.",
            "motivation": "High motivation was associated with higher vaccination rates (61.0% vs 46.3%).",
            "ability": "High ability (knowing where to get the vaccine) was strongly associated with vaccination (71.8% vs 43.2%).",
            "regional": "Geographic variations showed higher uptake in Nasarawa (67.1%) compared to Abuja (48.5%) and Adamawa (28.3%).",
            "interactions": "Effects of education, motivation and ability were moderated by exposure to vaccine advertising."
        }
        
        if data_path:
            self.load_data(data_path)
            
    def load_data(self, data_path):
        """Load the HPV vaccination survey data"""
        try:
            import pandas as pd
            import os
            
            if not os.path.exists(data_path):
                print(f"File not found: {data_path}")
                return False
                
            # Load data based on file type
            if data_path.lower().endswith('.csv'):
                self.data = pd.read_csv(data_path)
            elif data_path.lower().endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(data_path)
            else:
                print("Unsupported file format. Please use CSV or Excel.")
                return False
                
            # Basic data validation
            if len(self.data) == 0:
                print("The data file is empty")
                return False
                
            print(f"Data loaded successfully: {len(self.data)} rows, {len(self.data.columns)} columns")
            print(f"Columns: {', '.join(self.data.columns.tolist())}")
            
            # Try to preprocess the data automatically
            self.preprocess_data()
            
            return True
        except Exception as e:
            import traceback
            print(f"Error loading data: {str(e)}")
            print(traceback.format_exc())
            return False
    
    def preprocess_data(self):
        """Perform basic preprocessing on the data"""
        if self.data is None:
            return
            
        try:
            # Drop rows with too many missing values (more than 50%)
            self.data = self.data.dropna(thresh=len(self.data.columns) * 0.5)
            
            # Identify potential vaccination status column if not obvious
            vaccination_col = None
            for col in self.data.columns:
                col_lower = col.lower()
                if 'vaccin' in col_lower or 'status' in col_lower:
                    vaccination_col = col
                    break
                    
            if vaccination_col:
                # Try to convert to binary if it's not already numeric
                if self.data[vaccination_col].dtype != 'int64' and self.data[vaccination_col].dtype != 'float64':
                    # Map common yes/no formats to 1/0
                    try:
                        self.data[vaccination_col] = self.data[vaccination_col].map({
                            'yes': 1, 'no': 0, 
                            'Yes': 1, 'No': 0,
                            'TRUE': 1, 'FALSE': 0,
                            'true': 1, 'false': 0,
                            'True': 1, 'False': 0,
                            '1': 1, '0': 0
                        }).fillna(self.data[vaccination_col])
                    except:
                        pass
            
            print("Data preprocessing completed")
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
    
    def train_model(self):
        """Train a logistic regression model to predict vaccination status informed by literature"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False
        
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            # Make a copy of data to avoid modifying the original
            data_for_model = self.data.copy()
            
            # For this specific dataset, we know the target column is 'hpvax'
            # If it doesn't exist, try several other likely column names
            potential_targets = ['hpvax', 'HPVax', 'hpv_vax', 'vaccination_status', 'vaccinated', 'heardhpvvax']
            
            target_column = None
            for col in potential_targets:
                if col in data_for_model.columns:
                    target_column = col
                    print(f"Using '{target_column}' as vaccination status column")
                    break
                    
            if target_column is None:
                print("Could not find vaccination status column. Columns available are:")
                print(", ".join(data_for_model.columns))
                return False
            
            # Check if the column has valid values
            unique_values = data_for_model[target_column].dropna().unique()
            print(f"Unique values in {target_column}: {unique_values}")
            
            # Convert target to binary numeric (Yes=1, No=0, "I don't know"=NaN)
            if data_for_model[target_column].dtype == 'object':
                print("Converting vaccination status to binary values (Yes=1, No=0)")
                
                # Create a mapping for the specific values we've found
                target_mapping = {'Yes': 1, 'No': 0}
                
                # Treat "I don't know" and similar values as NaN
                dont_know_values = ["I don't know", "Don't know", "Unknown", "Not sure"]
                
                # Apply the mapping
                data_for_model[target_column] = data_for_model[target_column].map(
                    lambda x: target_mapping.get(x, float('nan') if x in dont_know_values else x)
                )
                
                print(f"After mapping: {data_for_model[target_column].value_counts(dropna=False).to_dict()}")
            
            # Ensure target is numeric
            try:
                data_for_model[target_column] = pd.to_numeric(data_for_model[target_column])
            except Exception as e:
                print(f"Could not convert all values in {target_column} to numeric. Will drop non-numeric values.")
                # For any remaining non-numeric values, set to NaN
                data_for_model[target_column] = pd.to_numeric(data_for_model[target_column], errors='coerce')
            
            # Check if we have enough valid target values after conversion
            valid_targets = data_for_model[target_column].dropna()
            print(f"Valid values after conversion: {len(valid_targets)} (out of {len(data_for_model)})")
            
            if len(valid_targets) < 10:
                print(f"Not enough valid values in {target_column} column: only {len(valid_targets)} non-null values")
                return False
            
            print(f"Target column distribution: {data_for_model[target_column].value_counts().to_dict()}")
            
            # Identify potential feature columns for this specific dataset
            known_features = [
                'education', 'knowwhere', 'seenads', 'age', 'heardhpvvax',
                'State', 'important', 'difficult', 'often', 'Group'
            ]
            
            feature_columns = []
            for col in known_features:
                if col in data_for_model.columns and col != target_column:
                    feature_columns.append(col)
            
            print(f"Using features: {feature_columns}")
            
            # Prepare feature columns for modeling
            for col in feature_columns:
                # Try to convert categorical features to numeric
                if data_for_model[col].dtype == 'object':
                    try:
                        # For State, use one-hot encoding
                        if col == 'State':
                            state_dummies = pd.get_dummies(data_for_model[col], prefix='state')
                            data_for_model = pd.concat([data_for_model, state_dummies], axis=1)
                            feature_columns.remove('State')
                            feature_columns.extend(state_dummies.columns.tolist())
                        # For boolean-like columns, map to 0/1
                        elif data_for_model[col].nunique() <= 5:
                            # Get unique values
                            unique_vals = data_for_model[col].dropna().unique()
                            print(f"Column {col} has values: {unique_vals}")
                            
                            if len(unique_vals) == 2:
                                # Binary column - create mapping (typical Yes/No pattern)
                                if 'Yes' in unique_vals and 'No' in unique_vals:
                                    mapping = {'Yes': 1, 'No': 0}
                                elif 'yes' in unique_vals and 'no' in unique_vals:
                                    mapping = {'yes': 1, 'no': 0}
                                else:
                                    # Just map first value to 0, second to 1
                                    first_val, second_val = unique_vals
                                    mapping = {first_val: 0, second_val: 1}
                                    
                                print(f"Mapping for {col}: {mapping}")
                                data_for_model[col] = data_for_model[col].map(mapping)
                            
                            # For columns like education with multiple values, create a simple ordinal mapping
                            elif col == 'education':
                                edu_order = {
                                    'None': 0,
                                    'Primary': 1,
                                    'Secondary': 2,
                                    'Graduate': 3, 
                                    'Postgraduate': 4
                                }
                                # Map education values, handling any that aren't in our predefined mapping
                                data_for_model[f'{col}_numeric'] = data_for_model[col].map(
                                    lambda x: edu_order.get(x, float('nan'))
                                )
                                # Replace original column in feature list with numeric version
                                feature_columns.remove(col)
                                feature_columns.append(f'{col}_numeric')
                    except Exception as e:
                        print(f"Error converting {col}: {str(e)}")
                        # If conversion fails, drop the column
                        print(f"Could not convert {col} to numeric, dropping from features")
                        if col in feature_columns:
                            feature_columns.remove(col)
            
            # Filter to only include columns that were successfully converted to numeric
            valid_features = []
            for col in feature_columns:
                if col in data_for_model.columns:
                    if pd.api.types.is_numeric_dtype(data_for_model[col]):
                        valid_features.append(col)
                    else:
                        print(f"Dropping non-numeric feature: {col}")
            
            feature_columns = valid_features
            print(f"Final features after processing: {feature_columns}")
            
            if not feature_columns:
                print("No valid numeric feature columns after processing")
                return False
            
            # Handle missing values in features
            for col in feature_columns:
                if data_for_model[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(data_for_model[col]):
                        # For numeric columns, fill with median
                        data_for_model[col] = data_for_model[col].fillna(data_for_model[col].median())
                    else:
                        # For categorical, fill with mode
                        data_for_model[col] = data_for_model[col].fillna(data_for_model[col].mode()[0])
            
            # Define features and target
            X = data_for_model[feature_columns]
            y = data_for_model[target_column]
            
            # Remove any rows with NaN in target
            valid_indices = ~y.isnull()
            X = X[valid_indices]
            y = y[valid_indices]
            
            print(f"Final dataset for training: {len(X)} rows, {len(feature_columns)} features")
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Calculate feature importance
            importance = {}
            for i, feature in enumerate(feature_columns):
                importance[feature] = abs(self.model.coef_[0][i])
            
            # Sort features by importance
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            self.feature_importance = {k: float(v) for k, v in sorted_importance}
            
            print(f"Model trained successfully. Training accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")
            
            # Literature-informed feature grouping based on papers
            literature_informed_features = {
                "knowledge_factors": ["heardhpvvax", "agegiven"],
                "access_factors": ["knowwhere", "difficult"],
                "social_factors": ["seenads", "often"],
                "demographic_factors": ["education", "age", "gender"],
                "geographic_factors": ["State"]
            }
            
            # Track feature importance by theoretical group
            feature_importance_by_group = {}
            
            # After training, calculate group-level importance
            for group_name, features in literature_informed_features.items():
                # Get features that were actually used in the model
                used_features = [f for f in features if f in feature_columns]
                if used_features:
                    group_importance = sum(self.feature_importance.get(f, 0) for f in used_features)
                    feature_importance_by_group[group_name] = float(group_importance)
                    
                    # Add to key findings
                    self.key_findings[f"{group_name}_importance"] = (
                        f"Based on the literature, {group_name} have a combined importance "
                        f"of {group_importance:.3f} in predicting vaccination status."
                    )
            
            # Store the group importance values
            self.feature_importance_by_group = feature_importance_by_group
            
            # Calculate key findings based on model
            self.calculate_key_findings(data_for_model, target_column)
            
            # Add model insights
            self.key_findings = self.key_findings if hasattr(self, 'key_findings') else {}
            self.key_findings["model_insights"] = {
                "training_accuracy": float(train_score * 100),
                "test_accuracy": float(test_score * 100),
                "feature_importance": self.feature_importance
            }
            
            return True
        
        except Exception as e:
            import traceback
            print(f"Error training model: {str(e)}")
            print(traceback.format_exc())
            return False

    def calculate_key_findings(self, data, target_column):
        """
        Calculate key findings from the data related to HPV vaccination
        """
        self.key_findings = {}
        
        try:
            # Add COM-B framework insight
            self.key_findings['framework'] = (
                "According to the COM-B behavior change framework, HPV vaccination behavior is influenced by: "
                "1) Capability factors like knowledge of the vaccine and where to get it, "
                "2) Opportunity factors like state of residence and ease of access, and "
                "3) Motivation factors like perceived importance and intention to vaccinate. "
                "Our analysis shows all three components affect vaccination decisions."
            )
            
            # Overall vaccination rate
            vax_rate = data[target_column].mean() * 100
            self.key_findings['overall'] = f"Overall HPV vaccination rate in the dataset is {vax_rate:.1f}%."
            
            # Education and Vaccination
            if 'education' in data.columns:
                try:
                    # Group by education and calculate vaccination rate
                    education_vax = data.groupby('education')[target_column].mean().reset_index()
                    education_vax = education_vax.sort_values(target_column, ascending=False)
                    
                    # Use the original survey question
                    question = SURVEY_QUESTIONS.get('education')
                    self.key_findings['education'] = (
                        f"When analyzing responses to '{question}', we found different vaccination rates by education level: " + 
                        ", ".join([f"{row['education']}: {row[target_column]*100:.1f}%" 
                                  for _, row in education_vax.iterrows()])
                    )
                except Exception as e:
                    print(f"Error calculating education findings: {str(e)}")
            
            # Regional differences
            if 'State' in data.columns:
                try:
                    # Group by State and calculate vaccination rate
                    state_vax = data.groupby('State')[target_column].mean().reset_index()
                    state_vax = state_vax.sort_values(target_column, ascending=False)
                    
                    self.key_findings['regional'] = (
                        f"Regional analysis shows varying vaccination rates across states: " + 
                        ", ".join([f"{row['State']}: {row[target_column]*100:.1f}%" 
                                 for _, row in state_vax.iterrows()])
                    )
                except Exception as e:
                    print(f"Error calculating regional findings: {str(e)}")
            
            # Exposure and vaccination
            if 'seenads' in data.columns:
                try:
                    # Group by exposure and calculate vaccination rate
                    exp_vax = data.groupby('seenads')[target_column].mean().reset_index()
                    
                    # Use the original survey question
                    question = SURVEY_QUESTIONS.get('seenads')
                    yes_rate = exp_vax.loc[exp_vax['seenads'] == 'Yes', target_column].values[0]*100
                    no_rate = exp_vax.loc[exp_vax['seenads'] == 'No', target_column].values[0]*100
                    
                    self.key_findings['exposure'] = (
                        f"For the question '{question}', those who answered 'Yes' have " + 
                        f"a vaccination rate of {yes_rate:.1f}% compared to {no_rate:.1f}% " + 
                        f"for those who answered 'No'. This indicates a strong correlation between " +
                        f"exposure to vaccine information and vaccination uptake."
                    )
                except Exception as e:
                    print(f"Error calculating exposure findings: {str(e)}")
            
            # Knowledge of where to get vaccine
            if 'knowwhere' in data.columns:
                try:
                    # Try converting knowwhere to binary if it's not already
                    if data['knowwhere'].dtype != 'int64' and data['knowwhere'].dtype != 'float64':
                        # Group by different response levels
                        know_vax = data.groupby('knowwhere')[target_column].mean().reset_index()
                        
                        # Use the original survey question
                        question = SURVEY_QUESTIONS.get('knowwhere')
                        self.key_findings['ability'] = (
                            f"Responses to '{question}' show that knowledge of where to get the vaccine " +
                            f"strongly correlates with vaccination rates. " +
                            ", ".join([f"{row['knowwhere']}: {row[target_column]*100:.1f}%" 
                                     for _, row in know_vax.iterrows()])
                        )
                    else:
                        # Binary version
                        yes_rate = data[data['knowwhere'] == 1][target_column].mean() * 100
                        no_rate = data[data['knowwhere'] == 0][target_column].mean() * 100
                        
                        self.key_findings['ability'] = (
                            f"Knowing where to get the HPV vaccine increases vaccination rates from " +
                            f"{no_rate:.1f}% to {yes_rate:.1f}%, highlighting the importance of " +
                            f"providing clear access information to caregivers."
                        )
                except Exception as e:
                    print(f"Error calculating ability findings: {str(e)}")
                    
            # Vaccination importance
            if 'important' in data.columns:
                try:
                    # Group by importance and calculate vaccination rate
                    imp_vax = data.groupby('important')[target_column].mean().reset_index()
                    
                    # Use the original survey question
                    question = SURVEY_QUESTIONS.get('important')
                    self.key_findings['importance'] = (
                        f"Analyzing responses to '{question}' reveals that perceived importance " +
                        f"is a strong predictor of vaccination status, with rates ranging from " +
                        ", ".join([f"{row['important']}: {row[target_column]*100:.1f}%" 
                                 for _, row in imp_vax.iterrows()])
                    )
                except Exception as e:
                    print(f"Error calculating importance findings: {str(e)}")
                    
        except Exception as e:
            import traceback
            print(f"Error in calculate_key_findings: {str(e)}")
            print(traceback.format_exc())
    
    def process_query(self, query):
        """Process a query about HPV vaccination factors"""
        import re
        
        # Convert query to lowercase for easier matching
        query_lower = query.lower()
        
        # Check for keywords related to different aspects
        if re.search(r'education|educated|school|university|college|academic', query_lower):
            return self.key_findings["education"]
        elif re.search(r'expos|messag|advert|information|awareness', query_lower):
            return self.key_findings["exposure"]
        elif re.search(r'motiv|willing|interest|intend|desire', query_lower):
            return self.key_findings["motivation"]
        elif re.search(r'abil|know|where|how|access|get', query_lower):
            return self.key_findings["ability"]
        elif re.search(r'region|area|location|geograph|nasarawa|abuja|adamawa', query_lower):
            return self.key_findings["regional"]
        elif re.search(r'interact|relation|between|combined|together|mix', query_lower):
            return self.key_findings["interactions"]
        else:
            return "I'm not sure about that specific aspect. You can ask me about education, exposure to messaging, motivation, ability, regional differences, or interaction effects on HPV vaccination rates."

    def generate_key_findings_summary(self) -> str:
        """Generate a well-formatted summary of key findings"""
        if not self.key_findings:
            return "No key findings available yet. Please train the model first."
        
        summary = "Key Findings from HPV Vaccination Survey Data:\n\n"
        
        # Overall vaccination rate
        if "overall_vaccination_rate" in self.key_findings:
            summary += f"Overall Vaccination Rate:\n"
            summary += f"• {self.key_findings['overall_vaccination_rate']}\n\n"
        
        # Education findings
        if "education_finding" in self.key_findings:
            summary += f"Education:\n"
            summary += f"• {self.key_findings['education_finding']}\n\n"
        
        # Exposure findings
        if "exposure_finding" in self.key_findings:
            summary += f"Exposure to Messaging:\n"
            summary += f"• {self.key_findings['exposure_finding']}\n\n"
        
        # Motivation findings
        if "motivation_finding" in self.key_findings:
            summary += f"Motivation:\n"
            summary += f"• {self.key_findings['motivation_finding']}\n\n"
        
        # Ability findings
        if "ability_finding" in self.key_findings:
            summary += f"Ability:\n"
            summary += f"• {self.key_findings['ability_finding']}\n\n"
        
        # Regional findings
        if "regional_finding" in self.key_findings:
            summary += f"Regional Differences:\n"
            summary += f"• {self.key_findings['regional_finding']}\n\n"
        
        # Feature importance
        if self.feature_importance and len(self.feature_importance) > 0:
            summary += "Top Predictors of Vaccination:\n"
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                summary += f"• {feature}: {importance:.3f}\n"
        
        return summary

# Create a simple command-line interface for the chatbot
def main():
    print("HPV Vaccination Chatbot - Nigeria Research Data")
    print("="*50)
    print("This chatbot can answer questions about factors affecting HPV vaccination in Nigeria")
    print("Type 'exit' to quit")
    
    # Initialize the chatbot
    chatbot = HPVVaccinationChatbot()
    
    while True:
        query = input("\nAsk a question about HPV vaccination in Nigeria: ")
        
        if query.lower() == 'exit':
            print("Thank you for using the HPV Vaccination Chatbot!")
            break
        
        response = chatbot.process_query(query)
        print("\nChatbot:", response)

if __name__ == "__main__":
    main()