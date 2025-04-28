import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Import our custom modules
from hpv_data_analysis import HPVVaccinationAnalysis
from hpv_nlp_module import HPVQueryProcessor

class HPVVaccinationChatbot:
    """
    Main chatbot class integrating data analysis and NLP capabilities
    for answering questions about HPV vaccination behavioral factors
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the HPV Vaccination chatbot
        
        Parameters:
        data_path (str, optional): Path to the data file
        """
        # Initialize the analysis module
        self.analyzer = HPVVaccinationAnalysis()
        
        # Initialize the NLP module
        self.nlp = HPVQueryProcessor()
        
        # Track conversation history
        self.conversation_history = []
        
        # Store analysis results
        self.analysis_results = None
        
        # Load data if provided
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        Load data from a file
        
        Parameters:
        data_path (str): Path to the data file
        
        Returns:
        bool: Success indicator
        """
        try:
            # Load data using the analyzer
            success = self.analyzer.load_data(data_path)
            
            if success:
                print(f"Data loaded successfully from {data_path}")
                
                # Clean the data
                self.analyzer.clean_data()
                
                # Run exploratory analysis
                self.analysis_results = self.analyzer.exploratory_analysis()
                
                return True
            else:
                print(f"Failed to load data from {data_path}")
                return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def process_query(self, query, user_id="user"):
        """
        Process a user query and generate a response
        
        Parameters:
        query (str): User query
        user_id (str): User identifier for tracking conversation
        
        Returns:
        dict: Response with answer and visualizations
        """
        # Add query to conversation history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history.append({
            "user_id": user_id,
            "timestamp": timestamp,
            "query": query,
            "type": "query"
        })
        
        # Process the query using NLP module
        response_text, visualizations = self.nlp.generate_custom_response(query, self.analysis_results)
        
        # Add response to conversation history
        self.conversation_history.append({
            "user_id": "chatbot",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "response": response_text,
            "visualizations": visualizations,
            "type": "response"
        })
        
        # Return the response
        return {
            "text": response_text,
            "visualizations": visualizations
        }
    
    def train_model(self):
        """
        Train a predictive model on the data
        
        Returns:
        dict: Model training results
        """
        if self.analyzer.data is None:
            return {"error": "No data loaded. Please load data first."}
        
        try:
            # Run logistic regression analysis
            result = self.analyzer.logistic_regression_analysis()
            
            # Analyze interaction effects
            interaction_result = self.analyzer.analyze_interaction_effects()
            
            # Combine results
            combined_result = {
                "regression": result,
                "interaction": interaction_result
            }
            
            return combined_result
        except Exception as e:
            return {"error": f"Error training model: {str(e)}"}
    
    def get_insights(self):
        """
        Get key insights from the analysis
        
        Returns:
        dict: Key insights
        """
        return self.analyzer.get_key_insights()
    
    def save_conversation(self, file_path):
        """
        Save the conversation history to a file
        
        Parameters:
        file_path (str): Path to save the conversation
        
        Returns:
        bool: Success indicator
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            print(f"Conversation saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
            return False
    
    def generate_report(self, query=None, report_path="hpv_analysis_report.html"):
        """
        Generate an HTML report with analysis results
        
        Parameters:
        query (str, optional): Specific query to focus the report on
        report_path (str): Path to save the HTML report
        
        Returns:
        bool: Success indicator
        """
        try:
            # Get insights
            insights = self.get_insights()
            
            # Generate HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>HPV Vaccination Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .section {{ margin-bottom: 30px; }}
                    .insight {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                    .visualization {{ text-align: center; margin: 20px 0; }}
                    .visualization img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
                    .recommendation {{ background-color: #e8f4f8; padding: 10px; margin-bottom: 10px; border-left: 4px solid #3498db; }}
                </style>
            </head>
            <body>
                <h1>HPV Vaccination Behavioral Factors Analysis</h1>
                <p><em>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
                
                <div class="section">
                    <h2>Overview</h2>
                    <div class="insight">{insights["overall"]}</div>
                </div>
                
                <div class="section">
                    <h2>Education Effects</h2>
                    <div class="insight">{insights["education"]}</div>
                    <div class="visualization">
                        <img src="education_effect.png" alt="Education Effect on HPV Vaccination">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Exposure to HPV Messaging</h2>
                    <div class="insight">{insights["exposure"]}</div>
                    <div class="visualization">
                        <img src="exposure_effect.png" alt="Exposure Effect on HPV Vaccination">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Education and Exposure Relationship</h2>
                    <div class="insight">{insights["education_exposure"]}</div>
                    <div class="visualization">
                        <img src="education_exposure.png" alt="Education and Exposure Relationship">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Motivation and Ability Factors</h2>
                    <div class="insight">{insights["motivation_ability"]}</div>
                    <div class="visualization">
                        <img src="motivation_effect.png" alt="Motivation Effect on HPV Vaccination">
                        <img src="ability_effect.png" alt="Ability Effect on HPV Vaccination">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Regional Variations</h2>
                    <div class="insight">{insights["regional"]}</div>
                    <div class="visualization">
                        <img src="region_effect.png" alt="Regional Variations in HPV Vaccination">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Interaction Effects</h2>
                    <div class="insight">{insights["interaction"]}</div>
                    <div class="visualization">
                        <img src="interaction_effect.png" alt="Interaction Effects">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Fogg Behavior Model Analysis</h2>
                    <div class="insight">{insights["fogg_model"]}</div>
                    <div class="visualization">
                        <img src="fogg_model_interaction.png" alt="Fogg Behavior Model Interaction">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    {"".join([f'<div class="recommendation">{r}</div>' for r in insights["recommendations"]])}
                </div>
            </body>
            </html>
            """
            
            # Save the HTML report
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            print(f"Report generated and saved to {report_path}")
            return True
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return False

# Command line interface for the chatbot
def main():
    print("="*60)
    print("HPV Vaccination Behavioral Factors Chatbot")
    print("="*60)
    print("This chatbot helps analyze factors affecting HPV vaccination decisions")
    print("based on research data from Nigeria.")
    print("\nType 'help' for available commands or 'exit' to quit.")
    print("="*60)
    
    chatbot = HPVVaccinationChatbot()
    
    # Available commands
    commands = {
        "load": "Load data from a file (e.g., 'load data.csv')",
        "train": "Train a predictive model on the loaded data",
        "insights": "Get key insights from the analysis",
        "report": "Generate an HTML report with visualizations",
        "save": "Save conversation history (e.g., 'save conversation.json')",
        "help": "Show available commands",
        "exit": "Exit the chatbot"
    }
    
    while True:
        user_input = input("\nYou: ").strip()
        
        # Check for commands
        if user_input.lower() == 'exit':
            print("Thank you for using the HPV Vaccination Chatbot!")
            break
        
        elif user_input.lower() == 'help':
            print("\nAvailable commands:")
            for cmd, desc in commands.items():
                print(f"  {cmd:<10} - {desc}")
            continue
        
        elif user_input.lower().startswith('load '):
            file_path = user_input[5:].strip()
            success = chatbot.load_data(file_path)
            if success:
                print(f"Data loaded successfully from {file_path}")
            else:
                print(f"Failed to load data from {file_path}")
            continue
        
        elif user_input.lower() == 'train':
            print("Training model...")
            result = chatbot.train_model()
            if "error" in result:
                print(result["error"])
            else:
                print("Model trained successfully")
                print(f"Regression accuracy: {result['regression']['test_accuracy']:.4f}")
            continue
        
        elif user_input.lower() == 'insights':
            insights = chatbot.get_insights()
            print("\nKey Insights:")
            for key, value in insights.items():
                if key != "recommendations":
                    print(f"\n{key.replace('_', ' ').title()}:")
                    print(f"  {value}")
                else:
                    print("\nRecommendations:")
                    for rec in value:
                        print(f"  - {rec}")
            continue
        
        elif user_input.lower() == 'report':
            success = chatbot.generate_report()
            if success:
                print("Report generated successfully")
            else:
                print("Failed to generate report")
            continue
        
        elif user_input.lower().startswith('save '):
            file_path = user_input[5:].strip()
            success = chatbot.save_conversation(file_path)
            if success:
                print(f"Conversation saved to {file_path}")
            else:
                print(f"Failed to save conversation to {file_path}")
            continue
        
        # Process as a regular query
        response = chatbot.process_query(user_input)
        print("\nChatbot:")
        print(response["text"])
        
        if response["visualizations"]:
            print("\nRelevant visualizations:", ", ".join(response["visualizations"]))

if __name__ == "__main__":
    main()