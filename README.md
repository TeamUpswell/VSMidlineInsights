HPV Vaccination Behavioral Factors Chatbot
This project implements a chatbot for analyzing behavioral factors affecting HPV vaccination decisions among caregivers in Nigeria, based on research data from a 2024 cross-sectional survey.

Overview
The chatbot allows users to:

Load and analyze HPV vaccination survey data
Ask natural language questions about factors affecting vaccination decisions
Explore the relationships between education, exposure to messaging, motivation, and ability
Generate visualizations of key relationships
Create analytical reports with insights and recommendations
Key Features
Natural Language Processing: Understands and responds to questions about vaccination behaviors
Data Analysis: Performs statistical analysis on survey data to identify patterns and relationships
Interactive Visualizations: Creates charts to illustrate key findings
Predictive Modeling: Builds logistic regression models to identify the most important factors
Web Interface: Offers both command-line and web-based interfaces
Installation
Prerequisites
Python 3.8+
pip (Python package manager)
Dependencies
bash
pip install -r requirements.txt
The requirements.txt file includes the following packages:

pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
nltk
flask
Setting Up
Clone the repository:
bash
git clone [your-repository-url]
cd hpv-vaccination-chatbot
Install dependencies:
bash
pip install -r requirements.txt
Download NLTK resources:
python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Usage
Command-line Interface
Run the chatbot in command-line mode:

bash
python hpv_chatbot_main.py
Available commands:

load [file_path]: Load data from a CSV or Excel file
train: Train a predictive model on the loaded data
insights: Get key insights from the analysis
report: Generate an HTML report with visualizations
save [file_path]: Save conversation history to a JSON file
help: Show available commands
exit: Exit the chatbot
Example questions to ask the chatbot:

"How does education affect HPV vaccination rates?"
"What is the relationship between exposure to messaging and vaccination?"
"How does the Fogg Behavior Model apply to HPV vaccination decisions?"
"What regional differences exist in vaccination rates?"
"What recommendations would improve vaccination rates among educated caregivers?"
Web Interface
Run the Flask web application:

bash
python app.py
Then open a web browser and navigate to http://localhost:5000

File Structure
hpv-vaccination-chatbot/
├── app.py                     # Flask web application
├── hpv_chatbot_main.py        # Main chatbot integration module
├── hpv_data_analysis.py       # Data analysis module
├── hpv_nlp_module.py          # Natural language processing module
├── templates/                 # Flask HTML templates
│   └── index.html             # Web interface template
├── static/                    # Static files for web interface
│   ├── css/
│   └── js/
├── data/                      # Sample data files
├── notebooks/                 # Jupyter notebooks for exploration
└── README.md                  # This file
Data Format
The chatbot expects survey data in CSV or Excel format with the following key columns:

education: Education level of caregiver (categorical)
gender: Gender of caregiver (categorical)
age: Age group of caregiver (categorical)
state: State/region (categorical)
important: Motivation measure - importance of vaccination (categorical)
knowwhere: Ability measure - knowing where to get vaccine (categorical)
seenads: Exposure to HPV vaccine messaging (binary)
hpvax: HPV vaccination status (binary outcome variable)
Sample data matching the Nigerian survey format is included in the data directory.

Key Research Findings
The chatbot is based on research that found:

Education Paradox: Contrary to typical health behavior patterns, education showed a negative association with HPV vaccination, with caregivers with no formal education having higher vaccination rates (87.5%) than those with postgraduate education (39.9%).
Message Exposure: Exposure to HPV vaccine messaging had a powerful effect, with exposed caregivers 51 percentage points more likely to vaccinate their children.
Motivation and Ability: Both motivation (seeing vaccination as important) and ability (knowing where to get the vaccine) were positively associated with vaccination.
Regional Differences: Significant regional variations existed, with Nasarawa showing higher vaccination rates (67.1%) than Abuja (48.5%) and Adamawa (28.3%).
Interaction Effects: Education moderated the effect of exposure to messaging, with less educated caregivers benefiting more from HPV vaccine messaging.
Fogg Behavior Model: The data supported the Fogg Behavior Model, showing a multiplicative effect when motivation, ability, and prompts (exposure) occurred together.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This project is based on research data from a Nigerian HPV vaccination survey conducted in 2024
Thanks to the Behavioral Insights Lab for making the research findings available
