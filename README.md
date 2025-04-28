# HPV Vaccination Chatbot

This interactive chatbot analyzes factors affecting HPV vaccination rates in Nigeria using survey data and provides insights through a conversational interface.

## Features

- Interactive chat interface for querying HPV vaccination data
- Automated data loading and model training
- Integration with research literature
- Original survey question reference
- Responsive design with Bootstrap

## Setup and Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys (see `.env.template`)
4. Run the application: `python app.py`
5. Access the interface at http://localhost:5000

## Project Structure

- `app.py`: Main Flask application
- `hpv_data_analysis.py`: Data processing and analysis module
- `hpv_nlp_module.py`: Natural language processing functionality
- `auto_data_loader.py`: Automatic data loading and model training
- `survey_metadata.py`: Survey question and answer option metadata
- `templates/`: HTML templates for the web interface

## Deployment

The application is deployed to PythonAnywhere at [your-pythonanywhere-url].
