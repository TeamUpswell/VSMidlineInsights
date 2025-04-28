from flask import Flask, render_template, request, jsonify, send_file
import os
import io
import traceback
from datetime import datetime
from survey_metadata import SURVEY_QUESTIONS, SURVEY_OPTIONS, COMB_MAPPING

# Import our chatbot modules
from hpv_data_analysis import HPVVaccinationChatbot
from hpv_nlp_module import HPVQueryProcessor
from startup_data_generator import ensure_data_exists

try:
    from literature_repository import LITERATURE_SOURCES, LITERATURE_INSIGHTS, LITERATURE_QUOTES, LITERATURE_RECOMMENDATIONS
except ImportError:
    # Create empty placeholders if the module doesn't exist
    LITERATURE_SOURCES = []
    LITERATURE_INSIGHTS = {}
    LITERATURE_QUOTES = {}
    LITERATURE_RECOMMENDATIONS = []

# Initialize Flask app
app = Flask(__name__)

# Initialize global variables
data_analyzer = None
query_processor = HPVQueryProcessor(use_claude=False)
data_loaded = False
model_trained = False

def initialize_chatbot():
    """Initialize the chatbot with data and model"""
    global data_analyzer, data_loaded, model_trained
    
    try:
        # Initialize data analyzer
        data_analyzer = HPVVaccinationChatbot()
        
        # Make sure we have data (generate if needed)
        data_path = ensure_data_exists()
        
        # Load the data
        data_loaded = data_analyzer.load_data(data_path)
        print(f"Data loaded: {data_loaded}")
        
        # Train model if data was loaded
        if data_loaded:
            model_trained = data_analyzer.train_model()
            print(f"Model trained: {model_trained}")
        
        return data_analyzer
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        traceback.print_exc()
        return None

# Initialize chatbot before defining routes
data_analyzer = initialize_chatbot()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/check_data_status')
def check_data_status():
    """Check if data is loaded and model is trained"""
    global data_analyzer, data_loaded, model_trained
    
    # Make sure we have the correct state
    if data_analyzer is not None and hasattr(data_analyzer, 'data') and data_analyzer.data is not None:
        data_loaded = True
    
    if data_loaded and hasattr(data_analyzer, 'model') and data_analyzer.model is not None:
        model_trained = True
        
    data_size = len(data_analyzer.data) if data_loaded else 0
    
    return jsonify({
        'loaded': data_loaded,
        'model_trained': model_trained,
        'data_size': data_size
    })

@app.route('/survey_info')
def survey_info():
    try:
        print("Survey info route called")
        
        # Check if we have survey metadata loaded
        if not SURVEY_QUESTIONS:
            print("Warning: SURVEY_QUESTIONS is empty")
            
        return jsonify({
            "survey_questions": SURVEY_QUESTIONS,
            "answer_options": SURVEY_OPTIONS,
            "comb_framework": COMB_MAPPING
        })
    except Exception as e:
        import traceback
        print(f"Error in survey_info route: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global data_analyzer, query_processor
    
    query = request.json.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'})
    
    try:
        # Always try to provide data context if available
        data_context = None
        if data_analyzer is not None and hasattr(data_analyzer, 'data') and data_analyzer.data is not None:
            # Extract key statistics to provide context
            data_context = {
                "data_summary": {
                    "total_records": len(data_analyzer.data),
                    "vaccination_rate": float(data_analyzer.data['hpvax'].map({'Yes': 1, 'No': 0, "I don't know": None}).mean() * 100) 
                    if 'hpvax' in data_analyzer.data.columns else 0.0,
                    "survey_questions": SURVEY_QUESTIONS  # Add survey questions to context
                }
            }
            
            # Add key_findings if available
            if hasattr(data_analyzer, 'key_findings'):
                data_context["key_findings"] = data_analyzer.key_findings
        
        # Process the query
        response_data = query_processor.process_query(query, data_context)
        
        return jsonify({
            'answer': response_data['answer'],
            'source': response_data['source'],
            'topics': response_data['topics']
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error processing query: {str(e)}'})

@app.route('/load_data', methods=['POST'])
def load_data():
    global data_analyzer
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Save the file temporarily
        file_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(file_path)
        
        # Initialize data analyzer
        data_analyzer = HPVVaccinationChatbot()
        success = data_analyzer.load_data(file_path)
        
        if success:
            return jsonify({'success': 'Data loaded successfully'})
        else:
            return jsonify({'error': 'Error processing data file'})
            
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/train_model', methods=['POST'])
def train_model():
    global data_analyzer
    
    if data_analyzer is None:
        return jsonify({'error': 'No data loaded. Please upload data first.'})
    
    try:
        success = data_analyzer.train_model()
        
        if success:
            return jsonify({'success': 'Model trained successfully'})
        else:
            # Try to get a more specific error message
            import sys
            error_info = sys.exc_info()
            if error_info[1]:
                error_msg = str(error_info[1])
            else:
                error_msg = "Error training model. Check server logs for details."
            return jsonify({'error': error_msg})
            
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/literature')
def literature():
    """Return information about the research literature"""
    try:
        paper_insights = {}
        
        # Group insights by paper
        for source in LITERATURE_SOURCES:
            paper_title = source['title']
            paper_insights[paper_title] = {
                "authors": source['authors'],
                "year": source['year'],
                "key_findings": source['key_findings'],
                "citation": source['citation'],
                "topics": {}
            }
            
            # Organize insights by topic for this paper
            for topic, insights in LITERATURE_INSIGHTS.items():
                relevant_insights = [insight for insight in insights 
                                   if source['authors'].split(',')[0] in insight]
                if relevant_insights:
                    paper_insights[paper_title]["topics"][topic] = relevant_insights
        
        return jsonify({
            "sources": LITERATURE_SOURCES,
            "insights": LITERATURE_INSIGHTS,
            "quotes": LITERATURE_QUOTES,
            "recommendations": LITERATURE_RECOMMENDATIONS,
            "paper_insights": paper_insights
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)