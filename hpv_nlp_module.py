import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import json
import re

# Import survey metadata
from survey_metadata import SURVEY_QUESTIONS, SURVEY_OPTIONS, COMB_MAPPING, SURVEY_SECTIONS
from literature_repository import LITERATURE_INSIGHTS, LITERATURE_QUOTES, LITERATURE_SOURCES, LITERATURE_RECOMMENDATIONS

# Try to download NLTK resources, but handle if they're already available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class HPVQueryProcessor:
    """
    Natural Language Processing module for understanding and responding to
    queries about HPV vaccination factors using rule-based approach
    """
    
    def __init__(self, use_claude: bool = False, claude_api_key: Optional[str] = None):
        """Initialize the query processor with topic information"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.use_claude = False  # Always use rule-based processing
        
        # Load survey metadata
        self.survey_questions = SURVEY_QUESTIONS
        self.survey_options = SURVEY_OPTIONS
        
        # Define topics and their related keywords
        self.topics = {
            'education': [
                'education', 'school', 'university', 'college', 'academic', 
                'educated', 'literacy', 'qualification', 'degree', 'formal education',
                'primary school', 'ssce', 'gce', 'ond', 'hnd', 'bsc', 'postgraduate'
            ],
            'exposure': [
                'exposure', 'messaging', 'advertisement', 'campaign', 'information',
                'awareness', 'media', 'communication', 'outreach', 'publicity',
                'seen ads', 'heard', 'advertisements', 'messages', 'last 3 months',
                'often', 'frequency'
            ],
            'motivation': [
                'motivation', 'willing', 'interest', 'intend', 'desire',
                'attitude', 'belief', 'perception', 'intention', 'willingness',
                'important', 'agree', 'strongly agree', 'likely', 'very likely'
            ],
            'ability': [
                'ability', 'access', 'know', 'knowledge', 'skill',
                'capability', 'competence', 'whereabouts', 'location', 'facility',
                'know where', 'getting vaccinated', 'easy', 'difficult'
            ],
            'regional': [
                'region', 'area', 'location', 'geography', 'place',
                'state', 'city', 'urban', 'rural', 'community',
                'adamawa', 'abuja', 'nasarawa', 'fct'
            ],
            'interaction': [
                'interaction', 'relationship', 'between', 'combined', 'correlation',
                'association', 'combination', 'together', 'mutual', 'joint'
            ],
            'survey': [
                'survey', 'questionnaire', 'study', 'research', 'questions',
                'participants', 'caregivers', 'respondents', 'sample', 'data collection',
                'methodology', 'original survey', 'nigerian'
            ],
            'comb': [
                'comb', 'framework', 'com-b', 'theory', 'capability', 
                'opportunity', 'motivation', 'behavior', 'behavior change'
            ],
            'barriers': [
                'barriers', 'challenges', 'obstacles', 'difficulties', 'problems', 'issues',
                'hindrances', 'hurdles', 'impediments', 'constraints', 'limitations',
                'misconceptions', 'safety concerns', 'cultural concerns', 'religious concerns'
            ],
            'strategies': [
                'strategies', 'interventions', 'approaches', 'solutions', 'recommendations',
                'initiatives', 'programs', 'campaigns', 'methods', 'techniques', 'practices',
                'mobile clinics', 'school programs', 'community outreach'
            ],
            'research': [
                'research', 'study', 'paper', 'article', 'publication', 'journal', 
                'findings', 'evidence', 'data', 'literature', 'report', 'investigation',
                'uchendu', 'nigerian journal', 'public health'
            ]
        }
        
        # Add survey questions to enhance topic detection
        for col, question in self.survey_questions.items():
            topic = self._map_column_to_topic(col)
            if topic and topic in self.topics:
                # Add the question text to the relevant topic keywords
                self.topics[topic].append(question.lower())
        
        # Create a document for each topic for later comparison
        self.topic_docs = {topic: ' '.join(keywords) for topic, keywords in self.topics.items()}
        
        # Prepare the TF-IDF vectors
        self.prepare_document_vectors()

    def _map_column_to_topic(self, column):
        """Map a column name to one of our topics"""
        col_lower = column.lower()
        
        if col_lower in ['education', 'edu']:
            return 'education'
        elif col_lower in ['seenads', 'heardhpvvax', 'often']:
            return 'exposure'
        elif col_lower in ['important', 'hpvlikely']:
            return 'motivation'
        elif col_lower in ['knowwhere', 'difficult']:
            return 'ability'
        elif col_lower in ['state']:
            return 'regional'
        else:
            return None

    def prepare_document_vectors(self):
        """Create TF-IDF vectors for topic documents"""
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text, stop_words='english')
        documents = list(self.topic_docs.values())
        self.topic_vectors = self.vectorizer.fit_transform(documents)
    
    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stop words, and lemmatizing"""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in self.stop_words]
        return filtered_tokens
    
    def identify_topics(self, query):
        """Identify the most relevant topics in a query"""
        # Preprocess the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with each topic
        similarities = cosine_similarity(query_vector, self.topic_vectors)[0]
        
        # Get topics sorted by similarity
        topic_names = list(self.topic_docs.keys())
        topic_scores = [(topic, score) for topic, score in zip(topic_names, similarities)]
        sorted_topics = sorted(topic_scores, key=lambda x: x[1], reverse=True)
        
        # Return topics that have similarity above a threshold
        threshold = 0.1
        relevant_topics = [topic for topic, score in sorted_topics if score > threshold]
        
        return relevant_topics if relevant_topics else ['general']
    
    def _get_basic_response(self, topic: str) -> str:
        """Get the basic rule-based response with better formatting"""
        responses = {
            'education': f"""
Education has a complex relationship with HPV vaccination rates in Nigeria.

According to our survey question '{self.survey_questions.get('education')}', we found that caregivers with no formal education had higher vaccination rates compared to those with postgraduate education. This counterintuitive finding may be related to exposure to messaging.

Key findings:
• Caregivers with no formal education: 87.5% vaccination rate
• Caregivers with primary education: 77.2% vaccination rate
• Caregivers with postgraduate education: 39.9% vaccination rate

This inverse relationship suggests that factors beyond education level, such as exposure to messaging and community influence, may play crucial roles in vaccination decisions.
""",
            
            'exposure': f"""
Exposure to HPV vaccine messaging was strongly associated with vaccination rates.

When caregivers were asked '{self.survey_questions.get('seenads')}', those who answered 'Yes' had significantly higher vaccination rates than those who answered 'No'.

Key statistics:
• Exposed to messaging: 70.4% vaccination rate
• Not exposed to messaging: 19.1% vaccination rate

This difference of 51.3 percentage points represents one of the strongest predictors of vaccination behavior in our study, highlighting the critical importance of effective communication campaigns.

Frequency of exposure (measured by '{self.survey_questions.get('often')}') also showed a dose-response relationship, with more frequent exposure correlating with higher vaccination rates.
""",
            
            'motivation': f"""
High motivation was associated with higher vaccination rates.

Our survey assessed this using the question '{self.survey_questions.get('important')}'. Caregivers who strongly agreed had much higher vaccination rates for their daughters than those who disagreed.

Key findings:
• Caregivers with high motivation (strongly agree/agree): 61.0% vaccination rate
• Caregivers with low motivation (disagree/strongly disagree): 46.3% vaccination rate

Motivation appears to be a necessary but not sufficient condition for vaccination, as it often needs to be combined with high capability (knowing where to get the vaccine) and opportunity (exposure to messaging) factors to translate into behavior.
""",
            
            'ability': f"""
Knowing where to get the HPV vaccine (ability) was strongly associated with vaccination rates.

We measured this using the question '{self.survey_questions.get('knowwhere')}'. Caregivers who agreed or strongly agreed were much more likely to have vaccinated their daughters.

Key statistics:
• High ability (know where to get vaccine): 71.8% vaccination rate
• Low ability (don't know where to get vaccine): 43.2% vaccination rate

This 28.6 percentage point difference highlights the importance of not just motivating caregivers but ensuring they have the practical knowledge to access vaccination services.

The inverse relationship with perceived difficulty (measured by '{self.survey_questions.get('difficult')}') further supports this finding.
""",
            
            'barriers': """
Several barriers to HPV vaccination have been identified in Nigeria.

Key barriers include:
• Knowledge gaps about HPV and cervical cancer
• Misconceptions about vaccine safety and efficacy
• Cultural and religious concerns about vaccines
• Limited access to vaccination services
• Cost considerations
• Low awareness of where to get vaccines

Our data shows that addressing barriers to ability (knowing where to get vaccines) and opportunity (exposure to messaging) may be particularly effective in increasing vaccination rates.
""",

            'general': """
I can answer questions about factors affecting HPV vaccination rates in Nigeria based on survey data.

Key topics include:

• Education: How education level relates to vaccination rates
• Exposure: The impact of messaging and advertisements on vaccination
• Motivation: How perceived importance influences vaccination decisions
• Ability: The effect of knowing where to access vaccines
• Regional differences: How vaccination rates vary across Nigerian states
• Barriers: Common obstacles to HPV vaccination
• Strategies: Evidence-based approaches to increase vaccination rates

What specific aspect of HPV vaccination would you like to know more about?
"""
        }
        return responses.get(topic, responses['general'])

    def get_rule_based_response(self, topic: str) -> str:
        """Get pre-defined response for a topic, enhanced with literature"""
        response = self._get_basic_response(topic)
        
        # Enhance with literature insights if available
        if topic in LITERATURE_INSIGHTS and LITERATURE_INSIGHTS[topic]:
            # Add a quote
            if topic in LITERATURE_QUOTES:
                response += f"\n\nAccording to research by Uchendu et al. (2025): \"{LITERATURE_QUOTES[topic]}\"\n\n"
            
            # Add insights from literature
            response += "Research findings on this topic include:\n"
            for insight in LITERATURE_INSIGHTS[topic][:2]:  # Limit to 2 insights
                response += f"• {insight}\n"
        
        # If asking about research specifically, add paper details
        if topic == 'research':
            response += "\n\nKey research on HPV vaccination in Nigeria includes:\n"
            for source in LITERATURE_SOURCES:
                response += f"• {source['title']} ({source['year']}) by {source['authors']}\n"
                response += f"  Key finding: {source['key_findings'][0]}\n"
        
        # If asking about strategies, add recommendations
        if topic == 'strategies':
            response += "\n\nRecommended strategies from research include:\n"
            for rec in LITERATURE_RECOMMENDATIONS[:4]:  # Limit to 4 recommendations
                response += f"• {rec}\n"
        
        return response

    def format_response(self, response_text: str) -> str:
        """
        Format response text to improve readability
        
        Parameters:
        response_text (str): Raw response text
        
        Returns:
        str: Formatted response text
        """
        # 1. Split into paragraphs (honoring existing breaks)
        paragraphs = response_text.split("\n\n")
        
        # 2. Process each paragraph
        formatted_paragraphs = []
        for para in paragraphs:
            # Skip empty paragraphs
            if not para.strip():
                continue
                
            # Process paragraph
            para = para.strip()
            
            # Format bullet points for consistency
            if "•" in para or "-" in para or any(line.strip().startswith("*") for line in para.split("\n")):
                lines = para.split("\n")
                formatted_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line:
                        # Convert different bullet styles to consistent format
                        if line.startswith("- "):
                            line = "• " + line[2:]
                        elif line.startswith("* "):
                            line = "• " + line[2:]
                        elif line.startswith("-"):
                            line = "• " + line[1:]
                        elif line.startswith("*"):
                            line = "• " + line[1:]
                        formatted_lines.append(line)
                
                para = "\n".join(formatted_lines)
            
            formatted_paragraphs.append(para)
        
        # 3. Join paragraphs with double line breaks
        formatted_text = "\n\n".join(formatted_paragraphs)
        
        # 4. Ensure proper formatting of lists
        lines = formatted_text.split("\n")
        in_list = False
        for i in range(len(lines)):
            if lines[i].strip().startswith("•"):
                in_list = True
            elif in_list and not lines[i].strip().startswith("•") and lines[i].strip():
                # End of list, add extra spacing
                lines[i] = "\n" + lines[i]
                in_list = False
        
        formatted_text = "\n".join(lines)
        
        # 5. Format headings (text followed by colon at beginning of paragraph)
        formatted_text = re.sub(r'(?m)^([A-Za-z\s]+):(\s*)', r'**\1:**\2', formatted_text)
        
        # 6. Format key statistics with bold
        formatted_text = re.sub(r'(\d+(\.\d+)?%)', r'**\1**', formatted_text)
        
        return formatted_text

    def process_query(self, query: str, data_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a user query and return a response
        
        Parameters:
        query (str): User's question
        data_context (Dict): Optional context from data analysis
        
        Returns:
        Dict: Response with answer and other metadata
        """
        # We'll always use the rule-based approach now
        topics = self.identify_topics(query)
        response = self.get_rule_based_response(topics[0])
        
        # Customize response with data if available
        if data_context and "data_summary" in data_context:
            try:
                summary = data_context["data_summary"]
                
                # Add data-specific information to the response
                if topics[0] == 'education' and 'education_distribution' in summary:
                    response += f"\n\nIn your current dataset with {summary.get('total_records')} records, the education levels are distributed as follows: "
                    for edu, count in summary["education_distribution"].items():
                        response += f"{edu}: {count} records, "
                    response = response.rstrip(", ") + "."
                    
                elif topics[0] == 'regional' and 'region_distribution' in summary:
                    response += f"\n\nIn your current dataset with {summary.get('total_records')} records, the regional distribution is: "
                    for region, count in summary["region_distribution"].items():
                        response += f"{region}: {count} records, "
                    response = response.rstrip(", ") + "."
                    
                elif topics[0] == 'general':
                    response += f"\n\nYour current dataset contains {summary.get('total_records')} records with an overall vaccination rate of {summary.get('vaccination_rate', 0):.1f}%."
            except Exception as e:
                print(f"Error customizing response with data: {str(e)}")
                # Continue with basic response if customization fails
        else:
            # Mention that data is loaded but not provided in context
            response += "\n\n(Note: I have access to HPV vaccination survey data, but specific statistics weren't provided for this response.)"
        
        # Apply formatting before returning
        formatted_response = self.format_response(response)
        
        return {
            "answer": formatted_response,
            "source": "rule_based",
            "topics": topics
        }
        
    def _get_survey_info_response(self) -> str:
        """Get information about the original survey"""
        return (
            "The original survey was titled 'Acceptability of HPV Vaccine Among Nigerian Caregivers' "
            "and was administered via Facebook Messenger to caregivers of girls aged 9-17 in Adamawa, "
            "Abuja FCT, and Nasarawa states. The survey covered several sections including consent, "
            "knowledge about HPV vaccination, motivation to vaccinate, ability to access vaccines, "
            "and demographic information. Key questions assessed whether children had received the HPV "
            "vaccine, knowledge of where to get the vaccine, and attitudes toward vaccination importance."
        )
        
    def _get_comb_framework_response(self) -> str:
        """Get information about the COM-B framework as applied to this data"""
        return (
            "The COM-B framework suggests that behavior (B) occurs when capability (C), opportunity (O), "
            "and motivation (M) align. In our HPV vaccination data:\n\n"
            "- Capability includes knowledge of the HPV vaccine (heardhpvvax) and knowing where to get it (knowwhere)\n"
            "- Opportunity includes state of residence (State) and perceived difficulty (difficult)\n"
            "- Motivation includes perceived importance of vaccination (important) and likelihood of future vaccination (hpvlikely)\n\n"
            "Our data shows that all three components influence vaccination decisions, with exposure to "
            "information playing a particularly important mediating role."
        )
    
    def _get_literature_sources(self) -> str:
        """Return information about literature sources used in the chatbot"""
        response = "\n\n**Key Research Sources:**\n"
        for i, source in enumerate(LITERATURE_SOURCES[:3]):  # Limit to 3 sources
            response += f"{i+1}. {source['title']} ({source['year']}) - {source['citation']}\n"
        return response