"""
This file contains metadata about the original survey questions and their relationship
to columns in the dataset
"""

SURVEY_QUESTIONS = {
    "consent": "Would you be willing to take the survey?",
    "numbergirls": "How many girls between the ages of 9 and 17 do you care for? (if any)?",
    "agegirl": "What is the age of the girl that you care for?",
    "youngest": "What is the age of the youngest of the girls that you care for?",
    "heardhpvvax": "Have you ever heard of the HPV vaccine?",
    "agegiven": "At what age should the HPV vaccine ideally be given?",
    "hpvax": "Has the girl-child in your care received an HPV vaccination?",
    "hpvlikely": "How likely is it that this girl-child in your care will receive an HPV vaccination in the next 12 months?",
    "important": "To what extent do you agree with the statement: \"Getting the girl who is in my care vaccinated against HPV is important to me.\"",
    "difficult": "To what extent do you agree with the statement: \"Getting the girl child who is in my care vaccinated against HPV is easy for me.\"",
    "knowwhere": "To what extent do you agree with the statement: \"I know where to get the HPV vaccine for the girl in my care\"",
    "gender": "What is your gender?",
    "age": "How old are you?",
    "education": "What is the highest educational level or degree that you have attained?",
    "seenads": "Have you seen or heard any advertisements or messages about HPV vaccine in the last 3 months?",
    "often": "How often have you seen advertisements or heard messages about HPV vaccine?",
    "State": "State of residence" # This was likely collected via Facebook targeting
}

# Answer options from the survey
SURVEY_OPTIONS = {
    "heardhpvvax": {"Yes": 1, "No": 2},
    "agegiven": {
        "9-14": "A",
        "15-19": "B",
        "20-24": "C", 
        "25+": "D",
        "Don't know/Can't say": "E"
    },
    "hpvax": {
        "Yes": "A",
        "No": "B",
        "I don't know": "C"
    },
    "hpvlikely": {
        "Very likely": "A",
        "Somewhat likely": "B",
        "No opinion, don't know": "C",
        "Somewhat unlikely": "D",
        "Very unlikely": "E"
    },
    "important": {
        "Strongly agree": "A",
        "Agree": "B",
        "Disagree": "C",
        "Strongly disagree": "D",
        "Don't know/Can't say": "E"
    },
    "difficult": {
        "Strongly agree": "A",
        "Agree": "B",
        "Disagree": "C",
        "Strongly disagree": "D",
        "Don't know/Can't say": "E"
    },
    "knowwhere": {
        "Strongly agree": "A",
        "Agree": "B",
        "No Opinion / Don't know": "C",
        "Disagree": "D",
        "Strongly disagree": "E"
    },
    "education": {
        "No formal education": 1,
        "Primary School Certificate": 2,
        "SSCE/GCE": 3,
        "OND": 4,
        "HND/BSc": 5,
        "Postgraduate degree": 6,
        "Other": 7
    },
    "seenads": {"Yes": "A", "No": "B"},
    "often": {
        "Only once": "A",
        "2-3 times": "B",
        "More than 3 times": "C"
    }
}

# COM-B framework mapping
COMB_MAPPING = {
    "Capability": {
        "description": "Knowledge and skills related to HPV vaccination",
        "columns": ["education", "heardhpvvax", "knowwhere", "agegiven"]
    },
    "Opportunity": {
        "description": "Environmental factors that enable or hinder vaccination",
        "columns": ["State", "difficult"]
    },
    "Motivation": {
        "description": "Beliefs, intentions, and desires related to vaccination",
        "columns": ["important", "hpvlikely"]
    }
}

# Survey section descriptions
SURVEY_SECTIONS = {
    "Knowledge": ["heardhpvvax", "agegiven", "hpvax", "hpvlikely"],
    "Motivation": ["important"],
    "Ability": ["difficult", "knowwhere"],
    "Profile": ["gender", "age", "education", "seenads", "often"]
}