"""
Module for automatically loading HPV survey data and training the model
without requiring user upload through the UI
"""

import os
import pandas as pd
from hpv_data_analysis import HPVVaccinationChatbot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoDataLoader:
    """Handles automatic loading of data and model training"""
    
    def __init__(self, data_path=None):
        """
        Initialize the auto data loader
        
        Parameters:
        data_path (str): Path to the data file or directory. If None, uses default locations
        """
        self.chatbot = HPVVaccinationChatbot()
        self.data_path = data_path
        
        # Default data paths to check (in order of preference)
        self.default_data_paths = [
            os.path.join('data', 'hpv_survey_data.csv'),
            os.path.join('data', 'hpv_survey_synthetic_data.csv'),
            os.path.join('temp', 'hpv_survey_data.csv'),
        ]
    
    def load_data(self):
        """
        Automatically load data from available sources
        
        Returns:
        bool: Success indicator
        """
        # If specific path is provided, try that first
        if self.data_path and os.path.exists(self.data_path):
            logger.info(f"Loading data from specified path: {self.data_path}")
            return self.chatbot.load_data(self.data_path)
        
        # Otherwise, try default paths
        for path in self.default_data_paths:
            if os.path.exists(path):
                logger.info(f"Loading data from default path: {path}")
                success = self.chatbot.load_data(path)
                if success:
                    logger.info(f"Successfully loaded data from {path}")
                    return True
                else:
                    logger.warning(f"Failed to load data from {path}")
        
        # If we couldn't find data, try generating synthetic data
        logger.warning("No data found in default locations. Generating synthetic data...")
        try:
            from synthetic_data_generator import generate_synthetic_hpv_data, save_synthetic_data
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Generate synthetic data
            synthetic_df = generate_synthetic_hpv_data(num_samples=2000)
            
            # Save to default location
            synthetic_path = os.path.join('data', 'hpv_survey_synthetic_data.csv')
            save_synthetic_data(synthetic_df, synthetic_path)
            
            # Load the generated data
            success = self.chatbot.load_data(synthetic_path)
            if success:
                logger.info(f"Successfully generated and loaded synthetic data from {synthetic_path}")
                return True
            else:
                logger.error("Failed to load generated synthetic data")
                return False
                
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            return False
            
        return False
    
    def train_model(self):
        """
        Train the model on the loaded data
        
        Returns:
        bool: Success indicator
        """
        if not self.chatbot.data is not None:
            logger.warning("No data loaded. Loading data first...")
            success = self.load_data()
            if not success:
                logger.error("Failed to load data for model training")
                return False
        
        logger.info("Training model on loaded data...")
        success = self.chatbot.train_model()
        
        if success:
            logger.info("Model trained successfully")
        else:
            logger.error("Model training failed")
            
        return success
    
    def get_chatbot(self):
        """
        Get the initialized and trained chatbot
        
        Returns:
        HPVVaccinationChatbot: The chatbot instance with data loaded and model trained
        """
        return self.chatbot