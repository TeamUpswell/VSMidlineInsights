from hpv_data_analysis import HPVVaccinationChatbot
import sys

def test_model_training(data_file):
    """Test data loading and model training directly"""
    
    try:
        # Initialize the chatbot
        print(f"Initializing HPV Vaccination Chatbot...")
        chatbot = HPVVaccinationChatbot()
        
        # Load data
        print(f"Loading data from {data_file}...")
        success = chatbot.load_data(data_file)
        
        if not success:
            print("Failed to load data. Exiting.")
            return False
            
        # Print data summary
        print(f"\nData summary:")
        print(f"  - Rows: {len(chatbot.data)}")
        print(f"  - Columns: {len(chatbot.data.columns)}")
        print(f"  - Column names: {', '.join(chatbot.data.columns.tolist())}")
        
        # Train model
        print("\nTraining model...")
        success = chatbot.train_model()
        
        if not success:
            print("Failed to train model. Exiting.")
            return False
            
        print("\nModel training successful!")
        return True
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Get data file from command line or use default
    data_file = sys.argv[1] if len(sys.argv) > 1 else "test_hpv_data.csv"
    test_model_training(data_file)