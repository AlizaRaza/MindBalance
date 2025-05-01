import pandas as pd  # Import pandas for data manipulation

# Define the path to the dataset file
DATASET_PATH = 'Student Mental health.csv'

# Function to load the dataset
def load_dataset():
    try:
        # Attempt to read the CSV file
        dataset = pd.read_csv(DATASET_PATH)
        # Print success message with dataset size
        print(f"Dataset loaded successfully with {len(dataset)} rows.")
        # Return the loaded dataset
        return dataset
    except Exception as e:
        # Handle any errors that occur during loading
        print(f"Error loading dataset: {e}")
        # Return None if loading fails
        return None

# This block runs when the script is executed directly (not imported)
if __name__ == "__main__":
    print("Testing dataset_loader.py...")
    # Call the load_dataset function
    dataset = load_dataset()
    # If dataset was loaded successfully
    if dataset is not None:
        # Display the first few rows
        print(dataset.head())  # Display the first few rows of the dataset
    else:
        # Display error message if loading failed
        print("Dataset could not be loaded.")
