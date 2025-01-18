
import pandas as pd

# Define the file path
file_path = 'C:/Users/user/Documents/ΠΜΣ DWS/NLP/project/labeled_songs.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Print the column names
print("Column names in the dataset:")
print(df.columns.tolist())

# Print the distinct values of the 'emotion' column
if 'emotion' in df.columns:
    print("\nDistinct values in the 'emotion' column:")
    print(df['emotion'].unique())
else:
    print("\nThe column 'emotion' does not exist in the dataset.")