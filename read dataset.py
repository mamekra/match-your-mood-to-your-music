import pandas as pd

# Path to the downloaded CSV file
file_path = r'C:/Users/user/Documents/ΠΜΣ DWS/NLP/project/song_lyrics.csv'

# Read only the first 1,000,000 rows of the CSV file
df = pd.read_csv(file_path, nrows=100_000)

# Path to save the new CSV file
output_file_path = r'C:/Users/user/Documents/ΠΜΣ DWS/NLP/project/song_lyrics_subset.csv'

# Write the DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

# Print confirmation
print(f'New CSV file saved to: {output_file_path}')

print(df.head())