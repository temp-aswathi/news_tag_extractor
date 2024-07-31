# News tag extractor

## Overview
`NewsTagExtractor.py` is a Streamlit application designed for extracting and visualizing tags from news articles. The app processes input text to generate relevant tags, calculate sentiment and readability scores, and display results in an interactive UI.

## Features
- **Preprocessing:** Cleans and preprocesses the input text.
- **Tag Extraction:** Uses TF-IDF vectorization to extract top tags from the input text.
- **Sentiment Analysis:** Calculates the sentiment polarity of the content.
- **Readability Calculation:** Computes the readability score of the content.
- **Tag Filtering:** Filters tags based on part-of-speech (POS) tags and named entity types.
- **Embedding Visualization:** Visualizes tag and title embeddings using t-SNE.
- **Streamlit UI:** Provides an interactive user interface for input and displaying results.
  
In the visualisation, when we hover over each point, we will be able to see the tags, and the red dot is the `vector embedding` of the title

[Output.pdf](https://github.com/user-attachments/files/16436311/Output.pdf)

## Installation

Clone the repository:
```sh
git clone https://github.com/your-repo/NewsTagExtractor.git
cd NewsTagExtractor
```
Install the required packages:
```sh
pip install -r requirements.txt
```
Download the Spacy model:
```sh
python -m spacy download en_core_web_lg
```

## Usage
Run the Streamlit app:
```sh
streamlit run NewsTagExtractor.py
```

Fill in the input details:
- Title: Enter the title of the news article.
- URL: Provide the URL of the news article.
- Category: Select the category from the dropdown.
- Content: Paste the content of the news article.
  
Generate Tags:
- Click the "Generate Tags" button to process the input and generate tags.
  
## Code Structure
### TagExtractor Class
- `__init__(self, sheet_id: str)`: Initializes the class with data from a Google Sheets CSV.
- `preprocess_text(self, text: str) -> str`: Preprocesses the input text.
- `find_matching_strings(self, list1: list, list2: list, threshold: float = 0.95, max_matches: int = 100) -> list`: Finds close matches from list1 in list2.
- `extract_tags(self, text: str, vocab: list = None, ngrams: tuple = (1, 4)) -> list`: Extracts top tags from the input text using TF-IDF vectorization.
- `calculate_sentiment(self, text: str) -> float`: Calculates sentiment polarity of the input text.
- `calculate_readability(self, text: str) -> float`: Calculates readability score of the input text.
- `filter_tags(self, tags: list) -> list`: Filters tags based on POS tags and entity types.
- `visualize_embeddings(self, tags: list, title: str) -> None`: Visualizes the embeddings of tags and title using t-SNE.
- `process_input(self, title: str, url: str, category: str, content: str) -> pd.DataFrame`: Processes the input data and generates a DataFrame with tags and additional information.
 
### Streamlit UI
The main function runs the Streamlit app, providing an interface for user input and displaying results.

## Example
```sh
streamlit run NewsTagExtractor.py
```
- Enter the title, URL, category, and content of the news article in the sidebar.
- Click "Generate Tags" to extract and display the tags along with sentiment, readability, and visualization.
  
## Future Enhancements
Save output as JSON or store in a database.
Improve tag extraction accuracy.
