from bs4 import BeautifulSoup as bs
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import html
import spacy
import pandas as pd
from textblob import TextBlob
from difflib import get_close_matches
import textstat
from collections import defaultdict
import time
import logging
from rich.console import Console
from rich.text import Text
from rich.table import Table
from concurrent.futures import ThreadPoolExecutor

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings
warnings.filterwarnings('ignore')

# Load spaCy model and update stop words
nlp = spacy.load('en_core_web_lg')
nlp.Defaults.stop_words.update(["said", "says", "say", "upto", "come"])

class TagExtractor:
    """
    A class for extracting, filtering, and processing tags from articles.
    
    Attributes:
        sheet_id (str): Google Sheets ID for fetching tag data.
        all_tag_df (pd.DataFrame): DataFrame containing all tags and their article counts.
        all_tag_list (list): List of all tags.
        all_tag_set (set): Set of all tags for quick lookup.
        num_of_article (list): List of article counts for each tag.
        tag_article_counts (dict): Dictionary mapping tags to their article counts.
        xml_cache (dict): Cache for storing fetched XML data.
    """
    
    def __init__(self, sheet_id: str):
        """
        Initializes the TagExtractor with data from a Google Sheet.
        
        Args:
            sheet_id (str): Google Sheets ID for fetching tag data.
        """
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv'
        self.all_tag_df = pd.read_csv(url)
        self.all_tag_df['Tag'] = self.all_tag_df['Tag'].str.lower()
        self.all_tag_list = self.all_tag_df['Tag'].to_list()
        self.all_tag_set = set(self.all_tag_list)
        self.num_of_article = self.all_tag_df['#Articles'].to_list()
        self.xml_cache = {}
        
        # Initialize the tag_article_counts dictionary
        self.tag_article_counts = dict(zip(self.all_tag_df['Tag'], self.num_of_article))

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the text by removing punctuation and stop words.
        
        Args:
            text (str): The text to preprocess.
        
        Returns:
            str: The preprocessed text.
        """
        custom_stop_words = set(nlp.Defaults.stop_words) - {'and'}
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_punct and (token.text.lower() not in custom_stop_words)]
        return ' '.join(tokens)

    def calculate_similarity(self, word1: str, word2: str) -> float:
        """
        Calculates the similarity between two words using spaCy.
        
        Args:
            word1 (str): The first word.
            word2 (str): The second word.
        
        Returns:
            float: The similarity score between the two words.
        """
        return nlp(word1).similarity(nlp(word2))

    def find_matching_strings(self, list1: list, list2: list, threshold: float = 0.95, max_matches: int = 100) -> list:
        """
        Finds matching strings between two lists based on a similarity threshold.
        
        Args:
            list1 (list): The first list of strings.
            list2 (list): The second list of strings.
            threshold (float): The similarity threshold for matching.
            max_matches (int): The maximum number of matches to find.
        
        Returns:
            list: A list of matched strings.
        """
        list1 = {str1.lower().strip() for str1 in list1}
        list2 = {str2.lower().strip() for str2 in list2}
        matches = set()
        
        for str1 in list1:
            close_matches = get_close_matches(str1, list2, n=max_matches, cutoff=threshold)
            matches.update(close_matches)
        
        return list(matches)

    def calc_similarity_from_list(self, tags: list, ex_tags: str) -> dict:
        """
        Calculates similarity between tags and existing tags.
        
        Args:
            tags (list): The list of tags to compare.
            ex_tags (str): The existing tags as a comma-separated string.
        
        Returns:
            dict: A dictionary with similarity scores.
        """
        similarity_dict_curr_tag = {}
        ex_tags_list = ex_tags.split(', ')
        tags_lower = [tag.lower() for tag in tags]
        
        for word1 in ex_tags_list:
            word1_lower = word1.lower()
            max_similarity = 0.0
            max_similar_word = None
            for word2 in tags_lower:
                similarity = self.calculate_similarity(word1_lower, word2)
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similar_word = word2
                if similarity == 1.0:
                    similarity_dict_curr_tag[f"{word1_lower} <-> {word2}"] = similarity
            if max_similarity > 0.8:
                similarity_dict_curr_tag[f"{word1_lower} <-> {max_similar_word}"] = max_similarity
        return similarity_dict_curr_tag

    def extract_tags(self, text: str, vocab: list = None, ngrams: tuple = (1, 4)) -> list:
        """
        Extracts tags from the text using TF-IDF vectorization.
        
        Args:
            text (str): The text to extract tags from.
            vocab (list, optional): List of words to use as vocabulary.
            ngrams (tuple, optional): Tuple specifying the range of n-grams.
        
        Returns:
            list: A list of extracted tags.
        """
        cleaned_text = self.preprocess_text(text)
        vectorizer = TfidfVectorizer(vocabulary=vocab, ngram_range=ngrams, max_features=50)
        tfidf_matrix = vectorizer.fit_transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        top_tags = [feature_names[idx] for idx in tfidf_scores.argsort()[::-1]]
        return top_tags

    def calculate_sentiment(self, text: str) -> float:
        """
        Calculates the sentiment polarity of the text.
        
        Args:
            text (str): The text to analyze.
        
        Returns:
            float: The sentiment polarity score.
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def calculate_readability(self, text: str) -> float:
        """
        Calculates the readability score of the text.
        
        Args:
            text (str): The text to analyze.
        
        Returns:
            float: The readability score.
        """
        return textstat.flesch_reading_ease(text)

    def get_pos_and_entities(self, text: str):
        """
        Extracts part-of-speech tags and named entities from the text.
        
        Args:
            text (str): The text to analyze.
        
        Returns:
            tuple: A tuple containing lists of POS tags and named entities.
        """
        doc = nlp(text)
        pos_tags = [f"{token.text} ({token.pos_})" for token in doc]
        entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
        return pos_tags, entities
        
    def filter_tags(self, tags: list) -> list:
        """
        Filters and sorts tags based on part-of-speech and entity types.
        
        Args:
            tags (list): The list of tags to filter.
        
        Returns:
            list: A sorted list of filtered tags.
        """
        filtered_tags = []
    
        for tag in tags:
            tag = tag.strip()
            doc = nlp(tag)
    
            if len(tag.split()) == 1:
                pos_tags = [token.pos_ for token in doc]
                entities = [ent.label_ for ent in doc.ents]
    
                if any(pos in {'INTJ', 'CCONJ', 'NUM', 'ADJ', 'ADV', 'ADP'} for pos in pos_tags):
                    continue
                if 'DATE' in entities:
                    continue
    
            filtered_tags.append(tag)
    
        filtered_tags = sorted(set(filtered_tags), key=lambda x: x.lower())
        filtered_tags_with_counts = [(tag, self.tag_article_counts.get(tag, 0)) for tag in filtered_tags]
        sorted_filtered_tags = sorted(filtered_tags_with_counts, key=lambda x: x[1], reverse=True)
        return [tag for tag, count in sorted_filtered_tags]
        
    def call_xml(self, lnk: str) -> bs:
        """
        Fetches and parses XML data from a URL, with caching.
        
        Args:
            lnk (str): The URL to fetch XML from.
        
        Returns:
            bs: A BeautifulSoup object containing the XML data, or None if there's an error.
        """
        if lnk in self.xml_cache:
            logging.info(f"Using cached XML for {lnk}")
            return self.xml_cache[lnk]
        
        try:
            source = requests.get(lnk)
            if source.status_code == 200:
                root = bs(source.text, 'xml')
                self.xml_cache[lnk] = root  # Cache the fetched XML
                return root
            else:
                logging.error(f"Error fetching XML from {lnk}: Status Code {source.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching XML from {lnk}: {e}")
            return None

    def process_items(self, items: list) -> pd.DataFrame:
        """
        Processes a list of XML items and extracts relevant data.
        
        Args:
            items (list): List of BeautifulSoup XML item objects.
        
        Returns:
            pd.DataFrame: A DataFrame containing extracted data.
        """
        data = defaultdict(list)
        
        def process_item(item):
            try:
                title = html.unescape(item.find('Title').text)
                publish_date = item.find('PublishDate').text
                link = item.find('Link').text.replace("https://rss.", "https://rss1.")
                web_link = item.find('WebLink').text
                summary = item.find('Summary').text
                category_name = item.find('CategoryName').text
                parent_cat_name = item.find('ParentCategoryName').text
                existing_tags = item.find('Tags').text
    
                content = ""
                xml_root = self.call_xml(link)
                if xml_root:
                    content_elem = xml_root.find('Content')
                    if content_elem:
                        content = re.sub(r"<blockquote.*?>.*?<\/blockquote>|<script.*?>.*?<\/script>|<a[^>]*><\/a>|<.*?>|(&[^;]+;)", " ", html.unescape(content_elem.text)).replace('\\', "")
    
                match = re.match(r'.*/([^/]+)-\d+\.html', web_link)
                split_string = ' '.join(match.group(1).split('-')) if match else ''
                merged_content = f"{title} {split_string} {content}"
    
                word_count = len(merged_content.split())
                sentiment = self.calculate_sentiment(content)
                readability = self.calculate_readability(content)
    
                if category_name in ['Press Release', 'Partner Content']:
                    tags_list = list(self.extract_tags(merged_content)) + ['promotion']
                elif category_name == 'Lifestyle' and 'tarot card reading' in merged_content.lower():
                    tags_list = list(self.extract_tags(merged_content)) + ['tarot card readings', 'tarot card predictions']
                elif category_name == 'Lifestyle' and 'skincare' in split_string.lower():
                    tags_list = list(self.extract_tags(merged_content)) + ['beauty']
                elif category_name == 'Lifestyle' and ('cook' in title.lower() or 'recipe' in title.lower()):
                    tags_list = list(self.extract_tags(merged_content)) + ['cookery']
                elif category_name == 'India' and 'lottery' in content.lower() and 'winners' in content.lower():
                    tags_list = list(self.extract_tags(merged_content))
                    tags_list = [tag for tag in tags_list if not (len(tag.split()) == 1 and nlp(tag)[0].pos_ == 'NOUN')]
                    tags_list.append('lottery results')
                else:
                    tags_list = self.extract_tags(merged_content)
    
                joined_tag = ', '.join(tags_list)
                sim_from_tags = self.calc_similarity_from_list(tags_list, existing_tags)
                sim_from_taglist = self.find_matching_strings(tags_list, self.all_tag_list)
                filtered_tags = self.filter_tags(sim_from_taglist)
                
                data['PublishDate'].append(publish_date)
                data['Title'].append(title)
                data['Link'].append(link)
                data['WebLink'].append(web_link)
                data['Summary'].append(summary)
                data['Content'].append(content)
                data['CategoryName'].append(category_name)
                data['ParentCategoryName'].append(parent_cat_name)
                data['ExistingTags'].append(existing_tags)
                data['WordCount'].append(word_count)
                data['Sentiment'].append(sentiment)
                data['Readability'].append(readability)
                data['NewTags'].append(joined_tag)
                data['Similarity'].append(sim_from_tags)
                data['SimilarTagsFromList'].append(filtered_tags)
        
                logging.info(f"Processed item: {title[:50]}...")
            except Exception as e:
                logging.error(f"Error processing item: {e}")
        
        # Use ThreadPoolExecutor to parallelize item processing
        with ThreadPoolExecutor(max_workers=min(10, len(items))) as executor:
            executor.map(process_item, items)
        
        return pd.DataFrame(data)

    def print_tags(self, row: pd.Series) -> None:
        """
        Prints detailed information about tags using the Rich library.
        
        Args:
            row (pd.Series): A DataFrame row containing tag information.
        """
        console = Console()
    
        console.print(f"\nTitle: {row['Title']}", style="bold blue")
        console.print(f"Link: {row['Link']}")
        console.print(f"WebLink: {row['WebLink']}")
        console.print(f"Category: {row['CategoryName']}")
        console.print(f"Parent Category: {row['ParentCategoryName']}")
        console.print(f"Existing Tags: {row['ExistingTags']}")
    
        existing_tags_set = set(tag.strip().lower() for tag in row['ExistingTags'].split(', '))
    
        similar_tags = row['SimilarTagsFromList']
        similar_tags_list = similar_tags.split(', ') if isinstance(similar_tags, str) else similar_tags
    
        table_similar = Table(title="Similar Tags with POS, Entity Details, and Article Count")
        table_similar.add_column("[bold]Tag[/bold]", justify="left")
        table_similar.add_column("[bold]POS[/bold]", justify="left")
        table_similar.add_column("[bold]Entities[/bold]", justify="left")
        table_similar.add_column("[bold]Article Count[/bold]", justify="right")
    
        for tag in sorted(similar_tags_list):
            tag = tag.strip()
            pos_tags, entities = self.get_pos_and_entities(tag)
    
            pos_tags_str = ', '.join(pos_tags)
            entities_str = ', '.join(entities)
            article_count = self.tag_article_counts.get(tag, 0)
    
            if tag.lower() in existing_tags_set:
                table_similar.add_row(f"[bold grey93 on grey15]{tag}[/bold grey93 on grey15]",
                                      f"[bold grey93 on grey15]{pos_tags_str}[/bold grey93 on grey15]",
                                      f"[bold grey93 on grey15]{entities_str}[/bold grey93 on grey15]",
                                      f"[bold grey93 on grey15]{article_count}[/bold grey93 on grey15]")
            else:
                table_similar.add_row(f"[bright_black]{tag}[/bright_black]",
                                      f"[bright_black]{pos_tags_str}[/bright_black]",
                                      f"[bright_black]{entities_str}[/bright_black]",
                                      f"[bright_black]{article_count}[/bright_black]")
    
        console.print(table_similar)

    def run(self, main_link: str) -> pd.DataFrame:
        """
        Runs the tag extraction and processing workflow.
        
        Args:
            main_link (str): The URL of the main XML feed.
        
        Returns:
            pd.DataFrame: A DataFrame containing processed data.
        """
        root = self.call_xml(main_link)
        if not root:
            return pd.DataFrame()
        
        items = root.find_all('Item')
        data_df = self.process_items(items)
        
        for index, row in data_df.iterrows():
            self.print_tags(row)
        
        return data_df
        
# Usage
console = Console()
if __name__ == "__main__":
    sheet_id = '1STUzG08LqSKC4wxKtqB19qJ1J6iHhlkAQLwjReZmN8c'
    main_link = 'https://rss1.oneindia.com/xml4apps/www.oneindia.com/latest.xml'

    tag_extractor = TagExtractor(sheet_id)
    data_df = tag_extractor.run(main_link)
