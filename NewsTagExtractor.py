%%writefile NewsTagExtractor.py

import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from textblob import TextBlob
import textstat
from difflib import get_close_matches
import html
import re
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# Load Spacy model globally
nlp = spacy.load('en_core_web_lg')
nlp.Defaults.stop_words.update(["said", "says", "say", "upto", "come"])

class TagExtractor:
    def __init__(self, sheet_id: str):
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv'
        self.all_tag_df = pd.read_csv(url)
        self.all_tag_df['Tag'] = self.all_tag_df['Tag'].str.lower()
        self.all_tag_list = self.all_tag_df['Tag'].to_list()
        self.num_of_article = self.all_tag_df['#Articles'].to_list()
        self.tag_article_count = dict(zip(self.all_tag_df['Tag'], self.num_of_article))

    def preprocess_text(self, text) -> str:
        custom_stop_words = set(nlp.Defaults.stop_words) - {'and'}
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_punct and (token.text.lower() not in custom_stop_words)]
        return ' '.join(tokens)

    def find_matching_strings(self, list1: list, list2: list, threshold: float = 0.95, max_matches: int = 100) -> list:
        list1 = [str1.lower().strip() for str1 in list1]
        list2 = [str2.lower().strip() for str2 in list2]
        matching_strings = []
        for str1 in list1:
            matches = get_close_matches(str1, list2, n=max_matches, cutoff=threshold)
            matching_strings.extend([(match, str1) for match in matches])
        return [match[0] for match in matching_strings]

    def extract_tags(self, text: str, vocab: list = None, ngrams: tuple = (1, 4)) -> list:
        cleaned_text = self.preprocess_text(text)
        vectorizer = TfidfVectorizer(vocabulary=vocab, ngram_range=ngrams, max_features=100)
        tfidf_matrix = vectorizer.fit_transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        top_tags = [feature_names[idx] for idx in tfidf_scores.argsort()[::-1]]  
        return top_tags

    def calculate_sentiment(self, text: str) -> float:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def calculate_readability(self, text: str) -> float:
        return textstat.flesch_reading_ease(text)

    def filter_tags(self, tags: list) -> list:
        filtered_tags = []
        for tag in tags:
            tag = tag.strip()
            doc = nlp(tag)
            
            if len(tag.split()) == 1:
                pos_tags = [token.pos_ for token in doc]
                entities = [ent.label_ for ent in doc.ents]
                
                # Skip tags with specified POS tags or DATE entity
                if any(pos in {'INTJ', 'CCONJ', 'NUM', 'VERB', 'ADJ', 'ADV', 'ADP'} for pos in pos_tags) or \
                   any(ent in {'DATE'} for ent in entities):
                    continue
                
            filtered_tags.append(tag)       
        return filtered_tags
        
    def visualize_embeddings(self, tags: list, title: str) -> None:
        title_vector = nlp(title).vector
        tag_vectors = [nlp(tag).vector for tag in tags]
        all_vectors = np.array([title_vector] + tag_vectors)
        tags_with_title = ['Title'] + tags
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        reduced_vectors = tsne.fit_transform(all_vectors)
        df = pd.DataFrame(reduced_vectors, columns=['x', 'y'])
        df['tag'] = tags_with_title
        df['color'] = ['red'] + ['blue'] * len(tags)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.loc[df['color'] == 'blue', 'x'],
            y=df.loc[df['color'] == 'blue', 'y'],
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.8),
            hovertext=df.loc[df['color'] == 'blue', 'tag'],
            hoverinfo='text',
            name='Tags'
        ))
        fig.add_trace(go.Scatter(
            x=df.loc[df['color'] == 'red', 'x'],
            y=df.loc[df['color'] == 'red', 'y'],
            mode='markers',
            marker=dict(size=12, color='red', opacity=0.8),
            hovertext=df.loc[df['color'] == 'red', 'tag'],
            hoverinfo='text',
            name='Title'
        ))
        fig.update_layout(
            title='Tag and Title Embeddings Visualization',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            showlegend=True,
            legend_title='Legend'
        )
        st.plotly_chart(fig)

    def process_input(self, title: str, url: str, category: str, content: str) -> pd.DataFrame:
        data = defaultdict(list)
        try:
            title = html.unescape(title)
            content = re.sub(r"<blockquote.*?>.*?<\/blockquote>|<script.*?>.*?<\/script>|<a[^>]*><\/a>|<.*?>|(&[^;]+;)", " ", html.unescape(content)).replace('\\', "")
            match = re.match(r'.*/([^/]+)-\d+\.html', url)
            if match:
                extracted_string = match.group(1)
                split_string = ' '.join(extracted_string.split('-'))
            else:
                split_string = ''
            merged_content = f"{title} {split_string} {content}"
            word_count = len(merged_content.split())
            sentiment = self.calculate_sentiment(content)
            readability = self.calculate_readability(content)
            tags_list = []
            if category in ['Press Release', 'Partner Content']:
                tags_list = list(self.extract_tags(merged_content))
                tags_list.append('promotion')
            elif category == 'Lifestyle' and 'tarot card reading' in merged_content.lower():
                tags_list = list(self.extract_tags(merged_content))
                tags_list.extend(['tarot card readings', 'tarot card predictions'])
            elif category == 'Lifestyle' and ('cook' in title.lower() or 'recipe' in title.lower()):
                tags_list = list(self.extract_tags(merged_content))
                tags_list.append('cookery')
            elif category == 'Lifestyle' and 'skincare' in title.lower():
                tags_list = list(self.extract_tags(merged_content))
                tags_list.extend(['beauty'])
            elif category == 'India' and 'lottery' in title.lower() and 'winners' in title.lower():
                tags_list = list(self.extract_tags(merged_content))
                tags_list = [tag for tag in tags_list if not (len(tag.split()) == 1 and nlp(tag)[0].pos_ == 'NOUN')]
                tags_list.append('lottery results')
            else:
                tags_list = self.extract_tags(merged_content)
            joined_tag = ', '.join(tags_list)
            sim_from_taglist = self.find_matching_strings(tags_list, self.all_tag_list)
            sim_from_taglist = list(set(sim_from_taglist))
            filtered_tags = self.filter_tags(sim_from_taglist)
            tag_counts = {tag: self.tag_article_count.get(tag, 0) for tag in filtered_tags}
            
            data['Title'].append(title)
            data['URL'].append(url)
            data['Category'].append(category)
            data['Content'].append(content)
            data['WordCount'].append(word_count)
            data['Sentiment'].append(sentiment)
            data['Readability'].append(readability)
            data['TopKeywords'].append(tags_list)  # Updated to reflect the correct variable
            data['ExtractedTags'].append(joined_tag)
            data['FilteredTags'].append(filtered_tags)
            data['TagArticleCounts'].append(tag_counts)
        except Exception as e:
            st.error(f"Error processing input: {e}")
        return pd.DataFrame(data)

# Streamlit UI
def main():
    st.set_page_config(page_title="Tag Extractor", page_icon=":label:", layout="wide")
    st.title("📝 Tag Extractor")

    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        color: #1f77b4;
        font-weight: bold;
    }
    .tag {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Extract Tags with Advanced Filtering</p>', unsafe_allow_html=True)

    # Get input from the user
    st.sidebar.header("Input Details")
    title = st.sidebar.text_input("Enter the title:")
    url = st.sidebar.text_input("Enter the URL:")
    category = st.sidebar.selectbox("Select the category:", ["Press Release", "Partner Content", "Lifestyle", "India", "International", "Mumbai", "Thiruvananthapuram", "Chennai", "Entertainment", "Sports", "Other"])
    content = st.sidebar.text_area("Enter the content:")

    if st.sidebar.button("Generate Tags"):
        if title and url and category and content:
            sheet_id = '1STUzG08LqSKC4wxKtqB19qJ1J6iHhlkAQLwjReZmN8c'
            tag_extractor = TagExtractor(sheet_id)
            data_df = tag_extractor.process_input(title, url, category, content)

            if not data_df.empty:
                st.success("Tags generated successfully!")
                
                # Display the tags and details
                for index, row in data_df.iterrows():
                    st.subheader(f"Title: {row['Title']}")
                    st.write(f"URL: {row['URL']}")
                    st.write(f"Category: {row['Category']}")
                    st.write(f"Word Count: {row['WordCount']}")
                    st.write(f"Sentiment: {row['Sentiment']:.2f}")
                    st.write(f"Readability: {row['Readability']:.2f}")
                    st.write(f"Number of suggested tags: {len(row['FilteredTags'])}")

                    # Display Filtered Tags with square border
                    st.write("**Filtered Tags:**")
                    filtered_tags_html = ''.join(f'<span class="tag">{tag}</span>' for tag in row['FilteredTags'])
                    st.markdown(filtered_tags_html, unsafe_allow_html=True)

                    # Display the t-SNE plot
                    st.write('Visualization of Tags and Title:')
                    tags = row['FilteredTags']
                    if tags:
                        tag_extractor.visualize_embeddings(tags, row['Title'])
                    else:
                        st.warning('No tags to visualize.')

            else:
                st.error("No tags generated. Please check the input.")

        else:
            st.warning("Please fill out all fields.")
            
if __name__ == "__main__":
    main()
