import requests
import bs4
from tqdm.auto import tqdm
import pandas as pd
import ollama
import json
import numpy as np

def extract_data(soup, tag, class_name):
    """
    Searches for an HTML element by tag and class, then extracts its text.
    
    Parameters:
    - soup: BeautifulSoup object containing parsed HTML
    - tag: HTML tag name (e.g., 'h1', 'p', 'div')
    - class_name: CSS class name (or None if searching by tag only)
    
    Returns: Text content if element found, None otherwise
    """
    element = soup.find(tag, class_=class_name)
    return element.get_text() if element else None

def grab_gc_talk(data_dict, year, month):
    # Step 1: Download the conference index page
    base_url = "https://www.churchofjesuschrist.org/"
    response = requests.get(f"https://www.churchofjesuschrist.org/study/general-conference/{year}/{month}?lang=eng")

    # Step 2: Parse the HTML to find all talk links
    soup = bs4.BeautifulSoup(response.content, "html.parser")
    raw_urls = soup.find_all("a", class_="item-U_5Ca")  # Find all talk links
    urls = [a["href"] for a in raw_urls]
    
    # Skip the first link (it's usually not a talk)
    final_urls = urls[1:]
    print(f"Processing talks for {year}-{month}...")

    # Step 3: For each talk, download and extract the data
    for url in tqdm(final_urls):
        # Download the individual talk page
        talk = requests.get(base_url + url)
        talk_soup = bs4.BeautifulSoup(talk.content, "html.parser")
        
        # Extract each piece of information we need
        data_dict['title'].append(extract_data(talk_soup, 'h1', None))
        data_dict['author'].append(extract_data(talk_soup, 'p', 'author-name'))
        data_dict['role'].append(extract_data(talk_soup, 'p', 'author-role'))
        data_dict['kicker'].append(extract_data(talk_soup, 'p', 'kicker'))
        data_dict['text'].append(extract_data(talk_soup, 'div', 'body-block'))
        data_dict['year'].append(year)
        data_dict['month'].append(month)
    return data_dict

def clean_data(df):
    mapping = {
        "Relief Society": "RS",
        "First Presidency": "FP",
        "Apostle": "A",
        "Young Women": "YW",
        "Young Men": "YM",
        "Primary": "P",
        "Sunday School": "SS",
        "Seventy": "70",
        "Bishop": "B",
        "President of the Church": "Prophet",
        "Managing Director": "MD",
        "Ward": "Member"
    }

    for key, value in mapping.items():
        df.loc[df['role'].str.contains(key, case=False), 'role'] = value

    return df

def generate_embeddings(df):
    """Generate embeddings for text and store as JSON strings for CSV compatibility."""
    # Only generate embeddings for rows with text
    mask = df['text'].notnull()
    embeddings = df.loc[mask, 'text'].apply(
        lambda x: json.dumps(ollama.embed(model="embeddinggemma", input=x)["embeddings"][0])
    )
    df.loc[mask, 'embedding'] = embeddings
    return df

def load_embeddings_from_csv(csv_path):
    """Load CSV and convert JSON string embeddings back to numeric arrays."""
    df = pd.read_csv(csv_path)
    
    # Convert embedding JSON strings back to lists/arrays
    def parse_embedding(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return np.array(json.loads(value))
            except (json.JSONDecodeError, ValueError):
                return None
        return None
    
    df['embedding'] = df['embedding'].apply(parse_embedding)
    # Remove rows with invalid embeddings
    df = df.dropna(subset=['embedding']).reset_index(drop=True)
    return df

def data_pipeline(years=[2025], months=["04", "10"], save_csv=True, verbose=True):
    data_dict = {
        "title": [],      # Talk title
        "author": [],     # Speaker name
        "role": [],       # Speaker's position/responsibility
        "kicker": [],     # Summary/description
        "text": [],       # Full talk text
        "year": [],       # Conference year
        "month": []       # Conference month
    }

    if verbose:
        print("Starting data collection...")

    # Loop through the conferences we want to collect (April and October of 2024-2025)
    for year in years:
        for month in months:
            data_dict = grab_gc_talk(data_dict, year, month)

    # Step 4: Convert our dictionary into a pandas DataFrame for easier analysis
    df = pd.DataFrame(data_dict)

    if verbose:
        print("Cleaning data...")
    df = clean_data(df)

    if verbose:
        print("Generating embeddings...")
    df = generate_embeddings(df)
    if save_csv:
        if verbose:
            print("Saving to CSV...")
        df.to_csv("data/gc_talks.csv", index=False)
    return df