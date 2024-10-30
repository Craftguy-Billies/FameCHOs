import feedparser
import json
import ast
from openai import OpenAI
import trafilatura
from bs4 import BeautifulSoup
import requests
import os
import time
import re
import threading
import uuid
import random
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from datetime import datetime, timedelta
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring, parse, fromstring, ElementTree
import urllib.parse
import pytz

DEBUG = False

lines = 17

model = "meta/llama-3.1-405b-instruct"

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

def extract_list_content(input_string):
    start_index = input_string.find("[")
    end_index = input_string.rfind("]")

    if start_index != -1 and end_index != -1 and start_index < end_index:
        try:
            return ast.literal_eval(input_string[start_index:end_index + 1])
        except (ValueError, SyntaxError):
            return []
    else:
        return []

def extract_json_content(input_string):
    start_index = input_string.find("{")
    end_index = input_string.rfind("}")

    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_string = input_string[start_index:end_index + 1]

        try:
            return json.loads(json_string)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error decoding JSON: {e}")
            return {}
    else:
        print("No valid JSON found in the input string.")
        return {}

def prettify_element(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def get_first_youtube_embed(query, model, max_retries = 3):
    prompt = f"""
    For this news article title: {query}
    Generate me a short but concise youtube search query,(for example summarize the title into a main topic or short sentence, details can be ommited) such that optimally I can search of the exact issue in youtube results.
    The query can be in chinese or english. but make sure it is in moderate length that can get optimal search results.
    Return me a JSON object with single key "query", without premable and explanation.
    Again, only return me ONE JSON OBJECT with single key QUERY without premable and explanation.
    """

    completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                stream=True
            )
           
    refined_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            refined_response += chunk.choices[0].delta.content

    modified_string = extract_json_content(refined_response)
    print(modified_string)
    if isinstance(modified_string, dict):
        query = modified_string['query']
    else:
        query = "kpop music instrumental"
   
    # Set up Chrome options for Selenium
    chrome_options = Options()
    chrome_options.binary_location = r'/usr/bin/google-chrome'
    chrome_options.add_argument("--headless")  # Headless mode
    chrome_options.add_argument("--no-sandbox")  # Required for some CI environments
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
   
    # Start the browser
    driver = webdriver.Chrome(options=chrome_options)
   
    # Format the YouTube search URL
    search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
   
    # Load the page
    driver.get(search_url)
   
    try:
        # Wait for the first video link to be present, up to 10 seconds
        first_video = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//a[@href and contains(@href, "/watch?v=")]'))
        )
       
        # Get the video URL and extract the video ID
        video_url = f"https://www.youtube.com{first_video.get_attribute('href')}"
        video_url = video_url.split('&')[0]  # Clean up extra parameters
        video_id = video_url.split('v=')[1]
       
        # Construct the embed code
        embed_code = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
       
    except (NoSuchElementException, TimeoutException):
        # Handle the case where no video is found or page takes too long to load
        embed_code = ""
   
    # Close the browser
    driver.quit()
   
    return embed_code

def fetch_news(rss_urls):
    news_items = []
    seen_titles = set()

    # Get the current time and the time two weeks ago
    now = datetime.now()
    two_weeks_ago = now - timedelta(weeks=2)

    for url, source, category in rss_urls:
        feed = feedparser.parse(url)

        for entry in feed.entries:
            # Check if the entry has a publication date
            if 'published_parsed' in entry:
                if entry.published_parsed:
                    # Convert the published date to a datetime object
                    pub_date = datetime(*entry.published_parsed[:6])
                    # Only consider articles published within the last two weeks
                    if pub_date < two_weeks_ago or pub_date > now:
                        continue
                else:
                  continue

            # Check for duplicates based on title
            if entry.title not in seen_titles:
                news_item = {
                    'title': entry.title,
                    'link': entry.link,
                    'summary': entry.summary,
                    'source' : source,
                    'category': category
                }
               
                # Fetch the article content
                downloaded = trafilatura.fetch_url(entry.link)
                if downloaded:
                    website_text = trafilatura.extract(downloaded)
                    if website_text:
                        word_count = len(website_text.split())
                        # Filter out articles that are too short or too long
                        if word_count <= 300 or word_count >= 1300:
                            continue
               
                # Add the news item to the list
                news_items.append(news_item)
                seen_titles.add(entry.title)

    return news_items

def split_article_into_segments(article, lines_per_segment=13):
    lines = article.split('\n')
    segments = [lines[i:i + lines_per_segment] for i in range(0, len(lines), lines_per_segment)]

    if len(segments[-1]) < 3:
        segments.pop()
    return segments

def count_newlines_exceeds_limit(text: str, limit: int = 5) -> bool:
    newline_count = text.count('\n')
    return newline_count > limit

def search(query, max_results = 8):
    encoded_query = requests.utils.quote(query)
    url = f"https://www.google.com/search?q={encoded_query}&gl=hk"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Send the request
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')
    results = []

    for g in soup.find_all('div', class_='tF2Cxc')[:max_results]:
        title = ''
        snippet = ''
        link = ''

        # Extract the title
        if g.find('h3'):
            title = g.find('h3').text
       
        # Extract the snippet
        if g.find('span', class_='aCOpRe'):
            snippet = g.find('span', class_='aCOpRe').text
        elif g.find('div', class_='IsZvec'):
            snippet = g.find('div', class_='IsZvec').text
        elif g.find('div', class_='VwiC3b'):
            snippet = g.find('div', class_='VwiC3b').text
        elif g.find('div', class_='s3v9rd'):
            snippet = g.find('div', class_='s3v9rd').text

        # Extract the URL
        if g.find('a'):
            link = g.find('a')['href']

        # Append the result as a dictionary
        results.append({
            'title': title,
            'snippet': snippet,
            'link': link
        })

    return results
def organize(word, description, results, model, max_retries=3):
    prompt = f"""
    The word to be translated: {word}
    The description of this word: {description}
    Web Search Context: {results}

    Summarize the search result.
    As long as the search result is NOT 100 PERCENT SURE IT IS CORRECT (eg, do not contain the translations with brackets in Chinese), return me the original word as key-value pair.
    Otherwise, only return the MUST correct Chinese translation of that noun i needed (eg translation from Wikipedia).
    return me a JSON object that stores the original word and search result word in key-value pairs.
    REMEMBER: If there isn't a Chinese name found, return me the original name, do not phonetically translate or translate with the original word's english meaning.
    Do not leave the key-value pair blank in any cases. return same word as key-value pair if no correct translation match.
    if there is more than one translation, only return me one.
    REMEMBER: the JSON returned has only ONE key-value pair. the JSON object has 1 key-value item ONLY.
    REMEMBER: DO NOT translate the lyrics or official terms that are bracketed.
    Return the JSON with ONE key-value pair with no preamble or explanation. 
    YOU MUST ORGANIZE ME AN ANSWER. DO NOT RETURN NON JSON REPLIES.
    """

    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                stream=True
            )
           
            refined_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    refined_response += chunk.choices[0].delta.content

            modified_string = extract_json_content(refined_response)
            return modified_string

        except Exception as e:
            retries += 1

    return {}

def websearch(word, description, model, max_retries=3):
    prompt = f"""
    You are an expert in crafting web search queries to get the accurate translation of unknown nouns into Chinese.
    The aim is to search out the Chinese name of the unknown noun.
    The search query should also include the description of that noun if it helps to define the Chinese name.
    make sure your chinese (traditional) words in the query DOES MAKE SENSE. DO NOT give me a search query with uncommon or unrelated chinese vocabularies.
    A query structure MUST include the original word in that language, and at least one chinese supporting description (e.g. place names, names of people, games, store names, etc.). just ensure they are relevant to search for the correct translation.
   
    Don't put in irrelevant words, as this will affect the accuracy of the search results!
    Also, don't search entirely in Chinese, as this will not find the correct translation!
   
    Make sure the search query is not too long. (at most 5 chinese characters are maximum)
    AGAIN: at most 6 chinese characters are maximum
    AGAIN: make sure your Chinese words in the query DOES MAKE SENSE.
    REMEMBER: prioritize the use of the translation of Wikipedia!! (inside brackets) If the search does not have Wikipedia, return the original word.
    Return the JSON with a single key 'query' with no preamble or explanation. REMEMBER: the JSON returned has only ONE key-value pair with a single key 'query' with no preamble or explanation

    AGAIN: at most 6 chinese characters.
    Word to transform into a query: {word}
    The description of this word: {description}
    """

    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                stream=True
            )
           
            refined_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    refined_response += chunk.choices[0].delta.content

            modified_string = extract_json_content(refined_response)
            if isinstance(modified_string, dict):
                return modified_string

        except Exception as e:
            retries += 1

    return {"query": ""}

def is_valid_dict(refined):
    """Check if the refined response is a valid dictionary."""
    return isinstance(refined, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in refined.items())

def process_segments(segments, model, max_retries=3):
    translated = {}
    for segment in segments:
        retries = 0
        while retries < max_retries:
            refined = refine_response(segment, translated, model)
            if is_valid_dict(refined):
                translated.update(refined)
                break
            else:
                retries += 1
                if retries >= max_retries:
                    print(f"Failed to refine response for segment: {segment}")

    # web search
    translated_ver = {}
    print(translated)
    for key, value in translated.items():
        query = websearch(key, value, model)
        print(query)
        if not query.get("query"):
            continue
        results = search(query["query"])
        if not results:
            continue
        final = organize(key, value, results, model)
        print(translated_ver)

        if isinstance(final, dict):
            translated_ver.update(final)
        else:
            print(f"Invalid result for {key}: {final}")

    return translated_ver

def refine_response(segment, translated, model):
    prompt = f"""
    Here is the initial list object:
    {segment}

    - The list above contains a segment of a big HTML article file.
    - Now I need to turn it into a Chinese article. But before that, there are some specific nouns that require web search to get an accurate translation.
    - Your job is to scan through the array, understand the full context and extract important nouns ONLY that cannot be translated directly.
    - All non-well-known nouns including personal nouns (e.g. blogger's son's name, a French blogger website name, etc) should NOT be extracted. DO NOT return me these nouns for extra web search.
    - For other human names, if they are well known and can be found on web, return me the full name to be searched.
    - If the name is not their official real name (e.g. artist stage name), do not return those name.
    - DO NOT return abbreviations as they cannot be searched. Return me the accurate full name of the person ONLY, NOT the simplified surname that appears in other paragraphs (e.g. Understand that "Cheung" is referred as "Cheung Ka Long" but not another person)
    - If the word is common and you can translate it correctly, do not return that word for extra web search work.
    - There might be some idioms in the article,  return these words if it is not logical in direct translation.
    - Return me a JSON object contain the noun and it's background information for web search as the key-value pairs.
    - If there are no words required for web search translation, return me a single "None". It is better to return fewer words.
    - REMEMBER: if there is a special noun with capitalization with are not logical in meaning, please return that noun as well even it is seemingly a normal word.
    - Return me a JSON object contain the noun and it's background information for web search as the key-value pairs with no premable or explanation.

    Previously translated words:
    {translated}

    Return the JSON object or a single word "None" ONLY.
    """

    completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            stream=True
        )

    refined_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            refined_response += chunk.choices[0].delta.content

    if refined_response:
        if str(refined_response).lower() != "none":
            refined_response_json = extract_json_content(refined_response)
            return refined_response_json
        return None

    return None

def clean_title(title, extension, directory=None):
    # Sanitize the title by replacing invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    clean_title = re.sub(invalid_chars, '_', title)
    clean_title = clean_title.strip()

    # Handle reserved file names
    reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}
    if clean_title.upper() in reserved_names:
        clean_title = f"{clean_title}_file"
   
    # Default filename if title is empty
    if not clean_title:
        clean_title = "default_filename"
   
    # Ensure the extension starts with a '.'
    if not extension.startswith('.'):
        extension = f".{extension}"
   
    # Combine the cleaned title with the extension
    file_name = f"{clean_title}{extension}"

    # If a directory is provided, combine the directory with the file name
    if directory:
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, file_name)
   
    # If no directory is provided, just return the file name
    return file_name

def consideration_test(title, segment, dictionary, model):
    full_article = ""
    prompt = f"""
    改寫並翻譯成日常用語繁體中文版本。
    刪除原文的所有圖片說明
    改寫必須合理，需要文句通順。人性化，不可以太浮誇和誇張
    語氣：專業、資訊性、具說服力
    身份：一個新聞作者想要帶資訊給讀者
    刪除所有作者自己的身份描述。（作者的家人名稱、工作地點、懷孕狀況等全部刪除），並改爲符合身份的描述。
    要求：把原文內容翻譯成繁體中文，不能自行創作，要詳細。翻譯時必須先了解整句話的意思，不要按字詞意思直接翻譯。
    要求：如果翻譯不是必須，可以不用翻譯某些字詞。
    要求：改寫一切網站上的內容，包括文章作者的名字，變成一篇新聞作者想要帶資訊給讀者的文章。
    要求：不要使用「值得注意的是」，「另外」，「最後」，「總括來說」等連接詞。
    要求：公司名稱、藝人藝名、團體名稱，如果是英文名稱是廣為人知的，請不要翻譯（保留英文名稱），如要翻譯，請括號標註英文名稱。
    要求：翻譯完請重新檢查文章是否通順，避免中英夾雜。
    刪除不相關的資訊。這篇文章的標題是:{title}

    有一些名詞我已經透過網上搜尋得到正確翻譯，請先熟悉一下這些翻譯再給我一篇正確無誤的翻譯，請括號標註原文名稱（英文）。用括號標示本來（未翻譯）的名詞。如果是沒有翻譯對照的字，使用原文語言。
    名詞：{dictionary}
    如果是人的說話，把它改爲間接引用。去掉 “”。

    現在處理這段文字：
    {segment}

    如果該行文字是小標題，用<h2>來標記。如果是不相關的內容，刪除不相關的資訊。這篇文章的標題是:{title}
    只回覆我中文的html，不需其它任何字。
    不要回覆我任何其它字，我只需要處理好的中文的html structure回覆。
    """

    print(prompt)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            if "no" not in chunk.choices[0].delta.content.lower():
                print(chunk.choices[0].delta.content, end="")
                full_article += chunk.choices[0].delta.content
    full_article = recheck(title, full_article, model)
    return full_article

def engtit(website_text, model, max_retries=3, retry_delay=5):
    full_article = ""
    first_20_lines = "\n".join(website_text.splitlines()[:20])
    prompt = f"""
    the article: {first_20_lines}

    refine the title of this article to me in chinese ONLY. no premable and explanations needed.
    """

    retries = 0
    success = False

    while retries < max_retries and not success:
        try:
            print(prompt)
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content.encode('utf-8', errors='ignore').decode('utf-8')
                    print(content, end="")
                    full_article += content

            success = True

        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Moving to next segment.")

    return full_article

def recheck(title, article, model, max_retries=3, retry_delay=5):
    full_article = ""
    processes = split_article_into_segments(article, lines_per_segment=17)

    for process in processes:
        prompt = f"""
        文章：{process}

        刪除：
        - 宣傳部分，包括「查看更多文章」，「或許你會喜歡」等等。整行刪掉。
        - 格式的段落，比如整個<p> 只有一個 "---"，整個刪掉。
        - 圖片來源，記者報道等無關文章主旨的句子，整個刪掉。
        - 不相干的東西，如果與前文和文章主旨完全不相干，刪掉。
        - 如果有部分內容是不相關的新聞，刪除該部分的內容。這篇新聞的標題是: {title}
        - 如有連結或相關宣傳，必須刪除，不能留下
        - 刪除「值得注意的是」，「另外」，「最後」，「總括來說」等連接詞。

        改寫：
        - 身份：我是一個香港新聞記者，專業，客觀
        - 不需要強調身份，但所有不符合這個設定的句子需要改寫成符合我身份的描述。原文作者的家人名稱、工作地點、懷孕狀況等全部刪除。
        - 全部改寫原文內容的句式（paraphrase），但保留原文的意思，以免誤導讀者。如果是內容有括號，括號內容需要保留。
        - 不需進行任何翻譯。
        - 改寫必須合理，需要文句通順。
        - 如果內容許可，增加<ul> <ol> <table>等元素來協助描述。整理段落的內容來寫。

        只回覆我中文的html，不需其它任何字。
        不要回覆我任何其它字，我只需要處理好的中文的html structure回覆。
        """

        retries = 0
        success = False

        while retries < max_retries and not success:
            try:
                print(prompt)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt.strip()}],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=8192,
                    stream=True
                )

                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content.encode('utf-8', errors='ignore').decode('utf-8')
                        print(content, end="")
                        full_article += content

                success = True

            except Exception as e:
                print(f"Error: {e}")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Moving to next segment.")

    return full_article

def titler(website_text, model, max_retries=3, delay=2):
    attempt = 0
    while attempt < max_retries:
        try:
            full_article = ""
            prompt = f"""
            i want to write a news article for this article:
            {website_text}

            1. I want a news article title that is clickbait enough, in moderate length and humanized, natural tone without overexaggeration.    
            2. the news title should include the highlight theme of the news, instead of a short phrase.
            3. return me a single JSON object with a single key 'title' without a preamble and explanations.
            4. output in traditional Chinese.
            AGAIN: single JSON object with a single key 'title', NO preamble and explanation needed.
            """

            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8000,
                stream=True
            )

            article_title = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    article_title += chunk.choices[0].delta.content

            article_title = extract_json_content(article_title)["title"]
            if article_title:
                  return article_title

        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                sleep_time = delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
            else:
                raise


html_tag_regex = re.compile(r'^<.*>.*</.*>$')
contains_html_tag_regex = re.compile(r'<(ul|h2|table|td|li|ol|tr|th)>')
contains_html_close_tag_regex = re.compile(r'</(ul|h2|table|td|li|ol|tr|th)>')

def rewrite_h2(content, model):
    prompt = f"""
    for this h2 title, rewrite it. you can write with another writing style, or simply just change the sentence structure, as long as the overall meaning remains the same.
    {content}
    return me the <h2> with tag wrapped. no preamble and explanation.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8000,
        stream=True
    )

    h2_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            h2_content += chunk.choices[0].delta.content
    return h2_content

def metadataer(title, model):
    prompt = f"""
    i am writing a news article with this keyword: {title}
    now i need two HTML tags, <meta name="description" content=""> and <meta name="keywords" content="">
    i need you to help me fill in the content part, using NLP techniques, SEO optimized naturally with the title content.
    i only want you to return me the two HTML meta tags, properly formatted as HTML structure, and return me without premable and explanations.
    Output the description content and keyword content in traditional Chinese.
    AGAIN: NO preamble and explanations.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8000,
        stream=True
    )

    metadata = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            metadata += chunk.choices[0].delta.content

    return metadata

def process_line(line, model, last_was_h2):
    stripped_line = line.strip()

    # Check if the line is already wrapped with any HTML tags
    if html_tag_regex.match(stripped_line):
        if stripped_line.startswith('<h2>') and stripped_line.endswith('</h2>'):
            if last_was_h2:  # Skip this <h2> if the previous line was also an <h2>
                return '', True
            else:
                return rewrite_h2(stripped_line, model) + '\n', True
        return stripped_line + '\n', False

    # Check if the line contains HTML tags and avoid wrapping with <p>
    elif contains_html_tag_regex.search(stripped_line):
        return stripped_line + '\n', False

    # Check if the line contains HTML closing tags and avoid wrapping with <p>
    elif contains_html_close_tag_regex.search(stripped_line):
        return stripped_line + '\n', False

    # Check if the line starts and ends with "**" for <h2>
    elif stripped_line.startswith('**') and stripped_line.endswith('**'):
        h2_content = '<h2>' + stripped_line.strip('**') + '</h2>'
        if last_was_h2:  # Skip this <h2> if the previous line was also an <h2>
            return '', True
        else:
            return rewrite_h2(h2_content, model) + '\n', True

    # Otherwise, wrap with <p> for plain text and reset the <h2> flag
    return '<p>' + stripped_line + '</p>\n', False

def add_rss_item(template_path, title, link, category, description):
    tree = parse(template_path)
    root = tree.getroot()
    channel = root.find('channel')
    last_build_date = channel.find('lastBuildDate')
    hk_timezone = pytz.timezone('Asia/Hong_Kong')
    last_build_date.text = datetime.now(hk_timezone).strftime('%a, %d %b %Y %H:%M:%S %z')

    # Create a new item
    item = Element('item')
    item_title = SubElement(item, 'title')
    item_title.text = title
    item_link = SubElement(item, 'link')
    item_link.text = link
    item_description = SubElement(item, 'description')
    item_description.text = description
    item_category = SubElement(item, 'category')
    item_category.text = category

    item_pub_date = SubElement(item, 'pubDate')
    item_pub_date.text = datetime.now(hk_timezone).strftime('%a, %d %b %Y %H:%M:%S %z')

    # Prettify the item
    pretty_item_str = prettify_element(item)
    pretty_item = fromstring(pretty_item_str.encode('utf-8'))
    channel.append(pretty_item)
    tree.write(template_path, encoding='utf-8', xml_declaration=True)

def get_bottom_items(rss_file_path):
    root = parse(rss_file_path)
    items = root.findall('.//item')  
    bottom_items = items[-4:-1] if len(items) >= 4 else items  
    result = {item.find('title').text: item.find('link').text for item in bottom_items}    
    return result

def append_to_sitemap(loc, priority):
    # File path to the sitemap.xml
    file_path = 'sitemap.xml'

    # Parse the existing sitemap.xml file
    tree = parse(file_path)
    root = tree.getroot()

    # Declare the sitemap namespace
    sitemap_ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
    nsmap = {"ns0": sitemap_ns}

    # Create a new <url> element in the sitemap namespace
    new_url = Element(f"{{{sitemap_ns}}}url")

    # Add <loc> element
    loc_element = SubElement(new_url, f"{{{sitemap_ns}}}loc")
    loc_element.text = loc

    # Add <lastmod> element with the current time in Hong Kong timezone
    hk_timezone = pytz.timezone('Asia/Hong_Kong')
    current_time = datetime.now(hk_timezone)
    lastmod_element = SubElement(new_url, f"{{{sitemap_ns}}}lastmod")
    lastmod_element.text = current_time.strftime('%Y-%m-%dT%H:%M:%S%z')
    lastmod_element.text = lastmod_element.text[:-2] + ':' + lastmod_element.text[-2:]

    # Add <changefreq> element
    changefreq_element = SubElement(new_url, f"{{{sitemap_ns}}}changefreq")
    changefreq_element.text = "weekly"

    # Add <priority> element
    priority_element = SubElement(new_url, f"{{{sitemap_ns}}}priority")
    priority_element.text = priority

    # Append the new <url> element to the root <urlset> element
    root.append(new_url)

    # Internal prettify function with a different name
    def prettify_xml_tree(element, level=0):
        """Prettifies the XML tree in place by adding indentation and newlines."""
        indent = "\n" + level * "  "
        if len(element):  # If the element has children
            if not element.text or not element.text.strip():
                element.text = indent + "  "
            for elem in element:
                prettify_xml_tree(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
        else:
            if not element.text or not element.text.strip():
                element.text = ""
            if level and (not element.tail or not element.tail.strip()):
                element.tail = indent

    prettify_xml_tree(root)  # Call the renamed prettify function

    # Write the updated and prettified XML back to the file
    tree = ElementTree(root)
    tree.write(file_path, encoding='UTF-8', xml_declaration=True)

def get_current_hk_time():
    tz_hk = pytz.timezone('Asia/Hong_Kong')
    current_time = datetime.now(tz_hk)
    return current_time.isoformat()

def write_file(file_path, content, title, source, category, model):
    url = "https://www.famechos.me/news/" + title + '.html'
    with open(file_path, 'w', encoding='utf-8') as file:
        # Dynamic data for the schema
        schema_data = {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "Article",
                    "headline": title,
                    "description": title,
                    "url": url,
                    "image": "https://www.famechos.me/images/banner.jpg",
                    "datePublished": get_current_hk_time(),
                    "author": {
                        "@type": "Person",
                        "name": "Famechos"
                    },
                    "publisher": {
                        "@type": "Organization",
                        "name": "Famechos",
                        "url": "https://www.famechos.me"
                    }
                },
                {
                    "@type": "Organization",
                    "name": "Famechos",
                    "url": "https://www.famechos.me",
                    "logo": "https://www.famechos.me/icons/favicon.png"
                },
                {
                    "@type": "WebSite",
                    "name": "Famechos",
                    "url": "https://www.famechos.me"
                }
            ]
        }
 
        # Convert the dictionary to a JSON string
        schema_json = json.dumps(schema_data)
        file.write(f"<!DOCTYPE html>\n<head>\n<script type='application/ld+json'>\n{schema_json}\n</script>\n")
        file.write('<link rel="canonical" href="' + url + '"/>\n')
        metadata = metadataer(title, model)
        file.write(metadata + '\n')
   
# Dynamically construct the meta tags
        meta_tags = f'''
<link rel="stylesheet" href="../main-nav.css">
<link rel="stylesheet" href="../main-content.css">
<link rel="stylesheet" href="../main-small.css">
<link rel="stylesheet" href="../post.css">
<link rel="stylesheet" href="../main-footer.css">
<link rel="stylesheet" href="../news.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/earlyaccess/notosanstc.css">
        <meta property="og:url" content="{url}" />
        <meta property="og:title" content="{title}" />
        <meta property="og:description" content="{title}" />
        <meta property="og:image" content="https://www.famechos.me/images/banner.jpg" />
        <meta property="twitter:card" content="summary_large_image" />
        <meta property="twitter:title" content="{title}" />
        <meta property="twitter:description" content="{title}" />
        <meta property="twitter:image" content="https://www.famechos.me/images/banner.jpg" />
<meta name="theme-color" content="white">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta property="og:locale" content="zh_TW" />
        <meta property="og:site_name" content="Famechos" />
<meta property="og:type" content="article" />
<meta name="robots" content="index, follow" />
<meta name="author" content="Famechos" />
        <meta name="referrer" content="origin">
<meta name="apple-mobile-web-app-capable" content="yes"/>
        <meta name="apple-mobile-web-app-status-bar-style" content="black"/>
        <meta name="apple-mobile-web-app-title" content="Famechos"/>
        <meta name="apple-touch-fullscreen" content="yes"/>
        <link rel="icon" type="image/x-icon" href="https://www.famechos.me/img/famechos-icon.jpeg">
        <link rel="shortcut icon" type="image/x-icon" href="https://www.famechos.me/img/famechos-icon.jpeg">

        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-3046601377771213"
     crossorigin="anonymous"></script>
     
        '''

        style = r"<style> * {box-sizing: border-box;margin: 0;padding: 0;font-family: 'Noto Sans TC', sans-serif;scroll-behavior: smooth;}</style>"

        file.write(meta_tags + '\n' + style + '\n')
   
        file.write('\n</head>\n\n<body>\n')
        navbar = r'''
<nav class="nav">
    <div class="nav-outerbox">
      <div class="nav-inner-left">
        <div class="nav-name-logo">
          <a href="https://www.famechos.me/"><img class="famechos-logo" src="../img/famechos_logo.png"></a>
         
        </div>
        <div class="nav-separate">
          <p>|</p>
        </div>
        <div class="nav-type-1 nav-type"><a href="https://www.famechos.me/k-pop.html">
          <button class="nav-type-btn1 nav-type-btn">
          <h2 class="nav-type-text">K-POP</h2>
        </button></a>
 
        </div>
        <div class="nav-type-2 nav-type"><a href="https://www.famechos.me/j-pop.html">
          <button class="nav-type-btn2 nav-type-btn">
          <h2 class="nav-type-text">J-POP</h2>
        </button></a>
 
        </div>
        <div class="nav-type-3 nav-type"><a href="https://www.famechos.me/drama.html">
          <button class="nav-type-btn3 nav-type-btn">
          <h2 class="nav-type-text">影視</h2>
        </button></a>
         
        </div>
        <div class="nav-type-4 nav-type"><a href="https://www.famechos.me/others.html">
          <button class="nav-type-btn4 nav-type-btn">
          <h2 class="nav-type-text">其他</h2>
        </button></a>
        </div>  
      </div>
      <div class="nav-inner-right">
        <div class="nav-feedback-outer"><a href="https://www.famechos.me/about_us.html">
          <button class="feedback-btn nav-type-btn">
            <h2 class="nav-type-text">關於我們</h2>
          </button></a>
        </div>
        <div class="nav-setting"><a href="https://www.famechos.me/privacy_policy.html">
          <button class="setting-btn nav-type-btn">
            <h2 class="nav-type-text">私隱條款</h2>
          </button></a>
        </div>
        <div class="nav-list">
          <button id="list-btn" class="list-btn nav-type-btn">
            <i class="bi bi-list"></i>
          </button>
        </div>

      </div>

      <div id="list" class="nav-list-outer">
        <div class="nav-type-1 nav-type list-type"><a href="https://www.famechos.me/k-pop.html">
          <button class="list-type-btn">
          <p>K-POP</p>
          </button></a>
        </div>

        <div class="nav-type-2 nav-type list-type"><a href="https://www.famechos.me/j-pop.html">
          <button class="list-type-btn">
          <p>J-POP</p>
          </button></a>
        </div>

        <div class="nav-type-3 nav-type list-type"><a href="https://www.famechos.me/drama.html">
          <button class="list-type-btn">
          <p>影視</p>
          </button></a>
        </div>

        <div class="nav-type-4 nav-type list-type"><a href="https://www.famechos.me/others.html">
          <button class="list-type-btn">
          <p>其他</p>
          </button></a>
        </div>

        <div class="nav-type-5 nav-type list-type"><a href="https://www.famechos.me/about_us.html">
          <button class="list-type-btn">
          <p>關於我們</p>
          </button></a>
        </div>

        <div class="nav-type-6 nav-type list-type"><a href="https://www.famechos.me/privacy_policy.html">
          <button class="list-type-btn">
          <p>私隱條款</p>
          </button></a>

        </div>  

      </div>
     
    </div>
  </nav>


  <main>
    <div class="news-main-outer">
      <div class="news-title-outer">
        '''
        file.write(navbar)
        file.write('<h1>' + title + '</h1>')
        then = r'''
</div>

      <div class="news-content-outer">

        <div class="news-info-outer">'''

        hk_timezone = pytz.timezone('Asia/Hong_Kong')
        hk_time = datetime.now(hk_timezone)
        current_date = hk_time.strftime('%d/%m/%Y')

        catt = f'''
          <p class="news-content-type">{category}</p>
          <p class="news-date">{current_date}</p>
        </div>
'''
        file.write(then)
        file.write(catt)
        # Split content into lines
        lines = content.splitlines()

        def count_chinese_and_english(text):
            lines = text.splitlines()
            results = []
            for line in lines:
                chinese_chars = re.findall(r'[\u4e00-\u9fff]', line)
                english_words = re.findall(r'\b\w+\b', line)

                total_count = len(chinese_chars) + len(english_words)
                results.append((line, total_count))

            total_words = 0
            final_sentences = []

            for sentence, count in results:
                if total_words + count <= 80:
                    final_sentences.append(sentence)
                    total_words += count
                else:
                    break

            output = '。'.join(final_sentences)
            output += '...'

            return output

        def remove_html_tags(text):
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text()

        if lines:
            des = lines.pop(0)
            des += '\n' + '\n'.join(lines[:4])

        des = remove_html_tags(des)
        des = count_chinese_and_english(des)

        def should_append_header(processed_line, last_was_h2):
            """Determine if the current header should be appended."""
            is_current_h2 = '<h2>' in processed_line and '</h2>' in processed_line
            return processed_line and not (is_current_h2 and last_was_h2)

        last_was_h2 = False  # To track if the last processed line was an <h2>

        # Regular expression to match Chinese characters
        chinese_char_regex = re.compile(r'[\u4e00-\u9fff]')

        def count_chinese_characters(text):
            """Count the number of Chinese characters in the given text."""
            return len(chinese_char_regex.findall(text))
        h = ""
        for line in lines:
            if line.strip():  # Ignore empty lines
                processed_line, last_was_h2 = process_line(line, model, last_was_h2)

                if should_append_header(processed_line, last_was_h2):
                    h += processed_line
                    h += '\n'
                    file.write(processed_line)

        chinese_char_count = count_chinese_characters(h)

        if chinese_char_count < 100:
            return

        file.write('\n<p>資料來源：' + source + '</p>')
        file.write('\n</div><div class ="news-vid-outer">\n<div class="related-vid-text-outer title-bar ">\n')
        file.write('<p class="related-vid-text">相關影片：</p>\n</div>\n')
        embed_code = get_first_youtube_embed(title, model)
        if embed_code:
            file.write(embed_code + '\n\n')
        file.write('\n<p class="news-text-inner">資料來源： YouTube</p>\n</div>\n')

        footer = r'''

 <div class ="related-news-outer title-bar">
        <p class="related-news-text">相關新聞：</p>
      </div>
     
      <div class ="related-news-box">
        '''

        rss_file_path = category.lower() + '.xml'
        the_json = get_bottom_items(rss_file_path)
        r_news = ""
        if the_json:
            print("oui")
            for a, b in the_json.items():
                r_news += f'''
<p class ="related-news">
          <i class="bi bi-dot"></i>
          <a href="{b}">{a}</a>
        </p>
                '''
           
        print("rss THAT HTML: " + str(r_news))
        last = r'''
      </div>

    </div>

    </main>
 <footer class="footer">
    <div class="footer-outer">


      <div class="footer-up-outer">
        <div class="footer-right-name-box">
          <a href="https://www.famechos.me/"><img class="famechos-logo-footer" src="../img/famechos_logo.png"></a>
        </div>
        <div class="footer-right-text-box">
          <p class="footer-right-text">每日為你提供最新、最全面的日韓資訊。</p>
        </div>
        <div class="footer-button-box">
          <button class="facebook-btn footer-btn">
            <i class="bi bi-facebook footer-logo"></i>
          </button>
          <button class="ig-btn footer-btn">
            <i class="bi bi-instagram footer-logo"></i>
          </button>
          <button class="x-btn footer-btn">
            <i class="bi bi-twitter-x footer-logo"></i>
          </button>
          <button class="x-btn footer-btn">
            <i class="bi bi-envelope-fill footer-logo"></i>
          </button>
        </div>
      </div>

      <div class="footer-down-outer">
        <div class="kpop-footer-box footer-box">
          <a href="https://www.famechos.me/k-pop.html"><p class="footer-text">K-POP</p></a>
        </div>
        <div class="jpop-footer-box footer-box">
          <a href="https://www.famechos.me/j-pop.html"><p class="footer-text">J-POP</p></a>
        </div>
        <div class="drama-footer-box footer-box">
          <a href="https://www.famechos.me/drama.html"><p class="footer-text">影視</p></a>
        </div>
        <div class="other-footer-box footer-box">
          <a href="https://www.famechos.me/others.html"><p class="footer-text">其他</p></a>
        </div>


      </div>

      <div class="copyright-outer">
        <p class="copyright-text">
          Copyright © 2024 by <strong>Famechos.me</strong> All Rights Reserved.
        </p>
      </div>

     
    </div>



  </footer>

<script src="../nav-list.js"></script>

</body>
</html>
'''
        file.write(footer)
        file.write(r_news)
        file.write(last)
    append_to_sitemap(url, "0.90")
    add_rss_item(f'{category.lower()}.xml', title, url, category, des)
    add_rss_item('rss.xml', title, url, category, des)
    commit_changes()

def parse_full_text(url, title, source, category, model, lines = 22):
    full_article = ""
    downloaded = trafilatura.fetch_url(url)
    website_text = trafilatura.extract(downloaded)

    if website_text is None:
        print("Failed to scrap, skipped this URL.")
        return

    if not count_newlines_exceeds_limit(website_text):
        print("Bad formatting, skipped this URL.")
        return

    if website_text:
   
        # Split the article into segments
        segments = split_article_into_segments(website_text, lines_per_segment=lines)

        # Process each segment
        sample = process_segments(segments, model)

        title = engtit(website_text, model)

        for segment in segments:
            full_article += consideration_test(title, segment, sample, model)
            full_article += "\n"
        title = titler(full_article, model)

        file_path = clean_title(title, 'html', r"news")
        write_file(file_path, full_article, title, source, category, model)
   
def commit_changes():
    try:
        # Step 1: Set Git config to always merge changes (avoids rebase conflicts)
        subprocess.run(["git", "config", "pull.rebase", "false"], check=True)

        # Step 2: Fetch the latest changes from GitHub
        subprocess.run(["git", "fetch", "origin"], check=True)
       
        # Step 3: Add all local changes
        subprocess.run(["git", "add", "--all"], check=True)
       
        # Step 4: Commit local changes
        subprocess.run(["git", "commit", "-m", "讀萬卷書不如寫萬篇文"], check=True)
       
        # Step 5: Pull the latest changes from GitHub and merge
        subprocess.run(["git", "pull", "--strategy=recursive", "--strategy-option=theirs"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred during git operation: {e}")
        # Continue even if pull fails due to conflicts

    try:
        # Step 6: Push the changes, force if needed
        subprocess.run(["git", "push", "--force"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during git push: {e}")

rss_urls = [
    ['https://www.koreaherald.com/common/rss_xml.php?ct=105', 'Korea Herald', 'K-Pop'],
    ['https://tokyocheapo.com/feed/', 'Tokyo Cheapo', 'Others'],
    ['https://www.koreatimes.co.kr/www/rss/entertainment.xml', 'TheKoreaTimes', 'K-Pop'],
    ['https://en.yna.co.kr/RSS/culture.xml', 'Yonhap News Agency', 'Drama'],
    ['https://j-generation.com/feed/', 'J-GENERATION', 'J-Pop'],
    ['https://phoenixtalkspopculturejapan.wordpress.com/category/dramas/feed/', 'Phoenix Talks Pop Culture Japan', 'Drama'],
    ['https://jpopblog.com/feed/', 'Jpopblog.com', 'J-Pop'],
    ['https://thesoulofseoul.net/feed/', 'The Soul of Seoul', 'Others'],
    ['https://10mag.com/feed/', '10mag', 'Others']
]

def main():
    try:
        model = "meta/llama-3.1-405b-instruct"
        news = fetch_news(rss_urls)
        print(news)
        file_path = "./news.txt"
        try:
            with open(file_path, 'r') as file:
                existing_links = file.readlines()
            existing_links = [line.strip() for line in existing_links]
        except FileNotFoundError:
            with open(file_path, 'w') as file:
                pass
            existing_links = []
       
        random.shuffle(news)

        unique_news_count = 0
        for new in news:
            if new['link'] not in existing_links:
                parse_full_text(new['link'], new['title'], new['source'], new['category'], model, lines)
                with open(file_path, 'a') as file:
                    file.write(new['link'] + '\n')
                existing_links.append(new['link'])
                unique_news_count += 1
                if unique_news_count == 1:
                    break
            else:
                print('News already used: ' + str(new['title']))
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    except ValueError as e:
        print(f"Error: {e}")
   
if __name__ == "__main__":
    if not DEBUG:
        main()
