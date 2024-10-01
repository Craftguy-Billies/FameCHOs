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
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta

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

def get_first_youtube_embed(query):
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
    
    # Wait for the page to fully load
    time.sleep(3)  # You may adjust this if the page is slow
    
    # Find the first video link
    first_video = driver.find_element(By.XPATH, '//a[@href and contains(@href, "/watch?v=")]')
    print(first_video)
    
    if first_video:
        video_url = f"https://www.youtube.com{first_video.get_attribute('href')}"
        video_url = video_url.split('&')[0]
        video_id = video_url.split('v=')[1]
        embed_code = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
        
        # Close the browser
        driver.quit()
        
        return embed_code
    
    # If no video found, return None
    driver.quit()
    return "No video found."

def fetch_news(rss_urls):
    news_items = []
    seen_titles = set()

    # Get the current time and the time two weeks ago
    now = datetime.now()
    two_weeks_ago = now - timedelta(weeks=2)

    for url, source in rss_urls:
        feed = feedparser.parse(url)

        for entry in feed.entries:
            # Check if the entry has a publication date
            if 'published_parsed' in entry:
                # Convert the published date to a datetime object
                pub_date = datetime(*entry.published_parsed[:6])
                # Only consider articles published within the last two weeks
                if pub_date < two_weeks_ago or pub_date > now:
                    continue

            # Check for duplicates based on title
            if entry.title not in seen_titles:
                news_item = {
                    'title': entry.title,
                    'link': entry.link,
                    'summary': entry.summary,
                     'source' : source
                }
                
                # Fetch the article content
                downloaded = trafilatura.fetch_url(entry.link)
                if downloaded:
                    website_text = trafilatura.extract(downloaded)
                    if website_text:
                        word_count = len(website_text.split())
                        # Filter out articles that are too short or too long
                        if word_count <= 200 or word_count >= 900:
                            continue
                
                # Add the news item to the list
                news_items.append(news_item)
                seen_titles.add(entry.title)

    return news_items

def split_article_into_segments(article, lines_per_segment=13):
    lines = article.split('\n')
    segments = [lines[i:i + lines_per_segment] for i in range(0, len(lines), lines_per_segment)]
    return segments

def count_newlines_exceeds_limit(text: str, limit: int = 5) -> bool:
    newline_count = text.count('\n')
    return newline_count > limit

def search(query, max_results = 7):
    encoded_query = requests.utils.quote(query)
    url = f"https://www.google.com/search?q={encoded_query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Send the request
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')
    snippets = []
    
    for g in soup.find_all('div', class_='tF2Cxc')[:max_results]:
        snippet = ''
        if g.find('span', class_='aCOpRe'):
            snippet = g.find('span', class_='aCOpRe').text
        elif g.find('div', class_='IsZvec'):
            snippet = g.find('div', class_='IsZvec').text
        elif g.find('div', class_='VwiC3b'):
            snippet = g.find('div', class_='VwiC3b').text
        elif g.find('div', class_='s3v9rd'):
            snippet = g.find('div', class_='s3v9rd').text
        
        snippets.append(snippet)
    
    return snippets

def organize(word, description, results, model, max_retries=3):
    prompt = f"""
    The word to be translated: {word}
    The description of this word: {description} 
    Web Search Context: {results} 

    您是研究問題任務的人工智慧助手，負責綜合網路搜尋結果。
    嚴格使用以下網路搜尋上下文來回答問題。如果你不知道答案，就說你不知道。
    保持答案簡潔，但以研究報告的形式提供所有詳細資訊。
    僅直接引用上下文中提供的資料。

    現在我將給你一個AI可能無法正確翻譯的名詞，需要網路搜尋才能獲得正確的資訊。
    你的任務是透過網路搜尋上下文為我提供正確的翻譯名詞（中文）。你搜尋到的翻譯必須是原文字詞的翻譯。(例如 "Wutopia" 和 "Hutopia" 是不同的字詞，如果混淆會出現完全錯誤的結果) 如果沒有準確中文名字結果，請傳回該語言的原始名字，不要音譯。
    如果網絡搜尋結果沒有提及準確中文翻譯，不要用原文語義給我答案，傳回該語言的原始名字即可。

    Summarize the search result. 
    As long as search result NOT 100 PERCENT SURE IT IS CORRECT, return me the original word.
    Otherwise, only return the MUST correct chinese translation of that noun i needed.
    return me a JSON object that store the original word and search result word in key-value pairs. 
    REMEMBER: If there isn't a chinese name found, return me the original name, do not phonetic translate or translate with the original word's english meaning.
    Do not leave the key-value pair blank no matter what.
    if there are more than one translation, only return me one.
    REMEMBER: the JSON returned has only ONE key-value pair. no need other keys to label.
    Return the JSON with ONE key-value pair with no premable or explanation. 
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
    You are an expert in crafting web search queries for getting the accruate translation of unknown nouns into chinese.
    The aim is to search out the chinese name of the unknown noun.
    The search query should also include the description of that noun if it helps to define the chinese name.
    make sure your chinese (tranditional) words in query DOES MAKE SENSE. DO NOT give me a search query with uncommon or unrelated chinese vocabularies.
    A query structure MUST include the original word in that language, and at least one chinese supporting description (e.g. 地名、人名、遊戲、店名等)。just ensure they are relevant to search the correct translate.
    不要把無關的字眼放進去，這會影響搜尋結果的準確性！
    也不要整個搜尋都是中文，這樣是搜尋不到正確翻譯的！
    Make sure the search query is not too long. (at most 10 chinese characters are maximum)
    AGAIN: make sure your chinese words in query DOES MAKE SENSE.
    Return the JSON with a single key 'query' with no premable or explanation. 
    
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
            return modified_string

        except Exception as e:
            retries += 1

    return {"query": ""}

def is_valid_dict(refined):
    """Check if the refined response is a valid dictionary."""
    return isinstance(refined, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in refined.items())

def process_segments(segments, title, model, max_retries=3):
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
        if not query["query"]:
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
    - Now I need to turn it into a chinese article. But before that, there are some nouns require web search to get accurate translation.
    - Your job is scan through the array, understanding the full context and extract important nouns ONLY that cannot be translated directly.
    - All non-well-known nouns include personal nouns (e.g. blogger's son's name, a French blogger website name, etc) should NOT be extracted. DO NOT return me these nouns for extra web search.
    - For other human names, if they are well known and can be found in web, return me the full name to be searched.
    - DO NOT return abbreviations as it cannot be searched. Return me the accurate full name of the person ONLY, NOT the simplifed surname that appears in other paragraphs (e.g. Understand that "Cheung" is referred as "Cheung Ka Long" but not another person)
    - If the word is common and you can translate correctly, do not return that word for extra web search work.
    - There might be some idioms in the article,  return these words if it is not logical in direct translation.
    - Return me a JSON object contain the noun and it's background information for web search as the key-value pairs.
    - If there is no words required web search translation, return me a single "None". It is better to return less words.
    - REMEMBER: if there is a special noun with capitalization with are not logical in meanings, please return that noun as well even it is seemingly a normal word.
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

def consideration_test(segment, title, dictionary, model):
    full_article = ""
    prompt = f"""
    文章的大標題是{title}。給我整理成一篇文章。改寫並翻譯成香港語氣的中文版本。
    刪除原文的所有圖片說明
    改寫必須合理，需要文句通順。
    語氣：專業、分享感受
    身份：一個香港的新聞作者想要帶資訊給讀者
    刪除所有作者自己的身份描述。（作者的家人名稱、工作地點、懷孕狀況等全部刪除），並改爲符合身份的描述。
    要求：必須把原文內容全部翻譯，不能自行創作，要詳細。翻譯時必須先了解整句話的意思，不要按字詞意思直接翻譯。
    要求：改寫一切網站上的內容，包括文章作者的名字，變成一篇我作爲一個香港人的角度了解整個主體之後所寫的文章。

    有一些名詞我已經透過網上搜尋得到正確翻譯，請先熟悉一下這些翻譯再給我一篇正確無誤的文章，不能有漏。用括號標示本來（未翻譯）的名詞。如果是沒有翻譯對照的字，使用原文語言。
    名詞：{dictionary}
    如果是人的說話，把它改爲間接引用。去掉 “”。

    現在處理這段文字：
    {segment}

    如果該行文字是小標題，用<h2>來標記。
    只回覆我中文的html，不需其它任何字。
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
    full_article = recheck(full_article, model)
    return full_article

def recheck(article, model, max_retries=3, retry_delay=5):
    full_article = ""
    processes = split_article_into_segments(article, lines_per_segment=17)

    for process in processes:
        prompt = f"""
        現在我有一篇文章。當中有一些部分需要你刪除和改寫。
        文章：{process}

        刪除：
        - 宣傳部分，包括「查看更多文章」，「或許你會喜歡」等等。整行刪掉。
        - 格式的段落，比如整個<p> 只有一個 "---"，整個刪掉。
        - 圖片來源，記者報道等無關文章主旨的句子，整個刪掉。
        - 不相干的東西，如果與前文和文章主旨完全不相干，刪掉。

        改寫：
        - 身份：我是一個香港人，想要帶資訊給讀者，所有經歷都只有自己和朋友，沒有和家人孩子一起。
        - 不需要強調身份，但所有不符合這個設定的句子需要改寫成符合我身份的描述。原文作者的家人名稱、工作地點、懷孕狀況等全部刪除。
        - 全部改寫原文內容的句式（paraphrase），但保留原文的意思，以免誤導讀者。如果是內容有括號，括號內容需要保留。
        - 改寫必須合理，需要文句通順。
        - 如果內容許可，增加<ul> <ol> <table>等元素來協助描述。整理段落的內容來寫。

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

            i want a news article title that is clickbait enough, in moderate length and humanized tone.     
	    the news title should include the highlight theme of the news, instead of a short phrase.
            return me a single JSON object with single key 'title' without premable and explanations.
            output in traditional chinese
            AGAIN: NO premable and explanation needed.
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

# Regex to check if line is already wrapped with any HTML tags
html_tag_regex = re.compile(r'^<.*>.*</.*>$')

def process_line(line):
    # Check if the line is already wrapped with any HTML tags
    if html_tag_regex.match(line.strip()):
        return line  # Line is already wrapped with an HTML tag, return as is

    # Check if the line starts and ends with "**" for <h2>
    elif line.strip().startswith('**') and line.strip().endswith('**'):
        return '<h2>' + line.strip().strip('**') + '</h2>\n'

    # Otherwise, wrap with <p> for plain text
    else:
        return '<p>' + line.strip() + '</p>\n'

def write_file(file_path, content, title, source):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('<h1>' + title + '</h1>\n\n')
        embed_code = get_first_youtube_embed(title)
        if embed_code:
            file.write(embed_code + '\n\n')
        # Split content into lines
        lines = content.splitlines()

        # Remove the first line (pop)
        if lines:
            lines.pop(0)
        for line in lines:
            # Remove empty lines and process non-empty lines
            if line.strip():  # Ignore empty lines
                file.write(process_line(line))
                
        file.write('\n<p>資料來源： ' + source + '</p>')

def parse_full_text(url, title, source, model, lines = 22):
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
        # form a clickbait title
        title = titler(website_text, model)

        # Split the article into segments
        segments = split_article_into_segments(website_text, lines_per_segment=lines)

        # Process each segment
        sample = process_segments(segments, title, model)

        for segment in segments:
            full_article += consideration_test(segment, title, sample, model)
            full_article += "\n"

        file_path = clean_title(title, 'html', r"Translated News")
        write_file(file_path, full_article, title, source)
    
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
    ['https://www.koreaherald.com/common/rss_xml.php?ct=105', 'Korea Herald'],
    ['https://tokyocheapo.com/feed/', 'Tokyo Cheapo']
]

def main():
    try:
        model = "meta/llama-3.1-405b-instruct"
        news = fetch_news(rss_urls)
        print(news)
        file_path = "news.txt"
        for new in news:
            try:
                with open(file_path, 'r') as file:
                    existing_links = file.readlines()
                existing_links = [line.strip() for line in existing_links]
            except FileNotFoundError:
                with open(file_path, 'w') as file:
                    pass
                existing_links = []

            for new in news:
                if new['link'] not in existing_links:
                    parse_full_text(new['link'], new['title'], new['source'], model, lines)
                    with open(file_path, 'a') as file:
                        file.write(new['link'] + '\n')
                    commit_changes()
                else:
                    print('News already used: ' + str(new['title']))
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    except ValueError as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    if not DEBUG:
        main()
