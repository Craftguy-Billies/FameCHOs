# FameCHOs — Repository Architecture & File Breakdown

## Overview

**FameCHOs** is an AI-powered automated news aggregation, translation, and publishing pipeline for **[famechos.me](https://www.famechos.me)** — a Hong Kong-based Traditional Chinese-language news website covering K-Pop, J-Pop, Korean dramas, and Korean/Japanese culture/lifestyle. The system uses NVIDIA-hosted LLMs (Llama 3.1 Nemotron 70B / 405B) to scrape English-language news from 10 RSS feeds, translate/rewrite articles into Traditional Chinese with a Hong Kong author's voice, generate clickbait titles, embed YouTube videos, output full HTML pages, update XML sitemaps and RSS feeds, then auto-commit/push/deploy via GitHub Actions.

---

## System Architecture

```
[GitHub Actions cron: every hour]
        │
        ▼
   新聞筆.py (main pipeline — runs 1 article per invocation)
        │
        ├── fetch_news() ──► 10 RSS feeds
        ├── parse_full_text() per article:
        │     ├── trafilatura scrape full article text
        │     ├── split into segments (~22 lines each)
        │     ├── process_segments():
        │     │     ├── refine_response() ──► LLM identifies proper nouns needing translation
        │     │     ├── websearch() ──► LLM crafts Google search query (≤5 Chinese chars)
        │     │     ├── search() ──► BeautifulSoup scrapes Google result snippets
        │     │     └── organize() ──► LLM picks correct Chinese translation from results
        │     ├── consideration_test() ──► LLM translates segment to Chinese (HK voice)
        │     ├── recheck() ──► LLM post-processing: remove promotional content, ensure voice
        │     ├── engtit() ──► LLM generates English title from raw text
        │     ├── titler() ──► LLM generates clickbait Traditional Chinese title
        │     └── write_file() ──► outputs HTML to news/
        │           ├── append_to_sitemap() ──► sitemap.xml
        │           ├── append_to_news_sitemap() ──► news_sitemap.xml
        │           ├── add_rss_item() ──► category-specific XML + rss.xml (master)
        │           └── commit_changes() ──► git add --all, commit, push --force
        │
        └── GitHub Actions FTP Deploy ──► famechos.me hosting (ftpupload.net)
```

---

## File-by-File Breakdown

### Python Scripts

#### `新聞筆.py` (~1200 lines) — **Active Pipeline**
The primary news pipeline run by GitHub Actions (`main.yml`) every hour. Key characteristics:

| Aspect | Detail |
|---|---|
| **LLM Model** | `nvidia/llama-3.1-nemotron-70b-instruct` (via `openai.OpenAI` client pointed at NVIDIA endpoint) |
| **RSS Sources** | 10 feeds: Korea Herald, KoreaTimes, Yonhap, Tokyo Cheapo, J Rock News, J-GENERATION, Phoenix Talks Pop Culture Japan, jpopblog.com, The Soul of Seoul, 10mag |
| **Articles per run** | 1 |
| **Word count filter** | 500–1500 words |
| **Output** | Full HTML article with JSON-LD Schema.org `NewsArticle`, OpenGraph/Twitter Card meta, Google AdSense, YouTube embed, related-news section |

**Key functions:**
- `fetch_news(rss_urls)` — Pulls and deduplicates RSS entries from all 10 feeds, filters by last 2 weeks
- `split_article_into_segments(article)` — Splits article text into ~22-line segments (removes last segment if < 3 lines)
- `process_segments(segments, model)` — Two-phase proper-noun translation pipeline:
  1. `refine_response()` — LLM scans each segment for translatable proper nouns
  2. `websearch()` + `search()` + `organize()` — Google search → parse snippets → LLM determines correct Chinese name
- `consideration_test(title, segment, dictionary, model)` — Core translation engine; rewrites English text as Traditional Chinese news from a Hong Kong author's perspective, using the proper-noun dictionary
- `engtit(website_text, model)` — Generates an English title from the raw article text
- `titler(website_text, model)` — Generates a clickbait Traditional Chinese title
- `recheck(article, model)` — Post-processing LLM pass that removes promotional/marketing language
- `write_file(file_path, content, title, source, category, model)` — Writes the final HTML file with all SEO/meta/schema/Youtube/related-news, then updates sitemaps, RSS feeds, and commits
- `get_first_youtube_embed(query, model)` — Uses Selenium headless Chrome to scrape the first YouTube search result for the article topic
- `append_to_sitemap(loc, priority)` — Appends `<url>` entry to `sitemap.xml`
- `append_to_news_sitemap(url, date)` — Appends Google News-specific entry to `news_sitemap.xml`
- `add_rss_item(template_path, title, link, category, description)` — Appends `<item>` to a category RSS file AND the master `rss.xml`
- `commit_changes()` — `git add --all && git commit -m "讀萬卷書不如寫萬篇文" && git pull --strategy=recursive --strategy-option=theirs && git push --force`
- `count_chinese_characters(text)` — Regex-based Chinese character counter (filters out articles with < 100 Chinese characters)

---

#### `autonews.py` (~1222 lines) — **Original Pipeline (Legacy)**
The original version of the pipeline. Very similar to `新聞筆.py` but differs in:

| Feature | `autonews.py` | `新聞筆.py` |
|---|---|---|
| RSS sources | 4 | 10 |
| Articles/run | 5 | 1 |
| Word count filter | 300–1300 | 500–1500 |
| Title generation | Single LLM call (Chinese only) | Two-phase: English title → Chinese clickbait title |
| Schema.org type | `Article` | `NewsArticle` |
| AdSense | No | Yes (`ca-pub-3046601377771213`) |
| News sitemap | No | Yes |
| Chinese char validation | No | Yes (< 100 chars = skip) |
| Search results | List of strings | List of dicts with `title`/`snippet`/`link` |

`autonews.py` is **not actively run** by GitHub Actions but is retained for reference/fallback.

---

#### `compressor.py` (57 lines) — **Image Compressor**
Standalone utility that compresses all JPEG/PNG images in `images/` to ≤ 220KB.
- Uses Pillow (`PIL`)
- Reduces JPEG quality incrementally (starting at 85, decreasing by 5)
- Converts PNGs to JPEG (RGB mode), deletes original PNG
- Not run by any GitHub Actions workflow

---

#### `cleanup_sitemap.py` (46 lines) — **Sitemap Maintenance**
Run by GitHub Actions `reset.yml` every 2 days.
- Parses `news_sitemap.xml`
- Removes `<url>` entries where `news:publication_date` is older than 2 days
- Keeps the Google News sitemap lean and compliant

---

### XML Files

#### `rss.xml` (~5,457 lines) — **Master RSS Feed**
Aggregated RSS 2.0 feed containing **all** articles across all categories. Uses PubSubHubbub (`pubsubhubbub.appspot.com`) for real-time push notifications. Each `<item>` includes `title`, `link`, `category`, `pubDate` (HK timezone, RFC 2822), and `description` (first ~80 characters + "..."). Served at `https://www.famechos.me/rss.xml`.

#### `sitemap.xml` (~5,452 lines) — **Standard Sitemap**
XML sitemap for search engines (Google, Bing, etc.). Root URL gets `priority=1.00`; all news article URLs get `priority=0.90` and `changefreq=weekly`.

#### `news_sitemap.xml` (~2 lines) — **Google News Sitemap**
Google News-specific sitemap with `<news:news>` namespace. Typically near-empty because `cleanup_sitemap.py` prunes entries older than 2 days and `新聞筆.py` only processes 1 article/hour.

#### `k-pop.xml` (~2,415 lines) — **K-Pop Category RSS**
Category-specific RSS feed for K-Pop articles. Used for the "related news" section on K-Pop article pages.

#### `drama.xml` (~1,708 lines) — **Drama Category RSS**
Category-specific RSS feed for Korean/Japanese drama, film, and entertainment industry articles.

#### `others.xml` (~1,162 lines) — **Others Category RSS**
Category-specific RSS feed for travel, food, culture, lifestyle, and tech articles.

#### `j-pop.xml` (~202 lines) — **J-Pop Category RSS**
Category-specific RSS feed for Japanese music/rock articles. Smallest feed.

---

### Data Files

#### `news.txt` (~1,617 lines)
Plain-text deduplication database. One source URL per line. Before processing, the pipeline reads this file and skips any article whose URL is already present. Grows monotonically — never pruned. Contains URLs from all 10 source feeds, dating from October 2024 to present.

#### `requirements.txt`
Python dependencies: `feedparser` (RSS), `openai` (LLM client), `trafilatura` (web scraping), `beautifulsoup4` (HTML parsing), `requests`, `ipython`, `langchain` + `langgraph`, `typing-extensions`. Note: `selenium` and `Pillow` are installed separately in the GitHub Actions workflow YAML.

---

### Static Assets

#### `images/banner.jpg` (55 KB)
The site's main OG/social-media image. Used as `og:image` and `twitter:image` in every article's `<head>`.

#### `chromedriver-linux64/` + `chromedriver-linux64.zip`
ChromeDriver binaries for Selenium-based YouTube scraping in the GitHub Actions runner environment.

---

### HTML News Articles (`news/` directory)

**Count**: ~922 HTML files (and growing at ~1/hour).

Each article is a self-contained HTML page with:
- **JSON-LD structured data**: `NewsArticle`, `Organization`, `WebSite` (schema.org)
- **SEO meta tags**: `description`, `keywords`, `canonical`, `robots`, `author`, `referrer`
- **OpenGraph & Twitter Card**: `og:title`, `og:description`, `og:image`, `og:locale=zh_TW`, `twitter:card=summary_large_image`
- **Google AdSense**: Publisher ID `ca-pub-3046601377771213`
- **CDN resources**: Bootstrap Icons, Noto Sans TC font (Google Fonts)
- **6 CSS stylesheets**: `main-nav.css`, `main-content.css`, `main-small.css`, `post.css`, `main-footer.css`, `news.css`
- **Full navigation bar**: Hamburger menu, 4 category links (K-POP, J-POP, 影視, 其他), About/Privacy
- **Article body**: LLM-translated content in Traditional Chinese with Hong Kong writing style — mix of `<p>`, `<h2>`, `<ul>`, `<ol>`, `<table>` elements
- **YouTube embed**: One relevant video found via Selenium search
- **Related news**: 3 links pulled from the category-specific RSS feed
- **Footer**: Social media links (Instagram, Twitter/X, YouTube), copyright
- **External JS**: `nav-list.js` for mobile navigation

---

### CI/CD (`.github/workflows/`)

| Workflow | Trigger | Function |
|---|---|---|
| `main.yml` | Cron: every hour (`0 */1 * * *`) + manual | Runs `python 新聞筆.py`, installs Chrome + Selenium + Python deps, commits changes, FTP deploys all HTML/XML/CSS/JS/images to `ftpupload.net/htdocs/`, pings PubSubHubbub |
| `ftp.yml` | Manual only (`workflow_dispatch`) | FTP deploy of repo contents (no pipeline run) |
| `reset.yml` | Cron: every 2 days at midnight | Runs `python cleanup_sitemap.py`, commits and pushes |

---

## Content Flow Summary

1. **GitHub Actions** triggers `新聞筆.py` every hour
2. Script fetches 10 RSS feeds → filters recent unduplicated English articles
3. Picks 1 unprocessed article → scrapes full text via `trafilatura`
4. Splits text into segments → LLM identifies proper nouns → Google searches for correct Chinese translations
5. LLM translates each segment to Traditional Chinese (Hong Kong voice)
6. LLM generates clickbait Chinese title + English title
7. Selenium Chrome scrapes YouTube for a relevant video embed
8. Writes full HTML file with SEO meta, JSON-LD schema, AdSense, YouTube, related news
9. Updates `sitemap.xml`, `news_sitemap.xml`, category RSS, master `rss.xml`
10. `git add --all`, commits with message "讀萬卷書不如寫萬篇文", force-pushes
11. GitHub Actions FTP deploys all web-facing files to production hosting
12. Every 2 days: `cleanup_sitemap.py` prunes old entries from Google News sitemap

---

## Key Design Decisions & Observations

- **`git push --force`**: The pipeline force-pushes because it races with other workflow runs (hourly cron may overlap if a run takes > 1 hour). Uses `pull --strategy-option=theirs` before pushing to minimize conflicts.
- **No database**: Deduplication relies entirely on `news.txt` (a flat text file) and the absence of a URL in it. This is simple but could become slow as the file grows.
- **LLM as translation engine**: Rather than using a traditional MT system, the entire translation/localization is done via LLM prompting — this gives the Hong Kong writing style and clickbait tone.
- **Monolithic script**: Both `新聞筆.py` and `autonews.py` are single ~1200-line files with no modular imports between them. They share nearly identical code with only feature differences.
- **Hardcoded secrets**: API keys for NVIDIA are embedded directly in the Python source files (not environment variables).
- **1 article/hour**: The conservative rate likely avoids rate-limiting on the LLM API and Google search scraping, while ensuring a steady content drip.
