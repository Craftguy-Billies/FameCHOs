# FameCHOs Project Analysis

## What Is This?

**FameCHOs** (https://www.famechos.me/) is a **fully automated, AI-powered news aggregation website** focused on Korean and Japanese entertainment and culture. It scrapes RSS feeds from news sources, uses NVIDIA's LLM API to rewrite/translate articles into Chinese, embeds related YouTube videos, publishes HTML pages, updates RSS feeds and sitemaps, and auto-deploys via FTP — all running on a GitHub Actions cron schedule every hour.

---

## Architecture

### Core Scripts

| File | Purpose |
|------|---------|
| `新聞筆.py` | **Active news automation script.** Runs on GitHub Actions hourly. Uses NVIDIA API (Llama 3.1 405B). Processes 1 article per run from 10 RSS feeds. |
| `autonews.py` | Older/original version. Uses different NVIDIA model (Nemotron 70B). Processes up to 5 articles per run from 4 RSS feeds. Not actively scheduled. |
| `compressor.py` | Image compression utility (PIL). Compresses JPEG/PNG images in `./images/` to under 220KB. |
| `cleanup_sitemap.py` | Removes `news_sitemap.xml` entries older than 2 days. Runs every 2 days via GitHub Actions. |

### Data / Content Files

| File | Purpose |
|------|---------|
| `rss.xml` | Main combined RSS feed |
| `k-pop.xml`, `j-pop.xml`, `drama.xml`, `others.xml` | Category-specific RSS feeds |
| `sitemap.xml` | Standard XML sitemap (~300+ URLs) |
| `news_sitemap.xml` | Google News sitemap |
| `news.txt` | Tracks already-processed article URLs to avoid duplicates |
| `news/` | ~300+ generated HTML articles (Traditional Chinese) |

### Infrastructure

| File | Purpose |
|------|---------|
| `.github/workflows/main.yml` | **Primary pipeline** — runs hourly: executes `新聞筆.py`, commits, deploys via FTP |
| `.github/workflows/ftp.yml` | Manual FTP deployment trigger |
| `.github/workflows/reset.yml` | Runs `cleanup_sitemap.py` every 2 days |
| `chromedriver-linux64/` | ChromeDriver binaries for Selenium YouTube search |

---

## How It Works (Pipeline)

1. **Scheduled Trigger** — GitHub Actions runs every hour (`cron: '0 */1 * * *'`)
2. **Fetch News** — Pulls from 10 RSS feeds (Korea Herald, Korea Times, Yonhap, Tokyo Cheapo, JRockNews, J-GENERATION, Phoenix Talks Pop Culture Japan, Jpopblog, The Soul of Seoul, 10mag)
3. **Filter** — Only articles from last 2 weeks, between 500–1500 words, not previously processed
4. **AI Rewrite** — Article text is split into segments, each rewritten/translated into Chinese by NVIDIA's Llama 3.1 405B, given a clickbait-style Chinese title
5. **YouTube Embed** — Uses Selenium + headless Chrome to search YouTube for a related video, generates an `<iframe>` embed
6. **HTML Generation** — Creates a full HTML page with article, video, related news links, navigation, and footer (links to famechos.me)
7. **Sitemap/RSS Update** — Appends entries to `sitemap.xml`, `rss.xml`, and category-specific RSS feeds
8. **Git Auto-Commit** — Commits with message "讀萬卷書不如寫萬篇文"
9. **FTP Deploy** — Pushes all files (excluding source scripts) to `ftpupload.net` hosting
10. **PubSubHubbub Ping** — Notifies Google's hub of RSS updates

---

## Content Categories

- **K-Pop**: BTS, BLACKPINK, NewJeans, G-Dragon, Seventeen, ATEEZ, HYBE, SM Entertainment, etc.
- **J-Pop**: DIR EN GREY, HYDE, BAND-MAID, Perfume, Hatsune Miku, anime theme songs, visual kei bands
- **Drama (影視)**: K-Dramas, Netflix series, Korean films, Squid Game
- **Others (其他)**: Tokyo/Seoul travel guides, food, shopping, cultural events, language learning

All articles are in **Traditional Chinese**, targeting Chinese-speaking audiences interested in Korean/Japanese pop culture. The site domain is `https://www.famechos.me/`.

---

## Key Observations

### Security Concerns
1. **Hardcoded NVIDIA API keys** exposed in both Python scripts
2. **Git remote URL contains an embedded access token** (visible in `git remote -v`)
3. `git push --force` used in the commit function (destructive, could overwrite others' work)

### Code Quality
1. Significant code duplication between `autonews.py` and `新聞筆.py`
2. No tests, no modular structure
3. Mixed Chinese/English naming and comments
4. All content is AI-generated without explicit disclosure to readers

---

## Recommendations (for future work)
1. Extract shared code into a common library module
2. Move API keys to environment variables / GitHub Secrets
3. Replace `git push --force` with proper merge strategy
4. Add transparency about AI-generated content
5. Add proper error handling and retry logic
6. Remove embedded token from git remote URL
