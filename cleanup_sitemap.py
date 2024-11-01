import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pytz

# Constants
SITEMAP_FILE = 'news_sitemap.xml'
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Get the current time in UTC
now = datetime.now(pytz.utc)

# Define the time threshold (2 days ago)
time_threshold = now - timedelta(days=2)

# Parse the XML file
tree = ET.parse(SITEMAP_FILE)
root = tree.getroot()

# Namespace mapping for news
namespaces = {
    'news': 'http://www.google.com/schemas/sitemap-news/0.9',
    '': 'http://www.sitemaps.org/schemas/sitemap/0.9'
}

# Find all <url> elements
urls_to_remove = []

for url in root.findall('url', namespaces):
    # Find the <news:publication_date> element within the <url>
    publication_date_elem = url.find('news:news/news:publication_date', namespaces)
    
    if publication_date_elem is not None:
        # Parse the publication date
        publication_date_str = publication_date_elem.text.strip()
        publication_date = datetime.strptime(publication_date_str, DATE_FORMAT)
        
        # Compare the publication date to the current time minus 2 days
        if publication_date < time_threshold:
            # Mark this <url> for removal if older than 2 days
            urls_to_remove.append(url)

# Remove the outdated <url> entries
for url in urls_to_remove:
    root.remove(url)

# Write the updated XML back to the file
tree.write(SITEMAP_FILE, encoding='utf-8', xml_declaration=True)

print(f"Removed {len(urls_to_remove)} outdated URLs from the sitemap.")
