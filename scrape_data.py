import requests
from bs4 import BeautifulSoup

def scrape_latest_articles(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = []
    for item in soup.find_all('p'):
        articles.append(item.text.strip())

    return articles[:10]  # Return the first 10 extracted texts

# Example usage
new_articles = scrape_latest_articles("https://example.com/news")
