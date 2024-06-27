import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def fetchURL(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extractLinks(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for l in soup.find_all('a', href=True):
        link = l['href']
        fLink = urljoin(url, link)
        links.append(fLink)
    return links

def extractText(html):
    soup = BeautifulSoup(html, 'html.parser')
    # text = soup.get_text()
    text = ""
    main = soup.find(id="main-content")
    if main:
        text = main.get_text()
    # paragraphs = text.split('\n\n')  # Split by paragraphs; adjust if needed
    return text

def getData(url):
    base_html = fetchURL(url)
    links = extractLinks(base_html,url)
    text = ""
    for l in links:
        html = fetchURL(l)
        text += extractText(html)
    return text

data = getData('https://stanford-cs324.github.io/winter2022/lectures/')
filename = 'data0.txt'
with open(filename, "w", encoding="utf-8") as file:
    file.write(data)