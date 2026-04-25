#Alex Eliseev
#This is python script file that reads the URLs for real news articles from the gossipcop_real.csv downloaded from kaggle which contains
#over 15000 links to real news articles that were published. It scrapes the news page, saving its title + text body into a text document
#in the FakeNewsArticles folder.
#Script is basically the same as for the one for donwloading fakes just slightly different file paths.
#Uses the newspaper3k framework for python to scrape the news paper page for its body text
from newspaper import Article
import pandas as pd
import os
import time


csv_path = "DS8008FinalProject/Data/src/GossipCopDataset/RealNewsArticlesLinks/gossipcop_real.csv"
output_folder = "DS8008FinalProject/Data/train/RealNewsArticles"
df = pd.read_csv(csv_path)

for i, row in df.iterrows():
    #Grabbing next url from the list in the dataset from kaggle.
    url = row["news_url"]
    title = row["title"]
    if not str(url).startswith("http"):
        url = "https://" + str(url)
    
    #Trying to download the article, skipping if its < 200 words long because probably no longer available e.t.c
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if len(text) < 200:
            print("Skipping, article too short")
            continue
        # Save using ID (cleaner than URL filenames)
        filepath = os.path.join(output_folder, f"RealArticle{i}.txt")
        
        #Writing text contents to the .txt in the folder
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(title + " ")
            f.write(text)

        #Download delay to not get IP banned, the dataset does alternate between news sites and there only max 200 news articles that we
        #pulling from any indivudal news site so hopefully no IP ban this way. Will take a lil bit to download them all but oh well.
        time.sleep(1)
    
    #Main error is the article being no longer available.
    except Exception as e:
        print(f"[{i}] Failed:", url, "->", e)       