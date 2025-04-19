import psycopg2
import praw
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import time
import json
import os
from transformers import pipeline

# --- Load ticker symbols from file ---
with open("all_tickers.txt", "r") as f:
    VALID_TICKERS = set(t.strip().upper() for t in f.readlines())

# --- Setup sentiment analyzer ---
analyzer = SentimentIntensityAnalyzer()

# --- Setup local summarization model (BART) ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Ticker extraction logic ---
def extract_ticker(title, body):
    combined = (title + " " + body).upper()
    match = re.search(r'\$[A-Z]{2,5}', combined)
    if match:
        return match.group()[1:]  # Remove $
    for word in title.upper().split():
        if word in VALID_TICKERS:
            return word
    return None

# --- Summarize using ML model ---
def summarize(text):
    try:
        return summarizer(text[:1024], max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
    except Exception as e:
        return text[:200] + "..." if len(text) > 200 else text

def load_last_scrape_time():
    cursor.execute("SELECT last_scraped_utc FROM scrape_state WHERE id = 'wsb';")
    row = cursor.fetchone()
    return row[0].timestamp() if row and row[0] else 0.0  # Convert to float for comparison

def save_last_scrape_time(timestamp_float):
    dt = datetime.utcfromtimestamp(timestamp_float)
    cursor.execute("""
        INSERT INTO scrape_state (id, last_scraped_utc)
        VALUES ('wsb', %s)
        ON CONFLICT (id) DO UPDATE SET last_scraped_utc = EXCLUDED.last_scraped_utc;
    """, (dt,))

# --- Reddit API ---
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="wsb-scraper"
)

# --- PostgreSQL connection ---
conn = psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    port=5432,
    dbname="postgres",
    user="postgres.zvlnpzespbffztxlnlea",
    password=os.getenv("SUPABASE_PASSWORD"),
    sslmode="require"
)
cursor = conn.cursor()

# --- Scrape DD Posts ---
last_scraped = load_last_scrape_time()
latest_scraped = last_scraped
subreddit = reddit.subreddit("wallstreetbets")
for submission in subreddit.new(limit=100):
    if submission.link_flair_text != "DD":
        continue
    #if submission.created_utc <= last_scraped:
       # continue

    created_time = datetime.utcfromtimestamp(submission.created_utc)
    scraped_time = datetime.utcnow()
    sentiment = analyzer.polarity_scores(submission.selftext)
    ticker = extract_ticker(submission.title, submission.selftext)
    tldr = summarize(submission.selftext)

    # --- Insert into dd_post ---
    cursor.execute("""
        INSERT INTO dd_post (
            post_id, title, author, post_text, tldr, created_utc, scraped_at,
            sentiment_compound, sentiment_pos, sentiment_neu, sentiment_neg, detected_ticker
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING;
    """, (
        submission.id, submission.title, str(submission.author), submission.selftext,
        tldr, created_time, scraped_time,
        sentiment['compound'], sentiment['pos'], sentiment['neu'], sentiment['neg'],
        ticker
    ))

    # --- Scrape and insert comments ---
    comment_scores = []
    submission.comments.replace_more(limit=0)
    for comment in submission.comments:
        if hasattr(comment, "body"):
            csent = analyzer.polarity_scores(comment.body)

            # Only include and insert meaningful sentiment scores
            if abs(csent['compound']) > 0.1:
                comment_scores.append(csent['compound'])

                cursor.execute("""
                    INSERT INTO post_comment (
                        comment_id, post_id, author, comment_text, created_utc,
                        sentiment_compound, sentiment_pos, sentiment_neu, sentiment_neg, scraped_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (comment_id) DO NOTHING;
                """, (
                    comment.id, submission.id, str(comment.author), comment.body,
                    datetime.utcfromtimestamp(comment.created_utc),
                    csent['compound'], csent['pos'], csent['neu'], csent['neg'], scraped_time
                ))

    # --- Determine action ---
    avg_comment_score = sum(comment_scores) / len(comment_scores) if comment_scores else 0.0
    post_score = sentiment['compound']
    if post_score > 0.9 and avg_comment_score > 0.5:
        decision = "buy"
    elif post_score < -0.9 and avg_comment_score < -0.5:
        decision = "sell"
    else:
        decision = "hold"

    # --- Insert into action table ---
    cursor.execute("""
        INSERT INTO action (
            post_id, detected_ticker, post_score, avg_comment_score, decision, tldr
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING;
    """, (
        submission.id, ticker, post_score, avg_comment_score, decision, tldr
    ))

    print(f"✅ {submission.id} | {ticker} | Post: {post_score:.2f}, Comments: {avg_comment_score:.2f} → {decision.upper()}")

    latest_scraped = max(latest_scraped, submission.created_utc)

save_last_scrape_time(latest_scraped)

conn.commit()
cursor.close()
conn.close()