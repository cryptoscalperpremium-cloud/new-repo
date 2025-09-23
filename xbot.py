#!/usr/bin/env python3
"""
Crypto Influencer Bot
Automatically generates and posts crypto-related tweets using AI and real-time data.
"""

import asyncio
import aiohttp
import json
import logging
import random
import time
import tweepy
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, time as dt_time, timedelta
from typing import Optional, Dict, List, Tuple
from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "Bot is alive!"

def run():
    app.run(host='0.0.0.0', port=5000)

t = Thread(target=run)
t.start()

# ===== CONFIGURATION =====
# API Keys (In a real production environment, these should be in environment variables or secure storage)
TWITTER_CONSUMER_KEY = "mkVrWM7iDgHF06alazfi4YeD9"
TWITTER_CONSUMER_SECRET = "gwIhsX7KtZmnFoTAPPAArArTHcHU5bi0C0fNt56UVPe5dOqEJ0"
TWITTER_ACCESS_TOKEN = "1745084568875532288-pDZ35wQrBCIoCHQpRef9RWr7EY8r2n"
TWITTER_ACCESS_TOKEN_SECRET = "n7vCY15vIIBLK5VnRqBFFEQsiyR85QNllEkqrgLRYHb5X"

# CryptoPanic API (free tier)
CRYPTO_PANIC_API = "https://cryptopanic.com/api/v1/posts/?auth_token=8445d929d46605242784de13df62084d7620bb7f&kind=news"

# CoinGecko API (no key needed)
COINGECKO_API = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false"

# AI Model settings
MODEL_MAX_LENGTH = 100

# Bot Settings - Changed from 2 hours to 200 posts per day
POSTS_PER_DAY = 200
MIN_INTERVAL_MINUTES = 4  # Minimum time between posts (4 minutes)
MAX_INTERVAL_MINUTES = 12  # Maximum time between posts (12 minutes)

# Hashtags pool
CRYPTO_HASHTAGS = [
    "#Bitcoin", "#BTC", "#Ethereum", "#ETH", "#Crypto", "#Blockchain", 
    "#DeFi", "#NFT", "#Web3", "#HODL", "#Altcoin", "#Bullish", "#Bearish",
    "#CryptoNews", "#ToTheMoon", "#DYOR"
]

# ===== INITIALIZATION =====
# Fix for Windows Unicode encoding issue
def setup_logging():
    class UnicodeEscapeHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                # Escape Unicode characters that can't be encoded
                msg = msg.encode('unicode_escape').decode('ascii')
                self.stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("crypto_bot.log", encoding='utf-8'),
            UnicodeEscapeHandler()
        ]
    )

setup_logging()
logger = logging.getLogger("CryptoInfluencerBot")

# Initialize AI components (with fallback)
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    logger.info("AI model loaded successfully")
    AI_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not available. Using simple text generation.")
    AI_AVAILABLE = False
except Exception as e:
    logger.error(f"Failed to load AI model: {e}")
    AI_AVAILABLE = False

# Initialize Twitter API
try:
    twitter_client = tweepy.Client(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
    )
    logger.info("Twitter client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Twitter client: {e}")
    exit(1)

# ===== CORE FUNCTIONS =====
async def fetch_with_retry(session: aiohttp.ClientSession, url: str, retries: int = 3) -> Optional[Dict]:
    """Fetch data from API with retry mechanism."""
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API returned status {response.status} on attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"API fetch attempt {attempt + 1} failed: {e}")
        
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to fetch data from {url} after {retries} attempts")
    return None

async def get_crypto_news() -> List[Dict]:
    """Fetch latest crypto news from CryptoPanic."""
    async with aiohttp.ClientSession() as session:
        data = await fetch_with_retry(session, CRYPTO_PANIC_API)
        if data and 'results' in data:
            return data['results'][:5]  # Return top 5 news items
    return []

async def get_crypto_prices() -> List[Dict]:
    """Fetch current prices for top 10 cryptocurrencies."""
    async with aiohttp.ClientSession() as session:
        data = await fetch_with_retry(session, COINGECKO_API)
        if data:
            return data
    return []

def analyze_price_trends(prices: List[Dict]) -> str:
    """Analyze price trends and generate a summary."""
    if not prices:
        return ""
    
    trends = []
    for coin in prices[:3]:  # Analyze top 3 coins
        name = coin['symbol'].upper()
        price = coin['current_price']
        change = coin['price_change_percentage_24h'] or 0
        
        if change > 5:
            trend = f"{name} is pumping! (+{change:.1f}%)"
        elif change > 2:
            trend = f"{name} is rising (+{change:.1f}%)"
        elif change < -5:
            trend = f"{name} is dumping! ({change:.1f}%)"
        elif change < -2:
            trend = f"{name} is dropping ({change:.1f}%)"
        else:
            trend = f"{name} is stable ({change:+.1f}%)"
        
        trends.append(trend)
    
    return " ".join(trends)

def generate_ai_content(news: List[Dict], trends: str) -> str:
    """Generate engaging crypto content using AI or fallback method."""
    # Prepare context for AI
    news_titles = [item.get('title', '') for item in news[:3]]
    context = f"Crypto news: {' '.join(news_titles)}. Market trends: {trends}. "
    
    if AI_AVAILABLE:
        try:
            import torch
            prompt = context + "Exciting tweet about cryptocurrency:"
            
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=50, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=MODEL_MAX_LENGTH,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new content (after the prompt)
            ai_content = generated_text[len(prompt):].strip()
            
            # Clean up and ensure it's tweet-friendly
            ai_content = ai_content.split('.')[0]  # Take only the first sentence
            ai_content = ai_content.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            
            return ai_content
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
    
    # Fallback text generation if AI is not available
    fallback_phrases = [
        "The crypto market is showing interesting movements today!",
        "Exciting developments in the blockchain space!",
        "Crypto markets are active with notable price action.",
        "Interesting trends emerging in the cryptocurrency world.",
        "The blockchain ecosystem continues to evolve rapidly."
    ]
    
    return random.choice(fallback_phrases)

def select_hashtags() -> str:
    """Select relevant hashtags for the tweet."""
    return " ".join(random.sample(CRYPTO_HASHTAGS, 4))

async def generate_tweet() -> Optional[str]:
    """Generate a complete tweet using news, prices, and AI."""
    try:
        # Fetch data concurrently
        news, prices = await asyncio.gather(
            get_crypto_news(),
            get_crypto_prices()
        )
        
        if not news and not prices:
            logger.warning("No data available for tweet generation")
            return None
        
        # Analyze trends
        trends = analyze_price_trends(prices) if prices else ""
        
        # Generate content
        ai_content = generate_ai_content(news, trends)
        
        # Select hashtags
        hashtags = select_hashtags()
        
        # Construct tweet (ensure it's under 280 characters)
        if trends:
            tweet = f"{ai_content} {trends} {hashtags}"
        else:
            tweet = f"{ai_content} {hashtags}"
        
        # Truncate if necessary (rare but possible)
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        
        return tweet
        
    except Exception as e:
        logger.error(f"Tweet generation failed: {e}")
        return None

async def post_tweet() -> None:
    """Generate and post a tweet with error handling."""
    logger.info("Starting tweet generation cycle...")
    
    tweet = await generate_tweet()
    if not tweet:
        logger.warning("Skipping tweet post due to generation failure")
        return
    
    logger.info(f"Generated tweet: {tweet}")
    
    # Post to Twitter
    try:
        response = twitter_client.create_tweet(text=tweet)
        if response and response.data:
            logger.info(f"Tweet posted successfully! Tweet ID: {response.data['id']}")
        else:
            logger.warning("Tweet posted but no response data received")
    except tweepy.TooManyRequests:
        logger.warning("Rate limit exceeded. Skipping this tweet.")
    except Exception as e:
        logger.error(f"Failed to post tweet: {e}")

# ===== SCHEDULING =====
def schedule_next_post(scheduler: AsyncIOScheduler) -> None:
    """Schedule the next post at a random interval."""
    # Calculate random interval between MIN and MAX minutes
    interval_minutes = random.randint(MIN_INTERVAL_MINUTES, MAX_INTERVAL_MINUTES)
    
    # Schedule the next post
    next_time = datetime.now() + timedelta(minutes=interval_minutes)
    scheduler.add_job(
        post_tweet_and_reschedule,
        'date',
        run_date=next_time,
        args=[scheduler],
        id=f"tweet_job_{int(time.time())}"
    )
    
    logger.info(f"Next tweet scheduled in {interval_minutes} minutes at {next_time.strftime('%H:%M:%S')}")

async def post_tweet_and_reschedule(scheduler: AsyncIOScheduler) -> None:
    """Post a tweet and schedule the next one."""
    await post_tweet()
    schedule_next_post(scheduler)

async def main() -> None:
    """Main function to run the bot."""
    logger.info("Crypto Influencer Bot is running!")  # Removed emoji to fix encoding issue
    
    # Create scheduler
    scheduler = AsyncIOScheduler()
    scheduler.start()
    
    # Schedule the first post
    schedule_next_post(scheduler)
    
    # Keep the script running
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down Crypto Influencer Bot...")
        scheduler.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:

        logger.error(f"Unexpected error: {e}")


from flask import Flask
from pyngrok import ngrok

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot is alive and running!"

# Open a tunnel on port 5000
public_url = ngrok.connect(5000)
print("🌍 Public URL:", public_url)

# Run Flask app (non-blocking so bot + scheduler keep working)
import threading
threading.Thread(target=lambda: app.run(port=5000)).start()



