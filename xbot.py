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
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from flask import Flask
import threading
from pyngrok import ngrok

# ===== FLASK + NGROK =====
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Crypto Influencer Bot is alive!"

def run_flask():
    app.run(host='0.0.0.0', port=5000)


# Set your authtoken
ngrok.set_auth_token("336sN9ZnXMTWyOwcsXsWWamabsR_2gXbYhtXiFsD7tnQz8hDj")

# Start Flask server in a separate thread
threading.Thread(target=run_flask, daemon=True).start()

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("🌍 Public URL (use this in UptimeRobot):", public_url)

# ===== LOGGING =====
def setup_logging():
    class UnicodeEscapeHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
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

# ===== CONFIGURATION =====
TWITTER_CONSUMER_KEY = "mkVrWM7iDgHF06alazfi4YeD9"
TWITTER_CONSUMER_SECRET = "gwIhsX7KtZmnFoTAPPAArArTHcHU5bi0C0fNt56UVPe5dOqEJ0"
TWITTER_ACCESS_TOKEN = "1745084568875532288-pDZ35wQrBCIoCHQpRef9RWr7EY8r2n"
TWITTER_ACCESS_TOKEN_SECRET = "n7vCY15vIIBLK5VnRqBFFEQsiyR85QNllEkqrgLRYHb5X"
CRYPTO_PANIC_API = "https://cryptopanic.com/api/v1/posts/?auth_token=8445d929d46605242784de13df62084d7620bb7f&kind=news"
COINGECKO_API = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false"

MODEL_MAX_LENGTH = 100
POSTS_PER_DAY = 200
MIN_INTERVAL_MINUTES = 4
MAX_INTERVAL_MINUTES = 12

CRYPTO_HASHTAGS = [
    "#Bitcoin", "#BTC", "#Ethereum", "#ETH", "#Crypto", "#Blockchain", 
    "#DeFi", "#NFT", "#Web3", "#HODL", "#Altcoin", "#Bullish", "#Bearish",
    "#CryptoNews", "#ToTheMoon", "#DYOR"
]

# ===== AI MODEL =====
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    logger.info("AI model loaded successfully")
    AI_AVAILABLE = True
except Exception as e:
    logger.warning(f"AI not available: {e}")
    AI_AVAILABLE = False

# ===== TWITTER CLIENT =====
try:
    twitter_client = tweepy.Client(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
    )
    logger.info("Twitter client initialized successfully")
except Exception as e:
    logger.error(f"Twitter client failed: {e}")
    exit(1)

# ===== CORE FUNCTIONS =====
async def fetch_with_retry(session: aiohttp.ClientSession, url: str, retries: int = 3) -> Optional[Dict]:
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API returned status {response.status} on attempt {attempt+1}")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
        await asyncio.sleep(2 ** attempt)
    logger.error(f"Failed to fetch data from {url}")
    return None

async def get_crypto_news() -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        data = await fetch_with_retry(session, CRYPTO_PANIC_API)
        if data and 'results' in data:
            return data['results'][:5]
    return []

async def get_crypto_prices() -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        data = await fetch_with_retry(session, COINGECKO_API)
        return data if data else []

def analyze_price_trends(prices: List[Dict]) -> str:
    trends = []
    for coin in prices[:3]:
        name = coin['symbol'].upper()
        price = coin['current_price']
        change = coin['price_change_percentage_24h'] or 0
        if change > 5: trends.append(f"{name} is pumping! (+{change:.1f}%)")
        elif change > 2: trends.append(f"{name} is rising (+{change:.1f}%)")
        elif change < -5: trends.append(f"{name} is dumping! ({change:.1f}%)")
        elif change < -2: trends.append(f"{name} is dropping ({change:.1f}%)")
        else: trends.append(f"{name} is stable ({change:+.1f}%)")
    return " ".join(trends)

def generate_ai_content(news: List[Dict], trends: str) -> str:
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
            ai_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ai_content = ai_content[len(prompt):].strip().split('.')[0]
            return ai_content
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
    fallback_phrases = [
        "The crypto market is showing interesting movements today!",
        "Exciting developments in the blockchain space!",
        "Crypto markets are active with notable price action.",
        "Interesting trends emerging in the cryptocurrency world.",
        "The blockchain ecosystem continues to evolve rapidly."
    ]
    return random.choice(fallback_phrases)

def select_hashtags() -> str:
    return " ".join(random.sample(CRYPTO_HASHTAGS, 4))

async def generate_tweet() -> Optional[str]:
    news, prices = await asyncio.gather(get_crypto_news(), get_crypto_prices())
    if not news and not prices: return None
    trends = analyze_price_trends(prices) if prices else ""
    ai_content = generate_ai_content(news, trends)
    hashtags = select_hashtags()
    tweet = f"{ai_content} {trends} {hashtags}" if trends else f"{ai_content} {hashtags}"
    return tweet[:277] + "..." if len(tweet) > 280 else tweet

async def post_tweet() -> None:
    tweet = await generate_tweet()
    if not tweet:
        logger.warning("Skipping tweet post due to generation failure")
        return
    logger.info(f"Generated tweet: {tweet}")
    try:
        response = twitter_client.create_tweet(text=tweet)
        if response and response.data:
            logger.info(f"Tweet posted! ID: {response.data['id']}")
    except tweepy.TooManyRequests:
        logger.warning("Rate limit exceeded. Skipping this tweet.")
    except Exception as e:
        logger.error(f"Failed to post tweet: {e}")

# ===== SCHEDULING =====
def schedule_next_post(scheduler: AsyncIOScheduler) -> None:
    interval_minutes = random.randint(MIN_INTERVAL_MINUTES, MAX_INTERVAL_MINUTES)
    next_time = datetime.now() + timedelta(minutes=interval_minutes)
    scheduler.add_job(post_tweet_and_reschedule, 'date', run_date=next_time, args=[scheduler], id=f"tweet_job_{int(time.time())}")
    logger.info(f"Next tweet scheduled in {interval_minutes} minutes at {next_time.strftime('%H:%M:%S')}")

async def post_tweet_and_reschedule(scheduler: AsyncIOScheduler) -> None:
    await post_tweet()
    schedule_next_post(scheduler)

async def main() -> None:
    logger.info("Crypto Influencer Bot is running!")
    scheduler = AsyncIOScheduler()
    scheduler.start()
    schedule_next_post(scheduler)
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down bot...")
        scheduler.shutdown()

# ===== RUN BOT =====
if __name__ == "__main__":
    asyncio.run(main())

