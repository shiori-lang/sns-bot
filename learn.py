"""
スタイル学習モジュール
- 自分のInstagram・X投稿をAPIで取得
- 競合URLからPlaywrightでスクレイピング
- Claudeでスタイル分析 → style_guide.json に保存
"""
import json
import os
import re
import asyncio
import logging
from pathlib import Path

import requests
import tweepy
import anthropic
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent))
STYLE_GUIDE_PATH = BASE_DIR / "style_guide.json"

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_USER_ID      = os.getenv("INSTAGRAM_USER_ID")
X_API_KEY              = os.getenv("X_API_KEY")
X_API_SECRET           = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN         = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET  = os.getenv("X_ACCESS_TOKEN_SECRET")
X_BEARER_TOKEN         = os.getenv("X_BEARER_TOKEN")
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ════════════════════════════════════════════════════
#  スタイルガイドの読み書き
# ════════════════════════════════════════════════════
def load_style_guide() -> dict:
    if STYLE_GUIDE_PATH.exists():
        return json.loads(STYLE_GUIDE_PATH.read_text(encoding="utf-8"))
    return {
        "own_posts": {"instagram": [], "x": []},
        "reference_posts": [],
        "style_analysis": "",
        "caption_examples": {"instagram": [], "x": [], "line": []}
    }

def save_style_guide(guide: dict):
    STYLE_GUIDE_PATH.write_text(
        json.dumps(guide, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# ════════════════════════════════════════════════════
#  自分の投稿をAPIで取得
# ════════════════════════════════════════════════════
def fetch_own_instagram_posts(limit: int = 50) -> list[str]:
    """Instagram Graph APIで自分の投稿キャプションを取得"""
    try:
        url = f"https://graph.instagram.com/v21.0/{INSTAGRAM_USER_ID}/media"
        r = requests.get(url, params={
            "fields": "caption,media_type,timestamp",
            "limit": limit,
            "access_token": INSTAGRAM_ACCESS_TOKEN
        })
        data = r.json().get("data", [])
        captions = [
            d["caption"] for d in data
            if d.get("caption") and d.get("media_type") in ("IMAGE", "CAROUSEL_ALBUM")
        ]
        logger.info(f"Instagram: {len(captions)}件取得")
        return captions
    except Exception as e:
        logger.error(f"Instagram取得エラー: {e}")
        return []

def fetch_own_x_posts(limit: int = 100) -> list[str]:
    """X APIで自分のツイートを取得"""
    try:
        client = tweepy.Client(
            bearer_token=X_BEARER_TOKEN,
            consumer_key=X_API_KEY,
            consumer_secret=X_API_SECRET,
            access_token=X_ACCESS_TOKEN,
            access_token_secret=X_ACCESS_TOKEN_SECRET
        )
        me = client.get_me()
        user_id = me.data.id
        tweets = client.get_users_tweets(
            id=user_id,
            max_results=min(limit, 100),
            tweet_fields=["text", "created_at"],
            exclude=["retweets", "replies"]
        )
        texts = [t.text for t in (tweets.data or []) if not t.text.startswith("RT")]
        logger.info(f"X: {len(texts)}件取得")
        return texts
    except tweepy.errors.Forbidden as e:
        if "402" in str(e) or "credits" in str(e).lower() or "payment" in str(e).lower():
            raise RuntimeError("X_CREDIT_REQUIRED")
        raise
    except Exception as e:
        logger.error(f"X取得エラー: {e}")
        return []

# ════════════════════════════════════════════════════
#  競合URLからスクレイピング（Playwright）
# ════════════════════════════════════════════════════
async def scrape_url(url: str) -> list[str]:
    """URLからSNS投稿テキストを抽出"""
    from playwright.async_api import async_playwright

    texts = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)

            # Instagram投稿
            if "instagram.com" in url:
                texts = await _scrape_instagram(page, url)
            # X/Twitter投稿
            elif "x.com" in url or "twitter.com" in url:
                texts = await _scrape_x(page, url)
            # その他（汎用）
            else:
                texts = await _scrape_generic(page)

            await browser.close()
    except Exception as e:
        logger.error(f"スクレイピングエラー ({url}): {e}")
    return texts


async def _scrape_instagram(page, url: str) -> list[str]:
    """Instagramページから投稿テキストを取得"""
    texts = []
    try:
        if "/p/" in url or "/reel/" in url:
            # 個別投稿
            el = await page.query_selector("meta[property='og:description']")
            if el:
                content = await el.get_attribute("content")
                if content:
                    texts.append(content)
        else:
            # プロフィールページ → 複数投稿のキャプションをmeta descriptionから取得
            # リンクをクリックして各投稿を見ることは難しいためOGPで対応
            els = await page.query_selector_all("article")
            for el in els[:10]:
                text = await el.inner_text()
                if text.strip():
                    texts.append(text.strip()[:300])
    except Exception as e:
        logger.error(f"Instagram scrape error: {e}")
    return texts


async def _scrape_x(page, url: str) -> list[str]:
    """X(Twitter)から投稿テキストを取得"""
    texts = []
    try:
        if "/status/" in url:
            # 個別ツイート
            await page.wait_for_selector('[data-testid="tweetText"]', timeout=10000)
            els = await page.query_selector_all('[data-testid="tweetText"]')
            for el in els[:1]:
                texts.append(await el.inner_text())
        else:
            # プロフィールページ → 最新ツイートを取得
            await page.wait_for_selector('[data-testid="tweetText"]', timeout=15000)
            els = await page.query_selector_all('[data-testid="tweetText"]')
            for el in els[:20]:
                t = await el.inner_text()
                if t.strip():
                    texts.append(t.strip())
    except Exception as e:
        logger.error(f"X scrape error: {e}")
    return texts


async def _scrape_generic(page) -> list[str]:
    """汎用ページからテキスト抽出"""
    try:
        body = await page.inner_text("body")
        # 長すぎる場合は先頭3000文字
        return [body[:3000]]
    except Exception:
        return []

# ════════════════════════════════════════════════════
#  Claudeでスタイル分析
# ════════════════════════════════════════════════════
def analyze_style(posts: dict) -> str:
    """投稿例からスタイルを分析"""
    own_insta  = posts.get("own_instagram", [])
    own_x      = posts.get("own_x", [])
    refs       = posts.get("reference", [])

    sample_own_insta = "\n---\n".join(own_insta[:10])
    sample_own_x     = "\n---\n".join(own_x[:10])
    sample_refs      = "\n---\n".join([r["text"] for r in refs[:10]])

    prompt = f"""スーパー「みどりのマート」のSNS投稿スタイルを分析してください。

【自分のInstagram投稿例】
{sample_own_insta or '（なし）'}

【自分のX(Twitter)投稿例】
{sample_own_x or '（なし）'}

【参考にする競合・参考アカウントの投稿例】
{sample_refs or '（なし）'}

以下の観点で分析し、今後のキャプション生成に使えるスタイルガイドを作成してください：
1. 文体・トーン（丁寧語・タメ口・親しみやすさなど）
2. よく使う表現・フレーズ
3. 絵文字の使い方
4. ハッシュタグの傾向
5. 投稿の構成パターン
6. 競合との差別化ポイント

日本語で200〜400文字程度でまとめてください。"""

    result = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return result.content[0].text.strip()

# ════════════════════════════════════════════════════
#  メイン学習処理
# ════════════════════════════════════════════════════
async def run_learn_own_posts() -> tuple[int, int, str, str]:
    """自分の投稿を取得・分析して保存。(insta件数, x件数, 分析結果, 警告) を返す"""
    guide = load_style_guide()
    warning = ""

    insta_posts = fetch_own_instagram_posts(50)

    x_posts = []
    try:
        x_posts = fetch_own_x_posts(100)
    except RuntimeError as e:
        if "X_CREDIT_REQUIRED" in str(e):
            warning = "x_credit"
        else:
            raise
    except Exception as e:
        logger.error(f"X取得エラー: {e}")

    guide["own_posts"]["instagram"] = insta_posts
    if x_posts:
        guide["own_posts"]["x"] = x_posts

    analysis = analyze_style({
        "own_instagram": insta_posts,
        "own_x": x_posts,
        "reference": guide.get("reference_posts", [])
    })
    guide["style_analysis"] = analysis
    save_style_guide(guide)

    return len(insta_posts), len(x_posts), analysis, warning


async def run_learn_url(url: str) -> tuple[int, str]:
    """URLから投稿を取得・分析して保存。(件数, 分析結果) を返す"""
    guide = load_style_guide()

    texts = await scrape_url(url)
    if not texts:
        return 0, ""

    # 参考投稿に追加
    for t in texts:
        guide["reference_posts"].append({"source": url, "text": t})

    # 全データで再分析
    analysis = analyze_style({
        "own_instagram": guide["own_posts"].get("instagram", []),
        "own_x": guide["own_posts"].get("x", []),
        "reference": guide.get("reference_posts", [])
    })
    guide["style_analysis"] = analysis
    save_style_guide(guide)

    return len(texts), analysis
