"""
スタイル学習モジュール
- 自分のInstagram・X投稿をAPIで取得（テキスト＋画像）
- 競合URLからPlaywrightでスクレイピング（テキスト＋画像）
- Claude Visionで画像スタイルを分析
- Claudeでテキストスタイル分析 → style_guide.json に保存
"""
import io
import json
import os
import re
import base64
import asyncio
import logging
from pathlib import Path

import requests
import tweepy
import anthropic
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent))
STYLE_GUIDE_PATH = BASE_DIR / "style_guide.json"

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_USER_ID      = os.getenv("INSTAGRAM_USER_ID")
INSTAGRAM_USERNAME     = os.getenv("INSTAGRAM_USERNAME")   # instagrapi用（競合スクレイピング）
INSTAGRAM_PASSWORD     = os.getenv("INSTAGRAM_PASSWORD")   # instagrapi用（競合スクレイピング）
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
        "image_analysis": "",
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
def fetch_own_instagram_posts(limit: int = 50) -> list[dict]:
    """Instagram Graph APIで自分の投稿キャプションと画像URLを取得"""
    try:
        url = f"https://graph.instagram.com/v21.0/{INSTAGRAM_USER_ID}/media"
        r = requests.get(url, params={
            "fields": "caption,media_type,timestamp,media_url",
            "limit": limit,
            "access_token": INSTAGRAM_ACCESS_TOKEN
        })
        data = r.json().get("data", [])
        posts = []
        for d in data:
            if d.get("media_type") in ("IMAGE", "CAROUSEL_ALBUM"):
                posts.append({
                    "caption":   d.get("caption", ""),
                    "image_url": d.get("media_url", ""),
                })
        logger.info(f"Instagram: {len(posts)}件取得")
        return posts
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
#  画像ダウンロード
# ════════════════════════════════════════════════════
def _download_image(url: str) -> bytes | None:
    """URLから画像をダウンロード。失敗したらNone"""
    if not url:
        return None
    try:
        r = requests.get(url, timeout=15, headers={
            # モバイルUAの方がInstagram CDNで弾かれにくい
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        })
        if r.status_code == 200 and len(r.content) > 500:
            ct = r.headers.get("Content-Type", "")
            # Content-Typeがなくてもバイナリなら画像として扱う
            if "image" in ct or "octet-stream" in ct or not ct:
                return r.content
    except Exception as e:
        logger.warning(f"画像DL失敗 ({url[:60]}): {e}")
    return None

def _prepare_image_for_vision(img_bytes: bytes) -> str:
    """画像をClaude Vision用のbase64文字列に変換（800px以下にリサイズ）"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((800, 800), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()

# ════════════════════════════════════════════════════
#  instagrapi でInstagramプロフィールを取得
# ════════════════════════════════════════════════════
_ig_client = None  # セッションキャッシュ

def _get_ig_client():
    """instagrapiクライアントをログイン済み状態で返す（セッション再利用）"""
    global _ig_client
    if _ig_client:
        return _ig_client
    if not (INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD):
        raise RuntimeError("INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD が未設定です")

    from instagrapi import Client
    session_path = BASE_DIR / "ig_session.json"
    cl = Client()
    if session_path.exists():
        try:
            cl.load_settings(session_path)
            cl.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)
            logger.info("instagrapi: セッション再利用でログイン成功")
        except Exception as e:
            logger.warning(f"instagrapi: セッション再利用失敗、再ログイン: {e}")
            session_path.unlink(missing_ok=True)
            cl = Client()
            cl.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)
    else:
        cl.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)
        logger.info("instagrapi: 新規ログイン成功")

    cl.dump_settings(session_path)
    _ig_client = cl
    return cl


def fetch_instagram_profile_posts(profile_url: str, max_posts: int = 20) -> list[dict]:
    """
    instagrapiを使ってInstagramプロフィールから投稿（キャプション＋画像バイト）を取得。
    INSTAGRAM_USERNAME/PASSWORDが未設定の場合は空リストを返す。
    """
    if not (INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD):
        return []

    # プロフィールURLからユーザー名を抽出
    # 例: https://www.instagram.com/username/ → "username"
    username = profile_url.rstrip("/").split("?")[0].split("/")[-1].lstrip("@")
    if not username:
        return []

    try:
        cl = _get_ig_client()
        user_id = cl.user_id_from_username(username)
        medias = cl.user_medias(user_id, amount=max_posts)

        posts = []
        for media in medias:
            caption = media.caption_text or ""
            # 画像URLを取得（Photo / Video thumbnail / Carousel先頭）
            image_url = ""
            if media.thumbnail_url:
                image_url = str(media.thumbnail_url)
            elif media.image_versions2 and media.image_versions2.candidates:
                image_url = str(media.image_versions2.candidates[0].url)

            # 画像を直接ダウンロード（instagrapi認証済みセッションで取得）
            image_bytes = None
            try:
                tmp = cl.photo_download_by_pk(media.pk, folder="/tmp")
                with open(tmp, "rb") as f:
                    image_bytes = f.read()
                os.unlink(tmp)
            except Exception:
                # フォールバック: URL から直接ダウンロード
                if image_url:
                    image_bytes = _download_image(image_url)

            posts.append({
                "text":        caption,
                "image_url":   image_url,
                "image_bytes": image_bytes,
            })

        logger.info(f"instagrapi: @{username} から {len(posts)}件取得")
        return posts
    except Exception as e:
        logger.error(f"instagrapi取得エラー (@{username}): {e}")
        global _ig_client
        _ig_client = None  # 次回は再ログイン
        return []


# ════════════════════════════════════════════════════
#  競合URLからスクレイピング（Playwright）
# ════════════════════════════════════════════════════
async def scrape_url(url: str) -> list[dict]:
    """
    URLからSNS投稿テキストと画像URLを抽出。list[{"text": str, "image_url": str, "image_bytes": bytes|None}]を返す。
    Instagramプロフィールページはinstagrapiを優先使用。
    """
    # ── Instagram プロフィールページ → instagrapi 優先 ──
    if "instagram.com" in url and "/p/" not in url and "/reel/" not in url:
        if INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD:
            logger.info(f"instagrapiでInstagramプロフィールを取得: {url}")
            posts = await asyncio.to_thread(fetch_instagram_profile_posts, url)
            if posts:
                return posts
            logger.warning("instagrapi取得失敗、Playwrightにフォールバック")

    # ── Playwright スクレイピング ──
    from playwright.async_api import async_playwright

    posts = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)

            if "instagram.com" in url:
                posts = await _scrape_instagram(page, url)
            elif "x.com" in url or "twitter.com" in url:
                posts = await _scrape_x(page, url)
            else:
                posts = await _scrape_generic(page)

            await browser.close()
    except Exception as e:
        logger.error(f"スクレイピングエラー ({url}): {e}")
    return posts


async def _scrape_instagram(page, url: str) -> list[dict]:
    """Instagramページから投稿テキストと画像URLを取得"""
    posts = []
    try:
        if "/p/" in url or "/reel/" in url:
            # 個別投稿
            text_el = await page.query_selector("meta[property='og:description']")
            img_el  = await page.query_selector("meta[property='og:image']")
            text      = (await text_el.get_attribute("content") or "") if text_el else ""
            image_url = (await img_el.get_attribute("content")  or "") if img_el  else ""
            if text or image_url:
                posts.append({"text": text, "image_url": image_url})
        else:
            # プロフィールページ
            articles = await page.query_selector_all("article")
            for article in articles[:10]:
                text = (await article.inner_text()).strip()[:300]
                img  = await article.query_selector("img")
                image_url = (await img.get_attribute("src") or "") if img else ""
                # og:image フォールバック
                if not image_url:
                    og = await page.query_selector("meta[property='og:image']")
                    if og:
                        image_url = await og.get_attribute("content") or ""
                if text or image_url:
                    posts.append({"text": text, "image_url": image_url})
    except Exception as e:
        logger.error(f"Instagram scrape error: {e}")
    return posts


async def _scrape_x(page, url: str) -> list[dict]:
    """X(Twitter)から投稿テキストと画像URLを取得"""
    posts = []
    try:
        if "/status/" in url:
            await page.wait_for_selector('[data-testid="tweetText"]', timeout=10000)
            text_els = await page.query_selector_all('[data-testid="tweetText"]')
            img_els  = await page.query_selector_all('[data-testid="tweetPhoto"] img')
            text      = (await text_els[0].inner_text()) if text_els else ""
            image_url = (await img_els[0].get_attribute("src") or "") if img_els else ""
            if text or image_url:
                posts.append({"text": text, "image_url": image_url})
        else:
            await page.wait_for_selector('[data-testid="tweetText"]', timeout=15000)
            text_els    = await page.query_selector_all('[data-testid="tweetText"]')
            img_els_all = await page.query_selector_all('[data-testid="tweetPhoto"] img')
            for i, el in enumerate(text_els[:20]):
                t = (await el.inner_text()).strip()
                image_url = ""
                if i < len(img_els_all):
                    image_url = await img_els_all[i].get_attribute("src") or ""
                if t or image_url:
                    posts.append({"text": t, "image_url": image_url})
    except Exception as e:
        logger.error(f"X scrape error: {e}")
    return posts


async def _scrape_generic(page) -> list[dict]:
    """汎用ページからテキスト抽出"""
    try:
        body = await page.inner_text("body")
        return [{"text": body[:3000], "image_url": ""}]
    except Exception:
        return []

# ════════════════════════════════════════════════════
#  Claude Vision で画像スタイルを分析
# ════════════════════════════════════════════════════
def analyze_images_style(image_bytes_list: list[bytes], source_desc: str) -> str:
    """画像リストをClaude Visionで分析してビジュアルスタイルを返す"""
    if not image_bytes_list:
        return ""

    content = []
    for img_bytes in image_bytes_list[:10]:  # 最大10枚
        try:
            b64 = _prepare_image_for_vision(img_bytes)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
            })
        except Exception as e:
            logger.warning(f"画像変換エラー: {e}")

    if not content:
        return ""

    content.append({
        "type": "text",
        "text": f"""これらは{source_desc}のSNS投稿画像です。
以下の観点で画像スタイルを分析し、今後の投稿画像作成に活かせるガイドを作成してください:

1. 色調・カラーパレット（温かみ・クール・明るい・鮮やか・落ち着いたなど）
2. 構図・レイアウト（クローズアップ・俯瞰・整然・にぎやかなど）
3. 照明・雰囲気（自然光・スタジオ光・明るい・ドラマチックなど）
4. 被写体の傾向（商品メイン・人物・店内・料理・価格POP等）
5. 全体の世界観（高級感・親しみやすい・元気・清潔感など）

100〜200文字でまとめてください。"""
    })

    result = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": content}]
    )
    return result.content[0].text.strip()

# ════════════════════════════════════════════════════
#  Claudeでテキストスタイル分析
# ════════════════════════════════════════════════════
def analyze_style(posts: dict) -> str:
    """投稿例（テキスト＋画像分析）からスタイルを分析"""
    own_insta  = posts.get("own_instagram", [])
    own_x      = posts.get("own_x", [])
    refs       = posts.get("reference", [])
    image_analysis = posts.get("image_analysis", "")

    sample_own_insta = "\n---\n".join(own_insta[:10])
    sample_own_x     = "\n---\n".join(own_x[:10])
    sample_refs      = "\n---\n".join([r["text"] for r in refs[:10] if r.get("text")])

    image_section = ""
    if image_analysis:
        image_section = f"\n【画像スタイル分析（Vision）】\n{image_analysis}\n"

    prompt = f"""スーパー「みどりのマート」のSNS投稿スタイルを分析してください。

【自分のInstagram投稿例（テキスト）】
{sample_own_insta or '（なし）'}

【自分のX(Twitter)投稿例】
{sample_own_x or '（なし）'}

【参考にする競合・参考アカウントの投稿例】
{sample_refs or '（なし）'}
{image_section}
以下の観点で分析し、今後のキャプション生成に使えるスタイルガイドを作成してください：
1. 文体・トーン（丁寧語・タメ口・親しみやすさなど）
2. よく使う表現・フレーズ
3. 絵文字の使い方
4. ハッシュタグの傾向
5. 投稿の構成パターン
6. ビジュアルスタイルの特徴（画像分析を反映）
7. 競合との差別化ポイント

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

    # Instagram: テキスト＋画像URL
    insta_posts = fetch_own_instagram_posts(50)
    insta_captions  = [p["caption"]   for p in insta_posts if p.get("caption")]
    insta_img_urls  = [p["image_url"] for p in insta_posts if p.get("image_url")]

    # X: テキストのみ
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

    # 自分のInstagram画像をダウンロードしてVisionで分析
    logger.info(f"Instagram画像 {len(insta_img_urls)}件をダウンロード中...")
    insta_images = []
    for url in insta_img_urls[:10]:
        img = _download_image(url)
        if img:
            insta_images.append(img)

    image_analysis = ""
    if insta_images:
        logger.info(f"{len(insta_images)}枚の画像をVisionで分析中...")
        image_analysis = analyze_images_style(insta_images, "自分のInstagram")

    # 参考投稿の画像も追加分析
    ref_images = []
    for ref in guide.get("reference_posts", [])[:10]:
        if ref.get("image_url"):
            img = _download_image(ref["image_url"])
            if img:
                ref_images.append(img)
    if ref_images:
        ref_analysis = analyze_images_style(ref_images, "参考アカウント")
        if ref_analysis:
            image_analysis = f"【自分の投稿】{image_analysis}\n【参考アカウント】{ref_analysis}" if image_analysis else ref_analysis

    guide["own_posts"]["instagram"] = insta_captions
    if x_posts:
        guide["own_posts"]["x"] = x_posts
    guide["image_analysis"] = image_analysis

    analysis = analyze_style({
        "own_instagram":  insta_captions,
        "own_x":          x_posts,
        "reference":      guide.get("reference_posts", []),
        "image_analysis": image_analysis,
    })
    guide["style_analysis"] = analysis
    save_style_guide(guide)

    return len(insta_captions), len(x_posts), len(insta_images), analysis, image_analysis, warning


async def run_learn_url(url: str) -> tuple[int, str]:
    """URLから投稿を取得・分析して保存。(件数, 分析結果) を返す"""
    guide = load_style_guide()

    posts = await scrape_url(url)
    if not posts:
        return 0, ""

    # 参考投稿に追加
    for p in posts:
        guide["reference_posts"].append({
            "source":    url,
            "text":      p.get("text", ""),
            "image_url": p.get("image_url", ""),
        })

    # 取得した画像をVisionで分析
    # instagrapi経由ならimage_bytesが既に入っている。なければURLからDL。
    images = []
    for p in posts[:10]:
        img = p.get("image_bytes")  # instagrapiで直接取得済み
        if not img and p.get("image_url"):
            img = _download_image(p["image_url"])
        if img:
            images.append(img)

    url_image_analysis = ""
    if images:
        logger.info(f"{len(images)}枚の競合画像をVisionで分析中...")
        url_image_analysis = analyze_images_style(images, f"{url}の投稿")

    # 既存の画像分析と統合
    existing_img = guide.get("image_analysis", "")
    if url_image_analysis:
        if existing_img:
            guide["image_analysis"] = f"{existing_img}\n【{url}】{url_image_analysis}"
        else:
            guide["image_analysis"] = url_image_analysis

    # 全データで再分析
    analysis = analyze_style({
        "own_instagram":  guide["own_posts"].get("instagram", []),
        "own_x":          guide["own_posts"].get("x", []),
        "reference":      guide.get("reference_posts", []),
        "image_analysis": guide.get("image_analysis", ""),
    })
    guide["style_analysis"] = analysis
    save_style_guide(guide)

    return len(posts), len(images), url_image_analysis, analysis
