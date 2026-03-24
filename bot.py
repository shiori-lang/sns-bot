"""
みどりのマート SNS自動投稿ボット（グループ対応版）
グループで @ボット名 でメンションして自然言語で指示する
"""
import os
import io
import asyncio
import logging
import base64
import json
import tempfile
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)

import google.generativeai as genai
import anthropic
import tweepy
from PIL import Image, ImageDraw
from learn import run_learn_own_posts, run_learn_url, run_learn_images, load_style_guide

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── APIキー ──────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_USER_ID      = os.getenv("INSTAGRAM_USER_ID")
X_API_KEY              = os.getenv("X_API_KEY")
X_API_SECRET           = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN         = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET  = os.getenv("X_ACCESS_TOKEN_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
IMGBB_API_KEY          = os.getenv("IMGBB_API_KEY", "")

# ── Gemini / Claude 初期化 ────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ロゴ保存パス（Railway volume対応: /data または本体ディレクトリ）
BASE_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent))
LOGO_DIR = BASE_DIR / "logos"
LOGO_DIR.mkdir(parents=True, exist_ok=True)
LOGO_PATH = LOGO_DIR / "logo_1.png"  # 後方互換

JST = timezone(timedelta(hours=9))

# メディアグループ遅延処理用タスク管理（job_queue の代替）
_media_group_tasks: dict[str, "asyncio.Task"] = {}

# 各プラットフォームの推奨画像サイズ（幅×高さ）
PLATFORM_IMAGE_SIZES = {
    "instagram": (1080, 1080),
    "x":         (1200, 675),
    "line":      (1040, 1040),
}

# ════════════════════════════════════════════════════
#  Claude で指示を解析
# ════════════════════════════════════════════════════
def parse_instruction(user_text: str) -> dict:
    """
    自然言語の指示からプラットフォーム・編集内容・キャプション指示を抽出
    """
    now_jst = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    result = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": f"""以下のSNS投稿指示を解析してJSON形式で返してください。
現在日時(JST): {now_jst}

指示: {user_text}

返すJSONの形式:
{{
  "platforms": ["instagram", "x", "line"],
  "add_logo": true,
  "generate_image": true,
  "image_prompt": "",
  "caption_instruction": "",
  "schedule_time": "",
  "post_type": "single",
  "items": []
}}

各フィールドの説明:
- platforms: 指定されたSNS（複数可）。「全部」なら3つ全て。未指定なら["instagram","x"]
- add_logo: 基本はtrue。「ロゴなし」「ロゴ不要」と明示された場合のみfalse
- generate_image: 新しく画像生成が必要か（写真は別途送られてくる場合はfalse）
- image_prompt: 画像生成する場合のプロンプト（日本語）。複数商品の場合はレイアウトも含めて詳細に
- caption_instruction: キャプション生成への指示（商品名・価格・セール内容など全情報を含める）
- schedule_time: 予約投稿の場合はISO 8601形式（例: "2026-03-25T09:00:00+09:00"）。今すぐなら空文字
- post_type: "single"（単品）/ "set"（セット・まとめ買い割引）/ "before_after"（値下げ前後）/ "multi"（複数商品紹介）
- items: 商品リスト。例: [{{"name":"りんご","price":"198円","original_price":"","qty":"3個"}}, ...]
  original_priceは値下げ前の価格（あれば）。セットや複数商品の場合に活用。

複数商品やセット割の場合のimage_promptの書き方：
- セット：「〇〇と△△を横並びに配置した特売POP風の画像。合計金額□□円を大きく表示。」
- まとめ買い：「〇〇を複数個並べ、まとめ買い割引□%OFFを強調したレイアウト。」
- 値下げ：「〇〇の商品写真。価格を大きく、〇〇円→△△円の値下げを視覚的に表現。」

JSONのみ返してください。"""
        }]
    )
    try:
        text = result.content[0].text.strip()
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception:
        return {
            "platforms": ["instagram", "x"],
            "add_logo": True,
            "generate_image": False,
            "image_prompt": "",
            "caption_instruction": user_text,
            "schedule_time": "",
        }


def parse_schedule_time(text: str) -> str:
    """テキストから投稿予定時刻をISO 8601形式(+09:00)で返す。読み取れなければ空文字"""
    now_str = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    try:
        result = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=[{"role": "user", "content":
                f"現在日時: {now_str} (JST)\nテキスト: {text}\n\n"
                "このテキストが示す日時をISO 8601形式（例: 2026-03-25T09:00:00+09:00）で返してください。\n"
                "読み取れない場合は空文字のみ返してください。ISO文字列または空文字のみ返答。"
            }]
        )
        out = result.content[0].text.strip()
        datetime.fromisoformat(out)  # validate
        return out
    except Exception:
        return ""

# ════════════════════════════════════════════════════
#  画像処理
# ════════════════════════════════════════════════════
def resize_for_platform(image_bytes: bytes, platform: str) -> bytes:
    """
    プラットフォームの推奨サイズにリサイズ。
    - Instagram / LINE（1:1）: センタークロップで正方形にぴったり
    - X（16:9）: 縦を保ちつつ横をレターボックス（黒バー）で埋める
      → 縦長・正方形の商品画像が上下で切れないようにする
    """
    target_w, target_h = PLATFORM_IMAGE_SIZES.get(platform, (1080, 1080))
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h

    # センタークロップ（全プラットフォーム共通）
    if src_ratio > tgt_ratio:
        new_h = target_h
        new_w = int(img.width * target_h / img.height)
    else:
        new_w = target_w
        new_h = int(img.height * target_w / img.width)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    x = (new_w - target_w) // 2
    y = (new_h - target_h) // 2
    img = img.crop((x, y, x + target_w, y + target_h))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def find_best_logo_position(base_img: Image.Image, logo_w: int, logo_h: int) -> tuple[int, int]:
    """
    画像全体をグリッドスキャンして視覚的に最も空いている場所を選択。
    低解像度版で各候補位置のピクセル分散（視覚的複雑さ）を計算し、
    最もシンプルな領域（ロゴが映える場所）を選ぶ。
    """
    margin = 20
    w, h = base_img.width, base_img.height

    # 高速化のため小さくリサイズしてグレースケール化
    scale = 4
    small = base_img.convert("L").resize((w // scale, h // scale), Image.LANCZOS)
    sw, sh = small.size
    lw, lh = logo_w // scale, logo_h // scale

    # グリッド状に候補位置を走査（画像端の余白内で均等サンプリング）
    step_x = max(1, (sw - lw - margin // scale) // 6)
    step_y = max(1, (sh - lh - margin // scale) // 6)

    best_pos = (w - logo_w - margin, h - logo_h - margin)  # デフォルト右下
    best_score = float("inf")

    xs = range(margin // scale, sw - lw - margin // scale, step_x)
    ys = range(margin // scale, sh - lh - margin // scale, step_y)

    for sy in ys:
        for sx in xs:
            region = small.crop((sx, sy, sx + lw, sy + lh))
            pixels = list(region.getdata())
            if not pixels:
                continue
            avg = sum(pixels) / len(pixels)
            # 分散（視覚的複雑さ）= 低いほどシンプルでロゴが映える
            variance = sum((p - avg) ** 2 for p in pixels) / len(pixels)
            # 端ほど優先（中央寄りにはならないようペナルティ）
            cx_ratio = abs((sx + lw / 2) / sw - 0.5)  # 0.5 = 端、0 = 中央
            edge_bonus = (1 - cx_ratio) * 30  # 中央に近いほど不利
            score = variance + edge_bonus

            if score < best_score:
                best_score = score
                best_pos = (sx * scale, sy * scale)

    return best_pos


def select_best_logo(base_img: Image.Image, logos: list[Path]) -> Image.Image:
    """
    複数ロゴの中から画像に最も映えるロゴを選択
    → 背景色と最もコントラストが高いロゴを選ぶ
    """
    if len(logos) == 1:
        return Image.open(logos[0]).convert("RGBA")

    gray = base_img.convert("L")
    # 画像全体の平均輝度
    pixels = list(gray.getdata())
    bg_brightness = sum(pixels) / len(pixels)

    best_logo = None
    best_contrast = -1

    for lp in logos:
        logo_img = Image.open(lp).convert("RGBA")
        # ロゴの非透明ピクセルの平均輝度
        r, g, b, a = logo_img.split()
        logo_gray = logo_img.convert("L")
        logo_pixels = [logo_gray.getpixel((x, y))
                       for x in range(logo_img.width)
                       for y in range(logo_img.height)
                       if logo_img.getpixel((x, y))[3] > 128]
        if not logo_pixels:
            continue
        logo_brightness = sum(logo_pixels) / len(logo_pixels)
        contrast = abs(bg_brightness - logo_brightness)
        if contrast > best_contrast:
            best_contrast = contrast
            best_logo = logo_img

    return best_logo or Image.open(logos[0]).convert("RGBA")


def _remove_white_background(logo: Image.Image, threshold: int = 240) -> Image.Image:
    """ロゴの白・明るい背景を透明化する（JPEGロゴ対応）"""
    logo = logo.convert("RGBA")
    data = logo.getdata()
    new_data = []
    for r, g, b, a in data:
        # ほぼ白（全チャンネルが threshold 以上）なら透明に
        if r >= threshold and g >= threshold and b >= threshold:
            new_data.append((r, g, b, 0))
        else:
            new_data.append((r, g, b, a))
    logo.putdata(new_data)
    return logo


def add_logo_to_image(image_bytes: bytes) -> bytes:
    """画像を分析してロゴを最適な位置に自動合成（複数ロゴ対応）"""
    logos = get_logo_paths()
    if not logos:
        logger.warning("ロゴファイルが見つかりません。/setlogoでロゴを登録してください。")
        return image_bytes
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        # 最適なロゴを自動選択
        logo = select_best_logo(base_img, logos)
        # 白背景を透明化（JPEG登録ロゴ対応）
        logo = _remove_white_background(logo)

        # ロゴを画像幅の18%にリサイズ
        logo_w = int(base_img.width * 0.18)
        ratio = logo_w / logo.width
        logo_h = int(logo.height * ratio)
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

        # 最適な位置を検出
        x, y = find_best_logo_position(base_img, logo_w, logo_h)

        base_img.paste(logo, (x, y), logo)
        buf = io.BytesIO()
        base_img.convert("RGB").save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"ロゴ合成エラー: {e}")
        return image_bytes

async def generate_image_gemini(prompt: str) -> bytes | None:
    """Gemini で画像生成（学習済み画像スタイルをプロンプトに反映）"""
    try:
        from google import genai as genai_new
        from google.genai import types as genai_types
    except ImportError:
        logger.error("google-genai SDK が未インストールです")
        return None

    guide = load_style_guide()
    image_analysis = guide.get("image_analysis", "")
    style_prefix = f"【参考スタイル】{image_analysis}\n\n" if image_analysis else ""

    full_prompt = (
        f"{style_prefix}"
        f"スーパー「みどりのマート」のInstagram投稿用（1:1正方形）の本格的な広告ポスター画像を生成してください。\n"
        f"スーパーの広告チラシ・ポスター風のグラフィックデザイン。\n"
        f"日本語キャッチコピー（大きめフォント）と英語サブタイトルを入れ、鮮やかな背景色で商業デザインらしく仕上げる。\n"
        f"価格・金額の数字は入れない。\n"
        f"テーマ: {prompt}"
    )
    try:
        client = genai_new.Client(api_key=GEMINI_API_KEY)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash-image",
            contents=[full_prompt],
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=genai_types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
        if not response.candidates:
            logger.error("Gemini: candidatesが空です")
            return None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                data = part.inline_data.data
                return data if isinstance(data, bytes) else base64.b64decode(data)
    except Exception as e:
        logger.error(f"Gemini画像生成エラー: {e}")
    return None

def _build_gemini_design_prompt(caption_instr: str, n_photos: int, style_section: str) -> str:
    """
    Claude Haiku でテーマを分析し、Gemini 向けの詳細デザインプロンプト（英語）を生成。
    背景テーマ・テキスト内容・レイアウトをテーマに合わせて動的に決定する。
    """
    try:
        resp = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            messages=[{"role": "user", "content": f"""
You are a professional SNS graphic designer for a Japanese supermarket "Midori no Mart".
Based on the post theme below, write a detailed English image generation prompt for Gemini AI.

Post theme (Japanese): {caption_instr}
Number of product photos: {n_photos}

The prompt must describe a commercial-quality Japanese supermarket SNS poster (1:1 square) like these styles:
- Snacks/drinks at night → starry/dark background, warm spotlight on products
- Limited edition items → bold blue geometric background, stamp/badge design
- Fruit/fresh items → bright outdoor nature background with fruit elements
- Valentine/seasonal → thematic color scheme (pink/red/etc), festive decorations
- Product feature/特集 → sunburst or radial background matching product color

Include all of these in your prompt:
1. Specific background theme and colors matching the product
2. Japanese headline text to overlay (3-8 chars catchphrase, bold, large)
3. English subtitle text (short phrase)
4. Product layout: {"center hero shot" if n_photos == 1 else f"grid of {n_photos} products, all clearly visible"}
5. Decorative elements (badges, lines, sparkles, etc.) appropriate to theme
6. Overall mood and style

{style_section}

Return ONLY the image generation prompt in English. No explanation.
"""}]
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.error(f"デザインプロンプト生成エラー: {e}")
        # フォールバック
        return (
            f"Professional Japanese supermarket SNS poster (1:1 square) for Midori no Mart. "
            f"Theme: {caption_instr}. "
            f"Vibrant commercial advertisement style, bold Japanese headline text, English subtitle, "
            f"{'center hero product shot' if n_photos == 1 else f'grid layout of {n_photos} products'}. "
            f"High quality graphic design, purchase-inspiring visuals. No prices."
        )


async def redesign_product_image(photos: list[bytes], caption_instr: str) -> bytes | None:
    """
    1枚または複数枚の商品写真を受け取り、学習済みスタイルでSNS用にデザインし直す。
    Claude Haiku でテーマ分析 → Gemini 画像生成の二段構え。
    """
    try:
        from google import genai as genai_new
        from google.genai import types as genai_types
    except ImportError:
        logger.error("google-genai SDK が未インストールです")
        return None

    guide = load_style_guide()
    image_analysis = guide.get("image_analysis", "")
    style_section = f"Reference style from past posts: {image_analysis}" if image_analysis else ""

    # Step 1: Claude Haiku でテーマに最適化したデザインプロンプトを生成
    prompt = await asyncio.to_thread(
        _build_gemini_design_prompt, caption_instr, len(photos), style_section
    )
    logger.info(f"Generated design prompt: {prompt[:200]}")

    try:
        client = genai_new.Client(api_key=GEMINI_API_KEY)

        contents = [prompt]
        for b in photos[:4]:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            contents.append(genai_types.Part.from_bytes(
                data=buf.getvalue(), mime_type="image/jpeg"
            ))

        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash-image",
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=genai_types.ImageConfig(aspect_ratio="1:1"),
            ),
        )

        if not response.candidates:
            logger.error("Gemini redesign: candidatesが空です")
            return None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                data = part.inline_data.data
                return data if isinstance(data, bytes) else base64.b64decode(data)
    except Exception as e:
        logger.error(f"Gemini画像デザイン生成エラー: {e}")
    return None


# ════════════════════════════════════════════════════
#  Claude でキャプション生成
# ════════════════════════════════════════════════════
def generate_caption(instruction: str, platform: str,
                     post_type: str = "single", items: list = None) -> str:
    rules = {
        "instagram": "Instagramらしく絵文字を使い、ハッシュタグ10個程度。200文字以内。",
        "x":         "X(Twitter)用に280文字以内。ハッシュタグ3個以内。",
        "line":      "LINEの公式アカウントらしく親しみやすく。ハッシュタグなし。150文字以内。",
    }
    rule = rules.get(platform, "")

    # スタイルガイドを読み込む
    guide = load_style_guide()
    style_analysis  = guide.get("style_analysis", "")
    image_analysis  = guide.get("image_analysis", "")
    examples = guide.get("own_posts", {}).get(platform, [])[:3]
    example_text = "\n---\n".join(examples) if examples else ""

    style_section = ""
    if style_analysis:
        style_section = f"\n【学習済みスタイルガイド】\n{style_analysis}\n"
    if image_analysis:
        style_section += f"\n【画像スタイルの傾向（この雰囲気に合うキャプションを）】\n{image_analysis}\n"
    if example_text:
        style_section += f"\n【過去の投稿例（このスタイルを参考に）】\n{example_text}\n"

    # 商品情報セクション（複数商品・セット・値下げ対応）
    items_section = ""
    if items:
        type_label = {
            "set":          "セット・組み合わせ割引",
            "before_after": "値下げ・タイムセール",
            "multi":        "複数商品まとめ紹介",
        }.get(post_type, "商品紹介")
        lines = []
        for it in items:
            name  = it.get("name", "")
            price = it.get("price", "")
            orig  = it.get("original_price", "")
            qty   = it.get("qty", "")
            line  = name
            if qty:   line += f" {qty}"
            if orig:  line += f" {orig}→{price}"
            elif price: line += f" {price}"
            lines.append(line)
        items_section = f"\n【投稿タイプ: {type_label}】\n商品: {' / '.join(lines)}\n"

    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"スーパー「みどりのマート」の{platform}向けキャプションを日本語で作成。\n"
                f"ルール: {rule}\n"
                f"{style_section}"
                f"{items_section}"
                f"指示: {instruction}\n"
                f"キャプション本文のみ出力。"
            )
        }]
    )
    return msg.content[0].text.strip()

# ════════════════════════════════════════════════════
#  SNS 投稿
# ════════════════════════════════════════════════════
def upload_to_imgbb(image_bytes: bytes) -> str | None:
    if not IMGBB_API_KEY:
        return None
    try:
        b64 = base64.b64encode(image_bytes).decode()
        r = requests.post("https://api.imgbb.com/1/upload",
                          data={"key": IMGBB_API_KEY, "image": b64})
        return r.json()["data"]["url"]
    except Exception as e:
        logger.error(f"imgbbアップロードエラー: {e}")
        return None

def post_instagram(image_bytes: bytes, caption: str) -> bool:
    try:
        url = upload_to_imgbb(image_bytes)
        if not url:
            return False
        r = requests.post(
            f"https://graph.instagram.com/v21.0/{INSTAGRAM_USER_ID}/media",
            params={"image_url": url, "caption": caption,
                    "access_token": INSTAGRAM_ACCESS_TOKEN}
        )
        cid = r.json().get("id")
        if not cid:
            return False
        r = requests.post(
            f"https://graph.instagram.com/v21.0/{INSTAGRAM_USER_ID}/media_publish",
            params={"creation_id": cid, "access_token": INSTAGRAM_ACCESS_TOKEN}
        )
        return "id" in r.json()
    except Exception as e:
        logger.error(f"Instagram投稿エラー: {e}")
        return False

def post_x(image_bytes: bytes, caption: str) -> bool:
    try:
        auth = tweepy.OAuth1UserHandler(X_API_KEY, X_API_SECRET,
                                        X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)
        api_v1 = tweepy.API(auth)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp = f.name
        media = api_v1.media_upload(tmp)
        os.unlink(tmp)
        if not media or not getattr(media, "media_id", None):
            logger.error("X: メディアアップロード失敗 - media_id なし")
            return False
        client = tweepy.Client(consumer_key=X_API_KEY,
                               consumer_secret=X_API_SECRET,
                               access_token=X_ACCESS_TOKEN,
                               access_token_secret=X_ACCESS_TOKEN_SECRET)
        client.create_tweet(text=caption, media_ids=[media.media_id])
        return True
    except Exception as e:
        logger.error(f"X投稿エラー: {e}")
        return False

def post_line(image_bytes: bytes, caption: str) -> bool:
    try:
        img_url = upload_to_imgbb(image_bytes)
        if not img_url:
            return False
        r = requests.post(
            "https://api.line.me/v2/bot/message/broadcast",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                     "Content-Type": "application/json"},
            json={"messages": [
                {"type": "image", "originalContentUrl": img_url,
                 "previewImageUrl": img_url},
                {"type": "text", "text": caption}
            ]}
        )
        return r.status_code == 200
    except Exception as e:
        logger.error(f"LINE投稿エラー: {e}")
        return False

# ════════════════════════════════════════════════════
#  メディアグループ（複数枚アルバム）まとめて処理
# ════════════════════════════════════════════════════
async def _process_media_group_direct(app, grp_key: str):
    """2秒バッファ後に複数写真をまとめてデザイン生成する（asyncio.create_task で呼ばれる）"""
    grp = app.bot_data.pop(grp_key, None)
    if not grp:
        return

    chat_id    = grp["chat_id"]
    message_id = grp["message_id"]
    photos     = grp["photos"]
    user_text  = grp.get("user_text", "")
    parsed     = grp.get("parsed") or {}

    # parsed が空（キャプションは最初のメッセージだけに付くため）→ ここで解析
    if not parsed and user_text:
        try:
            parsed = await asyncio.to_thread(parse_instruction, user_text)
        except Exception as e:
            logger.error(f"parse_instruction error in media group job: {e}")
            parsed = {}

    await app.bot.send_message(chat_id=chat_id,
        text=f"🎨 {len(photos)}枚の写真から商品を切り出してデザインし直し中...（少し時間がかかります）")

    platforms     = parsed.get("platforms", ["instagram", "x"])
    do_add_logo   = parsed.get("add_logo", True)
    caption_instr = parsed.get("caption_instruction", user_text)
    image_prompt  = parsed.get("image_prompt", "") or caption_instr
    schedule_time = parsed.get("schedule_time", "")
    post_type     = parsed.get("post_type", "single")
    items         = parsed.get("items", [])

    use_as_is = any(kw in user_text for kw in ("そのまま", "この写真で", "この画像で", "加工なし"))
    if use_as_is:
        image_bytes = photos[0]
    else:
        image_bytes = await redesign_product_image(photos, caption_instr)
        if not image_bytes:
            await app.bot.send_message(chat_id=chat_id,
                text="⚠️ デザイン生成に失敗しました。1枚目の写真を使います。")
            image_bytes = photos[0]

    # ロゴ未登録警告
    if do_add_logo and not get_logo_paths():
        await app.bot.send_message(chat_id=chat_id,
            text="⚠️ ロゴ未登録です。/setlogo でロゴを登録するとすべての投稿に自動でロゴが入ります。")

    await app.bot.send_message(chat_id=chat_id, text="✍️ キャプション生成中...")
    captions = {p: generate_caption(caption_instr, p, post_type, items) for p in platforms}

    class _FakeMsg:
        def __init__(self, chat_id, message_id, bot):
            self.chat_id    = chat_id
            self.message_id = message_id
            self._bot       = bot
        async def reply_text(self, text, **kw):
            await self._bot.send_message(chat_id=self.chat_id, text=text, **kw)
        async def reply_photo(self, photo, **kw):
            await self._bot.send_photo(chat_id=self.chat_id, photo=photo, **kw)

    # bot_data / user_data を持つダミーコンテキストを作成
    class _FakeCtx:
        def __init__(self, application):
            self.bot       = application.bot
            self.bot_data  = application.bot_data
            self.user_data = {}
            self.job_queue = application.job_queue  # None でも可
        def _get_pending_key(self): return None

    fake_msg = _FakeMsg(chat_id, message_id, app.bot)
    fake_ctx = _FakeCtx(app)
    await _send_preview(fake_msg, fake_ctx, image_bytes, captions, platforms,
                        image_prompt, caption_instr, do_add_logo, schedule_time,
                        post_type, items)


# ════════════════════════════════════════════════════
#  投稿実行（確認後 / 予約）
# ════════════════════════════════════════════════════
async def execute_scheduled_post(context: ContextTypes.DEFAULT_TYPE):
    """JobQueueから呼ばれる予約投稿ジョブ"""
    job  = context.job
    data = job.data
    chat_id     = data["chat_id"]
    image_bytes = data["image_bytes"]
    captions    = data["captions"]
    platforms   = data["platforms"]
    pending_key = data.get("pending_key")

    results = []
    for p in platforms:
        caption = captions.get(p, "")
        sized = resize_for_platform(image_bytes, p)
        if p == "instagram":
            ok = post_instagram(sized, caption)
            results.append(f"📸 Instagram: {'✅ 成功' if ok else '❌ 失敗'}")
        elif p == "x":
            ok = post_x(sized, caption)
            results.append(f"🐦 X: {'✅ 成功' if ok else '❌ 失敗'}")
        elif p == "line":
            ok = post_line(sized, caption)
            results.append(f"💚 LINE: {'✅ 成功' if ok else '❌ 失敗'}")

    if pending_key:
        context.bot_data.pop(pending_key, None)

    await context.bot.send_message(
        chat_id=chat_id,
        text="📅 予約投稿 完了\n\n" + "\n".join(results)
    )


async def execute_posting(context: ContextTypes.DEFAULT_TYPE,
                           chat_id: int, message_id: int):
    pending_key  = f"pending_{chat_id}_{message_id}"
    data         = context.bot_data.get(pending_key, {})
    image_bytes  = data.get("image_bytes")
    sized_images = data.get("sized_images", {})
    captions     = data.get("captions", {})
    platforms    = data.get("platforms", [])

    if not image_bytes and not sized_images:
        await context.bot.send_message(chat_id=chat_id, text="❌ 画像データが見つかりません。もう一度やり直してください。")
        context.bot_data.pop(pending_key, None)
        context.bot_data.pop(f"pending_chat_{chat_id}", None)
        return

    results = []
    for p in platforms:
        caption = captions.get(p, "")
        sized = sized_images.get(p) or (resize_for_platform(image_bytes, p) if image_bytes else None)
        if not sized:
            results.append(f"{'📸' if p=='instagram' else '🐦' if p=='x' else '💚'} {p.upper()}: ❌ 画像なし")
            continue
        if p == "instagram":
            ok = post_instagram(sized, caption)
            results.append(f"📸 Instagram: {'✅ 成功' if ok else '❌ 失敗'}")
        elif p == "x":
            ok = post_x(sized, caption)
            results.append(f"🐦 X: {'✅ 成功' if ok else '❌ 失敗'}")
        elif p == "line":
            ok = post_line(sized, caption)
            results.append(f"💚 LINE: {'✅ 成功' if ok else '❌ 失敗'}")

    # bot_data を掃除
    context.bot_data.pop(pending_key, None)
    context.bot_data.pop(f"pending_chat_{chat_id}", None)

    await context.bot.send_message(
        chat_id=chat_id,
        text="📊 投稿結果\n\n" + "\n".join(results)
    )

# ════════════════════════════════════════════════════
#  Telegram ハンドラー
# ════════════════════════════════════════════════════
async def get_bot_username(context: ContextTypes.DEFAULT_TYPE) -> str:
    if "_username" not in context.bot_data:
        me = await context.bot.get_me()
        context.bot_data["_username"] = me.username
    return context.bot_data.get("_username", "")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """写真＋テキストのメッセージを処理"""
    msg: Message = update.message
    if not msg:
        return

    # グループでは @メンション か リプライのみ反応（ロゴ登録含め全処理をスキップ）
    if msg.chat.type in ("group", "supergroup"):
        bot_username = await get_bot_username(context)
        mentioned = (
            (msg.text and f"@{bot_username}" in (msg.text or ""))
            or (msg.caption and f"@{bot_username}" in (msg.caption or ""))
            or (msg.reply_to_message
                and msg.reply_to_message.from_user
                and msg.reply_to_message.from_user.username == bot_username)
            or bool(context.user_data.get("waiting_logo"))
            or (context.user_data.get("learning_images") is not None)
        )
        if not mentioned:
            return

    # キャンセルワードで全待機状態を解除
    _cancel_words = ("キャンセル", "cancel", "Cancel", "やめる", "やめて")
    if (msg.text or "").strip() in _cancel_words:
        cleared = False
        if context.user_data.pop("learning_images", None) is not None:
            context.user_data.pop("learning_label", None)
            cleared = True
        if context.user_data.pop("waiting_logo", None):
            cleared = True
        if cleared:
            await msg.reply_text("✅ キャンセルしました。")
            return

    # ── /learnimage 画像受け取り中 ──────────────────────────
    if context.user_data.get("learning_images") is not None:
        if msg.photo:
            photo = msg.photo[-1]
            file = await photo.get_file()
            data = await file.download_as_bytearray()
            context.user_data["learning_images"].append(bytes(data))
            count = len(context.user_data["learning_images"])
            await msg.reply_text(f"🖼 {count}枚受け取りました。続けて送るか「完了」で分析開始。")
            return
        elif (msg.text or "").strip() in ("完了", "done", "Done", "OK", "ok"):
            images = context.user_data.pop("learning_images")
            label  = context.user_data.pop("learning_label", "競合参考画像")
            if not images:
                await msg.reply_text("画像が1枚もありません。先に画像を送ってください。")
                return
            await msg.reply_text(f"🔍 {len(images)}枚の画像をVisionで分析中...")
            image_analysis, analysis = await asyncio.to_thread(run_learn_images, images, label)
            if image_analysis:
                await msg.reply_text(f"✅ 学習完了！\n\n🖼 画像スタイル分析:\n\n{image_analysis}")
            else:
                await msg.reply_text("❌ 画像の分析に失敗しました。")
            return

    # ── ロゴ登録待機中のアルバム → 各写真をロゴとして保存（商品投稿フローを通さない）──
    if msg.photo and msg.media_group_id and context.user_data.get("waiting_logo"):
        await _save_logo(msg)
        # 同じアルバムの残り写真も処理できるよう waiting_logo は True のまま維持
        # 3秒後に自動解除
        clear_key = f"clear_logo_{msg.chat_id}"
        old = _media_group_tasks.pop(clear_key, None)
        if old and not old.done():
            old.cancel()
        _ud = context.user_data
        async def _clear_logo_after():
            await asyncio.sleep(3)
            _ud.pop("waiting_logo", None)
        _media_group_tasks[clear_key] = asyncio.create_task(_clear_logo_after())
        return

    # ── メディアグループ（アルバム）は user_text チェックより先に処理 ──────
    # Telegram はアルバムを「1枚ずつの別メッセージ」で送るため、
    # キャプションのない2枚目以降がロゴ登録に誤判定されるのを防ぐ
    if msg.photo and msg.media_group_id:
        grp_key  = f"media_group_{msg.media_group_id}"
        bot_username_mg = await get_bot_username(context)
        raw_caption = (msg.caption or "").replace(f"@{bot_username_mg}", "").strip()

        if grp_key not in context.bot_data:
            context.bot_data[grp_key] = {
                "photos": [], "chat_id": msg.chat_id,
                "message_id": msg.message_id,
                "user_text": raw_caption,
                "parsed": None,
            }
        elif raw_caption:
            # 後から届いたキャプション付きメッセージで上書き
            context.bot_data[grp_key]["user_text"] = raw_caption

        file = await msg.photo[-1].get_file()
        context.bot_data[grp_key]["photos"].append(bytes(await file.download_as_bytearray()))

        # 既存タスクをキャンセルして再スケジュール（最後の写真から2秒後に処理）
        old = _media_group_tasks.pop(grp_key, None)
        if old and not old.done():
            old.cancel()
        _app = context.application
        async def _run_after_delay(key=grp_key):
            await asyncio.sleep(2)
            await _process_media_group_direct(_app, key)
        _media_group_tasks[grp_key] = asyncio.create_task(_run_after_delay())
        return  # バッファに積んだだけ

    # ロゴ登録待機中なら写真をロゴとして保存
    if context.user_data.get("waiting_logo") and msg.photo:
        context.user_data["waiting_logo"] = False
        await _save_logo(msg)
        return

    # テキスト取得（caption or text）
    raw_text = msg.caption or msg.text or ""
    # @メンション部分を除去
    bot_username = await get_bot_username(context)
    user_text = raw_text.replace(f"@{bot_username}", "").strip()

    if not user_text:
        # テキストなしで写真だけ → ロゴ登録として扱う（DM のみ。グループは /setlogo 待機中のみここに来る）
        if msg.photo:
            await _save_logo(msg)
        else:
            await msg.reply_text(
                "📝 指示を一緒に送ってください。\n例: この写真でインスタとXに投稿して。お題は新鮮なイチゴの特売"
            )
        return

    # ── 予約投稿の日時入力待ち ──────────────────────────────
    ws_key = context.bot_data.get(f"waiting_schedule_{msg.chat_id}")
    if ws_key and not msg.photo:
        # キャンセル
        if user_text.strip() in ("キャンセル", "cancel", "Cancel"):
            del context.bot_data[f"waiting_schedule_{msg.chat_id}"]
            await msg.reply_text("予約入力をキャンセルしました。")
            return
        pending = context.bot_data.get(ws_key)
        if pending:
            # 既に検出済みの schedule_time をそのまま使う or 新しく解析
            iso_time = pending.get("schedule_time", "") if user_text.strip() in ("確定", "ok", "OK") else ""
            if not iso_time:
                await msg.reply_text("🕐 時間を解析中...")
                iso_time = parse_schedule_time(user_text)

            if not iso_time:
                await msg.reply_text(
                    "⏰ 日時を読み取れませんでした。もう一度入力してください。\n"
                    "例: 明日の朝9時 / 3月25日 14:00\n「キャンセル」で中止"
                )
                return

            dt = datetime.fromisoformat(iso_time)
            # 予約ジョブを登録
            if not context.job_queue:
                await msg.reply_text("❌ 予約投稿機能が利用できません（APScheduler未設定）。")
                return

            job_name = f"scheduled_{msg.chat_id}"
            # 同一チャットの既存予約を上書き
            for old_job in context.job_queue.get_jobs_by_name(job_name):
                old_job.schedule_removal()

            context.job_queue.run_once(
                execute_scheduled_post,
                when=dt,
                data={
                    "chat_id":     pending["chat_id"],
                    "image_bytes": pending["image_bytes"],
                    "captions":    pending["captions"],
                    "platforms":   pending["platforms"],
                    "pending_key": ws_key,
                },
                name=job_name,
            )
            del context.bot_data[f"waiting_schedule_{msg.chat_id}"]
            await msg.reply_text(
                f"📅 {dt.strftime('%Y年%m月%d日 %H:%M')} に予約しました ✅\n"
                "投稿時刻になったら自動で投稿します。"
            )
        return

    # ── 画像修正ボタン（🎨）押下後のテキスト入力 ──────────────
    pending_key = context.bot_data.get(f"pending_chat_{msg.chat_id}")
    if pending_key:
        # chat_id_msg_id 部分を抽出してeditimg_キーを確認
        key_suffix = pending_key.replace("pending_", "", 1)  # "chat_id_msg_id"
        editimg_key  = f"editimg_{key_suffix}"
        rewrite_key  = f"rewrite_{key_suffix}"

        if context.bot_data.pop(editimg_key, None) and not msg.photo:
            # 画像修正モード：テキスト指示を画像修正に特化して渡す
            pending = context.bot_data.get(pending_key, {})
            forced = {"target": "image", "image_note": user_text, "caption_note": ""}
            await _handle_modification(msg, context, user_text, pending, pending_key, force_mod=forced)
            return

        if context.bot_data.pop(rewrite_key, None) and not msg.photo:
            # キャプション修正モード
            pending = context.bot_data.get(pending_key, {})
            forced = {"target": "caption", "image_note": "", "caption_note": user_text}
            await _handle_modification(msg, context, user_text, pending, pending_key, force_mod=forced)
            return

    # ── プレビュー後の自由テキスト修正指示 ──────────────────────
    if pending_key and pending_key in context.bot_data and not msg.photo:
        pending = context.bot_data[pending_key]
        await _handle_modification(msg, context, user_text, pending, pending_key)
        return

    await msg.reply_text("🤖 指示を解析中...")

    # 指示解析
    parsed = parse_instruction(user_text)
    platforms        = parsed.get("platforms", ["instagram", "x"])
    do_add_logo      = parsed.get("add_logo", True)
    do_generate      = parsed.get("generate_image", False)
    image_prompt     = parsed.get("image_prompt", "")
    caption_instr    = parsed.get("caption_instruction", user_text)
    schedule_time    = parsed.get("schedule_time", "")
    post_type        = parsed.get("post_type", "single")
    items            = parsed.get("items", [])

    # 画像取得
    image_bytes = None

    # 写真が添付されている場合（media_group_id 付きは上で処理済みのため単体写真のみ到達）
    if msg.photo:
        # ── 単体写真 ──
        file = await msg.photo[-1].get_file()
        raw_photo = bytes(await file.download_as_bytearray())
        use_as_is = any(kw in user_text for kw in ("そのまま", "この写真で", "この画像で", "加工なし"))
        if use_as_is:
            image_bytes = raw_photo
        else:
            await msg.reply_text("🎨 商品を切り出してデザインし直し中...（少し時間がかかります）")
            image_bytes = await redesign_product_image([raw_photo], caption_instr)
            if not image_bytes:
                await msg.reply_text("⚠️ デザイン生成に失敗しました。元の写真を使います。")
                image_bytes = raw_photo
            if not image_prompt:
                image_prompt = caption_instr

    # 写真なし＆生成指示あり → Geminiでテキストから画像生成
    elif do_generate and image_prompt:
        await msg.reply_text("🎨 Geminiで画像生成中...")
        image_bytes = await generate_image_gemini(image_prompt)
        if not image_bytes:
            await msg.reply_text("❌ 画像生成に失敗しました。")
            return

    if not image_bytes:
        await msg.reply_text(
            "📷 画像が必要です。写真を添付して送ってください。\n"
            "または「〇〇の画像を生成して」と指示してください。"
        )
        return

    # ロゴ未登録の場合だけ警告（合成は _send_preview 内でリサイズ後に行う）
    if do_add_logo and not get_logo_paths():
        await msg.reply_text(
            "⚠️ ロゴ未登録です。/setlogo でロゴを登録するとすべての投稿に自動でロゴが入ります。"
        )

    # キャプション生成
    await msg.reply_text("✍️ キャプション生成中...")
    captions = {p: generate_caption(caption_instr, p, post_type, items) for p in platforms}

    await _send_preview(msg, context, image_bytes, captions, platforms,
                        image_prompt, caption_instr, do_add_logo, schedule_time,
                        post_type, items)


async def _send_preview(msg, context, image_bytes, captions, platforms,
                        image_prompt="", caption_instr="", do_add_logo=True, schedule_time="",
                        post_type="single", items=None):
    """プレビュー画像とキャプションを送信（各プラットフォームのリサイズ済み画像を表示）"""
    platform_emoji = {"instagram": "📸", "x": "🐦", "line": "💚"}
    platform_sizes = {"instagram": "1080×1080", "x": "1200×675", "line": "1040×1040"}

    # 各プラットフォーム向けにリサイズ → ロゴ合成（ロゴはリサイズ後に合成してクロップ欠けを防ぐ）
    logos = get_logo_paths()
    sized_images = {}
    for p in platforms:
        resized = resize_for_platform(image_bytes, p)
        if do_add_logo and logos:
            resized = add_logo_to_image(resized)
        sized_images[p] = resized

    key = f"pending_{msg.chat_id}_{msg.message_id}"
    context.bot_data[key] = {
        "image_bytes":  image_bytes,    # 元画像（修正再生成用）
        "sized_images": sized_images,   # リサイズ済み（投稿用）
        "captions":     captions,
        "platforms":    platforms,
        "chat_id":      msg.chat_id,
        "message_id":   msg.message_id,
        "image_prompt": image_prompt,
        "caption_instr": caption_instr,
        "do_add_logo":   do_add_logo,
        "schedule_time": schedule_time,
        "post_type":     post_type,
        "items":         items or [],
    }
    context.bot_data[f"pending_chat_{msg.chat_id}"] = key

    # 各プラットフォームのリサイズ済み画像を1枚ずつ送信
    for p in platforms:
        emoji = platform_emoji.get(p, "")
        size  = platform_sizes.get(p, "")
        await msg.reply_photo(
            photo=io.BytesIO(sized_images[p]),
            caption=f"{emoji} {p.upper()}  {size}"
        )

    # キャプション＋ボタン（parse_mode なし ← Claude生成テキストが含まれるため）
    caption_lines = ["📋 投稿プレビュー\n"]
    for p in platforms:
        emoji = platform_emoji.get(p, "")
        caption_lines.append(f"{emoji} {p.upper()}\n{captions[p]}\n")

    if schedule_time:
        try:
            dt = datetime.fromisoformat(schedule_time)
            caption_lines.append(f"⏰ 検出された投稿時刻: {dt.strftime('%Y年%m月%d日 %H:%M')}")
        except Exception:
            pass

    caption_lines.append("返信でテキスト修正指示も送れます")

    keyboard = [
        [InlineKeyboardButton("✅ 今すぐ投稿",    callback_data=f"post_{msg.chat_id}_{msg.message_id}")],
        [InlineKeyboardButton("🎨 画像を修正",    callback_data=f"editimg_{msg.chat_id}_{msg.message_id}"),
         InlineKeyboardButton("✏️ キャプション修正", callback_data=f"rewrite_{msg.chat_id}_{msg.message_id}")],
        [InlineKeyboardButton("⏰ 予約投稿",      callback_data=f"schedule_{msg.chat_id}_{msg.message_id}")],
        [InlineKeyboardButton("❌ キャンセル",    callback_data=f"cancel_{msg.chat_id}_{msg.message_id}")],
    ]
    await msg.reply_text(
        "\n".join(caption_lines),
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def _handle_modification(msg, context, user_text: str, pending: dict, pending_key: str,
                               force_mod: dict = None):
    """プレビュー後の自然言語修正指示を処理。force_mod を渡すと Claude 判定をスキップ。"""
    await msg.reply_text("🤖 修正内容を解析中...")

    image_prompt   = pending.get("image_prompt", "")
    caption_instr  = pending.get("caption_instr", "")
    platforms      = pending.get("platforms", ["instagram", "x"])
    do_add_logo    = pending.get("do_add_logo", True)
    schedule_time  = pending.get("schedule_time", "")
    image_bytes    = pending.get("image_bytes")

    if force_mod:
        mod = force_mod
    else:
        # Claudeで修正種別を判定
        result = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": f"""以下の修正指示がSNS投稿の「画像」「キャプション」「両方」のどれに関するものかを判定してください。

修正指示: {user_text}

JSONのみ返してください:
{{"target": "image" or "caption" or "both", "image_note": "画像への追加指示", "caption_note": "キャプションへの追加指示"}}"""}]
        )
        try:
            text = result.content[0].text.strip()
            text = re.sub(r"```json|```", "", text).strip()
            mod = json.loads(text)
        except Exception:
            mod = {"target": "both", "image_note": user_text, "caption_note": user_text}

    target = mod.get("target", "both")

    # 画像を修正
    if target in ("image", "both") and image_prompt:
        new_prompt = f"{image_prompt}。修正: {mod.get('image_note', user_text)}"
        await msg.reply_text("🎨 画像を再生成中...")
        new_image = await generate_image_gemini(new_prompt)
        if new_image:
            image_bytes = new_image   # ロゴは _send_preview 内でリサイズ後に合成
            image_prompt = new_prompt
        else:
            await msg.reply_text("⚠️ 画像の再生成に失敗しました。元の画像でキャプションのみ更新します。")

    # キャプションを修正
    if target in ("caption", "both"):
        new_instr = f"{caption_instr}。修正: {mod.get('caption_note', user_text)}"
        await msg.reply_text("✍️ キャプション修正中...")
        post_type = pending.get("post_type", "single")
        items     = pending.get("items", [])
        captions = {p: generate_caption(new_instr, p, post_type, items) for p in platforms}
        caption_instr = new_instr
    else:
        captions = pending.get("captions", {})

    # 古いpendingを削除して新しいプレビューを表示
    del context.bot_data[pending_key]
    post_type = pending.get("post_type", "single")
    items     = pending.get("items", [])
    await _send_preview(msg, context, image_bytes, captions, platforms,
                        image_prompt, caption_instr, do_add_logo, schedule_time,
                        post_type, items)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    # callback_data から (chat_id, msg_id) を安全に取り出すヘルパー
    try:
        parts = data.split("_", 2)
        chat_id, msg_id = parts[1], parts[2]
    except (IndexError, ValueError):
        logger.error(f"不正なcallback_data: {data}")
        await query.edit_message_text("❌ 不正なデータです。もう一度やり直してください。")
        return

    if data.startswith("cancel_"):
        pending_key = f"pending_{chat_id}_{msg_id}"
        context.bot_data.pop(pending_key, None)
        context.bot_data.pop(f"pending_chat_{chat_id}", None)
        await query.edit_message_text("❌ キャンセルしました。")
        return

    if data.startswith("post_"):
        await query.edit_message_text("⏳ 投稿中...")
        await execute_posting(context, int(chat_id), int(msg_id))
        return

    if data.startswith("schedule_"):
        pending_key = f"pending_{chat_id}_{msg_id}"
        pending = context.bot_data.get(pending_key)
        if not pending:
            await query.edit_message_text("❌ データが見つかりません。もう一度やり直してください。")
            return
        # 既に検出済みの schedule_time があれば提示
        hint = ""
        if pending.get("schedule_time"):
            try:
                dt = datetime.fromisoformat(pending["schedule_time"])
                hint = f"\n（検出済み: {dt.strftime('%Y年%m月%d日 %H:%M')} — そのまま送信で確定）"
            except Exception:
                pass
        context.bot_data[f"waiting_schedule_{chat_id}"] = pending_key
        await query.edit_message_text(
            f"📅 投稿する日時を入力してください。{hint}\n"
            "例: 明日の朝9時 / 3月25日 14:00 / 2026-03-25 09:00"
        )
        return

    if data.startswith("editimg_"):
        context.bot_data[f"editimg_{chat_id}_{msg_id}"] = True
        await query.edit_message_text(
            "🎨 どう修正しますか？修正内容を送ってください。\n"
            "例: もっと明るい雰囲気にして / 背景を白にして / 野菜を大きく映して"
        )
        return

    if data.startswith("rewrite_"):
        context.bot_data[f"rewrite_{chat_id}_{msg_id}"] = True
        await query.edit_message_text(
            "📝 新しいキャプション指示を送ってください："
        )
        return


def get_logo_paths() -> list[Path]:
    """登録済みロゴ一覧を返す"""
    return sorted(LOGO_DIR.glob("logo_*.png"))

async def set_logo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ロゴ画像を登録する（最大4枚）"""
    msg = update.message
    if msg.photo:
        await _save_logo(msg)
    else:
        context.user_data["waiting_logo"] = True
        count = len(get_logo_paths())
        await msg.reply_text(
            f"📷 ロゴ画像を送ってください。\n"
            f"現在 {count}/4 枚登録済み。"
        )

async def clear_logos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """登録済みロゴをすべて削除する"""
    msg = update.message
    logos = get_logo_paths()
    if not logos:
        await msg.reply_text("ロゴは登録されていません。")
        return
    for lp in logos:
        lp.unlink(missing_ok=True)
    await msg.reply_text(f"🗑 {len(logos)}枚のロゴを削除しました。\n/setlogo で新しいロゴを登録してください。")

def _image_hash(img: Image.Image) -> str:
    """画像を8x8グレースケールに縮小してハッシュ化（重複検出用）"""
    import hashlib
    small = img.convert("L").resize((8, 8), Image.LANCZOS)
    return hashlib.md5(small.tobytes()).hexdigest()

async def _save_logo(msg):
    file = await msg.photo[-1].get_file()
    img_bytes = bytes(await file.download_as_bytearray())
    new_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    new_hash = _image_hash(new_img)

    # 重複チェック
    existing = get_logo_paths()
    for lp in existing:
        existing_hash = _image_hash(Image.open(lp).convert("RGBA"))
        if existing_hash == new_hash:
            await msg.reply_text(f"⚠️ このロゴは既に登録済みです。（{len(existing)}/4枚）")
            return

    if len(existing) >= 4:
        save_path = existing[0]  # 一番古いものを上書き
        new_img.save(save_path, "PNG")
        await msg.reply_text(f"✅ ロゴを更新しました！（4/4枚）")
    else:
        save_path = LOGO_DIR / f"logo_{len(existing)+1}.png"
        new_img.save(save_path, "PNG")
        count = len(get_logo_paths())
        await msg.reply_text(f"✅ ロゴを登録しました！（{count}/4枚）")


async def learn_own_posts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """自分のInstagram・X投稿を学習"""
    msg = update.message
    await msg.reply_text("📚 自分の投稿を取得・分析中...（少し時間がかかります）")
    try:
        insta_count, x_count, img_count, analysis, image_analysis, warning = await run_learn_own_posts()
        x_status = f"{x_count}件" if x_count else "⚠️ クレジット不足（後で追加可）"
        img_status = f"（画像 {img_count}枚 を Vision 分析）" if img_count else "（画像URL取得不可）"
        header = (
            f"✅ 学習完了！\n\n"
            f"📸 Instagram: {insta_count}件 {img_status}\n"
            f"🐦 X: {x_status}"
        )
        if warning == "x_credit":
            header += (
                "\n\n⚠️ Xのクレジットが不足しています\n"
                "X Developer Console でクレジットを追加すると\n"
                "Xの過去投稿も学習できます。"
            )
        await msg.reply_text(header)
        if image_analysis:
            await msg.reply_text(f"🖼 画像スタイル分析:\n\n{image_analysis}")
        await msg.reply_text(f"📊 総合スタイル分析:\n\n{analysis}")
    except Exception as e:
        logger.error(f"学習エラー: {e}")
        await msg.reply_text(f"❌ エラー: {e}")


async def learn_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """競合・参考URLから学習"""
    msg = update.message
    if not context.args:
        await msg.reply_text(
            "URLを指定してください。\n"
            "例: `/learnurl https://www.instagram.com/competitor/`",
            parse_mode="Markdown"
        )
        return
    url = context.args[0]
    await msg.reply_text(f"🔍 {url} を解析中...")
    try:
        count, img_count, image_analysis, analysis = await run_learn_url(url)
        if count == 0:
            hint = ""
            if "instagram.com" in url:
                hint = "\n\nInstagram はログインが必要なためプロフィールページのスクレイピングができません。\n個別投稿URL（/p/xxxxx）なら取得できる場合があります。"
            elif "x.com" in url or "twitter.com" in url:
                hint = "\n\nX はログインが必要なためスクレイピングできません。"
            await msg.reply_text(f"❌ 投稿を取得できませんでした。{hint}")
            return
        await msg.reply_text(f"✅ 参考投稿 {count}件 を学習しました！" + (f"（画像 {img_count}枚 を Vision 分析）" if img_count else ""))
        if image_analysis:
            await msg.reply_text(f"🖼 画像スタイル分析:\n\n{image_analysis}")
        await msg.reply_text(f"📊 更新されたスタイル分析:\n\n{analysis}")
    except Exception as e:
        logger.error(f"URL学習エラー: {e}")
        await msg.reply_text(f"❌ エラー: {e}")


async def learn_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /learnimage で画像を学習。
    コマンドと一緒に画像を送るか、コマンド送信後に複数枚送れる。
    使い方:
      /learnimage [ラベル]  → 以降に送った画像を学習
      画像にキャプション /learnimage でも可
    """
    msg = update.message
    label = " ".join(context.args) if context.args else "競合参考画像"

    # コマンドと同時に画像が送られた場合
    if msg.photo:
        context.user_data["learning_images"] = []
        context.user_data["learning_label"] = label
        photo = msg.photo[-1]
        file = await photo.get_file()
        data = await file.download_as_bytearray()
        context.user_data["learning_images"].append(bytes(data))
        await msg.reply_text(
            f"🖼 1枚受け取りました。続けて画像を送ってください。\n"
            f"分析を開始するには「完了」と送信してください。\n"
            f"ラベル: {label}"
        )
        return

    # 画像なし → 受付モード開始
    context.user_data["learning_images"] = []
    context.user_data["learning_label"] = label
    await msg.reply_text(
        f"📸 学習させたい競合・参考アカウントの画像を送ってください（複数可）。\n"
        f"送り終わったら「完了」と送信してください。\n"
        f"ラベル: {label}"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🛒 *みどりのマート SNSボット*\n\n"
        "グループに追加して @メンション で使えます。\n\n"
        "*使い方:*\n"
        "`@bot この写真でインスタとXに投稿して。桃の特売200円`\n"
        "`@bot みどまのロゴを入れてLINEにも投稿して`\n\n"
        "*コマンド:*\n"
        "/setlogo - ロゴ画像を登録\n"
        "/learnposts - 自分のInstagram・X投稿を学習\n"
        "/learnurl <URL> - 競合・参考アカウントのURLを学習\n"
        "/learnimage [ラベル] - 競合の画像を直接送って学習\n"
        "/start - このメッセージ",
        parse_mode="Markdown"
    )


# ════════════════════════════════════════════════════
#  メイン
# ════════════════════════════════════════════════════
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("setlogo", set_logo))
    app.add_handler(CommandHandler("clearlogos", clear_logos))
    app.add_handler(CommandHandler("learnposts", learn_own_posts))
    app.add_handler(CommandHandler("learnurl", learn_url))
    app.add_handler(CommandHandler("learnimage", learn_image))
    app.add_handler(CallbackQueryHandler(callback_handler))
    # テキスト・写真メッセージ（グループ・DM両対応）
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO) & ~filters.COMMAND,
        handle_message
    ))

    logger.info("🚀 ボット起動中...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    main()
