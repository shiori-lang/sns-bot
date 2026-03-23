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
from learn import run_learn_own_posts, run_learn_url, load_style_guide

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
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""以下のSNS投稿指示を解析してJSON形式で返してください。
現在日時(JST): {now_jst}

指示: {user_text}

返すJSONの形式:
{{
  "platforms": ["instagram", "x", "line"],  // 指定されたSNS（複数可）。「全部」なら3つ全て。未指定なら["instagram","x"]
  "add_logo": true or false,  // 基本はtrue。「ロゴなし」「ロゴ不要」と明示された場合のみfalse
  "generate_image": true or false,  // 新しく画像生成が必要か（写真は別途送られてくる場合はfalse）
  "image_prompt": "",  // 画像生成する場合のプロンプト（日本語）
  "caption_instruction": "",  // キャプション生成への指示（お題・内容など）
  "schedule_time": ""  // 予約投稿の場合はISO 8601形式（例: "2026-03-25T09:00:00+09:00"）。今すぐ投稿なら空文字
}}

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
    """プラットフォームの推奨サイズにリサイズ（センタークロップ）"""
    target_w, target_h = PLATFORM_IMAGE_SIZES.get(platform, (1080, 1080))
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h

    if src_ratio > tgt_ratio:
        # 横長 → 左右をクロップ
        new_w = int(img.height * tgt_ratio)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    elif src_ratio < tgt_ratio:
        # 縦長 → 上下をクロップ
        new_h = int(img.width / tgt_ratio)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))

    img = img.resize((target_w, target_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def find_best_logo_position(base_img: Image.Image, logo_w: int, logo_h: int) -> tuple[int, int]:
    """
    画像の4隅を比較して最も明るい（被りにくい）コーナーを選ぶ
    ロゴは白・明るい背景に映えるため、最も暗い（コントラストが取れる）隅を選ぶ
    """
    margin = 20
    w, h = base_img.width, base_img.height
    gray = base_img.convert("L")

    corners = {
        "top_left":     (margin, margin),
        "top_right":    (w - logo_w - margin, margin),
        "bottom_left":  (margin, h - logo_h - margin),
        "bottom_right": (w - logo_w - margin, h - logo_h - margin),
    }

    # 各隅の領域の平均輝度を計算（輝度が低い＝暗い→ロゴが目立つ）
    best_pos = corners["bottom_right"]
    best_score = float("inf")

    for name, (cx, cy) in corners.items():
        region = gray.crop((cx, cy, cx + logo_w, cy + logo_h))
        import numpy as np
        arr = list(region.getdata())
        avg = sum(arr) / len(arr) if arr else 128
        # 端に近いほどスコアを下げる（端の方が自然）
        score = avg
        if score < best_score:
            best_score = score
            best_pos = (cx, cy)

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
    """Gemini で画像生成"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(
            [f"スーパーのSNS投稿用の魅力的な商品画像を生成: {prompt}"],
            generation_config=genai.GenerationConfig(
                response_modalities=["image", "text"]
            )
        )
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                return base64.b64decode(part.inline_data.data)
    except Exception as e:
        logger.error(f"Gemini画像生成エラー: {e}")
    return None

# ════════════════════════════════════════════════════
#  Claude でキャプション生成
# ════════════════════════════════════════════════════
def generate_caption(instruction: str, platform: str) -> str:
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

    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"スーパー「みどりのマート」の{platform}向けキャプションを日本語で作成。\n"
                f"ルール: {rule}\n"
                f"{style_section}"
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
        text="📅 *予約投稿 完了*\n\n" + "\n".join(results),
        parse_mode="Markdown"
    )


async def execute_posting(context: ContextTypes.DEFAULT_TYPE,
                           chat_id: int, message_id: int):
    data      = context.bot_data.get(f"pending_{chat_id}_{message_id}", {})
    image_bytes = data.get("image_bytes")
    captions    = data.get("captions", {})
    platforms   = data.get("platforms", [])

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

    await context.bot.send_message(
        chat_id=chat_id,
        text="📊 *投稿結果*\n\n" + "\n".join(results),
        parse_mode="Markdown"
    )

# ════════════════════════════════════════════════════
#  Telegram ハンドラー
# ════════════════════════════════════════════════════
async def get_bot_username(context: ContextTypes.DEFAULT_TYPE) -> str:
    if not hasattr(context.bot_data, "_username"):
        me = await context.bot.get_me()
        context.bot_data["_username"] = me.username
    return context.bot_data.get("_username", "")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """写真＋テキストのメッセージを処理"""
    msg: Message = update.message
    if not msg:
        return

    # ロゴ登録待機中なら写真をロゴとして保存
    if context.user_data.get("waiting_logo") and msg.photo:
        context.user_data["waiting_logo"] = False
        await _save_logo(msg)
        return

    # グループでは @メンション か リプライのみ反応
    if msg.chat.type in ("group", "supergroup"):
        bot_username = await get_bot_username(context)
        mentioned = (
            msg.text and f"@{bot_username}" in (msg.text or "")
        ) or (
            msg.caption and f"@{bot_username}" in (msg.caption or "")
        ) or (
            msg.reply_to_message and
            msg.reply_to_message.from_user and
            msg.reply_to_message.from_user.username == bot_username
        )
        if not mentioned:
            return

    # テキスト取得（caption or text）
    raw_text = msg.caption or msg.text or ""
    # @メンション部分を除去
    bot_username = await get_bot_username(context)
    user_text = raw_text.replace(f"@{bot_username}", "").strip()

    if not user_text:
        # テキストなしで写真だけ → ロゴ登録として扱う
        if msg.photo:
            await _save_logo(msg)
        else:
            await msg.reply_text(
                "📝 指示を一緒に送ってください。\n例: `@bot この写真でインスタとXに投稿して。お題は新鮮なイチゴの特売`",
                parse_mode="Markdown"
            )
        return

    # ── 予約投稿の日時入力待ち ──────────────────────────────
    ws_key = context.bot_data.get(f"waiting_schedule_{msg.chat_id}")
    if ws_key and not msg.photo:
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
                    "例: `明日の朝9時` / `3月25日 14:00`",
                    parse_mode="Markdown"
                )
                return

            dt = datetime.fromisoformat(iso_time)
            # 予約ジョブを登録
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

    # ── プレビュー後の修正指示を自然言語で処理 ──────────────────
    pending_key = context.bot_data.get(f"pending_chat_{msg.chat_id}")
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

    # 画像取得
    image_bytes = None

    # 写真が添付されている場合
    if msg.photo:
        file = await msg.photo[-1].get_file()
        image_bytes = bytes(await file.download_as_bytearray())

    # 写真なし＆生成指示あり → Gemini生成
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

    # ロゴ合成（デフォルトで常に入れる。「ロゴなし」指示の場合のみスキップ）
    if do_add_logo:
        logos = get_logo_paths()
        if logos:
            image_bytes = add_logo_to_image(image_bytes)
        else:
            await msg.reply_text(
                "⚠️ ロゴ未登録です。`/setlogo` でロゴを登録するとすべての投稿に自動でロゴが入ります。"
            )

    # キャプション生成
    await msg.reply_text("✍️ キャプション生成中...")
    captions = {p: generate_caption(caption_instr, p) for p in platforms}

    await _send_preview(msg, context, image_bytes, captions, platforms, image_prompt, caption_instr, do_add_logo, schedule_time)


async def _send_preview(msg, context, image_bytes, captions, platforms,
                        image_prompt="", caption_instr="", do_add_logo=True, schedule_time=""):
    """プレビュー画像とキャプションを送信"""
    # データ保存
    key = f"pending_{msg.chat_id}_{msg.message_id}"
    context.bot_data[key] = {
        "image_bytes": image_bytes,
        "captions": captions,
        "platforms": platforms,
        "chat_id": msg.chat_id,
        "message_id": msg.message_id,
        "image_prompt": image_prompt,
        "caption_instr": caption_instr,
        "do_add_logo": do_add_logo,
        "schedule_time": schedule_time,
    }
    # 最新のpendingを記録（修正フロー用）
    context.bot_data[f"pending_chat_{msg.chat_id}"] = key

    platform_emoji = {"instagram": "📸", "x": "🐦", "line": "💚"}
    platform_sizes = {"instagram": "1080×1080", "x": "1200×675", "line": "1040×1040"}
    preview_lines = ["📋 *投稿プレビュー*\n"]
    for p in platforms:
        emoji = platform_emoji.get(p, "")
        size = platform_sizes.get(p, "")
        preview_lines.append(f"{emoji} *{p.upper()}* `{size}`\n{captions[p]}\n")

    if schedule_time:
        try:
            dt = datetime.fromisoformat(schedule_time)
            preview_lines.append(f"⏰ *検出された投稿時刻:* {dt.strftime('%Y年%m月%d日 %H:%M')}")
        except Exception:
            pass

    preview_lines.append("_返信で修正指示を送ることもできます_")

    await msg.reply_photo(photo=io.BytesIO(image_bytes))
    keyboard = [
        [InlineKeyboardButton("✅ 今すぐ投稿", callback_data=f"post_{msg.chat_id}_{msg.message_id}")],
        [InlineKeyboardButton("⏰ 予約投稿", callback_data=f"schedule_{msg.chat_id}_{msg.message_id}")],
        [InlineKeyboardButton("❌ キャンセル", callback_data=f"cancel_{msg.chat_id}_{msg.message_id}")],
    ]
    await msg.reply_text(
        "\n".join(preview_lines),
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def _handle_modification(msg, context, user_text: str, pending: dict, pending_key: str):
    """プレビュー後の自然言語修正指示を処理"""
    await msg.reply_text("🤖 修正内容を解析中...")

    image_prompt   = pending.get("image_prompt", "")
    caption_instr  = pending.get("caption_instr", "")
    platforms      = pending.get("platforms", ["instagram", "x"])
    do_add_logo    = pending.get("do_add_logo", True)
    schedule_time  = pending.get("schedule_time", "")
    image_bytes    = pending.get("image_bytes")

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
            if do_add_logo and get_logo_paths():
                new_image = add_logo_to_image(new_image)
            image_bytes = new_image
            image_prompt = new_prompt

    # キャプションを修正
    if target in ("caption", "both"):
        new_instr = f"{caption_instr}。修正: {mod.get('caption_note', user_text)}"
        await msg.reply_text("✍️ キャプション修正中...")
        captions = {p: generate_caption(new_instr, p) for p in platforms}
        caption_instr = new_instr
    else:
        captions = pending.get("captions", {})

    # 古いpendingを削除して新しいプレビューを表示
    del context.bot_data[pending_key]
    await _send_preview(msg, context, image_bytes, captions, platforms, image_prompt, caption_instr, do_add_logo, schedule_time)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("cancel_"):
        _, chat_id, msg_id = data.split("_", 2)
        pending_key = f"pending_{chat_id}_{msg_id}"
        context.bot_data.pop(pending_key, None)
        context.bot_data.pop(f"pending_chat_{chat_id}", None)
        await query.edit_message_text("❌ キャンセルしました。")
        return

    if data.startswith("post_"):
        _, chat_id, msg_id = data.split("_", 2)
        await query.edit_message_text("⏳ 投稿中...")
        await execute_posting(context, int(chat_id), int(msg_id))
        return

    if data.startswith("schedule_"):
        _, chat_id, msg_id = data.split("_", 2)
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
            "例: `明日の朝9時` / `3月25日 14:00` / `2026-03-25 09:00`",
            parse_mode="Markdown"
        )
        return

    if data.startswith("editimg_"):
        _, chat_id, msg_id = data.split("_", 2)
        context.bot_data[f"editimg_{chat_id}_{msg_id}"] = True
        await query.edit_message_text(
            "🎨 どう修正しますか？修正内容を送ってください。\n"
            "例: `もっと明るい雰囲気にして` / `背景を白にして` / `野菜を大きく映して`"
        )
        return

    if data.startswith("rewrite_"):
        _, chat_id, msg_id = data.split("_", 2)
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
        insta_count, x_count, analysis, warning = await run_learn_own_posts()
        x_status = f"{x_count}件" if x_count else "⚠️ クレジット不足（後で追加可）"
        header = (
            f"✅ 学習完了！\n\n"
            f"📸 Instagram: {insta_count}件\n"
            f"🐦 X: {x_status}"
        )
        if warning == "x_credit":
            header += (
                "\n\n⚠️ Xのクレジットが不足しています\n"
                "X Developer Console でクレジットを追加すると\n"
                "Xの過去投稿も学習できます。"
            )
        await msg.reply_text(header)
        await msg.reply_text(f"📊 スタイル分析結果:\n\n{analysis}")
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
        count, analysis = await run_learn_url(url)
        if count == 0:
            await msg.reply_text("❌ 投稿を取得できませんでした。URLを確認してください。")
            return
        await msg.reply_text(f"✅ 参考投稿 {count}件 を学習しました！")
        await msg.reply_text(f"📊 更新されたスタイル分析:\n\n{analysis}")
    except Exception as e:
        logger.error(f"URL学習エラー: {e}")
        await msg.reply_text(f"❌ エラー: {e}")


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
    app.add_handler(CommandHandler("learnposts", learn_own_posts))
    app.add_handler(CommandHandler("learnurl", learn_url))
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
