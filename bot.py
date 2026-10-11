import os
import asyncio
import io
import json
import re
import glob
import shutil
import tempfile
import time
import urllib.parse
import logging
import httpx
import yt_dlp
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def first_run_setup():
    """Birinchi ishga tushishda kalitlarni so'raydi va .env ga saqlaydi."""
    if os.path.exists(ENV_PATH):
        return
    print("Spark AI birinchi marta ishga tushyapti. Kalitlarni kiriting.\n")
    tg = input("Telegram bot token (BotFather'dan): ").strip()
    ai = input("Gemini API kalit (aistudio.google.com dan): ").strip()
    with open(ENV_PATH, "w") as f:
        f.write(f"TELEGRAM_TOKEN={tg}\nGEMINI_API_KEY={ai}\n")
    print("\nSaqlandi. Keyingi safar so'ramaydi.\n")


first_run_setup()
load_dotenv(ENV_PATH)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
# Model ishlamasa, keyingisini avtomatik sinab ko'radi
MODELS = [
    m for m in [
        os.getenv("GEMINI_MODEL"),
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-flash-latest",
    ] if m
]
working_model = None
working_minimal = True

SYSTEM_PROMPT = (
    "Sen Spark AI san, Telegramdagi do'stona yordamchi bot. "
    "Foydalanuvchi qaysi tilda yozsa, shu tilda javob ber "
    "(asosan o'zbekcha). Javoblaring qisqa, aniq va tushunarli bo'lsin."
)

MAX_MESSAGES = 20  # har bir foydalanuvchi uchun eslab qolinadigan xabarlar soni

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

client = genai.Client(api_key=GEMINI_API_KEY)
histories: dict[int, list[types.Content]] = {}


INTRO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spark_intro.mp4")
intro_file_id = None  # birinchi yuborishdan keyin Telegram'ga saqlanadi, keyingilari bir zumda ketadi


async def send_intro(chat_id, context: ContextTypes.DEFAULT_TYPE):
    """Spark mashinasi dvigatel ovozi bilan kelib, fonarini 2 marta miltillatadigan video."""
    global intro_file_id
    caption = "Spark AI tayyor! ⚡ (ovozni eshitish uchun videoni bosing 🔊)"
    try:
        if intro_file_id:
            await context.bot.send_video(chat_id, intro_file_id, caption=caption, supports_streaming=True)
            return True
        if not os.path.exists(INTRO_PATH):
            return False
        with open(INTRO_PATH, "rb") as f:
            msg = await context.bot.send_video(
                chat_id, f, caption=caption, duration=7, width=720, height=400,
                supports_streaming=True, **UPLOAD_KW,
            )
        intro_file_id = msg.video.file_id
        return True
    except Exception as e:
        logging.error("Intro video xatosi: %s", e)
        return False


async def spark(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.UPLOAD_VIDEO)
    ok = await send_intro(update.effective_chat.id, context)
    if not ok:
        await update.message.reply_text("Spark mashinasi hozir kelolmadi, spark_intro.mp4 fayli yo'q ekan 🚗")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Salom! Men Spark AI ⚡\n"
        "Menga istalgan savol bering, yordam beraman.\n\n"
        "/logo - logo yaratish (masalan: /logo Spark AI, ko'k va sariq)\n"
        "/musiqa - qo'shiq topib MP3 yuborish (masalan: /musiqa Qalbim)\n"
        "/video - video topib yuborish (yoki YouTube havolasini tashlang)\n"
        "/spark - Spark mashinasini chaqirish 🚗⚡\n"
        "/reset - suhbatni tozalash"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    histories.pop(update.effective_user.id, None)
    await update.message.reply_text("Suhbat tozalandi ✨")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    if TRIGGER_RE.search(text):
        kind_name, query = await detect_intent(text)
        if kind_name in ("music", "video") and query:
            await handle_query(update, query, "a" if kind_name == "music" else "v")
            return

    history = histories.setdefault(user_id, [])
    history.append(types.Content(role="user", parts=[types.Part(text=text)]))
    del history[:-MAX_MESSAGES]
    while history and history[0].role != "user":
        history.pop(0)

    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    global working_model, working_minimal
    answer = None
    fatal = False
    candidates = [working_model] if working_model else MODELS
    for model in candidates:
        for minimal in ([working_minimal, False] if working_minimal else [False]):
            kwargs = dict(system_instruction=SYSTEM_PROMPT, max_output_tokens=2000)
            if minimal:
                # "o'ylash"ni minimal qilish javobni ancha tezlashtiradi
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")
            try:
                t0 = time.time()
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=history,
                    config=types.GenerateContentConfig(**kwargs),
                )
                answer = response.text or "Javob olib bo'lmadi, boshqacha so'rab ko'ring."
                working_model, working_minimal = model, minimal
                logging.info("Javob %.1f soniyada keldi (%s)", time.time() - t0, model)
                break
            except Exception as e:
                err = str(e)
                logging.error("Gemini xatosi (%s): %s", model, err[:300])
                if minimal and ("INVALID_ARGUMENT" in err or "thinking" in err.lower()):
                    working_minimal = False
                    continue
                if "NOT_FOUND" in err or "404" in err:
                    working_model = None
                    break
                fatal = True
                break
        if answer is not None or fatal:
            break

    if answer is None:
        history.pop()
        await update.message.reply_text(
            "Kechirasiz, hozir xatolik bo'ldi. Birozdan keyin qayta urinib ko'ring."
        )
        return

    history.append(types.Content(role="model", parts=[types.Part(text=answer)]))

    # Telegram bitta xabarga 4096 belgigacha ruxsat beradi
    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i : i + 4000])


IMAGE_MODELS = [
    m for m in [
        os.getenv("GEMINI_IMAGE_MODEL"),
        "gemini-3.1-flash-image",
        "gemini-2.5-flash-image",
    ] if m
]

LOGO_STYLE = (
    "professional logo design, flat vector style, clean bold shapes, minimal, "
    "centered composition, high contrast, modern brand identity, "
    "plain solid background, sharp edges, no mockup"
)


async def make_prompt(user_text: str) -> str:
    """Foydalanuvchi so'rovini (o'zbekcha ham bo'ladi) batafsil inglizcha promptga aylantiradi."""
    instruction = (
        "Convert the request below into ONE detailed English prompt for an AI image "
        "generator to make a professional logo. Keep any brand name EXACTLY as written "
        "and say it should appear as clean, correctly spelled text. Mention colors, "
        "symbol idea and style. Output only the prompt, max 60 words.\n\nRequest: "
        + user_text
    )
    try:
        r = await client.aio.models.generate_content(
            model=working_model or MODELS[0],
            contents=instruction,
            config=types.GenerateContentConfig(max_output_tokens=300),
        )
        base = (r.text or "").strip() or user_text
    except Exception as e:
        logging.error("Prompt yaratishda xato: %s", e)
        base = user_text
    return f"{base}. {LOGO_STYLE}"


async def gen_image_gemini(prompt: str):
    for model in IMAGE_MODELS:
        try:
            r = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
            for cand in r.candidates or []:
                for part in cand.content.parts or []:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        logging.info("Rasm Gemini (%s) orqali yaratildi", model)
                        return part.inline_data.data
        except Exception as e:
            logging.error("Gemini rasm xatosi (%s): %s", model, str(e)[:300])
    return None


async def gen_image_pollinations(prompt: str):
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt[:900])
        + "?width=1024&height=1024&nologo=true&model=flux"
    )
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
            r = await c.get(url)
        data = r.content
        if r.status_code == 200 and (data[:3] == b"\xff\xd8\xff" or data[:4] == b"\x89PNG"):
            logging.info("Rasm Pollinations orqali yaratildi")
            return data
        logging.error("Pollinations javobi yaroqsiz: %s", r.status_code)
    except Exception as e:
        logging.error("Pollinations xatosi: %s", e)
    return None


async def logo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    request = " ".join(context.args).strip()
    if not request:
        await update.message.reply_text(
            "Logo uchun tavsif yozing. Masalan:\n"
            "/logo Spark AI, to'q ko'k va sariq, chaqmoq belgisi bilan"
        )
        return

    await update.message.reply_text("Logo tayyorlayapman, 20-40 soniya kuting ⚡")
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.UPLOAD_PHOTO)

    prompt = await make_prompt(request)
    t0 = time.time()
    data = await gen_image_gemini(prompt) or await gen_image_pollinations(prompt)

    if not data:
        await update.message.reply_text(
            "Kechirasiz, hozir logo yarata olmadim. Birozdan keyin qayta urinib ko'ring."
        )
        return

    logging.info("Logo %.1f soniyada tayyor bo'ldi", time.time() - t0)
    await update.message.reply_photo(photo=io.BytesIO(data), caption="Mana logoyingiz ✨")


YT_RE = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/|embed/))([\w-]{11})")
MAX_UPLOAD = 49 * 1024 * 1024  # oddiy Telegram bot uchun ~50 MB chegara
HAS_FFMPEG = shutil.which("ffmpeg") is not None
MAX_AUDIO_SEC = 3600
MAX_VIDEO_SEC = 1800
UPLOAD_KW = dict(read_timeout=600, write_timeout=600, connect_timeout=60, pool_timeout=60)


def fmt_dur(sec):
    if not sec:
        return ""
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _search(query, n=5):
    opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return [
        {"id": e["id"], "title": e.get("title") or "?", "duration": e.get("duration")}
        for e in (info.get("entries") or [])
        if e and e.get("id")
    ]


def _download(video_id, kind, outdir):
    url = f"https://www.youtube.com/watch?v={video_id}"
    limit = MAX_AUDIO_SEC if kind == "a" else MAX_VIDEO_SEC
    opts = {
        "quiet": True,
        "noplaylist": True,
        "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
        "socket_timeout": 30,
        "retries": 3,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration < {limit}"),
    }
    if kind == "a":
        if HAS_FFMPEG:
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ]
        else:
            opts["format"] = "bestaudio[ext=m4a]/bestaudio/best"
    else:
        opts["format"] = (
            "b[ext=mp4][filesize<48M]/b[ext=mp4][filesize_approx<48M]/"
            "b[ext=mp4][height<=360]/b[ext=mp4]/b"
        )
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = [f for f in glob.glob(os.path.join(outdir, f"{video_id}.*")) if not f.endswith(".part")]
    if not files:
        raise RuntimeError("Fayl yuklanmadi (juda uzun yoki mavjud emas)")
    return files[0], info


def format_menu(video_id):
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("🎵 MP3", callback_data=f"a:{video_id}"),
            InlineKeyboardButton("🎬 Video", callback_data=f"v:{video_id}"),
        ]]
    )


async def handle_query(update: Update, query: str, kind: str):
    m = YT_RE.search(query)
    if m:
        await update.message.reply_text("Qaysi formatda yuklab beray?", reply_markup=format_menu(m.group(1)))
        return
    msg = await update.message.reply_text("Qidiryapman... 🔎")
    try:
        results = await asyncio.to_thread(_search, query)
    except Exception as e:
        logging.error("Qidiruv xatosi: %s", e)
        await msg.edit_text("Qidirishda xatolik bo'ldi. Birozdan keyin urinib ko'ring.")
        return
    if not results:
        await msg.edit_text("Hech narsa topilmadi. Boshqacha yozib ko'ring.")
        return
    rows = [
        [InlineKeyboardButton(f"{i}. {r['title'][:48]} ({fmt_dur(r['duration'])})", callback_data=f"{kind}:{r['id']}")]
        for i, r in enumerate(results, 1)
    ]
    await msg.edit_text("Mana topilganlari, birini tanlang:", reply_markup=InlineKeyboardMarkup(rows))


async def musiqa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = " ".join(context.args).strip()
    if not q:
        await update.message.reply_text("Qo'shiq nomini yozing. Masalan:\n/musiqa Shaxzoda Qalbim")
        return
    await handle_query(update, q, "a")


async def video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = " ".join(context.args).strip()
    if not q:
        await update.message.reply_text("Video nomini yoki YouTube havolasini yozing. Masalan:\n/video qiziqarli lavhalar")
        return
    await handle_query(update, q, "v")


async def link_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_query(update, update.message.text, "a")


TRIGGER_RE = re.compile(
    r"(qo['\u2019\u02bb`]?shiq|qoshiq|musiqa|mp3|kuy|klip|video|youtube|yutub|"
    r"\u044e\u0442\u0443\u0431|\u043f\u0435\u0441\u043d|\u043c\u0443\u0437\u044b\u043a|\u0432\u0438\u0434\u0435\u043e|\u049b\u045e\u0448\u0438\u049b)",
    re.IGNORECASE,
)


async def detect_intent(text: str):
    """Oddiy gapdan 'musiqa topish' yoki 'video topish' niyatini ajratadi."""
    prompt = (
        "Foydalanuvchi xabarini tahlil qil. Agar u qo'shiq/musiqa topib berishni so'rasa "
        'intent="music", video topib berishni so\'rasa intent="video", boshqa hamma holatda intent="chat". '
        "query ga faqat qo'shiq nomi va ijrochisini (yoki video nomini) yoz, ortiqcha so'zlarsiz "
        "('youtubedan', 'top', 'yuklab ber' kabilarni olib tashla). "
        'Faqat JSON qaytar: {"intent": "...", "query": "..."}\n\nXabar: ' + text
    )
    try:
        r = await client.aio.models.generate_content(
            model=working_model or MODELS[0],
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=200, response_mime_type="application/json"
            ),
        )
        data = json.loads((r.text or "{}").strip().strip("`").replace("json\n", "", 1))
        return data.get("intent", "chat"), (data.get("query") or "").strip()
    except Exception as e:
        logging.error("Niyatni aniqlashda xato: %s", e)
        return "chat", ""


async def media_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    kind, vid = q.data.split(":", 1)
    chat_id = q.message.chat_id

    status = await q.message.reply_text("Yuklab olinmoqda, kuting... ⏳")
    await context.bot.send_chat_action(
        chat_id, ChatAction.UPLOAD_VOICE if kind == "a" else ChatAction.UPLOAD_VIDEO
    )
    tmp = tempfile.mkdtemp(prefix="spark_")
    try:
        t0 = time.time()
        path, info = await asyncio.to_thread(_download, vid, kind, tmp)
        size = os.path.getsize(path)
        logging.info("Yuklandi: %.1f MB, %.1f soniya", size / 1048576, time.time() - t0)
        if size > MAX_UPLOAD:
            await status.edit_text(
                f"Fayl juda katta ({size / 1048576:.0f} MB). Oddiy Telegram bot 50 MB gacha yubora oladi. "
                "Audio (MP3) formatini tanlab ko'ring."
            )
            return
        title = info.get("title") or ""
        with open(path, "rb") as f:
            if kind == "a":
                await context.bot.send_audio(
                    chat_id, f, title=title, performer=info.get("uploader"),
                    duration=info.get("duration"), caption="@sparkaiuzbot ⚡", **UPLOAD_KW,
                )
            else:
                await context.bot.send_video(
                    chat_id, f, caption=f"{title}\n@sparkaiuzbot ⚡",
                    duration=info.get("duration"), supports_streaming=True, **UPLOAD_KW,
                )
        await status.delete()
    except Exception as e:
        logging.error("Yuklash xatosi: %s", str(e)[:400])
        await status.edit_text(
            "Yuklab bo'lmadi. Video juda uzun, yopiq yoki YouTube bloklagan bo'lishi mumkin. Boshqasini sinab ko'ring."
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    # Python 3.14 da event loop avtomatik yaratilmaydi, shuning uchun o'zimiz yaratamiz
    asyncio.set_event_loop(asyncio.new_event_loop())
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("spark", spark))
    app.add_handler(CommandHandler(["logo", "rasm"], logo))
    app.add_handler(CommandHandler(["musiqa", "mp3"], musiqa))
    app.add_handler(CommandHandler("video", video))
    app.add_handler(CallbackQueryHandler(media_callback, pattern=r"^[av]:"))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(YT_RE) & ~filters.COMMAND, link_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    logging.info("Spark AI ishga tushdi ⚡")
    app.run_polling()


if __name__ == "__main__":
    main()
