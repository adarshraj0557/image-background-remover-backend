from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from rembg import remove, new_session
from contextlib import asynccontextmanager
from PIL import Image
from mangum import Mangum
import io
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from datetime import datetime
from collections import defaultdict
import json
from dotenv import load_dotenv
import redis
from redis.exceptions import ConnectionError
import hashlib  # For image hashing
import gc  # For manual cleanup

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

# Global session for shared model (preload on startup)
model_session = None

# Single-threaded executor to prevent concurrent rembg
executor = ThreadPoolExecutor(max_workers=1)




usage_stats = defaultdict(lambda: {"requests": 0, "uploads": 0, "last_request": None})
redis_client = None

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
ALLOWED_TYPES = os.getenv("ALLOWED_FILE_TYPES", "image/jpeg,image/jpg,image/png,image/webp").split(",")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1hr cache

# Startup: Preload session and connect Redis
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_session, redis_client
    # Preload lighter model session (u2net_human_seg for low mem)
    try:
        model_session = new_session("u2net_human_seg")  # ~120MB vs 176MB for u2net
        logger.info("Loaded u2net_human_seg model (~120MB)")
    except Exception as e:
        logger.error(f"Model preload failed: {e}. Falling back to default.")
        model_session = new_session()  # Default u2net if seg fails

    # Redis (from prior setup)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        try:
            redis_client.ping()
            logger.info("Connected to Redis")
        except ConnectionError:
            logger.error("Redis connection failed")
    yield
app = FastAPI(
    title="Background Remover SaaS API",
    description="AI-powered background removal service",
    version="1.0.0",
    lifespan=lifespan
)    
handler = Mangum(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Resize image if too large (reduces mem)
def resize_if_needed(image_bytes, max_size=1024):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        img.save(output, format="PNG", optimize=True)
        return output.getvalue()
    return image_bytes

# Hash image for caching
def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

@app.get("/")
def index():
    return "Server is running"

@app.get("/health")
def health():
    return {"ok": True, "timestamp": datetime.utcnow().isoformat()}

@app.get("/stats")
def get_stats():
    if redis_client:
        total_requests = sum(int(v) for v in redis_client.hvals("usage:requests") or [])
        total_uploads = sum(int(v) for v in redis_client.hvals("usage:uploads") or [])
        return {
            "total_requests": total_requests,
            "total_uploads": total_uploads,
            "unique_users": len(redis_client.hkeys("usage:requests") or [])
        }
    # Fallback
    total_requests = sum(stats["requests"] for stats in usage_stats.values())
    total_uploads = sum(stats["uploads"] for stats in usage_stats.values())
    return {
        "total_requests": total_requests,
        "total_uploads": total_uploads,
        "unique_users": len(usage_stats)
    }

@limiter.limit("50/minute")  # Tighter limit to prevent spikes
@app.post("/remove-bg")
async def remove_bg(request: Request, file: UploadFile = File(...)):
    client_ip = get_remote_address(request)

    # Track usage (Redis or in-mem)
    if redis_client:
        redis_client.hincrby("usage:requests", client_ip, 1)
        redis_client.hset("usage:last", client_ip, datetime.utcnow().isoformat())
    else:
        usage_stats[client_ip]["requests"] += 1
        usage_stats[client_ip]["last_request"] = datetime.utcnow()

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_TYPES)}"
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    try:
        _ = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Cache check (hash after resize to save space)
    resized_raw = resize_if_needed(raw)
    img_hash = get_image_hash(resized_raw)
    if redis_client:
        cached = redis_client.get(f"cache:bg:{img_hash}")
        if cached:
            logger.info(f"Cache hit for {client_ip}")
            return Response(content=cached, media_type="image/png")

    try:
        logger.info(f"Processing image for {client_ip}: {file.filename}")
        loop = asyncio.get_event_loop()

        # Run rembg with optimizations
        def process_image():
            # Use preloaded session, disable matting
            out = remove(
                resized_raw,
                session=model_session,
                only_matting=False,  # Disable to save ~100-200MB
                alpha_matting=False,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10
            )
            return out

        out = await loop.run_in_executor(executor, process_image)

        # Cache result
        if redis_client and len(out) < 5 * 1024 * 1024:  # Cache if <5MB
            redis_client.setex(f"cache:bg:{img_hash}", CACHE_TTL, out)

        # Track success
        if redis_client:
            redis_client.hincrby("usage:uploads", client_ip, 1)
        else:
            usage_stats[client_ip]["uploads"] += 1

        gc.collect()  # Force cleanup
        logger.info(f"Successfully processed image for {client_ip} (used ~{len(out)/1024/1024:.1f}MB output)")
        return Response(content=out, media_type="image/png")

    except Exception as e:
        gc.collect()
        logger.error(f"Processing error for {client_ip}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")