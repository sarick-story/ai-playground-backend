import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the chat module
from .chat import app as chat_app

# Create the main FastAPI app
app = FastAPI()

# Determine allowed origins based on environment
# For development, we allow localhost origins
# For production, we would specify the actual domain
allowed_origins = [
    "http://localhost:3000",  # Next.js default port
    "http://localhost:8000",  # Backend port (for testing)
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Add production domain if available
if os.environ.get("FRONTEND_URL"):
    allowed_origins.append(os.environ.get("FRONTEND_URL"))

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # More restrictive than "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Only methods we need
    allow_headers=["Content-Type", "Authorization", "x-vercel-ai-data-stream"],  # Required headers
    expose_headers=["x-vercel-ai-data-stream"],  # Expose this header to the client
)

# Mount the chat app at /api to ensure paths match
app.mount("/api", chat_app)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.index:app", host="0.0.0.0", port=port, reload=True)