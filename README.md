# AI Playground Backend

This is the backend server for the MCP Playground application. It provides the API endpoints needed by the frontend to interact with OpenAI and the Story MCP Hub.

## Prerequisites

1. Install UV (Python Package Manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Python 3.12 or higher is required. You can check your Python version with:
```bash
python --version
```

3. Clone the Story MCP Hub repository (if not already done):
```bash
# Navigate to the parent directory of ai-playground-backend
cd ..
git clone https://github.com/piplabs/story-mcp-hub.git
```

## Environment Setup

1. Create a `.env` file in the root directory with the following variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

2. Install dependencies:
```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

## Running the Application

Start the backend server:
```bash
python -m api.index
```

The backend will run on http://localhost:8000. You can verify it's running by visiting http://localhost:8000/health in your browser, which should return `{"status": "ok"}`.

## API Endpoints

The backend provides the following main endpoints:

- `/api/chat`: Handles chat interactions with OpenAI models
- `/health`: Health check endpoint

## Project Structure
```
ai-playground-backend/
├── api/              # Backend API endpoints
│   ├── __init__.py   # Package initialization
│   ├── index.py      # Main FastAPI application
│   ├── chat.py       # Chat API endpoints
│   ├── mcp_agent.py  # MCP Agent implementation
│   └── utils/        # Utility functions
├── pyproject.toml    # Project dependencies and configuration
└── .env              # Environment variables (not in version control)
```

## Development

The backend is built with:
- FastAPI for the API framework
- Uvicorn as the ASGI server
- OpenAI for AI capabilities
- MCP for blockchain integration

## Deployment

### Google Cloud Run Deployment

Export your OpenAI API key and run the deployment script:

```bash
# Export your OpenAI API key
export OPENAI_API_KEY=your_openai_api_key_here

# Run the deployment script
./deploy.sh
```

The script will:
1. Build and push the Docker image to Google Container Registry
2. Deploy the application to Cloud Run
3. Configure the necessary environment variables
4. Display the service URL upon completion

## Troubleshooting

- If you encounter any issues with UV, make sure it's properly installed and in your PATH
- Ensure the Story MCP Hub repository is in the same parent directory
- Verify that all environment variables are properly set
- If you get dependency errors, try updating your dependencies with `uv pip install -e .`
