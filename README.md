# AI Playground Backend

This is the backend for the AI Playground, providing APIs for the frontend to communicate with various AI models and services.

## Features

- Integration with different MCPs (Model Context Protocols)
- Support for Storyscan blockchain analytics
- Support for Story SDK operations with wallet integration
- Streaming responses for real-time interaction

## Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)
- Story Protocol account and API key
- Pinata account and JWT (for IPFS operations)
- RPC Provider URL for blockchain access

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ai-playground-backend
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install .
```

Alternatively, you can install all dependencies including development tools with:

```bash
pip install -e ".[dev]"
```

4. Set up environment variables:

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

Edit the `.env` file to include:

```
OPENAI_API_KEY=your-openai-api-key
RPC_PROVIDER_URL=your-ethereum-rpc-url
WALLET_PRIVATE_KEY=your-wallet-private-key  # Optional: Only needed for server-side signing
PINATA_JWT=your-pinata-jwt  # Optional: Only needed for IPFS operations
```

## Usage

### Running the server

Start the server with:

```bash
python -m api.index
```

This will start the server on port 8000 by default. You can specify a different port by setting the `PORT` environment variable.

### Available MCPs

1. **Storyscan MCP** - For blockchain analytics
2. **SDK MCP** - For Story SDK operations with wallet integration

## SDK MCP Setup

The SDK MCP requires the following dependencies:

```bash
pip install story-protocol-python-sdk web3
```

If you encounter the error `ModuleNotFoundError: No module named 'story_protocol_python_sdk'`, make sure to install the package using:

```bash
pip install story-protocol-python-sdk
```

## Docker Support

You can also run the application using Docker:

```bash
docker-compose up --build
```

This will start both the backend and necessary services.

## Development

### Local Development

For local development, you can use the provided `localdeploy.sh` script:

```bash
./localdeploy.sh
```

This will start the server with hot-reloading enabled for faster development.

### Working with MCPs

To work with both the Storyscan MCP and the SDK MCP, you'll need to:

1. Clone the MCP repositories in the same parent directory as this project
2. Ensure the directory structure is as follows:
   - `/your/workspace/ai-playground-backend`
   - `/your/workspace/story-mcp-hub/storyscan-mcp`
   - `/your/workspace/story-mcp-hub/story-sdk-mcp`

3. Set the environment variables in your `.env` file:
   ```
   MCP_SERVER_PATH=/app/story-mcp-hub/storyscan-mcp/server.py
   SDK_MCP_SERVER_PATH=/app/story-mcp-hub/story-sdk-mcp/server.py
   ```

## Troubleshooting

### Common Issues

1. **Missing SDK dependencies**: If you see errors about missing modules, ensure you have installed all required packages with `pip install story-protocol-python-sdk web3`.

2. **StoryClient initialization errors**: The StoryClient initializer requires `web3`, `account`, and `chain_id` parameters. If you see errors like `StoryClient.__init__() got an unexpected keyword argument`, check that your code is using the correct parameter format.

3. **RPC connection issues**: If you see errors connecting to the blockchain, check your RPC provider URL in the `.env` file.

## License

[MIT License](LICENSE)
