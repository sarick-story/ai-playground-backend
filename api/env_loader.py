"""
Environment variable loader for the AI Playground Backend.

This module ensures that environment variables are properly loaded
from the .env file, regardless of where the application is started from.
"""

import os
from pathlib import Path


def load_environment_variables():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        
        # Find the .env file - it should be in the ai-playground-backend directory
        current_file = Path(__file__)
        backend_dir = current_file.parent.parent  # Go up two levels from api/env_loader.py
        env_file = backend_dir / ".env"
        
        if env_file.exists():
            load_dotenv(env_file)
            print(f"✅ Loaded environment variables from {env_file}")
            return True
        else:
            # Try other common locations
            alt_locations = [
                Path.cwd() / ".env",  # Current working directory
                Path.cwd().parent / ".env",  # Parent directory
            ]
            
            for alt_env in alt_locations:
                if alt_env.exists():
                    load_dotenv(alt_env)
                    print(f"✅ Loaded environment variables from {alt_env}")
                    return True
            
            print("⚠️  No .env file found in expected locations")
            return False
            
    except ImportError:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"⚠️  Error loading environment variables: {e}")
        return False


def ensure_openai_api_key():
    """Ensure OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Make sure your .env file contains: OPENAI_API_KEY=your_key_here")
        return False
    
    # Don't print the full key for security
    masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
    print(f"✅ OpenAI API key loaded: {masked_key}")
    return True


# Load environment variables when this module is imported
if __name__ != "__main__":
    load_environment_variables()