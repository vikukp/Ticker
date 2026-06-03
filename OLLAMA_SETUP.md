# Using Local Ollama LLM

This guide explains how to set up and use a local Ollama LLM with the Ticker Research project.

## Prerequisites

1. **Install Ollama**
   - Download from: https://ollama.ai
   - Install for macOS, Linux, or Windows

2. **Start Ollama**
   ```bash
   # On macOS
   /Applications/Ollama.app/Contents/MacOS/Ollama serve
   
   # Or simply run: ollama serve
   # This will start the Ollama server on http://localhost:11434
   ```

3. **Pull a Model**
   ```bash
   # Pull a model (examples: mistral, llama2, neural-chat)
   ollama pull mistral
   
   # Or try other models:
   ollama pull llama2
   ollama pull neural-chat
   ollama pull openchat
   ```

## Configuration

1. **Copy the example env file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file:**
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=mistral  # Change to your chosen model
   ```

3. **Verify Ollama is running:**
   ```bash
   # Test the connection
   curl http://localhost:11434/api/tags
   ```

## Running the App

1. **Start the Streamlit app:**
   ```bash
   python3 run.py
   ```

2. **Access the app:**
   - Open http://localhost:8501 in your browser

## Supported Ollama Models

- **Mistral** (7B) - Good balance of speed and quality
  ```bash
  ollama pull mistral
  ```

- **Llama 2** (7B/13B) - Strong reasoning capabilities
  ```bash
  ollama pull llama2
  ```

- **Neural Chat** (7B) - Optimized for conversation
  ```bash
  ollama pull neural-chat
  ```

- **Openchat** (3.5) - Fast and lightweight
  ```bash
  ollama pull openchat
  ```

- **Dolphin Mixtral** (8x7B) - Powerful but slower
  ```bash
  ollama pull dolphin-mixtral
  ```

## Performance Tips

- **For faster responses:** Use smaller models like `openchat` or `neural-chat`
- **For better quality:** Use larger models like `llama2` or `mistral`
- **Memory requirements:** Ensure your machine has 8GB+ RAM for smooth operation
- **GPU acceleration:** If you have NVIDIA/AMD GPU, Ollama will use it automatically

## Troubleshooting

### Connection Error
If you get "Connection refused" error:
1. Ensure Ollama is running: `ollama serve`
2. Check if the base URL is correct in `.env`
3. Test with: `curl http://localhost:11434/api/tags`

### Model Not Found
If Ollama can't find the model:
1. Pull the model: `ollama pull mistral`
2. List available models: `ollama list`

### Slow Response Times
- Use a smaller model (`openchat`, `neural-chat`)
- Free up system memory
- Close other applications
