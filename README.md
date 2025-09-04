# Setup Instructions
## Install and start Ollama:
Install Ollama (macOS)
```
brew install ollama
```

Or download from https://ollama.ai

## Start Ollama server
```
ollama serve
```

Pull a good model for technical writing (in another terminal)
```
ollama pull llama3.1
# or
ollama pull mistral
# or  
ollama pull codellama
```
## Setup requirements and depedencies
```
chmod +x setup.sh

./setup.sh

source slack_faq_env/bin/activate
```

## Run the FAQ generator:
```
python slack_to_faq_with_llm.py data
```

## Advanced Usage
```
# Use specific model
python slack_to_faq_with_llm.py data FAQ.md mistral:latest

# Generate with different output file
python slack_to_faq_with_llm.py data Team_Solutions.md

# If Ollama is running on different port
# Edit the base_url in OllamaClient initialization
```
