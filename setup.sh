# setup.sh
#!/bin/bash

echo "Setting up Slack to FAQ converter..."

# Create virtual environment
python3 -m venv slack_faq_env
source slack_faq_env/bin/activate

# Install requirements
pip install -r requirements.txt

echo "Setup complete! 🎉"
echo ""
echo "Usage:"
echo "1. Extract your Slack export ZIP file"
echo "2. Run: python slack_to_faq_with_llm.py /path/to/slack/export"
echo ""
echo "Example:"
echo "  python slack_to_faq_with_llm.py data"
