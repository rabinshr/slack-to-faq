# slack_to_faq_with_llm.py
import json
import os
import sys
import requests
from datetime import datetime
from collections import defaultdict
import re

class OllamaClient:
    """Client for interacting with local Ollama LLM"""
    
    def __init__(self, base_url="http://localhost:11434", model="llama3.1"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        
    def test_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                print(f"✅ Ollama connected. Available models: {', '.join(available_models)}")
                
                if self.model not in available_models:
                    print(f"⚠️  Model '{self.model}' not found. Using first available model.")
                    if available_models:
                        self.model = available_models[0]
                        print(f"   Switched to: {self.model}")
                
                return True
            else:
                print(f"❌ Ollama responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {self.base_url}")
            print("   Make sure Ollama is running: 'ollama serve'")
            return False
        except Exception as e:
            print(f"❌ Error testing Ollama connection: {e}")
            return False
    
    def generate_response(self, prompt, max_tokens=1000):
        """Generate response from Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,  # Lower temperature for more consistent technical writing
                    "top_p": 0.9
                }
            }
            
            response = self.session.post(f"{self.base_url}/api/generate", json=payload)
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return None

class SlackConversationProcessor:
    """Processes Slack conversations and prepares them for LLM analysis"""
    
    def __init__(self, export_directory):
        self.export_directory = export_directory
        self.users = {}
        
    def load_and_process_conversations(self):
        """Load daily files and extract meaningful conversations"""
        json_files = []
        
        # Recursively find all JSON files in the directory tree
        for root, dirs, files in os.walk(self.export_directory):
            for file in files:
                if file.endswith('.json'):
                    # Store the full path relative to export_directory for processing
                    relative_path = os.path.relpath(os.path.join(root, file), self.export_directory)
                    json_files.append(relative_path)
        
        if not json_files:
            print(f"❌ No JSON files found in {self.export_directory} (searched recursively)")
            return []
        
        print(f"📂 Processing {len(json_files)} daily export files (found recursively)...")
        
        all_messages = []
        user_profiles = {}
        
        # Load all messages
        for relative_filename in sorted(json_files):
            filepath = os.path.join(self.export_directory, relative_filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                
                for msg in messages:
                    if isinstance(msg, dict):
                        user_id = msg.get('user')
                        user_profile = msg.get('user_profile', {})
                        
                        if user_id and user_profile:
                            user_profiles[user_id] = user_profile
                        
                        msg['_source_file'] = relative_filename
                
                all_messages.extend(messages)
                
            except Exception as e:
                print(f"⚠️  Error processing {relative_filename}: {e}")
                continue
        
        self.users = user_profiles
        print(f"✅ Loaded {len(all_messages)} messages from {len(json_files)} files")
        
        # Extract conversations
        conversations = self.extract_threaded_conversations(all_messages)
        return conversations
    
    def extract_threaded_conversations(self, messages):
        """Extract meaningful threaded conversations"""
        # Filter regular messages
        regular_messages = [msg for msg in messages 
                          if msg.get('type') == 'message' and not msg.get('subtype') and msg.get('text')]
        
        # Group by threads
        threads = defaultdict(list)
        
        for msg in regular_messages:
            thread_ts = msg.get('thread_ts')
            if thread_ts:
                threads[thread_ts].append(msg)
        
        print(f"🧵 Found {len(threads)} threaded conversations")
        
        # Process threads into structured conversations
        conversations = []
        
        for thread_ts, thread_messages in threads.items():
            thread_messages.sort(key=lambda x: float(x.get('ts', 0)))
            
            # Find root message
            root_message = next((msg for msg in thread_messages if msg.get('ts') == thread_ts), thread_messages[0])
            
            root_text = self.clean_text(root_message.get('text', ''))
            if len(root_text) < 10:  # Skip very short messages
                continue
            
            # Get responses
            responses = [msg for msg in thread_messages if msg != root_message]
            substantial_responses = [
                {
                    'user': self.get_user_name(resp.get('user', '')),
                    'text': self.clean_text(resp.get('text', '')),
                    'timestamp': datetime.fromtimestamp(float(resp.get('ts', 0)))
                }
                for resp in responses
                if len(self.clean_text(resp.get('text', ''))) > 15  # Filter short responses
            ]
            
            if substantial_responses:
                conversation = {
                    'question': root_text,
                    'question_user': self.get_user_name(root_message.get('user', '')),
                    'timestamp': datetime.fromtimestamp(float(root_message.get('ts', 0))),
                    'source_file': root_message.get('_source_file', 'unknown'),
                    'responses': substantial_responses,
                    'technical_score': self.calculate_technical_value(root_text, substantial_responses)
                }
                conversations.append(conversation)
        
        # Sort by technical value
        conversations.sort(key=lambda x: x['technical_score'], reverse=True)
        
        print(f"✅ Extracted {len(conversations)} meaningful conversations")
        return conversations
    
    def get_user_name(self, user_id):
        """Get clean user name"""
        if user_id in self.users:
            profile = self.users[user_id]
            return (profile.get('display_name') or 
                   profile.get('real_name') or 
                   profile.get('name') or 
                   user_id)
        return user_id
    
    def clean_text(self, text):
        """Clean message text for LLM processing"""
        if not text:
            return ""
        
        # Replace user mentions
        text = re.sub(r'<@(\w+)>', lambda m: f"@{self.get_user_name(m.group(1))}", text)
        
        # Replace channel mentions
        text = re.sub(r'<#(\w+)\|([^>]+)>', r'#\2', text)
        
        # Clean URLs but preserve them
        text = re.sub(r'<(https?://[^|>]+)(\|([^>]+))?>', 
                     lambda m: f"[{m.group(3)}]({m.group(1)})" if m.group(3) else m.group(1), 
                     text)
        
        # Clean HTML entities
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        
        return text.strip()
    
    def calculate_technical_value(self, question, responses):
        """Calculate the technical documentation value of a conversation"""
        all_text = question + ' ' + ' '.join([r['text'] for r in responses])
        
        score = 0
        
        # Technical indicators
        if re.search(r'```|`[^`]+`', all_text):  # Code blocks
            score += 15
        
        # Technical terms
        technical_terms = len(re.findall(
            r'\b(api|config|auth|deploy|install|error|debug|server|database|endpoint|webhook|integration)\b', 
            all_text, re.IGNORECASE))
        score += technical_terms * 3
        
        # Solution indicators
        if re.search(r'\b(solution|fix|solved|working|works|resolved)\b', all_text, re.IGNORECASE):
            score += 10
        
        # Detailed responses
        score += min(len(all_text) / 50, 20)
        
        # Multiple contributors
        score += len(responses) * 2
        
        return score

class LLMTechnicalWriter:
    """Uses LLM to generate professional technical documentation"""
    
    def __init__(self, ollama_client):
        self.ollama = ollama_client
        
    def create_conversation_prompt(self, conversation):
        """Create a prompt for the LLM to generate technical documentation"""
        
        # Format the conversation for the LLM
        conversation_text = f"ORIGINAL QUESTION:\n{conversation['question']}\n\n"
        conversation_text += "TEAM RESPONSES:\n"
        
        for i, response in enumerate(conversation['responses'], 1):
            conversation_text += f"{i}. {response['user']}: {response['text']}\n"
        
        prompt = f"""You are a professional technical writer creating FAQ documentation from team conversations. 

Your task is to transform the following Slack conversation into a clear, comprehensive FAQ entry that follows technical writing best practices.

CONVERSATION TO ANALYZE:
{conversation_text}

INSTRUCTIONS:
1. Create a clear, specific question that captures the core technical problem
2. Write a comprehensive answer that synthesizes all the information from the responses
3. Structure the answer with proper headings, steps, code blocks, and formatting
4. Use professional technical writing tone (clear, concise, authoritative)
5. Group relevant topics together into the same section
6. Include prerequisites, step-by-step instructions, code examples, and important notes as applicable
7. Remove conversational elements (thanks, btw, etc.) but preserve all technical content
8. Format code properly with markdown code blocks
9. Add warnings or important notes where relevant
10. Generalise the answer to be applicable to a wider audience, if specific examples are available use them as examples

OUTPUT FORMAT:
Return ONLY the FAQ entry in this exact format:

QUESTION: [Clear, specific technical question]

ANSWER: [Comprehensive, well-structured technical answer using markdown formatting]

Generate the FAQ entry now:"""

        return prompt
    
    def generate_faq_entry(self, conversation):
        """Generate a single FAQ entry using LLM"""
        prompt = self.create_conversation_prompt(conversation)
        
        print(f"🤖 Generating FAQ for: {conversation['question'][:60]}...")
        
        response = self.ollama.generate_response(prompt, max_tokens=1500)
        
        if not response:
            # Fallback if LLM fails
            return {
                'question': conversation['question'],
                'answer': self.create_fallback_answer(conversation),
                'metadata': self.extract_metadata(conversation)
            }
        
        # Parse LLM response
        try:
            # Split into question and answer
            if "QUESTION:" in response and "ANSWER:" in response:
                parts = response.split("ANSWER:", 1)
                question_part = parts[0].replace("QUESTION:", "").strip()
                answer_part = parts[1].strip()
            else:
                # Fallback parsing
                lines = response.split('\n')
                question_part = conversation['question']  # Use original
                answer_part = response
            
            return {
                'question': question_part or conversation['question'],
                'answer': answer_part,
                'metadata': self.extract_metadata(conversation)
            }
            
        except Exception as e:
            print(f"⚠️  Error parsing LLM response: {e}")
            return {
                'question': conversation['question'],
                'answer': self.create_fallback_answer(conversation),
                'metadata': self.extract_metadata(conversation)
            }
    
    def create_fallback_answer(self, conversation):
        """Create fallback answer if LLM processing fails"""
        responses = conversation['responses']
        
        # Find the most comprehensive response
        best_response = max(responses, key=lambda x: len(x['text']))
        
        answer = f"**Solution:**\n\n{best_response['text']}"
        
        # Add additional context if available
        other_responses = [r for r in responses if r != best_response and len(r['text']) > 30]
        if other_responses:
            answer += f"\n\n**Additional Information:**\n\n"
            for resp in other_responses[:2]:  # Limit to 2 additional responses
                answer += f"- {resp['text']}\n"
        
        return answer
    
    def extract_metadata(self, conversation):
        """Extract metadata for documentation purposes"""
        return {
            'contributors': list(set([r['user'] for r in conversation['responses']] + [conversation['question_user']])),
            'date': conversation['timestamp'].strftime('%Y-%m-%d'),
            'source': conversation['source_file'],
            'technical_score': conversation.get('technical_score', 0)
        }
    
    def categorize_conversation(self, conversation):
        """Use LLM to categorize the conversation"""
        prompt = f"""You are a technical documentation specialist. Analyze this conversation and determine the most appropriate category for a technical FAQ.

CONVERSATION:
Question: {conversation['question']}
Responses: {' '.join([r['text'][:200] for r in conversation['responses']])}
"""

        response = self.ollama.generate_response(prompt, max_tokens=50)
        
        # Validate response
        valid_categories = [
            # 'Authentication & Security', 'API Integration & Webhooks', 'Payment Processing',
            # 'Database & Data Management', 'Configuration & Setup', 'Deployment & Infrastructure',
            # 'Development Workflow', 'Troubleshooting & Debugging', 'Performance & Monitoring',
            'General Technical'
        ]
        
        if response and response.strip() in valid_categories:
            return response.strip()
        
        # Fallback to keyword-based categorization
        return self.fallback_categorize(conversation)
    
    def fallback_categorize(self, conversation):
        """Fallback categorization using keywords"""
        text = conversation['question'] + ' ' + ' '.join([r['text'] for r in conversation['responses']])
        text = text.lower()
        
        # category_keywords = {
        #     # 'Authentication & Security': ['auth', 'login', 'password', 'token', 'security', 'permission'],
        #     # 'API Integration & Webhooks': ['api', 'webhook', 'endpoint', 'integration', 'rest', 'json'],
        #     # 'Payment Processing': ['payment', 'billing', 'cybersource', 'stripe', 'transaction'],
        #     # 'Database & Data Management': ['database', 'sql', 'query', 'data', 'table', 'migration'],
        #     # 'Configuration & Setup': ['config', 'setup', 'install', 'configure', 'environment'],
        #     # 'Troubleshooting & Debugging': ['error', 'bug', 'issue', 'problem', 'debug', 'fix'],
        #     # 'Deployment & Infrastructure': ['deploy', 'server', 'hosting', 'production', 'infrastructure'],
        #     # 'Development Workflow': ['code', 'git', 'repository', 'development', 'workflow'],
        #     # 'Performance & Monitoring': ['performance', 'slow', 'monitor', 'optimize', 'memory']
        # }
        
        # scores = {}
        # for category, keywords in category_keywords.items():
        #     score = sum(1 for keyword in keywords if keyword in text)
        #     scores[category] = score
        
        return 'General Technical'

class TechnicalFAQGenerator:
    """Generates comprehensive technical FAQ using LLM"""
    
    def __init__(self, ollama_client):
        self.ollama = ollama_client
        self.writer = LLMTechnicalWriter(ollama_client)
        
    def process_conversations(self, conversations, batch_size=5):
        """Process conversations in batches to avoid overwhelming the LLM"""
        
        print(f"🤖 Processing {len(conversations)} conversations with LLM...")
        
        categorized_faqs = defaultdict(list)
        
        for i, conversation in enumerate(conversations):
            print(f"   Progress: {i+1}/{len(conversations)}")
            
            # Generate FAQ entry
            faq_entry = self.writer.generate_faq_entry(conversation)
            
            # Categorize
            category = self.writer.categorize_conversation(conversation)
            faq_entry['category'] = category
            
            categorized_faqs[category].append(faq_entry)
            
            # Small delay to avoid overwhelming Ollama
            if i % batch_size == 0 and i > 0:
                print(f"   Processed {i} conversations...")
        
        return dict(categorized_faqs)
    
    def generate_category_introduction(self, category, faq_items):
        """Generate an introduction for each category using LLM"""
        
        # Sample some questions from the category
        sample_questions = [item['question'] for item in faq_items[:5]]
        
        prompt = f"""You are a technical writer creating documentation. Write a brief, professional introduction for the "{category}" section of a technical FAQ.

This section contains {len(faq_items)} questions covering topics like:
{chr(10).join(['- ' + q[:100] + '...' for q in sample_questions])}

Write 2-3 sentences that:
1. Keep it professional and concise
2. Be sure to group relevant topic together such as cross-posting, payment methods, customer groups, etc.

Write only the introduction text, no formatting or headers:"""

        response = self.ollama.generate_response(prompt, max_tokens=200)
        
        return response or f"This section covers {category.lower()} related questions and solutions."
    
    def generate_comprehensive_faq(self, categorized_faqs):
        """Generate the complete FAQ document"""
        
        content = []
        
        # Calculate statistics
        total_faqs = sum(len(items) for items in categorized_faqs.values())
        all_contributors = set()
        for items in categorized_faqs.values():
            for item in items:
                all_contributors.update(item['metadata']['contributors'])
        
        # Professional header
        content.append(f"""# Technical FAQ & Solutions Guide

*Professional technical documentation compiled from team knowledge*

## Document Overview

This comprehensive FAQ provides solutions to common technical questions and challenges encountered by our team. Each entry has been professionally rewritten and structured for clarity and actionability.

### Quick Stats
- **Total Solutions:** {total_faqs}
- **Categories:** {len(categorized_faqs)}
- **Contributors:** {len(all_contributors)} team members
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Table of Contents

""")
        
        # Generate TOC
        for category in categorized_faqs.keys():
            if categorized_faqs[category]:
                anchor = category.lower().replace(' ', '-').replace('&', 'and')
                count = len(categorized_faqs[category])
                content.append(f"- [{category}](#{anchor}) ({count} solutions)")
        
        content.append("\n---\n")
        
        # Generate category sections
        category_order = [
            # 'Configuration & Setup',
            # 'Authentication & Security',
            # 'API Integration & Webhooks',
            # 'Payment Processing',
            # 'Database & Data Management',
            # 'Deployment & Infrastructure',
            # 'Development Workflow',
            # 'Troubleshooting & Debugging',
            # 'Performance & Monitoring',
            'General Technical'
        ]
        
        processed = set()
        
        for category in category_order:
            if category in categorized_faqs and categorized_faqs[category]:
                content.append(self.generate_category_section(category, categorized_faqs[category]))
                processed.add(category)
        
        # Add remaining categories
        for category, items in categorized_faqs.items():
            if category not in processed and items:
                content.append(self.generate_category_section(category, items))
        
        # Professional footer
        content.append(f"""
---

## About This Documentation

This FAQ is automatically generated from real team conversations using advanced language processing to ensure:

- **Accuracy:** Solutions are tested and verified by team members
- **Clarity:** Content is rewritten in professional technical writing style  
- **Completeness:** Comprehensive coverage of common scenarios and edge cases
- **Currency:** Regular updates from ongoing team discussions

## Contributing

To improve this documentation:

1. **Ask detailed questions** in team channels with specific context
2. **Provide thorough solutions** with step-by-step instructions
3. **Include code examples** and configuration details
4. **Use threading** to keep conversations organized

## Support

For questions not covered here:
- Search this document using Ctrl+F (Cmd+F)
- Check recent team channels for updated information
- Ask in the appropriate technical channel with full context

---

*Generated by automated technical writing pipeline*  
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
""")
        
        return '\n'.join(content)
    
    def generate_category_section(self, category, faq_items):
        """Generate a complete category section"""
        
        # # Generate introduction
        # intro = self.generate_category_introduction(category, faq_items)
        
        # section = [
        #     f"## {category}\n",
        #     f"{intro}\n",
        #     "---\n"
        # ]
        section = []
        
        # Add FAQ items
        for i, item in enumerate(faq_items, 1):
            contributors_text = ", ".join(item['metadata']['contributors'])
            
            section.extend([
                f"### {item['question']}\n",
                # f"*Solution verified by: {contributors_text} | Date: {item['metadata']['date']}*\n",
                f"{item['answer']}\n",
                "---\n"
            ])
        
        return '\n'.join(section)

def main():
    if len(sys.argv) < 2:
        print("""
🤖 Slack to Technical FAQ with LLM

Usage: python slack_to_faq_with_llm.py <export_directory> [output_file] [model_name]

Examples:
  python slack_to_faq_with_llm.py ./data/
  python slack_to_faq_with_llm.py ./data/ Technical_FAQ.md
  python slack_to_faq_with_llm.py ./data/ FAQ.md llama3.1

Prerequisites:
  1. Install Ollama: https://ollama.ai
  2. Start Ollama: ollama serve
  3. Pull a model: ollama pull llama3.1
        """)
        return
    
    export_directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'Technical_FAQ.md'
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'llama3.1'
    
    if not os.path.exists(export_directory):
        print(f"❌ Directory not found: {export_directory}")
        return
    
    print("🚀 Technical FAQ Generation with LLM")
    print("="*60)
    
    # Step 1: Initialize Ollama
    print("🤖 Step 1: Connecting to Ollama...")
    ollama = OllamaClient(model=model_name)
    
    if not ollama.test_connection():
        print("\n💡 To fix this:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull llama3.1")
        return
    
    # Step 2: Process Slack conversations
    print("\n📁 Step 2: Processing Slack conversations...")
    processor = SlackConversationProcessor(export_directory)
    conversations = processor.load_and_process_conversations()
    
    if not conversations:
        print("❌ No suitable conversations found")
        return
    
    # Filter for high-value conversations
    high_value_conversations = [c for c in conversations if c['technical_score'] > 10]
    print(f"📊 Selected {len(high_value_conversations)} high-value conversations for documentation")
    
    # Step 3: Generate FAQ with LLM
    print(f"\n🤖 Step 3: Generating professional FAQ entries...")
    generator = TechnicalFAQGenerator(ollama)
    categorized_faqs = generator.process_conversations(high_value_conversations)
    
    print(f"✅ Generated FAQ entries:")
    for category, items in categorized_faqs.items():
        print(f"   📂 {category}: {len(items)} entries")
    
    # Step 4: Generate final document
    print(f"\n📝 Step 4: Compiling final documentation...")
    final_content = generator.generate_comprehensive_faq(categorized_faqs)
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    # Summary
    total_entries = sum(len(items) for items in categorized_faqs.values())
    file_size = os.path.getsize(output_file) / 1024
    
    print("="*60)
    print("🎉 Professional Technical FAQ Complete!")
    print("="*60)
    print(f"📄 Output: {output_file}")
    print(f"📚 Total entries: {total_entries}")
    print(f"📂 Categories: {len(categorized_faqs)}")
    print(f"💾 Size: {file_size:.1f} KB")
    print(f"🤖 Model used: {ollama.model}")
    print(f"\n✨ Your professional technical FAQ is ready!")
    print(f"💡 Each entry has been rewritten by AI for clarity and completeness.")

if __name__ == "__main__":
    main()
