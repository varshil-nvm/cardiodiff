import re

def apply_prompt_template(style, user_query):
    """Wraps user queries in structural prompt templates for instructional clarity."""
    if style == 'direct':
        return f"User Query: {user_query}"
    elif style == 'few_shot':
        return f"Task: Categorize incoming text into Tech, Science, or Art.\n\nExample 1: 'Quantum computing advances computing power.' -> Category: Science\nExample 2: 'The oil painting depicts a beautiful sunset.' -> Category: Art\nExample 3: 'This smartphone features a dynamic display refresh rate.' -> Category: Tech\n\nInput: '{user_query}' -> Category:"
    elif style == 'roleplay':
        return f"System Prompt: You are a creative Shakespearean poet who explains modern concepts.\nUser: Explain this topic to me: {user_query}"
    else:
        return user_query

def mock_llm_inference(full_prompt, temperature=0.7):
    """Deterministic rules-based response mapping to simulate LLM execution styles."""
    prompt_lower = full_prompt.lower()
    
    # Check for Few-Shot classification patterns first
    if 'task: categorize incoming text' in prompt_lower:
        # Extract the input query segment to prevent matching rules on few-shot examples
        query_part = prompt_lower.split("input:")[-1] if "input:" in prompt_lower else prompt_lower
        if any(w in query_part for w in ['ai', 'neural', 'compute', 'gpu', 'smartphone', 'tech']):
            return "Tech (Inferred via pattern matching)"
        elif any(w in query_part for w in ['galaxy', 'star', 'physics', 'nebula', 'stellar', 'science']):
            return "Science (Inferred via pattern matching)"
        else:
            return "Art (Inferred via pattern matching)"
            
    # Check for Roleplay patterns
    if 'shakespearean' in prompt_lower or 'system prompt:' in prompt_lower:
        topic = re.sub(r'.*explain this topic to me:\s*', '', prompt_lower, flags=re.IGNORECASE).strip()
        if not topic or topic == prompt_lower:
            topic = "the cosmos"
        
        # High temperature increases variation/flamboyance
        if temperature > 0.5:
            return f"Hark! Thy mind seeketh knowledge of '{topic}'. Verily, 'tis a wondrous marvel of our modern epoch, spinning threads of golden light across the endless tapestry of time!"
        else:
            return f"Alas, the concept of '{topic}' is but a fleeting shadow, yet my pen shall write of its virtues with a solemn heart.\n"
            
    # Default Direct response fallback
    if 'ai' in prompt_lower or 'machine learning' in prompt_lower:
        return "Generative AI models are statistical frameworks trained to approximate target distributions and generate new synthetic configurations."
    else:
        return "Thy input has been processed by the GenLab playhouse! Try typing about 'AI' or configuring 'Few-Shot' templates for unique mock dynamics."
