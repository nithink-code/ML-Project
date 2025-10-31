"""
Test script to demonstrate the intelligent response generation
This shows how the system analyzes user input and generates context-aware responses
"""

def analyze_message(content: str, interview_type: str = "technical"):
    """Simulate the answer characteristics analysis"""
    user_content = content.lower()
    message_length = len(content.split())
    
    characteristics = {
        "is_detailed": message_length > 30,
        "is_brief": message_length < 10,
        "mentions_example": any(keyword in user_content for keyword in ["example", "instance", "time when", "once", "project"]),
        "mentions_technology": any(tech in user_content for tech in ["python", "javascript", "java", "react", "node", "sql", "api", "database", "algorithm", "framework", "library"]),
        "mentions_team": any(keyword in user_content for keyword in ["team", "colleague", "coworker", "manager", "collaborate"]),
        "mentions_challenge": any(keyword in user_content for keyword in ["challenge", "difficult", "problem", "issue", "struggle", "obstacle"]),
        "mentions_success": any(keyword in user_content for keyword in ["success", "achieved", "accomplished", "improved", "solved"]),
        "shows_uncertainty": any(phrase in user_content for phrase in ["i think", "maybe", "not sure", "probably", "i guess"]),
        "asks_clarification": "?" in content
    }
    
    return characteristics


def generate_intelligent_response(interview_type, characteristics, user_msg):
    """Generate responses that actually react to user's message content"""
    
    if interview_type == "technical":
        if characteristics["is_brief"]:
            return "Could you elaborate on that? I'd like to understand your approach in more detail."
        elif characteristics["shows_uncertainty"]:
            return "I see you're thinking through this. What would be your first step in tackling this problem?"
        elif characteristics["mentions_technology"]:
            # Extract technology mentions for context-aware follow-up
            techs = [t for t in ["Python", "JavaScript", "Java", "React", "Node", "SQL", "API", "database"] 
                    if t.lower() in user_msg.lower()]
            tech_mention = techs[0] if techs else "that technology"
            return f"Interesting choice with {tech_mention}. What made you decide to use it for this particular use case?"
        elif characteristics["mentions_example"]:
            return "Thanks for that concrete example. Can you walk me through the technical decisions you made and why?"
        elif characteristics["is_detailed"]:
            return "That's a thorough answer. How would you optimize this solution for better performance or scalability?"
        else:
            return "Can you describe your thought process when approaching problems like this?"
    
    elif interview_type == "behavioral":
        if characteristics["is_brief"]:
            return "That's a good start. Can you tell me more about the specific situation and what actions you took?"
        elif characteristics["mentions_team"]:
            return "Team dynamics are important. How did you handle any disagreements or different viewpoints within the team?"
        elif characteristics["mentions_challenge"]:
            return "I appreciate you sharing that challenge. What did you learn from that experience, and how has it shaped your approach since?"
        elif characteristics["mentions_success"]:
            return "That's a great outcome! How did you measure that success, and what was the impact on the organization?"
        elif characteristics["mentions_example"]:
            return "Thank you for that example. What would you do differently if you faced a similar situation today?"
        elif characteristics["is_detailed"]:
            return "That's very insightful. How do you apply those learnings in your current work?"
        else:
            return "Can you provide a specific example that illustrates your approach?"
    
    else:  # general
        if characteristics["is_brief"]:
            return "I'd love to hear more about that. What aspects of it are most important to you?"
        elif characteristics["asks_clarification"]:
            return "Great question! Let me rephrase: I'm interested in understanding your background and what drives your career decisions."
        elif characteristics["mentions_technology"] or characteristics["mentions_example"]:
            return "That experience sounds valuable. How has it prepared you for the kind of challenges you want to tackle next?"
        elif characteristics["is_detailed"]:
            return "Thank you for that comprehensive answer. What are you most passionate about in your field?"
        else:
            return "Interesting. How does that align with where you see your career heading in the next few years?"


# Test cases demonstrating different response types
test_cases = [
    {
        "type": "technical",
        "user_input": "I used Python and FastAPI to build a REST API for our microservices architecture. We had to handle high concurrency and implemented caching with Redis.",
        "description": "Detailed technical answer with multiple technologies"
    },
    {
        "type": "technical",
        "user_input": "Yes, I know Python.",
        "description": "Very brief answer"
    },
    {
        "type": "technical",
        "user_input": "I think maybe using a hash map could work, but I'm not sure.",
        "description": "Uncertain answer"
    },
    {
        "type": "behavioral",
        "user_input": "Once I had to work with a difficult team member who didn't want to collaborate. It was a challenge but we eventually solved the problem.",
        "description": "Mentions team, challenge, and success"
    },
    {
        "type": "behavioral",
        "user_input": "I worked on it.",
        "description": "Very brief behavioral answer"
    },
    {
        "type": "general",
        "user_input": "I'm passionate about machine learning and have worked on several projects using TensorFlow and PyTorch. For example, I built a recommendation system that improved user engagement by 30%.",
        "description": "Detailed answer with example and technology"
    },
    {
        "type": "general",
        "user_input": "What do you mean by that?",
        "description": "Asks for clarification"
    }
]


if __name__ == "__main__":
    print("=" * 80)
    print("INTELLIGENT RESPONSE GENERATION TEST")
    print("=" * 80)
    print("\nThis demonstrates how the system analyzes user input and generates")
    print("contextually appropriate responses based on the content.\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test['description']}")
        print(f"Interview Type: {test['type'].upper()}")
        print(f"{'=' * 80}")
        
        print(f"\n👤 USER: {test['user_input']}")
        
        # Analyze the message
        characteristics = analyze_message(test['user_input'], test['type'])
        
        # Show detected characteristics
        active_chars = [k for k, v in characteristics.items() if v]
        if active_chars:
            print(f"\n🔍 DETECTED CHARACTERISTICS:")
            for char in active_chars:
                print(f"   ✓ {char.replace('_', ' ').title()}")
        
        # Generate response
        response = generate_intelligent_response(test['type'], characteristics, test['user_input'])
        
        print(f"\n🤖 AI INTERVIEWER: {response}")
        print()
    
    print("=" * 80)
    print("\nKEY FEATURES:")
    print("✓ Analyzes message length (brief vs detailed)")
    print("✓ Detects technology mentions and references them")
    print("✓ Identifies examples, challenges, successes")
    print("✓ Recognizes team collaboration discussion")
    print("✓ Spots uncertainty and adjusts tone")
    print("✓ Responds to clarification questions")
    print("✓ Generates context-specific follow-ups")
    print("=" * 80)
