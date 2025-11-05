"""
Test the interview pattern analysis function
"""
from server import analyze_interview_pattern

# Test Technical Interview
technical_messages = [
    {"role": "assistant", "content": "Tell me about a recent technical project."},
    {"role": "user", "content": "I built a REST API using Python and FastAPI. I implemented authentication with JWT tokens, designed the database schema with proper indexing for performance, and wrote comprehensive unit tests. The system handles about 1000 requests per second with optimization techniques like caching and database query optimization."},
    {"role": "assistant", "content": "How did you handle scalability?"},
    {"role": "user", "content": "I used Redis for caching frequently accessed data, implemented connection pooling for the database, and designed the API to be stateless so it could scale horizontally. I also added rate limiting to prevent abuse and implemented asynchronous processing for long-running tasks."},
    {"role": "assistant", "content": "What about error handling?"},
    {"role": "user", "content": "I implemented comprehensive error handling with custom exceptions, proper HTTP status codes, and detailed error messages. I also added logging and monitoring to track issues in production."}
]

# Test Behavioral Interview
behavioral_messages = [
    {"role": "assistant", "content": "Tell me about a time you faced a challenge."},
    {"role": "user", "content": "In my previous role, we had a situation where our team missed a critical deadline. I took action by organizing daily standup meetings to track progress, identified the bottlenecks in our workflow, and worked with the team to redistribute tasks based on everyone's strengths. As a result, we delivered the project two weeks ahead of our revised deadline and improved our team collaboration process."},
    {"role": "assistant", "content": "How do you handle conflicts with team members?"},
    {"role": "user", "content": "When I experienced conflict with a colleague over project direction, I scheduled a one-on-one meeting to understand their perspective. I listened actively, acknowledged valid concerns, and we collaborated to find a solution that incorporated both our ideas. This action strengthened our working relationship and the final result exceeded our manager's expectations."}
]

# Test General Interview
general_messages = [
    {"role": "assistant", "content": "Tell me about your background."},
    {"role": "user", "content": "I have five years of experience in software development. My career goal is to become a technical lead. I'm passionate about learning new technologies and I'm motivated by solving complex problems. My strengths include quick learning and attention to detail."},
    {"role": "assistant", "content": "What interests you about this role?"},
    {"role": "user", "content": "I'm interested in the opportunity to grow my skills in cloud technologies and work with a talented team. The company's mission aligns with my values, and I'm excited about the challenges this role presents."}
]

print("=" * 80)
print("TECHNICAL INTERVIEW EVALUATION")
print("=" * 80)
tech_eval = analyze_interview_pattern("technical", technical_messages)
print(f"Overall Score: {tech_eval['overall_score']}%")
print(f"Technical Score: {tech_eval['technical_score']}%")
print(f"Communication Score: {tech_eval['communication_score']}%")
print(f"Problem Solving Score: {tech_eval['problem_solving_score']}%")
print(f"\nStrengths:")
for s in tech_eval['strengths']:
    print(f"  - {s}")
print(f"\nAreas for Improvement:")
for a in tech_eval['areas_for_improvement']:
    print(f"  - {a}")
print(f"\nDetailed Feedback:\n{tech_eval['detailed_feedback']}")

print("\n" + "=" * 80)
print("BEHAVIORAL INTERVIEW EVALUATION")
print("=" * 80)
behavioral_eval = analyze_interview_pattern("behavioral", behavioral_messages)
print(f"Overall Score: {behavioral_eval['overall_score']}%")
print(f"Technical Score: {behavioral_eval['technical_score']}%")
print(f"Communication Score: {behavioral_eval['communication_score']}%")
print(f"Problem Solving Score: {behavioral_eval['problem_solving_score']}%")
print(f"\nStrengths:")
for s in behavioral_eval['strengths']:
    print(f"  - {s}")
print(f"\nAreas for Improvement:")
for a in behavioral_eval['areas_for_improvement']:
    print(f"  - {a}")
print(f"\nDetailed Feedback:\n{behavioral_eval['detailed_feedback']}")

print("\n" + "=" * 80)
print("GENERAL INTERVIEW EVALUATION")
print("=" * 80)
general_eval = analyze_interview_pattern("general", general_messages)
print(f"Overall Score: {general_eval['overall_score']}%")
print(f"Technical Score: {general_eval['technical_score']}%")
print(f"Communication Score: {general_eval['communication_score']}%")
print(f"Problem Solving Score: {general_eval['problem_solving_score']}%")
print(f"\nStrengths:")
for s in general_eval['strengths']:
    print(f"  - {s}")
print(f"\nAreas for Improvement:")
for a in general_eval['areas_for_improvement']:
    print(f"  - {a}")
print(f"\nDetailed Feedback:\n{general_eval['detailed_feedback']}")

print("\n" + "=" * 80)
print("SCORE COMPARISON")
print("=" * 80)
print(f"Technical Interview - Overall: {tech_eval['overall_score']}% (Tech: {tech_eval['technical_score']}%, Comm: {tech_eval['communication_score']}%, PS: {tech_eval['problem_solving_score']}%)")
print(f"Behavioral Interview - Overall: {behavioral_eval['overall_score']}% (Tech: {behavioral_eval['technical_score']}%, Comm: {behavioral_eval['communication_score']}%, PS: {behavioral_eval['problem_solving_score']}%)")
print(f"General Interview - Overall: {general_eval['overall_score']}% (Tech: {general_eval['technical_score']}%, Comm: {general_eval['communication_score']}%, PS: {general_eval['problem_solving_score']}%)")
print("\nNote: Scores should differ based on interview type and content!")
