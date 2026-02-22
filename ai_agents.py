"""
AI agents integrated in the project.
To use it digit in the terminal: python ai_agents.py reviewer app/logic.py
"""
import os
import sys
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY not found.")
    sys.exit(1)

client = genai.Client(api_key=api_key)


def run_agent(role, content):
    prompts = {
        "reviewer": """
            Act as a Senior Software Engineer.
            Review the code below. Look for logic errors, security risks, and style issues.
            Return a Markdown bulleted list. If OK, say "OK".
        """,
        "documenter": """
            Act as a Technical Writer.
            Generate Google Style Docstrings and a README summary for this code.
        """,
        "tester": """
            Act as a QA Engineer.
            Write pytest unit tests for this code. Return ONLY code.
        """,
    }

    if role not in prompts:
        return f"❌ Error: Role '{role}' not recognized."

    full_prompt = f"{prompts[role]}\n\n--- CODE ---\n{content}"

    try:
        response = client.models.generate_content(
            # model='gemini-1.5-flash',
            model="gemini-3-flash-preview",
            contents=full_prompt,
        )
        return response.text
    except Exception as e:
        return f"❌ Agent Error: {str(e)}"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ai_agents.py [reviewer|documenter|tester] [file_path]")
        sys.exit(1)

    role = sys.argv[1]
    file_path = sys.argv[2]

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r") as f:
        code_content = f.read()

    print(f"--- 🤖 Agent {role.upper()} working on {file_path}... ---")
    print(run_agent(role, code_content))
