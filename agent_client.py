import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.run.response import RunResponse

# Determine Ollama host (default to localhost if not set)
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
print(f"🔌 Connecting to Ollama at: {ollama_host}")

# Define the Speech Pathology Agent
knowledge_agent_ai = Agent(
    model=Ollama(id="llama3.1:8b", host=ollama_host),
    tools=[], # No tools needed for text rewriting and analysis
    instructions=dedent("""\
        You are an expert Speech Language Pathologist AI assistant specialized in English, Hindi, and Telugu.
        
        Your task is to analyze the provided speech transcription and return a strictly valid JSON object containing:
        1. "text": The corrected, fluent version of the speech. Maintain >90% similarity to the original text. Remove ONLY disfluencies (um, uh, stutters). DO NOT paraphrase or change vocabulary.
        2. "metrics": A dictionary with "words" (count), "disfluencies" (count of um, uh, etc.), and "rate" (fluency score 0-100).
        3. "analysis": Concise HTML-formatted text analyzing speech patterns and grammar.
        4. "suggestions": Concise HTML-formatted therapy tips for improvement.
        5. "soap": A dictionary containing "s" (Subjective), "o" (Objective), "a" (Assessment), and "p" (Plan) for clinical documentation.
        6. "level": The difficulty level assigned based on disfluency ("Beginner", "Intermediate", "Advanced").

        Ensure the output is ONLY the JSON string, no markdown formatting.
        If the input is in Hindi or Telugu, provide the corrected text in that SAME language (do not translate to English), and analysis/suggestions in the requested language or English.
        Keep analysis and suggestions concise and effective.

        ADAPTIVE DIFFICULTY LOGIC:
        - If disfluency is high (>10%): Level = "Beginner". Suggestions should focus on basic breathing, slow pacing, and anxiety reduction.
        - If disfluency is moderate (3-10%): Level = "Intermediate". Suggestions should focus on soft onsets, continuous phonation, and phrasing.
        - If disfluency is low (<3%): Level = "Advanced". Suggestions should focus on intonation, prosody, and public speaking confidence.
        """),
    add_datetime_to_instructions=True,
    show_tool_calls=False,
    markdown=True,
    stream=False,
)



def knowledge_agent_client(prompt: str):
    try:
        response = knowledge_agent_ai.run(message=prompt, stream=False)
        if isinstance(response, RunResponse):
            return response.content  
        else:
            print("Error: Invalid response from knowledge_agent_ai.")
            return None  
    except Exception as e:
        print(f"Error while querying knowledge_agent: {str(e)}")
        return None


if __name__ == "__main__":

    # Example usage with a relevant query
    message = "Um I don't know what to say Actually yeah. Um this weekend I'm going to my cousin's birthday party."
    print(f'{knowledge_agent_client(message)}')
