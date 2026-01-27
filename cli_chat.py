import sys
import os
from pathlib import Path

# Add 'app' directory to sys.path so that modules in app/ can import each other
# e.g. 'from agent.agent import Agent' will work because 'agent' is found inside 'app'
current_dir = Path(__file__).parent.absolute()
app_dir = current_dir / "app"
sys.path.insert(0, str(app_dir))

# Try to import the agent
try:
    from agent.smart_menu_agent import SmartMenuAgent
    from llm.select_llm import LLM
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

from dotenv import load_dotenv

def main():
    print("Initializing FoodPlanGPT CLI...")
    
    # Attempt to load environment variables
    # 1. Check root .env
    root_env = current_dir / ".env"
    # 2. Check app/agent/.env (legacy location observed in project)
    agent_env = app_dir / "agent" / ".env"
    
    env_loaded = False
    if root_env.exists():
        load_dotenv(root_env)
        print(f"Loaded environment from {root_env}")
        env_loaded = True
    
    if agent_env.exists():
        # Load agent env as well (override or complement)
        load_dotenv(agent_env, override=False)
        print(f"Loaded environment from {agent_env}")
        env_loaded = True
        
    if not env_loaded:
        print("WARNING: No .env file found. API calls may fail.")

    try:
        # Initialize LLM
        # Using configuration from smart_menu_agent.py's __main__ block
        # You can adjust model/provider here or use environment variables in LLM class
        
        # Check if CHATANYWHERE_API is set if using chatanywhere
        if not os.getenv("CHATANYWHERE_API") and not os.getenv("GEMINI_API_KEY"):
             print("Note: Ensure your API keys (CHATANYWHERE_API or others) are set in .env")

        llm = LLM(model="gpt-5-mini", provider="chatanywhere")
        
        print("\nStarting Agent Session...")
        agent = SmartMenuAgent(llm)
        agent.cli()
        
    except Exception as e:
        print(f"\nError running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
