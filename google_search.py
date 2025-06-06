import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv
import json
import tempfile
import shutil # For rmtree
from pathlib import Path # For path manipulation
from dataclasses import asdict
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger
from langchain_openai import ChatOpenAI
from urllib.parse import urlparse

load_dotenv()

# --- Import project functions ---
from models import CompanyFacts
from google_harvester import get_facts_from_google
# <<< CHANGE: Re-import your URL selection logic
from search_common import select_best_url_with_llm, get_Google_Search_candidates # Corrected import name

Logger.set_debug(1)


async def main(company_name: str, headful: bool = False): # Added headful parameter
    google_api_key_local = os.getenv("GOOGLE_API_KEY")
    google_cx_local = os.getenv("GOOGLE_CX")
    openai_api_key_local = os.getenv("OPENAI_API_KEY")

    # <<< CHANGE: Fully automated Step 0
    # --- Step 0: Automated URL Discovery ---
    if not openai_api_key_local:
        print("Error: OPENAI_API_KEY is required for URL selection. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Discovering best URL for '{company_name}'...")
    url_selection_llm_model = os.getenv("URL_SELECTION_MODEL", "gpt-4.1-mini")
    url_selection_llm = ChatOpenAI(model=url_selection_llm_model, temperature=0)
    print(f"Using URL selection LLM: {url_selection_llm_model}") # Added for clarity

    if not (google_api_key_local and google_cx_local):
        print("Error: GOOGLE_API_KEY and GOOGLE_CX are required for URL discovery. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    google_candidates = await get_Google_Search_candidates(company_name, google_api_key_local, google_cx_local)
    # FIX: Handle potential None return from select_best_url_with_llm and ensure two values are unpacked safely
    url_selection_result = select_best_url_with_llm(company_name, google_candidates, llm=url_selection_llm)
    if isinstance(url_selection_result, tuple) and len(url_selection_result) == 2:
        company_url, source_of_url = url_selection_result
    elif isinstance(url_selection_result, str): # If it somehow returned just a string URL
        company_url = url_selection_result
        source_of_url = "LLM Selection via search_common (source assumed)" # Default source
    else: # If it returned None or an unexpected format
        company_url = None
        source_of_url = "LLM Selection Failed or Invalid Format"

    if not company_url:
        print(f"❌ Could not determine a reliable URL for '{company_name}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"✅ URL selected: {company_url} (Source: {source_of_url})")

    # ── Step 1: Intelligent facts from Google ────────────────────────
    facts = CompanyFacts(official_website=company_url)
    if google_api_key_local and google_cx_local:
        print(f"\nGathering facts for '{company_name}' from Google...")
        google_facts = await get_facts_from_google(company_name, google_api_key_local, google_cx_local)
        facts.merge_with(google_facts)
        print("Facts after Google search:")
        print(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
    else:
        # This case should ideally not be reached if URL discovery succeeded,
        # but kept for robustness if logic changes.
        print("Warning: GOOGLE_API_KEY or GOOGLE_CX not set. Skipping Google fact gathering.", file=sys.stderr)

    # ── Step 2: Scrape homepage only if necessary ────────────────────
    missing_fields_list = facts.missing_fields() # Store the list
    if not missing_fields_list:
        print("\n✅ All facts found via Google – no scraping necessary.")
        # Print final facts here as well for consistency if we return early
        print("\nFinal facts (from Google Search only):")
        print(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
        return

    print(f"\n🔥 Google search incomplete. Need to scrape for: {missing_fields_list}")
    
    fields_to_gather_text = "\n    • ".join(missing_fields_list)
    root_url_for_prompt = "null"
    if company_url:
        try:
            parsed_found_url = urlparse(company_url)
            root_url_for_prompt = f"{parsed_found_url.scheme}://{parsed_found_url.netloc}"
        except Exception as e:
            print(f"Warning: Could not parse company_url '{company_url}': {e}. Using 'null' for root_url_for_prompt.", file=sys.stderr)


    # --- MCP Agent part ---
    base_mcp_config_path = Path(__file__).parent / "sequential_mcp_config.json"
    if not base_mcp_config_path.exists():
        print(f"Error: Base MCP config file not found at {base_mcp_config_path}", file=sys.stderr)
        print("\nFinal facts (MCP agent could not run due to missing config):")
        print(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
        sys.exit(1)
    
    agent_llm = None
    if openai_api_key_local:
        try:
            agent_model_name = os.getenv("AGENT_MODEL_NAME", "gpt-4.1-mini")
            agent_llm = ChatOpenAI(model=agent_model_name, temperature=0)
            print(f"LLM for MCPAgent initialized with model: {agent_model_name}")
        except Exception as e:
            print(f"Error initializing LLM for MCPAgent: {e}. Agent cannot run.", file=sys.stderr)
            print("\nFinal facts (without agent step due to LLM init error):")
            print(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
            sys.exit(1)
    else:
        print("Warning: OPENAI_API_KEY is not available. MCPAgent cannot be initialized.", file=sys.stderr)
        print("\nFinal facts (from Google Search only, agent step skipped):")
        print(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
        sys.exit(0)

    temp_profile_dir_for_cleanup = None
    temp_mcp_launcher_file_for_cleanup = None 
    client = None # Ensure client is defined in this scope

    try:
        # Common setup for both headful and headless:
        # 1. Create a temporary directory for Playwright's user data and its specific config.
        temp_profile_dir = Path(tempfile.mkdtemp(prefix="mcp_playwright_profile_"))
        temp_profile_dir_for_cleanup = temp_profile_dir

        # 2. Create the Playwright-specific runtime configuration content.
        is_actually_headless = not headful 
        
        if headful:
            print("ℹ️ Configuring for HEADFUL mode.")
        else:
            print("ℹ️ Configuring for HEADLESS mode.")

        playwright_runtime_config_content = {
            "browser": {
                "userDataDir": str(temp_profile_dir.resolve()), 
                "launchOptions": {
                    "headless": is_actually_headless, 
                    "args": [] 
                }
            }
        }
        if headful: 
            playwright_runtime_config_content["browser"]["launchOptions"]["args"].extend([
                "--window-position=0,0", 
                "--window-size=800,600"  
            ])
            
        # 3. Write this Playwright-specific config to a file within the temp_profile_dir.
        playwright_runtime_config_file_path = temp_profile_dir / "playwright_runtime_config.json"
        with open(playwright_runtime_config_file_path, 'w') as f_pw_config:
            json.dump(playwright_runtime_config_content, f_pw_config, indent=2)
        print(f"    Created Playwright runtime config: {playwright_runtime_config_file_path} with headless: {is_actually_headless}")

        # 4. Load the base MCP launcher configuration.
        with open(base_mcp_config_path, 'r') as f_read_base_mcp:
            mcp_launcher_data = json.load(f_read_base_mcp)

        # 5. Modify the MCP launcher config for the Playwright server.
        if "mcpServers" in mcp_launcher_data and "playwright" in mcp_launcher_data["mcpServers"]:
            playwright_server_entry = mcp_launcher_data["mcpServers"]["playwright"]
            current_args = playwright_server_entry.get("args", [])
            
            new_args = []
            skip_next_arg = False
            for arg in current_args:
                if skip_next_arg:
                    skip_next_arg = False
                    continue
                if arg == "--config": 
                    skip_next_arg = True
                    continue
                if arg.startswith("--headless"): 
                    continue
                new_args.append(arg)
            
            playwright_server_entry["args"] = new_args + ["--config", str(playwright_runtime_config_file_path.resolve())]
            print(f"    Modified MCP launcher args for Playwright: {playwright_server_entry['args']}")
        else:
            raise ValueError("Invalid base MCP configuration: 'playwright' server entry or 'mcpServers' key missing.")

        # 6. Write the modified MCP launcher configuration to a new temporary file.
        fd_mcp_launcher, temp_mcp_launcher_path_str = tempfile.mkstemp(suffix='.json', prefix='mcp_launcher_runtime_')
        with os.fdopen(fd_mcp_launcher, 'w') as tmp_mcp_write:
            json.dump(mcp_launcher_data, tmp_mcp_write, indent=2)
        
        mcp_config_path_to_use = Path(temp_mcp_launcher_path_str) 
        temp_mcp_launcher_file_for_cleanup = mcp_config_path_to_use 
        print(f"    Using temporary MCP launcher: {mcp_config_path_to_use}")

        # 7. Initialize MCPClient.
        client = MCPClient.from_config_file(str(mcp_config_path_to_use.resolve()))

    except Exception as e_config_master:
        print(f"Error during MCP configuration: {e_config_master}", file=sys.stderr)
        print(f"    Falling back to original base MCP config: {base_mcp_config_path}")
        if temp_profile_dir_for_cleanup and temp_profile_dir_for_cleanup.exists():
            shutil.rmtree(temp_profile_dir_for_cleanup, ignore_errors=True)
            temp_profile_dir_for_cleanup = None # Reset after cleanup
        if temp_mcp_launcher_file_for_cleanup and temp_mcp_launcher_file_for_cleanup.exists():
            os.remove(temp_mcp_launcher_file_for_cleanup)
            temp_mcp_launcher_file_for_cleanup = None # Reset after cleanup
        # Fallback client initialization
        client = MCPClient.from_config_file(str(base_mcp_config_path.resolve()))


    # This try block is for agent execution, separated from config setup
    try:
        if not client: # Should not happen if fallback above works, but as a safeguard
            print("Error: MCPClient could not be initialized. Exiting agent step.", file=sys.stderr)
            # Print facts gathered so far before exiting
            print("\nFinal facts (MCP agent could not run due to client init failure):")
            print(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
            sys.exit(1)

        agent = MCPAgent(llm=agent_llm, client=client, max_steps=30)
        
        company_info = f"The target company is '{company_name}'. The primary URL for your investigation is '{root_url_for_prompt}', identified via '{source_of_url}'."
        agent_behavior = """You are a specialized Web Intelligence Agent, equipped with Playwright browser tools.
Your role is to act as an expert data analyst and web researcher.
Be concise in your thoughts and actions. Use the browser_snapshot tool sparingly as it consumes a lot of context."""
        mission_steps = f"""1. **Navigate**: Open the base URL: {root_url_for_prompt}
2. **Explore**: Systematically search the site. Prioritize pages like /contact, /about, /imprint (or /impressum), /legal, and /company.
3. **Crucial Constraint**: All gathered information MUST pertain to the company's headquarters (Hauptsitz). Do NOT include details from regional branches or subsidiaries. Verify that contact details (email, phone) are for the headquarters.
4. **Gather**: Meticulously extract ONLY the following missing facts:
    • {fields_to_gather_text}"""
        output_format = """Your final response MUST be a single, valid JSON object and nothing else.
Do not include any explanatory text, greetings, or summaries before or after the JSON.
If, after a thorough search, a fact cannot be found, its value in the JSON MUST be `null`.

For example, if you were asked for 'founded' and 'haupt_tel', your response must be:
{
  "founded": "1925",
  "haupt_tel": "+41 58 570 31 11"
}"""
        agent_prompt = f"{agent_behavior}\n\n{company_info}\n\n## Your Mission\n\n{mission_steps}\n\n## Response Format\n\n{output_format}\n\nBegin your analysis now.\n"
        
        print(f"\nÜbermittelter Prompt an den Agenten (Auszug):")
        print(f"Please gather ONLY the following missing facts from the website {root_url_for_prompt}:\n    • {fields_to_gather_text}")
        print("-" * 20)

        agent_result_str = "" 
        try:
            agent_result_str = await agent.run(agent_prompt, max_steps=30) 
            print(f"\nRaw result from Agent:\n{agent_result_str}")
            
            agent_facts_dict = json.loads(agent_result_str) 
            
            agent_company_facts = CompanyFacts() 
            for key, value in agent_facts_dict.items():
                if hasattr(agent_company_facts, key):
                    setattr(agent_company_facts, key, value)
                else:
                    print(f"Warning: Agent returned unexpected key '{key}' with value '{value}'. Ignoring.", file=sys.stderr)

            facts.merge_with(agent_company_facts) 

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from agent: {e}", file=sys.stderr)
            print(f"Agent raw output was: {agent_result_str}")
        except Exception as e: 
            print(f"An error occurred with the agent or processing its result: {e}", file=sys.stderr)

        print("\n✅ Final combined facts:")
        print(json.dumps(asdict(facts), indent=2, ensure_ascii=False))

    finally:
        # Cleanup temporary MCP launcher file
        if temp_mcp_launcher_file_for_cleanup and temp_mcp_launcher_file_for_cleanup.exists():
            try:
                os.remove(temp_mcp_launcher_file_for_cleanup)
                print(f"    Cleaned up temporary MCP launcher: {temp_mcp_launcher_file_for_cleanup}")
            except Exception as e_cleanup_launcher:
                print(f"    Error cleaning up temporary MCP launcher {temp_mcp_launcher_file_for_cleanup}: {e_cleanup_launcher}", file=sys.stderr)
        
        # Cleanup temporary profile directory for Playwright
        if temp_profile_dir_for_cleanup and temp_profile_dir_for_cleanup.exists():
            try:
                shutil.rmtree(temp_profile_dir_for_cleanup)
                print(f"    Cleaned up temporary profile directory: {temp_profile_dir_for_cleanup}")
            except Exception as e_cleanup_profile_dir:
                print(f"    Error cleaning up temporary profile directory {temp_profile_dir_for_cleanup}: {e_cleanup_profile_dir}", file=sys.stderr)


def console_main() -> None:
    """Entry point for the console script."""
    parser = argparse.ArgumentParser(
        description="Gather company facts using Google Search and optionally a web scraping agent."
    )
    parser.add_argument("company", help="The company name to search for")
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run browser in headful mode (default: headless)"
    )
    args = parser.parse_args()

    asyncio.run(main(args.company, args.headful)) # Pass headful argument


if __name__ == "__main__":
    console_main()
