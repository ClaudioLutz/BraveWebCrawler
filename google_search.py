import asyncio
import os
import sys
import argparse
import logging # Added logging
from dotenv import load_dotenv
import json
import tempfile
import shutil # For rmtree
from pathlib import Path # For path manipulation
from dataclasses import asdict
from mcp_use import MCPAgent, MCPClient
# from mcp_use.logging import Logger # Commented out, using standard logging
from langchain_openai import ChatOpenAI
from urllib.parse import urlparse

# --- Initialize Logger ---
logger = logging.getLogger(__name__)

load_dotenv()

# --- Import project functions ---
from models import CompanyFacts
from google_harvester import get_facts_from_google
# <<< CHANGE: Re-import your URL selection logic
from search_common import select_best_url_with_llm, get_Google_Search_candidates # Corrected import name

# Logger.set_debug(1) # Commented out, using standard logging configuration


async def main(company_name: str, headful: bool = False): # Added headful parameter
    google_api_key_local = os.getenv("GOOGLE_API_KEY")
    google_cx_local = os.getenv("GOOGLE_CX")
    openai_api_key_local = os.getenv("OPENAI_API_KEY")

    # <<< CHANGE: Fully automated Step 0
    # --- Step 0: Automated URL Discovery ---
    if not openai_api_key_local:
        logger.error("OPENAI_API_KEY is required for URL selection. Exiting.")
        sys.exit(1)

    logger.info(f"Discovering best URL for '{company_name}'...")
    url_selection_llm_model = os.getenv("URL_SELECTION_MODEL", "gpt-4.1-mini")
    url_selection_llm = ChatOpenAI(model=url_selection_llm_model, temperature=0)
    logger.info(f"Using URL selection LLM: {url_selection_llm_model}")

    if not (google_api_key_local and google_cx_local):
        logger.error("GOOGLE_API_KEY and GOOGLE_CX are required for URL discovery. Exiting.")
        sys.exit(1)
        
    google_candidates = await get_Google_Search_candidates(company_name, google_api_key_local, google_cx_local)
    # FIX: Handle potential None return from select_best_url_with_llm and ensure two values are unpacked safely
    url_selection_result = select_best_url_with_llm(company_name, google_candidates, llm=url_selection_llm)
    if isinstance(url_selection_result, tuple) and len(url_selection_result) == 2:
        company_url, source_of_url = url_selection_result
    elif isinstance(url_selection_result, str): # If it somehow returned just a string URL
        company_url = url_selection_result
        source_of_url = "LLM Selection via search_common (source assumed)" # Default source
        logger.info(f"URL selection returned a string, source assumed: {source_of_url}")
    else: # If it returned None or an unexpected format
        company_url = None
        source_of_url = "LLM Selection Failed or Invalid Format"
        logger.warning(f"URL selection failed or returned an invalid format. Result: {url_selection_result}")

    if not company_url:
        logger.error(f"❌ Could not determine a reliable URL for '{company_name}'. Exiting.")
        sys.exit(1)

    logger.info(f"✅ URL selected: {company_url} (Source: {source_of_url})")

    # ── Step 1: Intelligent facts from Google ────────────────────────
    facts = CompanyFacts(official_website=company_url)
    if google_api_key_local and google_cx_local:
        logger.info(f"\nGathering facts for '{company_name}' from Google...")
        google_facts = await get_facts_from_google(company_name, google_api_key_local, google_cx_local)
        facts.merge_with(google_facts)
        logger.info("Facts after Google search:")
        logger.info(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
    else:
        # This case should ideally not be reached if URL discovery succeeded,
        # but kept for robustness if logic changes.
        logger.warning("GOOGLE_API_KEY or GOOGLE_CX not set. Skipping Google fact gathering.")

    # ── Step 2: Scrape homepage only if necessary ────────────────────
    missing_fields_list = facts.missing_fields() # Store the list
    if not missing_fields_list:
        logger.info("\n✅ All facts found via Google – no scraping necessary.")
        # Print final facts here as well for consistency if we return early
        logger.info("\nFinal facts (from Google Search only):")
        logger.info(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
        return

    logger.info(f"\n🔥 Google search incomplete. Need to scrape for: {missing_fields_list}")
    
    fields_to_gather_text = "\n    • ".join(missing_fields_list)
    root_url_for_prompt = "null"
    if company_url:
        try:
            parsed_found_url = urlparse(company_url)
            root_url_for_prompt = f"{parsed_found_url.scheme}://{parsed_found_url.netloc}"
        except Exception as e:
            logger.warning(f"Could not parse company_url '{company_url}': {e}. Using 'null' for root_url_for_prompt.", exc_info=True)


    # --- MCP Agent part ---
    base_mcp_config_path = Path(__file__).parent / "sequential_mcp_config.json"
    if not base_mcp_config_path.exists():
        logger.error(f"Base MCP config file not found at {base_mcp_config_path}")
        logger.info("\nFinal facts (MCP agent could not run due to missing config):")
        logger.info(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
        sys.exit(1)
    
    agent_llm = None
    if openai_api_key_local:
        try:
            agent_model_name = os.getenv("AGENT_MODEL_NAME", "gpt-4.1-mini")
            agent_llm = ChatOpenAI(model=agent_model_name, temperature=0)
            logger.info(f"LLM for MCPAgent initialized with model: {agent_model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM for MCPAgent: {e}. Agent cannot run.", exc_info=True)
            logger.info("\nFinal facts (without agent step due to LLM init error):")
            logger.info(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
            sys.exit(1)
    else:
        logger.warning("OPENAI_API_KEY is not available. MCPAgent cannot be initialized.")
        logger.info("\nFinal facts (from Google Search only, agent step skipped):")
        logger.info(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
        sys.exit(0) # Graceful exit as this is a configured skip

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
            logger.info("ℹ️ Configuring for HEADFUL mode.")
        else:
            logger.info("ℹ️ Configuring for HEADLESS mode.")

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
        logger.info(f"    Created Playwright runtime config: {playwright_runtime_config_file_path} with headless: {is_actually_headless}")

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
            logger.info(f"    Modified MCP launcher args for Playwright: {playwright_server_entry['args']}")
        else:
            # This will be caught by e_config_master below
            raise ValueError("Invalid base MCP configuration: 'playwright' server entry or 'mcpServers' key missing.")

        # 6. Write the modified MCP launcher configuration to a new temporary file.
        fd_mcp_launcher, temp_mcp_launcher_path_str = tempfile.mkstemp(suffix='.json', prefix='mcp_launcher_runtime_')
        with os.fdopen(fd_mcp_launcher, 'w') as tmp_mcp_write:
            json.dump(mcp_launcher_data, tmp_mcp_write, indent=2)
        
        mcp_config_path_to_use = Path(temp_mcp_launcher_path_str) 
        temp_mcp_launcher_file_for_cleanup = mcp_config_path_to_use 
        logger.info(f"    Using temporary MCP launcher: {mcp_config_path_to_use}")

        # 7. Initialize MCPClient.
        client = MCPClient.from_config_file(str(mcp_config_path_to_use.resolve()))

    except Exception as e_config_master:
        logger.error(f"Error during MCP configuration: {e_config_master}", exc_info=True)
        logger.info(f"    Falling back to original base MCP config: {base_mcp_config_path}")
        if temp_profile_dir_for_cleanup and temp_profile_dir_for_cleanup.exists():
            shutil.rmtree(temp_profile_dir_for_cleanup, ignore_errors=True)
            temp_profile_dir_for_cleanup = None # Reset after cleanup
        if temp_mcp_launcher_file_for_cleanup and temp_mcp_launcher_file_for_cleanup.exists():
            os.remove(temp_mcp_launcher_file_for_cleanup)
            temp_mcp_launcher_file_for_cleanup = None # Reset after cleanup
        # Fallback client initialization
        try:
            client = MCPClient.from_config_file(str(base_mcp_config_path.resolve()))
            logger.info("Successfully initialized MCPClient with fallback base config.")
        except Exception as e_fallback_client:
            logger.error(f"Failed to initialize MCPClient even with fallback config: {e_fallback_client}", exc_info=True)
            client = None # Ensure client is None if fallback also fails


    # This try block is for agent execution, separated from config setup
    try:
        if not client: # Should not happen if fallback above works, but as a safeguard
            logger.error("MCPClient could not be initialized. Exiting agent step.")
            # Print facts gathered so far before exiting
            logger.info("\nFinal facts (MCP agent could not run due to client init failure):")
            logger.info(json.dumps(asdict(facts), indent=2, ensure_ascii=False))
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
        
        logger.info(f"\nÜbermittelter Prompt an den Agenten (Auszug):")
        logger.info(f"Please gather ONLY the following missing facts from the website {root_url_for_prompt}:\n    • {fields_to_gather_text}")
        logger.info("-" * 20)

        agent_result_str = "" 
        try:
            agent_result_str = await agent.run(agent_prompt, max_steps=30) 
            logger.info(f"\nRaw result from Agent:\n{agent_result_str}")
            
            agent_facts_dict = json.loads(agent_result_str) 
            
            agent_company_facts = CompanyFacts() 
            for key, value in agent_facts_dict.items():
                if hasattr(agent_company_facts, key):
                    setattr(agent_company_facts, key, value)
                else:
                    logger.warning(f"Agent returned unexpected key '{key}' with value '{value}'. Ignoring.")

            facts.merge_with(agent_company_facts) 

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from agent: {e}. Agent raw output was: {agent_result_str}", exc_info=True)
        except Exception as e: 
            logger.error(f"An error occurred with the agent or processing its result: {e}", exc_info=True)

        logger.info("\n✅ Final combined facts:")
        logger.info(json.dumps(asdict(facts), indent=2, ensure_ascii=False))

    finally:
        # Cleanup temporary MCP launcher file
        if temp_mcp_launcher_file_for_cleanup and temp_mcp_launcher_file_for_cleanup.exists():
            try:
                os.remove(temp_mcp_launcher_file_for_cleanup)
                logger.info(f"    Cleaned up temporary MCP launcher: {temp_mcp_launcher_file_for_cleanup}")
            except Exception as e_cleanup_launcher:
                logger.error(f"    Error cleaning up temporary MCP launcher {temp_mcp_launcher_file_for_cleanup}: {e_cleanup_launcher}", exc_info=True)
        
        # Cleanup temporary profile directory for Playwright
        if temp_profile_dir_for_cleanup and temp_profile_dir_for_cleanup.exists():
            try:
                shutil.rmtree(temp_profile_dir_for_cleanup)
                logger.info(f"    Cleaned up temporary profile directory: {temp_profile_dir_for_cleanup}")
            except Exception as e_cleanup_profile_dir:
                logger.error(f"    Error cleaning up temporary profile directory {temp_profile_dir_for_cleanup}: {e_cleanup_profile_dir}", exc_info=True)


def console_main() -> None:
    """Entry point for the console script."""
    # --- Basic Logging Configuration ---
    # Outputs to stdout by default. stderr can be used for errors if preferred.
    # Basic config should be called only once.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

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
