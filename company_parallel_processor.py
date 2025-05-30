import asyncio
import os
import sys
import argparse
import csv # Added for CSV handling
import glob # Added for finding files
import json # For parsing agent's JSON result
import multiprocessing # Added for parallel processing
import shutil # Added for directory cleanup
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger
from langchain_openai import ChatOpenAI
from pydantic.types import SecretStr # Added for API key handling
# import httpx # Moved to search_common
from urllib.parse import urlparse # Still needed
from typing import Dict, Any # Moved to search_common, but still needed here for type hints
import tempfile
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# --- Import functions and constants from search_common ---
from search_common import (
    select_best_url_with_llm,
    get_brave_search_candidates,
    get_wikidata_homepage,
    # SEARCH_URL, # Not directly used in this file anymore
    # WIKIDATA_SEARCH, # Not directly used in this file anymore
    # WIKIDATA_ENTITY, # Not directly used in this file anymore
    BLACKLIST # Referenced by moved functions, effectively via search_common
)

# Expected keys from the agent's JSON output, used for CSV header and data extraction
EXPECTED_JSON_KEYS = [
    "official_website", "founded", "Hauptsitz", "Firmenidentifikationsnummer",
    "HauptTelefonnummer", "HauptEmailAdresse", "Geschäftsbericht"
]

# Enable mcp_use debug logging
Logger.set_debug(1)

# Functions select_best_url_with_llm, get_brave_search_candidates, get_wikidata_homepage
# have been moved to search_common.py and will be imported from there.

async def process_company_data(company_name: str, company_number: str) -> Dict[str, Any]:
    """
    Processes a single company: finds its URL and then uses an agent to extract information.
    Returns a dictionary with keys from EXPECTED_JSON_KEYS.
    """
    result_data = {key: "null" for key in EXPECTED_JSON_KEYS}
    tmp_profile_dir = None  # Initialize for cleanup in finally block

    try: # Main try block for the entire function's core logic
        llm_url_selector = None
        openai_api_key_val = os.getenv("OPENAI_API_KEY")
        if openai_api_key_val:
            try:
                llm_url_selector = ChatOpenAI(
                    model="gpt-4.1-mini", temperature=0, api_key=SecretStr(openai_api_key_val)
                )
                print(f"LLM for URL selection initialized for '{company_name}'.")
            except Exception as e:
                print(f"Error initializing LLM for URL selection for '{company_name}': {e}. Proceeding without LLM URL selection.", file=sys.stderr)
                llm_url_selector = None # Ensure it's None
        else:
            print(f"Warning: OPENAI_API_KEY not set. LLM-based URL selection will be skipped for '{company_name}'.", file=sys.stderr)

        company_url = None
        source_of_url = "None"

        brave_api_key_val = os.getenv("BRAVE_API_KEY")
        if brave_api_key_val:
            print(f"\nAttempting to find URL for '{company_name}' using Brave Search...")
            # Use imported function, pass brave_api_key_val as brave_api_key argument
            brave_candidates = get_brave_search_candidates(company_name, brave_api_key=brave_api_key_val, count=5)
            if brave_candidates:
                if llm_url_selector:
                    print(f"Found {len(brave_candidates)} Brave candidates for '{company_name}'. Asking LLM to select...")
                    # Use imported function, pass llm_url_selector as llm argument
                    company_url = select_best_url_with_llm(company_name, brave_candidates, llm_url_selector)
                    if company_url: source_of_url = "Brave Search + LLM"
                    else: print(f"LLM did not select a URL for '{company_name}'. Applying Brave heuristic fallback.")
                else:
                    print(f"LLM not available for URL selection for '{company_name}'. Applying Brave heuristic fallback.")

                if not company_url: # Fallback logic if LLM didn't find or wasn't used
                    ch_urls = [c['url'] for c in brave_candidates if c['is_ch_domain'] and c['company_match_in_host']]
                    other_urls = [c['url'] for c in brave_candidates if not c['is_ch_domain'] and c['company_match_in_host']]
                    any_urls = [c['url'] for c in brave_candidates]
                    if ch_urls: company_url = ch_urls[0]
                    elif other_urls: company_url = other_urls[0]
                    elif any_urls: company_url = any_urls[0]
                    if company_url: source_of_url = "Brave Search (heuristic fallback)"
                    else: print(f"Brave heuristic fallback also failed to find a URL for '{company_name}'.")
            else:
                print(f"No candidates found from Brave Search for '{company_name}'.")
        else:
            print(f"Brave API key not available. Skipping Brave Search for '{company_name}'.")


        if not company_url:
            status_msg = f"Previous method ({source_of_url})" if source_of_url != "None" else "Brave Search skipped or failed"
            print(f"{status_msg} did not yield a URL for '{company_name}'. Trying Wikidata...")
            # Use imported function
            company_url = get_wikidata_homepage(company_name)
            if company_url: source_of_url = "Wikidata"

        root_url_for_prompt = "null"
        if company_url:
            print(f"\n==> Final URL identified for '{company_name}': {company_url} (Source: {source_of_url})")
            parsed_found_url = urlparse(company_url)
            root_url_for_prompt = f"{parsed_found_url.scheme}://{parsed_found_url.netloc}"
        else:
            print(f"\n==> Could not find URL for '{company_name}' using available methods.", file=sys.stderr)
        
        result_data["official_website"] = root_url_for_prompt # Store the found URL or "null"

        # --- MCP Agent part ---
        # Create a unique directory for this worker's Playwright profile
        tmp_profile_dir = Path(tempfile.gettempdir()) / f"mcp_playwright_profile_{os.getpid()}"
        try:
            tmp_profile_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{os.getpid()}] Created temp profile directory: {tmp_profile_dir} for {company_name}")
        except Exception as e_mkdir:
            print(f"[{os.getpid()}] Error creating temp profile directory {tmp_profile_dir}: {e_mkdir}", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "TEMP_DIR_CREATION_ERROR"
            # This error is critical; cleanup will be attempted in finally, then return.
            raise  # Re-raise to be caught by the outer try-except

        if not os.getenv("OPENAI_API_KEY"): # Agent critically needs this
            print(f"Error: OPENAI_API_KEY is not available. MCPAgent cannot be initialized for '{company_name}'.", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_OPENAI_KEY_MISSING"
            raise Exception("AGENT_OPENAI_KEY_MISSING")

        agent_llm = None
        openai_api_key_for_agent = os.getenv("OPENAI_API_KEY") 
        assert openai_api_key_for_agent is not None # Ensure for Pylance that it's not None here
        try:
            agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=SecretStr(openai_api_key_for_agent))
            print(f"[{os.getpid()}] LLM for MCPAgent initialized for '{company_name}'.")
        except Exception as e:
            print(f"[{os.getpid()}] Error initializing LLM for MCPAgent for '{company_name}': {e}. Agent cannot run.", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_LLM_INIT_FAILURE"
            raise # Re-raise
        
        if agent_llm is None: # Should not happen if above try succeeded, but as a safeguard
            print(f"[{os.getpid()}] Agent LLM is None after initialization attempt for '{company_name}'. Cannot proceed.", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_LLM_IS_NONE_POST_INIT"
            raise Exception("AGENT_LLM_IS_NONE_POST_INIT")

        # --- Dynamic MCP Configuration ---
        # 1. Create Per-Process Playwright Configuration File (runtime-playwright-config.json)
        playwright_config_content = {
            "browser": {
                "userDataDir": str(tmp_profile_dir.resolve()),  # Absolute path
                "launchOptions": {"headless": False}  # Explicitly set headless mode
            }
        }
        per_process_playwright_config_path = tmp_profile_dir / "runtime-playwright-config.json"
        with open(per_process_playwright_config_path, 'w') as f:
            json.dump(playwright_config_content, f, indent=2)
        print(f"[{os.getpid()}] Created per-process Playwright config: {per_process_playwright_config_path}")

        # 2. Create Per-Process MCP Launcher File (runtime-mcp-launcher.json)
        base_mcp_launcher_path = Path(__file__).parent / "parallel_mcp_launcher.json"
        if not base_mcp_launcher_path.exists():
            print(f"[{os.getpid()}] Error: Base MCP launcher template not found at {base_mcp_launcher_path} for '{company_name}'", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "BASE_MCP_LAUNCHER_MISSING"
            raise Exception("Base MCP launcher template not found")

        with open(base_mcp_launcher_path, 'r') as f:
            mcp_launcher_template = json.load(f)

        if "mcpServers" in mcp_launcher_template and \
           "playwright" in mcp_launcher_template["mcpServers"] and \
           "args" in mcp_launcher_template["mcpServers"]["playwright"]:
            args_list = mcp_launcher_template["mcpServers"]["playwright"]["args"]
            try:
                config_arg_index = args_list.index("--config")
                if config_arg_index + 1 < len(args_list):
                    args_list[config_arg_index + 1] = str(per_process_playwright_config_path.resolve())
                    print(f"[{os.getpid()}] MCP Launcher '--config' arg updated to: {args_list[config_arg_index + 1]}")
                else:
                    raise ValueError("MCP Launcher template '--config' argument is last, no value to replace.")
            except ValueError as e_template_val: # Handles '--config' not found or index issue
                print(f"[{os.getpid()}] Error processing '--config' in {base_mcp_launcher_path} for '{company_name}': {e_template_val}", file=sys.stderr)
                result_data[EXPECTED_JSON_KEYS[1]] = "MCP_LAUNCHER_TEMPLATE_CONFIG_ARG_ERROR"
                raise
        else:
            print(f"[{os.getpid()}] Error: MCP Launcher template {base_mcp_launcher_path} has unexpected structure for '{company_name}'", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "MCP_LAUNCHER_TEMPLATE_STRUCTURE_ERROR"
            raise Exception("MCP Launcher template has unexpected structure")

        dynamic_mcp_launcher_path = tmp_profile_dir / "runtime-mcp-launcher.json"
        with open(dynamic_mcp_launcher_path, 'w') as f:
            json.dump(mcp_launcher_template, f, indent=2)
        print(f"[{os.getpid()}] Created dynamic MCP launcher: {dynamic_mcp_launcher_path}")

        # 3. Initialize MCPClient with the dynamic launcher
        client = MCPClient.from_config_file(str(dynamic_mcp_launcher_path.resolve()))
        
        agent = MCPAgent(llm=agent_llm, client=client, max_steps=30)

        prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.

Die offizielle Webseite für "{company_name}" wurde als "{root_url_for_prompt}" identifiziert (Quelle: {source_of_url}).

Wenn eine URL ({root_url_for_prompt}) vorhanden ist und nicht 'null' oder 'nicht gefunden' lautet:
1. Öffne diese URL: {root_url_for_prompt}
2. Durchsuche diese Seite und relevante Unterseiten (z. B. /about, /unternehmen, /impressum, /geschichte, /contact, /legal)
   und sammle die unten genannten Fakten. Achte darauf, die aktuellsten Informationen zu finden.

Wenn KEINE URL gefunden wurde (d.h. als "{root_url_for_prompt}" angegeben ist) ODER Informationen auf der Webseite nicht auffindbar sind, gib für die entsprechenden Felder **null** zurück.

Fakten zu sammeln:
    • Gründungsjahr (JJJJ)
    • Offizielle Website (die bereits ermittelte Root-URL: "{root_url_for_prompt}")
    • Addresse Hauptsitz (vollständige Adresse)
    • Firmenidentifikationsnummer (meistens im Impressum, z.B. CHE-XXX.XXX.XXX)
    • Haupt-Telefonnummer (internationales Format wenn möglich)
    • Haupt-Emailadresse (allgemeine Kontakt-Email)
    • URL oder PDF-Link des AKTUELLSTEN Geschäftsberichtes/Jahresberichtes (falls öffentlich zugänglich)

Antworte **ausschließlich** mit genau diesem JSON, ohne jeglichen Text davor
oder danach:

{{
  "official_website": "{root_url_for_prompt}",
  "founded": "<jahr JJJJ oder null>",
  "Hauptsitz": "<vollständige Adresse oder null>",
  "Firmenidentifikationsnummer": "<ID oder null>",
  "HauptTelefonnummer": "<nummer oder null>",
  "HauptEmailAdresse": "<email oder null>",
  "Geschäftsbericht" : "<url/PDF-Link oder null>"
}}
"""
        print(f"\nÜbermittelter Prompt an den Agenten für '{company_name}' (Auszug):")
        print(f"[{os.getpid()}] Die offizielle Webseite für \"{company_name}\" wurde als \"{root_url_for_prompt}\" identifiziert (Quelle: {source_of_url}).")
        if company_url: print(f"[{os.getpid()}] 1. Öffne diese URL: {root_url_for_prompt}")
        print(f"[{os.getpid()}] {'-' * 20}")

        # Inner try-except for agent execution and JSON parsing
        try:
            agent_result_str = await agent.run(prompt, max_steps=30)
            print(f"[{os.getpid()}] Result from Agent for '{company_name}':\n{agent_result_str}")
            agent_json_result = json.loads(agent_result_str)
            
            for key in EXPECTED_JSON_KEYS:
                result_data[key] = agent_json_result.get(key, "null") # Use agent's value or default

        except json.JSONDecodeError:
            print(f"[{os.getpid()}] Error: Could not decode JSON from agent for '{company_name}'", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_JSON_DECODE_ERROR" # Indicate error
        except Exception as e_agent_run: # Catch other errors from agent.run()
            print(f"[{os.getpid()}] Error during agent execution for '{company_name}': {e_agent_run}", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = f"AGENT_EXECUTION_ERROR: {str(e_agent_run)[:50]}"
        
    except Exception as e_outer: # Catch-all for setup errors before agent run, or other unexpected issues
        print(f"[{os.getpid()}] Outer error processing company '{company_name}': {e_outer}", file=sys.stderr)
        if result_data.get(EXPECTED_JSON_KEYS[1], "null") == "null":
            result_data[EXPECTED_JSON_KEYS[1]] = f"PROCESSING_ERROR: {str(e_outer)[:50]}"
    
    finally:
        # Cleanup the temporary profile directory
        if tmp_profile_dir and tmp_profile_dir.exists():
            try:
                shutil.rmtree(tmp_profile_dir)
                print(f"[{os.getpid()}] Cleaned up temp profile directory: {tmp_profile_dir}")
            except Exception as e_clean:
                print(f"[{os.getpid()}] Error cleaning up temp profile directory {tmp_profile_dir}: {e_clean}", file=sys.stderr)
    
    return result_data


def process_company_data_wrapper(company_name: str, company_number: str):
    """
    Synchronous wrapper for the process_company_data async function.
    """
    return asyncio.run(process_company_data(company_name, company_number))


def console_main() -> None:
    """Entry point for the console script."""
    parser = argparse.ArgumentParser(
        description="Search for company information from the newest CSV in the 'input' folder using Brave/Wikidata/Agent and output to CSV, in parallel."
    )
    # parser.add_argument("input_csv", help="Path to the input CSV file (columns: company_number,company_name)") # Removed
    parser.add_argument("output_csv", help="Path to the output CSV file for results")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use for parallel processing. Defaults to the number of CPU cores."
    )
    args = parser.parse_args()

    # --- Determine the newest input CSV file ---
    input_dir = "input"
    if not os.path.isdir(input_dir):
        print(f"CRITICAL ERROR: Input directory '{input_dir}' not found. Please create it and place your CSV file(s) there. Exiting.", file=sys.stderr)
        sys.exit(1)

    list_of_csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not list_of_csv_files:
        print(f"CRITICAL ERROR: No CSV files found in the '{input_dir}' directory. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    input_csv_path = max(list_of_csv_files, key=os.path.getmtime)
    print(f"Using the newest input CSV file: {input_csv_path}")
    # --- End of determining input CSV ---

    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL ERROR: OPENAI_API_KEY is not set in .env or environment. The agent cannot run. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    if not os.getenv("BRAVE_API_KEY"):
        print("Warning: BRAVE_API_KEY is not set. Brave Search functionality will be skipped.", file=sys.stderr)

    output_rows_buffer = [] 
    header = ['company_number', 'company_name'] + EXPECTED_JSON_KEYS
    output_rows_buffer.append(header)

    companies_to_process = []
    try:
        with open(input_csv_path, 'r', encoding='utf-8-sig') as infile: 
            reader = csv.reader(infile)
            try:
                input_header = next(reader) 
                if not (input_header[0].strip().lower() == 'company_number' and \
                        input_header[1].strip().lower() == 'company_name'):
                    print(f"Warning: Input CSV header mismatch. Expected 'company_number, company_name', got '{','.join(input_header)}'. Assuming correct column order.", file=sys.stderr)
            except StopIteration:
                print(f"Error: Input CSV file {input_csv_path} is empty or has no header.", file=sys.stderr)
                sys.exit(1)
            
            for row in reader:
                companies_to_process.append(row)
            
    except FileNotFoundError: # Should be caught by earlier checks
        print(f"Error: Input CSV file not found at {input_csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during input CSV reading from {input_csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    total_companies = len(companies_to_process)
    if total_companies == 0:
        print(f"No company data found in {input_csv_path} (after header). Exiting.")
        try:
            with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
            print(f"Empty output CSV with headers written to {args.output_csv}")
        except IOError as e:
            print(f"Error writing empty output CSV to {args.output_csv}: {e}", file=sys.stderr)
        sys.exit(0)
        
    print(f"Found {total_companies} companies to process from {input_csv_path}.")

    if total_companies > 0:
        num_workers = args.workers 
        print(f"Using {num_workers} worker processes.")
        print(f"Starting parallel processing of {total_companies} companies with {num_workers} workers...")
        
        processed_args = []
        valid_companies_for_processing = [] 

        for i, row_data in enumerate(companies_to_process):
            if len(row_data) < 2:
                print(f"Skipping invalid row {i+2} in input CSV (not enough columns): {row_data}", file=sys.stderr)
                error_entry = ["INVALID_INPUT_ROW"] * (2 + len(EXPECTED_JSON_KEYS))
                if len(row_data) > 0: error_entry[0] = row_data[0]
                output_rows_buffer.append(error_entry) 
                continue

            company_number, company_name = row_data[0].strip(), row_data[1].strip()
            if not company_name:
                print(f"Skipping row {i+2} due to empty company_name (company_number: {company_number}).", file=sys.stderr)
                error_entry = [company_number, "EMPTY_COMPANY_NAME"] + ["null"] * len(EXPECTED_JSON_KEYS)
                output_rows_buffer.append(error_entry) 
                continue
            
            processed_args.append((company_name, company_number)) 
            valid_companies_for_processing.append([company_number, company_name]) 

        if processed_args: 
            with multiprocessing.Pool(processes=num_workers) as pool:
                async_results = pool.starmap_async(process_company_data_wrapper, processed_args)
                results = async_results.get()
            
            print("Parallel processing finished. Preparing results for CSV output.")

            for i, extracted_info_dict in enumerate(results):
                company_number, company_name = valid_companies_for_processing[i] 
                
                current_row_data = [company_number, company_name]
                for key in EXPECTED_JSON_KEYS:
                    current_row_data.append(extracted_info_dict.get(key, "null"))
                output_rows_buffer.append(current_row_data)
        else:
            print("No valid companies to process after filtering.")
            results = [] 

    else: 
        results = [] 
        print("No companies were processed as the input file was empty or contained no data rows.")

    print(f"\nWriting {len(output_rows_buffer)-1} data entries to {args.output_csv}...")
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(output_rows_buffer)
        print(f"Successfully wrote to {args.output_csv}")
    except IOError as e:
        print(f"Error writing to output CSV {args.output_csv}: {e}", file=sys.stderr)
    
    print(f"\nProcessing complete. All collected data written to {args.output_csv}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) 
    console_main()
