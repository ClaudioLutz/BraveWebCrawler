import asyncio
import os
import sys
import argparse
import csv
import glob
import json
import multiprocessing
import shutil
import time
import psutil
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger
from langchain_openai import ChatOpenAI
from pydantic.types import SecretStr
import httpx # Required for pre-check
import re # Required for pre-check (though is_url_relevant_to_company is imported)
from urllib.parse import urlparse
from typing import Dict, Any
import tempfile
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# --- Import functions and constants from search_common ---
from search_common import (
    select_best_url_with_llm,
    get_brave_search_candidates,
    get_wikidata_homepage,
    BLACKLIST,
    is_url_relevant_to_company # Import the relevance check function
)

# Timeout for agent processing per company (in seconds)
AGENT_PROCESSING_TIMEOUT = 60  # 1 minute

# NEW UTILITY FUNCTION
def rmtree_with_retry(path_to_remove: Path, attempts: int = 5, delay_seconds: float = 1.0):
    """
    Attempts to remove a directory tree, retrying on PermissionError.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path_to_remove)
            print(f"[{os.getpid()}] Successfully removed directory: {path_to_remove} on attempt {attempt + 1}")
            return
        except PermissionError as e:
            print(f"[{os.getpid()}] PermissionError removing {path_to_remove} on attempt {attempt + 1}/{attempts}: {e}", file=sys.stderr)
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
            else:
                print(f"[{os.getpid()}] Failed to remove directory {path_to_remove} after {attempts} attempts.", file=sys.stderr)
        except FileNotFoundError:
            print(f"[{os.getpid()}] Directory not found for removal (already deleted?): {path_to_remove}", file=sys.stderr)
            return
        except Exception as e:
            print(f"[{os.getpid()}] Unexpected error removing {path_to_remove} on attempt {attempt + 1}: {e}", file=sys.stderr)
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
            else:
                print(f"[{os.getpid()}] Failed to remove directory {path_to_remove} due to unexpected error after {attempts} attempts.", file=sys.stderr)
    print(f"[{os.getpid()}] Warning: Directory {path_to_remove} could not be deleted after all retries.", file=sys.stderr)

def kill_chrome_processes_using(path: Path):
    """
    Kill any chrome.exe processes that are locking files inside the specified profile directory.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'chrome.exe' in proc.info['name'].lower():
                cmdline_args = proc.info['cmdline']
                if cmdline_args:
                    cmdline_str = ' '.join(cmdline_args)
                    if str(path) in cmdline_str:
                        print(f"[{os.getpid()}] Terminating chrome process {proc.info['pid']} using {path}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=0.5)
                        except psutil.TimeoutExpired:
                            print(f"[{os.getpid()}] Chrome process {proc.info['pid']} did not terminate gracefully, killing.")
                            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            print(f"[{os.getpid()}] Unexpected error when checking process {proc.info.get('pid', 'N/A')}: {e}", file=sys.stderr)
            continue

EXPECTED_JSON_KEYS = [
    "official_website", "founded", "Hauptsitz", "Firmenidentifikationsnummer",
    "HauptTelefonnummer", "HauptEmailAdresse", "Geschäftsbericht"
]

Logger.set_debug(1)

async def process_company_data(company_name: str, company_number: str) -> Dict[str, Any]:
    """
    Processes a single company: finds its URL and then uses an agent to extract information.
    Returns a dictionary with keys from EXPECTED_JSON_KEYS.
    """
    result_data = {key: "null" for key in EXPECTED_JSON_KEYS}
    tmp_profile_dir = None
    client_mcp = None # Renamed from client to avoid conflict with httpx.AsyncClient

    try:
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
                llm_url_selector = None
        else:
            print(f"Warning: OPENAI_API_KEY not set. LLM-based URL selection will be skipped for '{company_name}'.", file=sys.stderr)

        company_url = None
        source_of_url = "None"
        brave_api_key_val = os.getenv("BRAVE_API_KEY")

        if brave_api_key_val:
            print(f"\nAttempting to find URL for '{company_name}' using Brave Search...")
            brave_candidates = get_brave_search_candidates(company_name, brave_api_key=brave_api_key_val, count=5)
            if brave_candidates:
                if llm_url_selector:
                    print(f"Found {len(brave_candidates)} Brave candidates for '{company_name}'. Asking LLM to select...")
                    company_url = select_best_url_with_llm(company_name, brave_candidates, llm_url_selector)
                    if company_url: source_of_url = "Brave Search + LLM"
                    else: print(f"LLM did not select a URL for '{company_name}'. Applying Brave heuristic fallback.")
                else:
                    print(f"LLM not available for URL selection for '{company_name}'. Applying Brave heuristic fallback.")

                if not company_url:
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
            company_url = get_wikidata_homepage(company_name)
            if company_url: source_of_url = "Wikidata"

        root_url_for_prompt = "null"
        if company_url:
            print(f"\n==> Final URL identified for '{company_name}': {company_url} (Source: {source_of_url})")
            parsed_found_url = urlparse(company_url)
            root_url_for_prompt = f"{parsed_found_url.scheme}://{parsed_found_url.netloc}"
        else:
            print(f"\n==> Could not find URL for '{company_name}' using available methods.", file=sys.stderr)
        
        result_data["official_website"] = root_url_for_prompt

        # --- URL Relevance Pre-check ---
        if root_url_for_prompt != "null":
            async with httpx.AsyncClient() as http_client_for_check: # Create client for the check
                print(f"[{os.getpid()}] Performing pre-agent relevance check for URL: {root_url_for_prompt} for company: {company_name}")
                is_relevant = await is_url_relevant_to_company(root_url_for_prompt, company_name, http_client_for_check)
                if not is_relevant:
                    print(f"[{os.getpid()}] URL '{root_url_for_prompt}' deemed NOT relevant to '{company_name}' by pre-check. Skipping agent.", file=sys.stderr)
                    return result_data # Return early, agent part skipped
                else:
                    print(f"[{os.getpid()}] URL '{root_url_for_prompt}' deemed RELEVANT to '{company_name}' by pre-check. Proceeding with agent.")
        # --- End of URL Relevance Pre-check ---

        tmp_profile_dir = Path(tempfile.gettempdir()) / f"mcp_playwright_profile_{os.getpid()}_{time.time_ns()}" # Added timestamp for more uniqueness
        try:
            tmp_profile_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{os.getpid()}] Created temp profile directory: {tmp_profile_dir} for {company_name}")
        except Exception as e_mkdir:
            print(f"[{os.getpid()}] Error creating temp profile directory {tmp_profile_dir}: {e_mkdir}", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "TEMP_DIR_CREATION_ERROR"
            raise

        if not os.getenv("OPENAI_API_KEY"):
            print(f"Error: OPENAI_API_KEY is not available. MCPAgent cannot be initialized for '{company_name}'.", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_OPENAI_KEY_MISSING"
            raise Exception("AGENT_OPENAI_KEY_MISSING")

        agent_llm = None
        openai_api_key_for_agent = os.getenv("OPENAI_API_KEY")
        assert openai_api_key_for_agent is not None
        try:
            agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=SecretStr(openai_api_key_for_agent))
            print(f"[{os.getpid()}] LLM for MCPAgent initialized for '{company_name}'.")
        except Exception as e:
            print(f"[{os.getpid()}] Error initializing LLM for MCPAgent for '{company_name}': {e}. Agent cannot run.", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_LLM_INIT_FAILURE"
            raise

        if agent_llm is None:
            print(f"[{os.getpid()}] Agent LLM is None after initialization attempt for '{company_name}'. Cannot proceed.", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_LLM_IS_NONE_POST_INIT"
            raise Exception("AGENT_LLM_IS_NONE_POST_INIT")

        playwright_config_content = {
            "browser": {
                "userDataDir": str(tmp_profile_dir.resolve()),
                "launchOptions": {
                    "headless": False,
                    "args": [
                        "--disable-breakpad",
                        "--disable-extensions",
                    ]
                }
            }
        }
        per_process_playwright_config_path = tmp_profile_dir / "runtime-playwright-config.json"
        with open(per_process_playwright_config_path, 'w') as f:
            json.dump(playwright_config_content, f, indent=2)
        print(f"[{os.getpid()}] Created per-process Playwright config: {per_process_playwright_config_path}")

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
            except ValueError as e_template_val:
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

        client_mcp = MCPClient.from_config_file(str(dynamic_mcp_launcher_path.resolve()))
        agent = MCPAgent(llm=agent_llm, client=client_mcp, max_steps=30)

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

        try:
            # --- Agent Execution with Timeout ---
            agent_task = agent.run(prompt, max_steps=30)
            agent_result_str = await asyncio.wait_for(agent_task, timeout=AGENT_PROCESSING_TIMEOUT)
            # --- End of Agent Execution with Timeout ---
            
            print(f"[{os.getpid()}] Result from Agent for '{company_name}':\n{agent_result_str}")
            agent_json_result = json.loads(agent_result_str)
            
            for key in EXPECTED_JSON_KEYS:
                result_data[key] = agent_json_result.get(key, "null")

        except asyncio.TimeoutError:
            print(f"[{os.getpid()}] Agent execution for '{company_name}' timed out after {AGENT_PROCESSING_TIMEOUT} seconds.", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_PROCESSING_TIMEOUT"
        except json.JSONDecodeError:
            print(f"[{os.getpid()}] Error: Could not decode JSON from agent for '{company_name}'", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_JSON_DECODE_ERROR"
        except Exception as e_agent_run:
            print(f"[{os.getpid()}] Error during agent execution for '{company_name}': {e_agent_run}", file=sys.stderr)
            result_data[EXPECTED_JSON_KEYS[1]] = f"AGENT_EXECUTION_ERROR: {str(e_agent_run)[:50]}"
        
    except Exception as e_outer:
        print(f"[{os.getpid()}] Outer error processing company '{company_name}': {e_outer}", file=sys.stderr)
        if result_data.get(EXPECTED_JSON_KEYS[1], "null") == "null": # Avoid overwriting specific error
            result_data[EXPECTED_JSON_KEYS[1]] = f"PROCESSING_ERROR: {str(e_outer)[:50]}"
    
    finally:
        if client_mcp:
            if hasattr(client_mcp, 'stop_servers'):
                try:
                    print(f"[{os.getpid()}] Attempting to stop MCPClient servers for '{company_name}'...")
                    await client_mcp.stop_servers()
                    print(f"[{os.getpid()}] MCPClient servers stopped for '{company_name}'.")
                except Exception as e_stop_servers:
                    print(f"[{os.getpid()}] Error stopping MCPClient servers for '{company_name}': {e_stop_servers}", file=sys.stderr)
            else:
                print(f"[{os.getpid()}] MCPClient for '{company_name}' does not have a 'stop_servers' method.", file=sys.stderr)
        
        if tmp_profile_dir and tmp_profile_dir.exists():
            print(f"[{os.getpid()}] Attempting to clean up temp profile directory: {tmp_profile_dir} for '{company_name}'")
            print(f"[{os.getpid()}] Attempting to terminate Chrome processes using profile: {tmp_profile_dir}")
            kill_chrome_processes_using(tmp_profile_dir)
            time.sleep(0.5)
            rmtree_with_retry(tmp_profile_dir, attempts=5, delay_seconds=1.0)
        elif tmp_profile_dir:
             print(f"[{os.getpid()}] Temp profile directory {tmp_profile_dir} for '{company_name}' was assigned but does not exist at cleanup time.")
        else:
            print(f"[{os.getpid()}] Temp profile directory was not created for '{company_name}', no cleanup needed for it.")
            
    return result_data

def process_company_data_wrapper(company_name: str, company_number: str):
    """
    Synchronous wrapper for the process_company_data async function.
    """
    return asyncio.run(process_company_data(company_name, company_number))

def console_main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for company information from the newest CSV in the 'input' folder using Brave/Wikidata/Agent and output to CSV, in parallel."
    )
    parser.add_argument("output_csv", help="Path to the output CSV file for results")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use for parallel processing. Defaults to the number of CPU cores."
    )
    args = parser.parse_args()

    input_dir = "input"
    if not os.path.isdir(input_dir):
        print(f"CRITICAL ERROR: Input directory '{input_dir}' not found. Exiting.", file=sys.stderr)
        sys.exit(1)

    list_of_csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not list_of_csv_files:
        print(f"CRITICAL ERROR: No CSV files found in '{input_dir}'. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    input_csv_path = max(list_of_csv_files, key=os.path.getmtime)
    print(f"Using the newest input CSV file: {input_csv_path}")

    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL ERROR: OPENAI_API_KEY is not set. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    if not os.getenv("BRAVE_API_KEY"):
        print("Warning: BRAVE_API_KEY is not set. Brave Search will be skipped.", file=sys.stderr)

    output_rows_buffer = [] 
    header = ['company_number', 'company_name'] + EXPECTED_JSON_KEYS
    output_rows_buffer.append(header)

    companies_to_process_tuples = [] # Store as tuples for starmap
    original_company_info_map = {} # To map results back if order changes or for errors

    try:
        with open(input_csv_path, 'r', encoding='utf-8-sig') as infile: 
            reader = csv.reader(infile)
            try:
                input_header = next(reader) 
                if not (input_header[0].strip().lower() == 'company_number' and \
                        input_header[1].strip().lower() == 'company_name'):
                    print(f"Warning: Input CSV header mismatch. Expected 'company_number, company_name', got '{','.join(input_header)}'.", file=sys.stderr)
            except StopIteration:
                print(f"Error: Input CSV file {input_csv_path} is empty or has no header.", file=sys.stderr)
                sys.exit(1)
            
            for i, row_data in enumerate(reader):
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
                
                companies_to_process_tuples.append((company_name, company_number))
                # Store original info for matching results, using a unique key if company_number might not be unique
                original_company_info_map[(company_name, company_number)] = [company_number, company_name]
            
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during input CSV reading from {input_csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    total_companies_for_pool = len(companies_to_process_tuples)
    if total_companies_for_pool == 0 and not output_rows_buffer: # If only invalid rows were found
        print(f"No valid company data found in {input_csv_path} to process. Exiting.")
        # Write header if output_rows_buffer is just header, or header + invalid rows
        if len(output_rows_buffer) <=1 : # Only header
             output_rows_buffer.append(header) # Ensure header is there if it was only invalid rows

    elif total_companies_for_pool == 0 and len(output_rows_buffer) > 1: # Only invalid rows, header already there
        print(f"No valid company data found in {input_csv_path} to process. Writing invalid rows found.")
        # Proceed to write output_rows_buffer which contains header + invalid rows

    elif total_companies_for_pool > 0 :
        num_workers = min(args.workers, total_companies_for_pool) # Don't use more workers than tasks
        print(f"Using {num_workers} worker processes for {total_companies_for_pool} valid companies.")
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            # starmap_async returns an AsyncResult object.
            # The order of results from .get() will match the order of tasks submitted.
            async_pool_results = pool.starmap_async(process_company_data_wrapper, companies_to_process_tuples)
            
            try:
                # .get() will block until all results are available.
                # No individual timeout here, timeout is handled within process_company_data_wrapper by asyncio.wait_for
                list_of_results_dicts = async_pool_results.get() 
                print("Parallel processing finished. Aggregating results for CSV output.")

                for i, extracted_info_dict in enumerate(list_of_results_dicts):
                    # Get original company_name and company_number based on the order
                    original_company_name, original_company_number = companies_to_process_tuples[i]
                    
                    current_row_data = [original_company_number, original_company_name]
                    for key in EXPECTED_JSON_KEYS:
                        current_row_data.append(extracted_info_dict.get(key, "null"))
                    output_rows_buffer.append(current_row_data)

            except Exception as e_pool: # Catch errors from pool.get() or during result processing
                print(f"Error during multiprocessing pool execution or result retrieval: {e_pool}", file=sys.stderr)
                # Add error entries for all companies that were supposed to be processed by the pool
                for cn, cnum in companies_to_process_tuples:
                    error_row = [cnum, cn] + [f"POOL_ERROR: {str(e_pool)[:50]}"] * len(EXPECTED_JSON_KEYS)
                    output_rows_buffer.append(error_row)
    else: # total_companies_for_pool == 0 and output_rows_buffer only has header
        print(f"No company data found in {input_csv_path} (after header). Exiting.")
        # Ensure output file has at least headers if it was completely empty
        if not output_rows_buffer: output_rows_buffer.append(header)


    # Final write of all buffered rows (header, invalid rows, processed rows, error rows)
    print(f"\nWriting {len(output_rows_buffer)-1 if len(output_rows_buffer)>0 else 0} data entries to {args.output_csv}...")
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(output_rows_buffer) # Write all rows
        print(f"Successfully wrote to {args.output_csv}")
    except IOError as e:
        print(f"Error writing to output CSV {args.output_csv}: {e}", file=sys.stderr)
    
    print(f"\nProcessing complete. All collected data written to {args.output_csv}")

if __name__ == "__main__":
    # It's good practice to set the start method, especially for cross-platform compatibility.
    # 'spawn' is generally safer than 'fork' as it doesn't inherit as much from the parent.
    # It's the default on Windows and macOS (Python 3.8+).
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # This can happen if the start method has already been set,
        # or if the current context doesn't support changing it.
        print("Note: Multiprocessing start method could not be set to 'spawn'. Using default.", file=sys.stderr)
        pass # Continue with the default or already set method.
    
    console_main()
