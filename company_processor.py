import asyncio
import os
import sys
import argparse
import csv
import glob
import json
import multiprocessing
import queue # For queue.Empty exception
import time # For process join timeout, though multiprocessing.Process.join has its own timeout
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger
from langchain_openai import ChatOpenAI
from pydantic.types import SecretStr
import httpx
import re
from urllib.parse import urlparse
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# --- Import functions and constants from search_common ---
from search_common import (
    select_best_url_with_llm,
    get_brave_search_candidates,
    get_wikidata_homepage,
    BLACKLIST,
    is_url_relevant_to_company # Added import
)

# Global variables for API Keys - these will be set ONCE in console_main
# These are still used by the main process to pass to subprocesses.
BRAVE_API_KEY_GLOBAL = None
OPENAI_API_KEY_GLOBAL = None

# Expected keys from the agent's JSON output, used for CSV header and data extraction
EXPECTED_JSON_KEYS = [
    "official_website", "founded", "Hauptsitz", "Firmenidentifikationsnummer",
    "HauptTelefonnummer", "HauptEmailAdresse", "Geschäftsbericht"
]

# Timeout for agent processing per company (in seconds)
AGENT_PROCESSING_TIMEOUT = 60  # 1 minute

# Enable mcp_use debug logging
Logger.set_debug(1)

# is_url_relevant_to_company is now imported from search_common

async def _process_company_data_internal(company_name: str, company_number: str, brave_api_key: str | None, openai_api_key: str) -> Dict[str, Any]:
    """
    Internal version of process_company_data.
    Processes a single company: finds its URL and then uses an agent to extract information.
    Returns a dictionary with keys from EXPECTED_JSON_KEYS.
    Accepts API keys as parameters.
    """
    result_data = {key: "null" for key in EXPECTED_JSON_KEYS}

    llm_url_selector = None
    if openai_api_key:
        try:
            llm_url_selector = ChatOpenAI(
                model="gpt-4.1-mini", temperature=0, api_key=SecretStr(openai_api_key)
            )
            print(f"LLM for URL selection initialized for '{company_name}'.")
        except Exception as e:
            print(f"Error initializing LLM for URL selection for '{company_name}': {e}. Proceeding without LLM URL selection.", file=sys.stderr)
            llm_url_selector = None
    else: # Should not happen if main process checks, but defensive
        print(f"Critical: openai_api_key not provided to _process_company_data_internal for '{company_name}'.", file=sys.stderr)
        result_data[EXPECTED_JSON_KEYS[1]] = "INTERNAL_OPENAI_KEY_MISSING"
        return result_data


    company_url = None
    source_of_url = "None"

    if brave_api_key:
        print(f"\nAttempting to find URL for '{company_name}' using Brave Search...")
        brave_candidates = get_brave_search_candidates(company_name, brave_api_key=brave_api_key, count=10)
        print(f"Brave search candidates for '{company_name}': {brave_candidates}")
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

    if root_url_for_prompt != "null":
        async with httpx.AsyncClient() as http_client:
            print(f"Performing pre-agent relevance check for URL: {root_url_for_prompt} for company: {company_name}")
            is_relevant = await is_url_relevant_to_company(root_url_for_prompt, company_name, http_client)
            if not is_relevant:
                print(f"URL '{root_url_for_prompt}' deemed NOT relevant to '{company_name}' by pre-check. Skipping agent.", file=sys.stderr)
                return result_data
            else:
                print(f"URL '{root_url_for_prompt}' deemed RELEVANT to '{company_name}' by pre-check. Proceeding with agent.")

    mcp_config_path = os.path.join(os.path.dirname(__file__), "sequential_mcp_config.json")
    if not os.path.exists(mcp_config_path):
        print(f"Error: MCP config file not found at {mcp_config_path} for '{company_name}'", file=sys.stderr)
        result_data[EXPECTED_JSON_KEYS[1]] = "MCP_CONFIG_MISSING"
        return result_data
    
    if not openai_api_key: # Redundant check, but good for safety within this function
        print(f"Error: openai_api_key is not available. MCPAgent cannot be initialized for '{company_name}'.", file=sys.stderr)
        result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_OPENAI_KEY_MISSING"
        return result_data

    agent_llm = None
    try:
        agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=SecretStr(openai_api_key))
        print(f"LLM for MCPAgent initialized for '{company_name}'.")
    except Exception as e:
        print(f"Error initializing LLM for MCPAgent for '{company_name}': {e}. Agent cannot run.", file=sys.stderr)
        result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_LLM_INIT_ERROR"
        return result_data

    client = MCPClient.from_config_file(mcp_config_path)
    agent = MCPAgent(llm=agent_llm, client=client, max_steps=30)

    prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.

Die offizielle Webseite für "{company_name}" wurde als "{root_url_for_prompt}" identifiziert (Quelle: {source_of_url}).

Wenn eine URL ({root_url_for_prompt}) vorhanden ist und nicht 'null' oder 'nicht gefunden' lautet:
1. Öffne diese URL: {root_url_for_prompt}
2. Durchsuche diese Seite und relevante Unterseiten (z. B. /about, /unternehmen, /impressum, /geschichte, /contact, /legal)
   und sammle die unten genannten Fakten. Achte darauf, die aktuellsten Informationen zu finden.

Wenn KEINE URL gefunden wurde (d.h. als "{root_url_for_prompt}" angegeben ist) ODER Informationen auf der Webseite nicht auffindbar sind, gib für die entsprechenden Felder **null** zurück.

WICHTIG: WENN DER "{company_name}" NICHT MIT "{root_url_for_prompt}" ZUSAMMENPASST MUSST DU IN ALLE FELDER NULL SCHREIBEN UND DIE SUCHE ABBRECHEN.

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
    print(f"Die offizielle Webseite für \"{company_name}\" wurde als \"{root_url_for_prompt}\" identifiziert (Quelle: {source_of_url}).")
    if company_url: print(f"1. Öffne diese URL: {root_url_for_prompt}")
    print("-" * 20)

    try:
        agent_result_str = await agent.run(prompt, max_steps=30)
        print(f"\nResult from Agent for '{company_name}':\n{agent_result_str}")
        agent_json_result = json.loads(agent_result_str)
        
        for key in EXPECTED_JSON_KEYS:
            result_data[key] = agent_json_result.get(key, "null")
        return result_data

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from agent for '{company_name}'", file=sys.stderr)
        result_data[EXPECTED_JSON_KEYS[1]] = "AGENT_JSON_DECODE_ERROR"
    except Exception as e:
        print(f"Error during agent execution for '{company_name}': {e}", file=sys.stderr)
        result_data[EXPECTED_JSON_KEYS[1]] = f"AGENT_EXECUTION_ERROR: {str(e)[:50]}"
    return result_data

def process_company_task_wrapper(result_queue: multiprocessing.Queue, company_name: str, company_number: str, brave_api_key: str | None, openai_api_key: str):
    """
    Wrapper function to run _process_company_data_internal in a separate process
    and put the result into a queue.
    """
    print(f"Subprocess started for {company_name} ({company_number})")
    try:
        # Ensure a new event loop for the subprocess if not managed by asyncio.run
        # For newer Python versions, asyncio.run manages this.
        # If issues arise, one might need:
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # result = loop.run_until_complete(_process_company_data_internal(...))
        # loop.close()
        
        result = asyncio.run(_process_company_data_internal(company_name, company_number, brave_api_key, openai_api_key))
        result_queue.put(result)
    except Exception as e:
        print(f"Critical error in subprocess for {company_name}: {e}", file=sys.stderr)
        # Put an error indication in the queue
        error_result = {key: "null" for key in EXPECTED_JSON_KEYS}
        error_result[EXPECTED_JSON_KEYS[1]] = f"SUBPROCESS_ERROR: {str(e)[:100]}"
        result_queue.put(error_result)
    finally:
        print(f"Subprocess finished for {company_name} ({company_number})")


def console_main() -> None:
    """Entry point for the console script."""
    global BRAVE_API_KEY_GLOBAL, OPENAI_API_KEY_GLOBAL # Still needed to read them initially

    parser = argparse.ArgumentParser(
        description="Search for company information from the newest CSV in the 'input' folder using Brave/Wikidata/Agent and output to CSV."
    )
    parser.add_argument("output_csv", help="Path to the output CSV file for results")
    args = parser.parse_args()

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

    BRAVE_API_KEY_GLOBAL = os.getenv("BRAVE_API_KEY")
    OPENAI_API_KEY_GLOBAL = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY_GLOBAL:
        print("CRITICAL ERROR: OPENAI_API_KEY is not set in .env or environment. The agent cannot run. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    if not BRAVE_API_KEY_GLOBAL:
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
            
    except FileNotFoundError:
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

    for i, row in enumerate(companies_to_process):
        if len(row) < 2:
            print(f"Skipping invalid row {i+2} in input CSV (not enough columns): {row}", file=sys.stderr)
            error_entry = ["INVALID_INPUT_ROW"] * (2 + len(EXPECTED_JSON_KEYS))
            if len(row) > 0: error_entry[0] = row[0]
            output_rows_buffer.append(error_entry)
            continue

        company_number, company_name = row[0].strip(), row[1].strip()
        if not company_name:
            print(f"Skipping row {i+2} due to empty company_name (company_number: {company_number}).", file=sys.stderr)
            error_entry = [company_number, "EMPTY_COMPANY_NAME"] + ["null"] * len(EXPECTED_JSON_KEYS)
            output_rows_buffer.append(error_entry)
            continue

        print(f"\n--- Processing company {i+1}/{total_companies}: '{company_name}' ({company_number}) ---")
        
        extracted_info_dict = {key: "null" for key in EXPECTED_JSON_KEYS} # Default error/timeout state

        # Pass the actual key values to the subprocess wrapper
        # OPENAI_API_KEY_GLOBAL is checked for existence at the start of console_main
        current_openai_api_key = OPENAI_API_KEY_GLOBAL 
        current_brave_api_key = BRAVE_API_KEY_GLOBAL # Can be None

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=process_company_task_wrapper,
            args=(result_queue, company_name, company_number, current_brave_api_key, current_openai_api_key)
        )
        process.start()
        process.join(timeout=AGENT_PROCESSING_TIMEOUT)

        if process.is_alive():
            print(f"Processing for '{company_name}' timed out after {AGENT_PROCESSING_TIMEOUT} seconds. Terminating process.", file=sys.stderr)
            process.terminate()
            process.join(timeout=5) # Give it a moment to terminate
            if process.is_alive(): # Force kill if still alive
                 print(f"Process for '{company_name}' did not terminate gracefully, attempting to kill.", file=sys.stderr)
                 process.kill() # SIGKILL
                 process.join() 
            extracted_info_dict[EXPECTED_JSON_KEYS[1]] = "AGENT_PROCESSING_TIMEOUT"
        else:
            try:
                # Get result from queue, with a small timeout in case process died before putting anything
                extracted_info_dict = result_queue.get(timeout=5) 
                if process.exitcode != 0:
                    print(f"Subprocess for '{company_name}' exited with code {process.exitcode}.", file=sys.stderr)
                    # If result from queue doesn't already indicate an error, mark one
                    if not any("ERROR" in str(val).upper() for val in extracted_info_dict.values()):
                         extracted_info_dict[EXPECTED_JSON_KEYS[1]] = f"SUBPROCESS_EXITED_WITH_CODE_{process.exitcode}"

            except queue.Empty:
                print(f"Result queue empty for '{company_name}' after process finished. Assuming error.", file=sys.stderr)
                extracted_info_dict[EXPECTED_JSON_KEYS[1]] = "QUEUE_EMPTY_AFTER_PROCESS_END"
            except Exception as e:
                print(f"Error retrieving result from queue for '{company_name}': {e}", file=sys.stderr)
                extracted_info_dict[EXPECTED_JSON_KEYS[1]] = f"QUEUE_RETRIEVAL_ERROR: {str(e)[:50]}"
        
        current_row_data = [company_number, company_name]
        for key in EXPECTED_JSON_KEYS:
            current_row_data.append(extracted_info_dict.get(key, "null"))
        output_rows_buffer.append(current_row_data)
        
        if (i + 1) % 5 == 0 or (i + 1) == total_companies:
            print(f"\nSaving progress: Writing {len(output_rows_buffer)-1} data entries to {args.output_csv}...")
            try:
                with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(output_rows_buffer)
                print(f"Successfully wrote to {args.output_csv}")
            except IOError as e:
                print(f"Error writing to output CSV {args.output_csv}: {e}. Data so far might be lost if script stops.", file=sys.stderr)
    
    print(f"\nProcessing complete. All collected data written to {args.output_csv}")

if __name__ == "__main__":
    # For multiprocessing to work correctly on all platforms, especially Windows
    multiprocessing.freeze_support() 
    # It's also good practice to set the start method if not default, though 'spawn' (default on Win) or 'fork' (default on Unix)
    # might have different implications for resource sharing. 'spawn' is generally safer.
    # try:
    #     multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass # If already set or not applicable
    console_main()
