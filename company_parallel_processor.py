import asyncio
import os
import sys
import argparse
import csv # Added for CSV handling
import json # For parsing agent's JSON result
import multiprocessing # Added for parallel processing
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger
from langchain_openai import ChatOpenAI
# import httpx # Moved to search_common
from urllib.parse import urlparse # Still needed
# from typing import List, Dict, Any # Moved to search_common

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

# Global variables for API Keys - these will be set ONCE in console_main
BRAVE_API_KEY_GLOBAL = None
OPENAI_API_KEY_GLOBAL = None

# Expected keys from the agent's JSON output, used for CSV header and data extraction
EXPECTED_JSON_KEYS = [
    "official_website", "ceo", "founder", "owner", "employees", "founded",
    "better_then_the_rest", "Hauptsitz", "Firmenidentifikationsnummer",
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
    # Initialize a default dictionary for returning data, especially in case of errors
    # official_website will be updated if found, otherwise remains "null" (string)
    result_data = {key: "null" for key in EXPECTED_JSON_KEYS}

    llm_url_selector = None
    if OPENAI_API_KEY_GLOBAL:
        try:
            llm_url_selector = ChatOpenAI(
                model="gpt-4.1-mini", temperature=0, api_key=OPENAI_API_KEY_GLOBAL
            )
            print(f"LLM for URL selection initialized for '{company_name}'.")
        except Exception as e:
            print(f"Error initializing LLM for URL selection for '{company_name}': {e}. Proceeding without LLM URL selection.", file=sys.stderr)
            llm_url_selector = None # Ensure it's None
    else:
        print(f"Warning: OPENAI_API_KEY not set. LLM-based URL selection will be skipped for '{company_name}'.", file=sys.stderr)

    company_url = None
    source_of_url = "None"

    if BRAVE_API_KEY_GLOBAL:
        print(f"\nAttempting to find URL for '{company_name}' using Brave Search...")
        # Use imported function, pass BRAVE_API_KEY_GLOBAL as brave_api_key argument
        brave_candidates = get_brave_search_candidates(company_name, brave_api_key=BRAVE_API_KEY_GLOBAL, count=5)
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
    mcp_config_path = os.path.join(os.path.dirname(__file__), "startpage_mcp.json")
    if not os.path.exists(mcp_config_path):
        print(f"Error: MCP config file not found at {mcp_config_path} for '{company_name}'", file=sys.stderr)
        result_data["ceo"] = "MCP_CONFIG_MISSING" # Indicate error in a field
        return result_data
    
    if not OPENAI_API_KEY_GLOBAL: # Agent critically needs this
        print(f"Error: OPENAI_API_KEY is not available. MCPAgent cannot be initialized for '{company_name}'.", file=sys.stderr)
        result_data["ceo"] = "AGENT_OPENAI_KEY_MISSING"
        return result_data

    agent_llm = None
    try:
        agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=OPENAI_API_KEY_GLOBAL)
        print(f"LLM for MCPAgent initialized for '{company_name}'.")
    except Exception as e:
        print(f"Error initializing LLM for MCPAgent for '{company_name}': {e}. Agent cannot run.", file=sys.stderr)
        result_data["ceo"] = "AGENT_LLM_INIT_ERROR"
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

Fakten zu sammeln:
    • Aktueller CEO / Geschäftsführer
    • Gründer (Komma-getrennt bei mehreren)
    • Inhaber (Besitzer der Firma)
    • Aktuelle Mitarbeiterzahl (Zahl oder Bereich, z. B. "200-250", "ca. 500")
    • Gründungsjahr (JJJJ)
    • Offizielle Website (die bereits ermittelte Root-URL: "{root_url_for_prompt}")
    • Was macht diese Firma besser als ihre Konkurrenz (Stichworte, maximal 10 Wörter)
    • Addresse Hauptsitz (vollständige Adresse)
    • Firmenidentifikationsnummer (meistens im Impressum, z.B. CHE-XXX.XXX.XXX oder HRB XXXXX etc.)
    • Haupt-Telefonnummer (internationales Format wenn möglich)
    • Haupt-Emailadresse (allgemeine Kontakt-Email)
    • URL oder PDF-Link des AKTUELLSTEN Geschäftsberichtes/Jahresberichtes (falls öffentlich zugänglich)

Antworte **ausschließlich** mit genau diesem JSON, ohne jeglichen Text davor
oder danach:

{{
  "official_website": "{root_url_for_prompt}",
  "ceo": "<name oder null>",
  "founder": "<name(s) oder null>",
  "owner": "<name(s) oder null>",
  "employees": "<zahl/bereich oder null>",
  "founded": "<jahr oder null>",
  "better_then_the_rest": "<text oder null>",
  "Hauptsitz": "<Strasse Nr, Postleitzahl, Ort>",
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
            result_data[key] = agent_json_result.get(key, "null") # Use agent's value or default
        return result_data

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from agent for '{company_name}'", file=sys.stderr)
        result_data["ceo"] = "AGENT_JSON_DECODE_ERROR" # Indicate error
    except Exception as e:
        print(f"Error during agent execution for '{company_name}': {e}", file=sys.stderr)
        result_data["ceo"] = f"AGENT_EXECUTION_ERROR: {str(e)[:50]}"
    return result_data


def process_company_data_wrapper(company_name: str, company_number: str):
    """
    Synchronous wrapper for the process_company_data async function.
    """
    return asyncio.run(process_company_data(company_name, company_number))


def console_main() -> None:
    """Entry point for the console script."""
    global BRAVE_API_KEY_GLOBAL, OPENAI_API_KEY_GLOBAL

    parser = argparse.ArgumentParser(
        description="Search for company information from a CSV using Brave/Wikidata/Agent and output to CSV."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file (columns: company_number,company_name)")
    parser.add_argument("output_csv", help="Path to the output CSV file for results")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use for parallel processing. Defaults to the number of CPU cores."
    )
    args = parser.parse_args()

    # load_dotenv() is at the top of the script.
    BRAVE_API_KEY_GLOBAL = os.getenv("BRAVE_API_KEY")
    OPENAI_API_KEY_GLOBAL = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY_GLOBAL:
        print("CRITICAL ERROR: OPENAI_API_KEY is not set in .env or environment. The agent cannot run. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    if not BRAVE_API_KEY_GLOBAL:
        print("Warning: BRAVE_API_KEY is not set. Brave Search functionality will be skipped.", file=sys.stderr)

    output_rows_buffer = [] # Holds all rows to be written
    header = ['company_number', 'company_name'] + EXPECTED_JSON_KEYS
    output_rows_buffer.append(header)

    companies_to_process = []
    try:
        with open(args.input_csv, 'r', encoding='utf-8-sig') as infile: # utf-8-sig handles potential BOM
            reader = csv.reader(infile)
            try:
                input_header = next(reader) # Read and store header
                # Basic validation of input CSV header
                if not (input_header[0].strip().lower() == 'company_number' and \
                        input_header[1].strip().lower() == 'company_name'):
                    print(f"Warning: Input CSV header mismatch. Expected 'company_number, company_name', got '{','.join(input_header)}'. Assuming correct column order.", file=sys.stderr)
            except StopIteration:
                print(f"Error: Input CSV file {args.input_csv} is empty or has no header.", file=sys.stderr)
                sys.exit(1)
            
            for row in reader:
                companies_to_process.append(row)
            
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_csv}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during input CSV reading: {e}", file=sys.stderr)
        sys.exit(1)

    total_companies = len(companies_to_process)
    if total_companies == 0:
        print(f"No company data found in {args.input_csv} (after header). Exiting.")
        # Create an empty output file with just headers
        try:
            with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
            print(f"Empty output CSV with headers written to {args.output_csv}")
        except IOError as e:
            print(f"Error writing empty output CSV to {args.output_csv}: {e}", file=sys.stderr)
        sys.exit(0)
        
    print(f"Found {total_companies} companies to process from {args.input_csv}.")

    # --- Parallel Processing with multiprocessing.Pool ---
    if total_companies > 0:
        num_workers = args.workers # Use the value from command-line argument or its default
        print(f"Using {num_workers} worker processes.")
        print(f"Starting parallel processing of {total_companies} companies with {num_workers} workers...")
        
        # Arguments for starmap_async need to be an iterable of argument tuples.
        # companies_to_process is already a list of [company_number, company_name]
        # We need to transform it to [(company_name, company_number), ...] for process_company_data_wrapper
        
        # Correctly prepare arguments for starmap_async:
        # Each element in processed_args should be a tuple (company_name, company_number)
        # that matches the arguments of process_company_data_wrapper.
        # The original companies_to_process has [company_number, company_name].
        processed_args = []
        valid_companies_for_processing = [] # To keep track of companies that are valid for processing

        for i, row_data in enumerate(companies_to_process):
            if len(row_data) < 2:
                print(f"Skipping invalid row {i+2} in input CSV (not enough columns): {row_data}", file=sys.stderr)
                error_entry = ["INVALID_INPUT_ROW"] * (2 + len(EXPECTED_JSON_KEYS))
                if len(row_data) > 0: error_entry[0] = row_data[0]
                output_rows_buffer.append(error_entry) # Add error row directly to buffer
                continue

            company_number, company_name = row_data[0].strip(), row_data[1].strip()
            if not company_name:
                print(f"Skipping row {i+2} due to empty company_name (company_number: {company_number}).", file=sys.stderr)
                error_entry = [company_number, "EMPTY_COMPANY_NAME"] + ["null"] * len(EXPECTED_JSON_KEYS)
                output_rows_buffer.append(error_entry) # Add error row directly to buffer
                continue
            
            # If valid, add to processed_args for the pool and keep original data for re-association
            processed_args.append((company_name, company_number)) # Correct order for wrapper
            valid_companies_for_processing.append([company_number, company_name]) # Store original for matching results

        if processed_args: # Only proceed with pool if there are valid companies
            with multiprocessing.Pool(processes=num_workers) as pool:
                async_results = pool.starmap_async(process_company_data_wrapper, processed_args)
                results = async_results.get()
            
            print("Parallel processing finished. Preparing results for CSV output.")

            # Reconstruct output_rows_buffer with the results
            # The header is already in output_rows_buffer
            # Each item in 'results' is the dictionary returned by process_company_data_wrapper
            # Match results with valid_companies_for_processing
            for i, extracted_info_dict in enumerate(results):
                company_number, company_name = valid_companies_for_processing[i] # Get original company_number and company_name
                
                current_row_data = [company_number, company_name]
                for key in EXPECTED_JSON_KEYS:
                    current_row_data.append(extracted_info_dict.get(key, "null"))
                output_rows_buffer.append(current_row_data)
        else:
            print("No valid companies to process after filtering.")
            results = [] # Ensure results is defined if no valid companies

    else: # Handle case where there were no companies to process (total_companies == 0)
        results = [] # Ensure results is defined
        print("No companies were processed as the input file was empty or contained no data rows.")

    # --- Final CSV Writing ---
    # This block writes all data (including headers, valid results, and error rows for invalid inputs) at once.
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
    # Set start method for multiprocessing for compatibility (especially on macOS/Windows)
    # This should be done only in the __main__ block.
    # 'spawn' is generally safer and more compatible than 'fork'.
    multiprocessing.set_start_method('spawn', force=True) 
    console_main()
