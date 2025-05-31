import asyncio
import os
import sys
import argparse
import csv
import glob
import json
import multiprocessing
import queue # For queue.Empty exception
import time
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

from search_common import (
    select_best_url_with_llm,
    get_brave_search_candidates,
    get_wikidata_homepage,
    BLACKLIST,
    is_url_relevant_to_company
)

BRAVE_API_KEY_GLOBAL = None
OPENAI_API_KEY_GLOBAL = None

EXPECTED_JSON_KEYS = [
    "official_website", "founded", "Hauptsitz", "Firmenidentifikationsnummer",
    "HauptTelefonnummer", "HauptEmailAdresse", "Geschäftsbericht"
]
# New: Define the status column name
PROCESSING_STATUS_COLUMN = "processing_status"

AGENT_PROCESSING_TIMEOUT = 35  # 35 seconds
Logger.set_debug(1)

async def _process_company_data_internal(company_name: str, company_number: str, brave_api_key: str | None, openai_api_key: str) -> Dict[str, Any]:
    result_data = {key: "null" for key in EXPECTED_JSON_KEYS}
    result_data[PROCESSING_STATUS_COLUMN] = "OK" # Default status

    llm_url_selector = None
    if openai_api_key:
        try:
            llm_url_selector = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=SecretStr(openai_api_key))
        except Exception as e:
            print(f"Error initializing LLM for URL selection for '{company_name}': {e}. Proceeding without LLM URL selection.", file=sys.stderr)
            # Not a fatal error for the whole process, but URL selection might be impacted.
            # Status will be updated later if URL finding fails.
    
    company_url = None
    source_of_url = "None"

    if brave_api_key:
        print(f"\nAttempting to find URL for '{company_name}' using Brave Search...")
        brave_candidates = get_brave_search_candidates(company_name, brave_api_key=brave_api_key, count=10)
        if brave_candidates:
            if llm_url_selector:
                company_url = select_best_url_with_llm(company_name, brave_candidates, llm_url_selector)
                if company_url: source_of_url = "Brave Search + LLM"
            if not company_url: # Fallback or if LLM not used/failed
                ch_urls = [c['url'] for c in brave_candidates if c['is_ch_domain'] and c['company_match_in_host']]
                other_urls = [c['url'] for c in brave_candidates if not c['is_ch_domain'] and c['company_match_in_host']]
                any_urls = [c['url'] for c in brave_candidates]
                if ch_urls: company_url = ch_urls[0]
                elif other_urls: company_url = other_urls[0]
                elif any_urls: company_url = any_urls[0]
                if company_url and source_of_url == "None": source_of_url = "Brave Search (heuristic fallback)"
    
    if not company_url:
        company_url = get_wikidata_homepage(company_name)
        if company_url: source_of_url = "Wikidata"

    root_url_for_prompt = "null"
    if company_url:
        parsed_found_url = urlparse(company_url)
        root_url_for_prompt = f"{parsed_found_url.scheme}://{parsed_found_url.netloc}"
        result_data["official_website"] = root_url_for_prompt
        result_data[PROCESSING_STATUS_COLUMN] = source_of_url # Update status with source
    else:
        result_data["official_website"] = "null"
        result_data[PROCESSING_STATUS_COLUMN] = "NO_URL_FOUND"
        # If no URL, no agent processing needed. Return early.
        print(f"\n==> Could not find URL for '{company_name}'. Status: {result_data[PROCESSING_STATUS_COLUMN]}", file=sys.stderr)
        return result_data

    # Pre-agent relevance check
    async with httpx.AsyncClient() as http_client:
        is_relevant = await is_url_relevant_to_company(root_url_for_prompt, company_name, http_client)
        if not is_relevant:
            error_msg = "PRE_CHECK_URL_MISMATCH"
            print(f"URL '{root_url_for_prompt}' deemed NOT relevant to '{company_name}' by pre-check. Skipping agent.", file=sys.stderr)
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
            # Keep official_website as the URL that failed the check, but other fields remain null.
            return result_data

    # MCP Agent part
    mcp_config_path = os.path.join(os.path.dirname(__file__), "sequential_mcp_config.json")
    if not os.path.exists(mcp_config_path):
        error_msg = "MCP_CONFIG_MISSING"
        result_data[PROCESSING_STATUS_COLUMN] = error_msg
        return result_data
    
    if not openai_api_key: # Should have been caught by llm_url_selector init, but defensive
        error_msg = "AGENT_OPENAI_KEY_MISSING"
        result_data[PROCESSING_STATUS_COLUMN] = error_msg
        return result_data

    agent_llm_for_mcp = None
    try:
        agent_llm_for_mcp = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=SecretStr(openai_api_key))
    except Exception as e:
        error_msg = "AGENT_LLM_INIT_ERROR"
        print(f"Error initializing LLM for MCPAgent for '{company_name}': {e}", file=sys.stderr)
        result_data[PROCESSING_STATUS_COLUMN] = error_msg
        return result_data

    client_mcp = MCPClient.from_config_file(mcp_config_path)
    agent = MCPAgent(llm=agent_llm_for_mcp, client=client_mcp, max_steps=30)

    prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.
Die offizielle Webseite für "{company_name}" wurde als "{root_url_for_prompt}" identifiziert (Quelle: {source_of_url}).
1. Öffne diese URL: {root_url_for_prompt}
2. Durchsuche diese Seite und relevante Unterseiten (z. B. /about, /unternehmen, /impressum, /geschichte, /contact, /legal)
   und sammle die unten genannten Fakten. Achte darauf, die aktuellsten Informationen zu finden.
WICHTIG: WENN DER "{company_name}" NICHT MIT "{root_url_for_prompt}" ZUSAMMENPASST MUSST DU IN ALLE FELDER AUSSER "official_website" NULL SCHREIBEN UND DIE SUCHE ABBRECHEN. "official_website" soll "{root_url_for_prompt}" bleiben.
Fakten zu sammeln:
    • Gründungsjahr (JJJJ)
    • Offizielle Website (die bereits ermittelte Root-URL: "{root_url_for_prompt}")
    • Addresse Hauptsitz (vollständige Adresse)
    • Firmenidentifikationsnummer (meistens im Impressum, z.B. CHE-XXX.XXX.XXX)
    • Haupt-Telefonnummer (internationales Format wenn möglich)
    • Haupt-Emailadresse (allgemeine Kontakt-Email)
    • URL oder PDF-Link des AKTUELLSTEN Geschäftsberichtes/Jahresberichtes (falls öffentlich zugänglich)
Antworte **ausschließlich** mit genau diesem JSON, ohne jeglichen Text davor oder danach:
{{
  "official_website": "{root_url_for_prompt}",
  "founded": "<jahr JJJJ oder null>",
  "Hauptsitz": "<vollständige Adresse oder null>",
  "Firmenidentifikationsnummer": "<ID oder null>",
  "HauptTelefonnummer": "<nummer oder null>",
  "HauptEmailAdresse": "<email oder null>",
  "Geschäftsbericht" : "<url/PDF-Link oder null>"
}}"""

    try:
        agent_result_str = await agent.run(prompt, max_steps=30)
        agent_json_result = json.loads(agent_result_str)
        
        for key in EXPECTED_JSON_KEYS:
            # official_website is already set and should be confirmed by agent or kept as is.
            # Agent's "official_website" is mainly for confirmation.
            if key == "official_website":
                # If agent returns a different URL, it might indicate an issue or a redirect.
                # For now, we trust the initially found root_url_for_prompt.
                # Could add logic here if agent's URL is significantly different.
                pass 
            else:
                result_data[key] = agent_json_result.get(key, "null")
        # If agent run was successful, status remains as the source_of_url or "OK" if not set by source.
        if result_data[PROCESSING_STATUS_COLUMN] == "OK" and source_of_url != "None":
             result_data[PROCESSING_STATUS_COLUMN] = source_of_url
        elif result_data[PROCESSING_STATUS_COLUMN] == "OK": # No specific source, but agent ran
             result_data[PROCESSING_STATUS_COLUMN] = "AGENT_OK"


    except json.JSONDecodeError:
        error_msg = "AGENT_JSON_DECODE_ERROR"
        result_data[PROCESSING_STATUS_COLUMN] = error_msg
    except Exception as e:
        error_msg = f"AGENT_EXECUTION_ERROR: {str(e)[:30]}"
        result_data[PROCESSING_STATUS_COLUMN] = error_msg
    
    # Ensure official_website is not overwritten by an error status if it was found
    if "ERROR" in result_data[PROCESSING_STATUS_COLUMN] or "TIMEOUT" in result_data[PROCESSING_STATUS_COLUMN]:
        if result_data["official_website"] == "null" and root_url_for_prompt != "null":
            result_data["official_website"] = root_url_for_prompt # Keep the URL if error happened after finding it
        elif "ERROR" in result_data["official_website"] or "TIMEOUT" in result_data["official_website"]:
             pass # official_website already contains an error, leave it
        # else official_website is already the correct URL or "null"
    
    return result_data

def process_company_task_wrapper(result_queue: multiprocessing.Queue, company_name: str, company_number: str, brave_api_key: str | None, openai_api_key: str):
    print(f"Subprocess started for {company_name} ({company_number})")
    try:
        result = asyncio.run(_process_company_data_internal(company_name, company_number, brave_api_key, openai_api_key))
        result_queue.put(result)
    except Exception as e:
        error_msg = f"SUBPROCESS_ERROR: {str(e)[:100]}"
        print(f"Critical error in subprocess for {company_name}: {e} ({error_msg})", file=sys.stderr)
        error_result = {key: "null" for key in EXPECTED_JSON_KEYS}
        error_result[PROCESSING_STATUS_COLUMN] = error_msg
        # official_website should be "null" as the error is at subprocess level
        error_result["official_website"] = "null" 
        result_queue.put(error_result)
    finally:
        print(f"Subprocess finished for {company_name} ({company_number})")

def console_main() -> None:
    global BRAVE_API_KEY_GLOBAL, OPENAI_API_KEY_GLOBAL

    parser = argparse.ArgumentParser(description="Search for company information...")
    parser.add_argument("output_csv", help="Path to the output CSV file for results")
    args = parser.parse_args()

    input_dir = "input"
    list_of_csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not list_of_csv_files:
        print(f"CRITICAL ERROR: No CSV files found in '{input_dir}'. Exiting.", file=sys.stderr)
        sys.exit(1)
    input_csv_path = max(list_of_csv_files, key=os.path.getmtime)
    print(f"Using the newest input CSV file: {input_csv_path}")

    BRAVE_API_KEY_GLOBAL = os.getenv("BRAVE_API_KEY")
    OPENAI_API_KEY_GLOBAL = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY_GLOBAL:
        print("CRITICAL ERROR: OPENAI_API_KEY is not set. Exiting.", file=sys.stderr)
        sys.exit(1)

    # New header including the processing status
    header = ['company_number', 'company_name'] + EXPECTED_JSON_KEYS + [PROCESSING_STATUS_COLUMN]
    output_rows_buffer = [header]

    companies_to_process = []
    try:
        with open(input_csv_path, 'r', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            next(reader) # Skip header
            for row in reader:
                companies_to_process.append(row)
    except Exception as e:
        print(f"Error reading input CSV {input_csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    total_companies = len(companies_to_process)
    print(f"Found {total_companies} companies to process from {input_csv_path}.")

    for i, row in enumerate(companies_to_process):
        if len(row) < 2:
            # Ensure row has enough elements for error reporting, matching the new header
            error_entry = ["INVALID_INPUT_ROW"] * len(header)
            if len(row) > 0: error_entry[0] = row[0]
            error_entry[-1] = "INVALID_INPUT_CSV_ROW_STRUCTURE" # Status for this specific error
            output_rows_buffer.append(error_entry)
            continue

        company_number, company_name = row[0].strip(), row[1].strip()
        if not company_name:
            error_entry = [company_number, "EMPTY_COMPANY_NAME"] + ["null"] * len(EXPECTED_JSON_KEYS) + ["EMPTY_COMPANY_NAME_IN_INPUT"]
            output_rows_buffer.append(error_entry)
            continue

        print(f"\n--- Processing company {i+1}/{total_companies}: '{company_name}' ({company_number}) ---")
        
        # Initialize with defaults, including status
        extracted_info_dict = {key: "null" for key in EXPECTED_JSON_KEYS}
        extracted_info_dict[PROCESSING_STATUS_COLUMN] = "PENDING"


        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=process_company_task_wrapper,
            args=(result_queue, company_name, company_number, BRAVE_API_KEY_GLOBAL, OPENAI_API_KEY_GLOBAL)
        )
        process.start()
        process.join(timeout=AGENT_PROCESSING_TIMEOUT)

        current_processing_status = "UNKNOWN_ERROR_MAIN_PROCESS" # Default if not updated

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive(): process.kill(); process.join()
            current_processing_status = "AGENT_PROCESSING_TIMEOUT_MAIN"
            print(f"Processing for '{company_name}' timed out. Status: {current_processing_status}", file=sys.stderr)
            # official_website remains "null" or its last known value if timeout happened late
        else:
            try:
                # This dict from queue should already have PROCESSING_STATUS_COLUMN
                extracted_info_dict = result_queue.get(timeout=5) 
                current_processing_status = extracted_info_dict.get(PROCESSING_STATUS_COLUMN, "STATUS_NOT_IN_QUEUE_RESULT")
                
                if process.exitcode != 0:
                    exit_code_error_msg = f"SUBPROCESS_EXITED_WITH_CODE_{process.exitcode}"
                    print(f"Subprocess for '{company_name}' exited with code {process.exitcode}. Status: {exit_code_error_msg}", file=sys.stderr)
                    # If status isn't already an error, update it
                    if "ERROR" not in current_processing_status.upper() and "TIMEOUT" not in current_processing_status.upper():
                        current_processing_status = exit_code_error_msg
            except queue.Empty:
                current_processing_status = "QUEUE_EMPTY_AFTER_PROCESS_END"
                print(f"Result queue empty for '{company_name}'. Status: {current_processing_status}", file=sys.stderr)
            except Exception as e:
                current_processing_status = f"QUEUE_RETRIEVAL_ERROR_MAIN: {str(e)[:30]}"
                print(f"Error retrieving result from queue for '{company_name}': {e}. Status: {current_processing_status}", file=sys.stderr)
        
        # Ensure the status is correctly set in the dict that will be written
        extracted_info_dict[PROCESSING_STATUS_COLUMN] = current_processing_status
        
        # Construct the row for CSV
        current_row_data = [company_number, company_name]
        for key in EXPECTED_JSON_KEYS:
            current_row_data.append(extracted_info_dict.get(key, "null"))
        current_row_data.append(extracted_info_dict.get(PROCESSING_STATUS_COLUMN, "FINAL_STATUS_MISSING")) # Add status
        output_rows_buffer.append(current_row_data)
        
        if (i + 1) % 5 == 0 or (i + 1) == total_companies:
            print(f"\nSaving progress to {args.output_csv}...")
            try:
                with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(output_rows_buffer)
            except IOError as e:
                print(f"Error writing to output CSV {args.output_csv}: {e}", file=sys.stderr)
    
    print(f"\nProcessing complete. All data written to {args.output_csv}")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    console_main()
