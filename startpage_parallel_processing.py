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
import httpx
import re
from urllib.parse import urlparse
from typing import Dict, Any
import tempfile
from pathlib import Path

load_dotenv()

# search_common imports removed as they are related to the old search method.
# For this Startpage-based version, these specific functions are not needed.

AGENT_PROCESSING_TIMEOUT = 120  # 35 seconds
EXPECTED_JSON_KEYS = [ # These are the data fields the agent is expected to return
    "official_website", "founded", "Hauptsitz", "Firmenidentifikationsnummer",
    "HauptTelefonnummer", "HauptEmailAdresse", "Geschäftsbericht", "extracted_company_name"
]
PROCESSING_STATUS_COLUMN = "processing_status" # Name for the new status column

def rmtree_with_retry(path_to_remove: Path, attempts: int = 5, delay_seconds: float = 1.0):
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

Logger.set_debug(1)

async def process_company_data(company_name: str, company_number: str) -> Dict[str, Any]:
    result_data = {key: "null" for key in EXPECTED_JSON_KEYS}
    # Initial status, agent will try to find URL and then process.
    result_data[PROCESSING_STATUS_COLUMN] = "PENDING_AGENT_PROCESSING"
    tmp_profile_dir = None
    client_mcp = None
    source_of_url = "Startpage Agent" # Default source if agent succeeds

    try:
        openai_api_key_val = os.getenv("OPENAI_API_KEY")
        # LLM for URL selection and Brave/Wikidata search logic removed.
        # URL pre-check logic removed. Agent will handle URL identification and relevance.

        # Agent setup will occur directly.
        # `company_url` and `root_url_for_prompt` will be determined by the agent's output or remain "null".
        # `result_data["official_website"]` will be populated from agent_json_result.

        tmp_profile_dir = Path(tempfile.gettempdir()) / f"mcp_playwright_profile_{os.getpid()}_{time.time_ns()}"
        try:
            tmp_profile_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            error_msg = "TEMP_DIR_CREATION_ERROR"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
            # Re-raise to be caught by outer try-except, which will then use this status
            raise Exception(f"{error_msg}: {e_mkdir}") 

        if not openai_api_key_val:
            error_msg = "AGENT_OPENAI_KEY_MISSING"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
            raise Exception(error_msg)

        agent_llm_for_mcp = None
        try:
            agent_llm_for_mcp = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=SecretStr(openai_api_key_val))
        except Exception as e:
            error_msg = "AGENT_LLM_INIT_FAILURE"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
            raise Exception(f"{error_msg}: {e}")

        # Calculate window position based on PID
        pid = os.getpid()
        base_window_width = 800  # Approximate width
        base_window_height = 600 # Approximate height
        windows_per_row = 2 # Two windows side-by-side
        max_rows_on_screen = 1 # Only one row of windows
        x_padding = 20 # Horizontal padding between windows
        y_padding = 40 # Vertical padding between windows (and for taskbar etc.)

        # Calculate column and row index for the window
        col_index = pid % windows_per_row
        row_index = (pid // windows_per_row) % max_rows_on_screen # Cycle through max_rows_on_screen

        x_pos = col_index * (base_window_width + x_padding)
        y_pos = row_index * (base_window_height + y_padding)

        playwright_config_content = {
            "browser": {
                "userDataDir": str(tmp_profile_dir.resolve()),
                "launchOptions": {
                    "headless": False,
                    "args": [
                        "--disable-breakpad",
                        "--disable-extensions",
                        f"--window-position={x_pos},{y_pos}",
                        f"--window-size={base_window_width},{base_window_height}"
                    ]
                }
            }
        }
        per_process_playwright_config_path = tmp_profile_dir / "runtime-playwright-config.json"
        with open(per_process_playwright_config_path, 'w') as f: json.dump(playwright_config_content, f, indent=2)

        base_mcp_launcher_path = Path(__file__).parent / "parallel_mcp_launcher.json"
        if not base_mcp_launcher_path.exists():
            error_msg = "BASE_MCP_LAUNCHER_MISSING"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
            raise Exception(error_msg)
        with open(base_mcp_launcher_path, 'r') as f: mcp_launcher_template = json.load(f)
        
        if "mcpServers" in mcp_launcher_template and "playwright" in mcp_launcher_template["mcpServers"] and "args" in mcp_launcher_template["mcpServers"]["playwright"]:
            args_list = mcp_launcher_template["mcpServers"]["playwright"]["args"]
            try:
                config_arg_index = args_list.index("--config")
                if config_arg_index + 1 < len(args_list):
                    args_list[config_arg_index + 1] = str(per_process_playwright_config_path.resolve())
                else: raise ValueError("MCP Launcher template '--config' argument is last")
            except ValueError as e_template_val:
                error_msg = "MCP_LAUNCHER_TEMPLATE_CONFIG_ARG_ERROR"
                result_data[PROCESSING_STATUS_COLUMN] = error_msg
                raise Exception(f"{error_msg}: {e_template_val}")
        else:
            error_msg = "MCP_LAUNCHER_TEMPLATE_STRUCTURE_ERROR"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
            raise Exception(error_msg)

        dynamic_mcp_launcher_path = tmp_profile_dir / "runtime-mcp-launcher.json"
        with open(dynamic_mcp_launcher_path, 'w') as f: json.dump(mcp_launcher_template, f, indent=2)
        
        client_mcp = MCPClient.from_config_file(str(dynamic_mcp_launcher_path.resolve()))
        agent = MCPAgent(llm=agent_llm_for_mcp, client=client_mcp, max_steps=30)

        prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen. Deine Aufgabe ist es, Informationen über das Unternehmen "{company_name}" zu finden.
1. Gehe zu https://www.startpage.com.
2. Suche nach dem folgenden Begriff: "{company_name} webseite schweiz".
3. Analysiere die Suchergebnisse und identifiziere die offizielle Webseite des Unternehmens "{company_name}".
   Klicke auf das relevanteste Suchergebnis, um zur Webseite zu gelangen.
   Wenn du keine klare offizielle Webseite findest oder die Ergebnisse nicht eindeutig sind, setze "official_website" auf "null" und gib die anderen Felder ebenfalls als "null" zurück.
4. Wenn du eine Webseite geöffnet hast, durchsuche diese Seite und relevante Unterseiten (z. B. /about, /unternehmen, /impressum, /geschichte, /contact, /legal)
   und sammle die unten genannten Fakten. Achte darauf, die aktuellsten Informationen zu finden.
Fakten zu sammeln:
    • Offizielle Website (die URL der Webseite, die du als offiziell identifiziert hast. Gib die Root-URL an, z.B. https://beispiel.com)
    • Gründungsjahr (JJJJ)
    • Addresse Hauptsitz (vollständige Adresse)
    • Firmenidentifikationsnummer (meistens im Impressum, z.B. CHE-XXX.XXX.XXX)
    • Haupt-Telefonnummer (internationales Format wenn möglich)
    • Haupt-Emailadresse (allgemeine Kontakt-Email)
    • URL oder PDF-Link des AKTUELLSTEN Geschäftsberichtes/Jahresberichtes (falls öffentlich zugänglich)
    • Der auf der Webseite genannte Firmenname (z.B. aus dem Impressum oder der "Über uns" Seite). Dies sollte der reine Textname sein, keine URL.
Antworte **ausschließlich** mit genau diesem JSON, ohne jeglichen Text davor oder danach:
{{
  "official_website": "<URL der gefundenen offiziellen Webseite oder null>",
  "founded": "<jahr JJJJ oder null>",
  "Hauptsitz": "<vollständige Adresse oder null>",
  "Firmenidentifikationsnummer": "<ID oder null>",
  "HauptTelefonnummer": "<nummer oder null>",
  "HauptEmailAdresse": "<email oder null>",
  "Geschäftsbericht" : "<url/PDF-Link oder null>",
  "extracted_company_name": "<reiner Text des auf Webseite gefundenen Firmennamens, keine URL, oder null>"
}}"""

        try:
            agent_task = agent.run(prompt, max_steps=25) # Max steps adjusted for potential search + site navigation
            agent_result_str = await asyncio.wait_for(agent_task, timeout=AGENT_PROCESSING_TIMEOUT)
            agent_json_result = json.loads(agent_result_str)
            
            # Populate all keys from agent result, including official_website
            for key in EXPECTED_JSON_KEYS:
                result_data[key] = agent_json_result.get(key, "null")

            if result_data["official_website"] and result_data["official_website"] != "null":
                # Attempt to normalize the URL to its root form if a valid one was returned
                try:
                    parsed_agent_url = urlparse(result_data["official_website"])
                    if parsed_agent_url.scheme and parsed_agent_url.netloc:
                         result_data["official_website"] = f"{parsed_agent_url.scheme}://{parsed_agent_url.netloc}"
                    else: # If parsing fails to get scheme/netloc, it might be an invalid or partial URL
                         result_data["official_website"] = "null" # Treat as not found
                         result_data[PROCESSING_STATUS_COLUMN] = "AGENT_BAD_URL_FORMAT"
                except Exception: # Catch any error during URL parsing
                    result_data["official_website"] = "null"
                    result_data[PROCESSING_STATUS_COLUMN] = "AGENT_URL_PARSE_ERROR"
            
            if result_data["official_website"] == "null" and result_data[PROCESSING_STATUS_COLUMN] == "PENDING_AGENT_PROCESSING":
                 # If agent explicitly returned null for website, or it was invalidated.
                result_data[PROCESSING_STATUS_COLUMN] = "AGENT_NO_URL_FOUND"
            elif result_data[PROCESSING_STATUS_COLUMN] == "PENDING_AGENT_PROCESSING": # If status hasn't been set to an error by URL parsing
                result_data[PROCESSING_STATUS_COLUMN] = f"{source_of_url} (AGENT_OK)"


        except asyncio.TimeoutError:
            error_msg = "AGENT_PROCESSING_TIMEOUT"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
        except json.JSONDecodeError:
            error_msg = "AGENT_JSON_DECODE_ERROR"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
        except Exception as e_agent_run:
            error_msg = f"AGENT_EXECUTION_ERROR: {str(e_agent_run)[:30]}"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg
        
    except Exception as e_outer:
        # This catches errors from setup before agent.run or if agent.run itself fails non-specifically
        # If status was already set to a specific error, don't overwrite with a generic one.
        if result_data[PROCESSING_STATUS_COLUMN] == "PENDING_AGENT_PROCESSING":
            # Only overwrite if it's still the initial pending status
            error_msg_outer = f"OUTER_PROCESSING_ERROR: {str(e_outer)[:30]}"
            result_data[PROCESSING_STATUS_COLUMN] = error_msg_outer
        print(f"[{os.getpid()}] Outer error processing company '{company_name}': {e_outer}. Final status: {result_data[PROCESSING_STATUS_COLUMN]}", file=sys.stderr)

    finally:
        if client_mcp:
            if hasattr(client_mcp, 'stop_servers'):
                try:
                    await client_mcp.stop_servers()
                except Exception as e_stop_servers:
                    print(f"[{os.getpid()}] Error stopping MCPClient servers for '{company_name}': {e_stop_servers}", file=sys.stderr)
        
        if tmp_profile_dir and tmp_profile_dir.exists():
            kill_chrome_processes_using(tmp_profile_dir)
            time.sleep(0.5) # Give a moment for processes to release locks
            rmtree_with_retry(tmp_profile_dir)
            
    return result_data

def process_company_data_wrapper(company_name: str, company_number: str):
    return asyncio.run(process_company_data(company_name, company_number))

def console_main() -> None:
    parser = argparse.ArgumentParser(description="Parallel company data processor.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes.")
    args = parser.parse_args()

    input_dir = "input"
    list_of_csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not list_of_csv_files: sys.exit(f"CRITICAL ERROR: No CSV files found in '{input_dir}'.")
    input_csv_path = max(list_of_csv_files, key=os.path.getmtime)
    print(f"Using newest input CSV: {input_csv_path}")

    if not os.getenv("OPENAI_API_KEY"): sys.exit("CRITICAL ERROR: OPENAI_API_KEY not set.")

    # Updated header
    header = ['company_number', 'company_name'] + EXPECTED_JSON_KEYS + [PROCESSING_STATUS_COLUMN]
    output_rows_buffer = [header]
    companies_to_process_tuples = []

    try:
        with open(input_csv_path, 'r', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            next(reader) # Skip input header
            for i, row_data in enumerate(reader):
                if len(row_data) < 2:
                    error_entry = ["INVALID_INPUT_ROW"] * len(header)
                    if len(row_data) > 0: error_entry[0] = row_data[0]
                    error_entry[-1] = "INVALID_INPUT_CSV_ROW_STRUCTURE"
                    output_rows_buffer.append(error_entry)
                    continue
                company_number, company_name = row_data[0].strip(), row_data[1].strip()
                if not company_name:
                    error_entry = [company_number, "EMPTY_COMPANY_NAME"] + ["null"] * len(EXPECTED_JSON_KEYS) + ["EMPTY_COMPANY_NAME_IN_INPUT"]
                    output_rows_buffer.append(error_entry)
                    continue
                companies_to_process_tuples.append((company_name, company_number))
    except Exception as e:
        sys.exit(f"Error reading input CSV {input_csv_path}: {e}")

    total_companies_for_pool = len(companies_to_process_tuples)
    if total_companies_for_pool > 0:
        num_workers = min(args.workers, total_companies_for_pool)
        print(f"Using {num_workers} worker processes for {total_companies_for_pool} valid companies.")
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            async_pool_results = pool.starmap_async(process_company_data_wrapper, companies_to_process_tuples)
            try:
                list_of_results_dicts = async_pool_results.get()
                print("Parallel processing finished. Aggregating results.")
                for i, extracted_info_dict in enumerate(list_of_results_dicts):
                    original_company_name, original_company_number = companies_to_process_tuples[i]
                    current_row_data = [original_company_number, original_company_name]
                    for key in EXPECTED_JSON_KEYS:
                        current_row_data.append(extracted_info_dict.get(key, "null"))
                    current_row_data.append(extracted_info_dict.get(PROCESSING_STATUS_COLUMN, "STATUS_KEY_MISSING_FROM_RESULT")) # Add status
                    output_rows_buffer.append(current_row_data)
            except Exception as e_pool:
                print(f"Error during multiprocessing pool execution: {e_pool}", file=sys.stderr)
                for cn, cnum in companies_to_process_tuples: # Log error for all tasks submitted to pool
                    error_row = [cnum, cn] + ["null"] * len(EXPECTED_JSON_KEYS) + [f"POOL_EXECUTION_ERROR: {str(e_pool)[:30]}"]
                    output_rows_buffer.append(error_row)
    
    # Final write
    print(f"\nWriting {len(output_rows_buffer)-1 if len(output_rows_buffer)>0 else 0} data entries to {args.output_csv}...")
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(output_rows_buffer)
        print(f"Successfully wrote to {args.output_csv}")
    except IOError as e:
        print(f"Error writing to output CSV {args.output_csv}: {e}", file=sys.stderr)
    
    print("\nProcessing complete.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # If already set or not applicable
    console_main()
