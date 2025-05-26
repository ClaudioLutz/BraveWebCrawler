import asyncio
import os
import sys
import argparse
import csv # Added for CSV handling
import json # For parsing agent's JSON result
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger
from langchain_openai import ChatOpenAI
import httpx
from urllib.parse import urlparse
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# --- Constants for Brave/Wikidata search ---
SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/w/api.php"
BLACKLIST = {"wikipedia.org", "facebook.com", "twitter.com", "linkedin.com", "pflegeheimvergleich.ch"}

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
Logger.set_debug(2)


def select_best_url_with_llm(company_name: str, search_results: List[Dict[str, Any]], llm: ChatOpenAI) -> str | None:
    """
    Uses a GPT model to select the most promising URL from a list of search results.
    Each item in search_results should be a dict with at least 'url', 'title', and 'description'.
    """
    if not search_results:
        return None

    formatted_results = []
    for i, result in enumerate(search_results):
        formatted_results.append(
            f"{i+1}. URL: {result.get('url', 'N/A')}\n"
            f"   Title: {result.get('title', 'N/A')}\n"
            f"   Description: {result.get('description', 'N/A')}"
        )

    prompt_text = f"""
You are an expert at identifying the official homepage of a company from a list of search engine results.
Given the company name "{company_name}" and the following search results, please select the number corresponding to the URL that is most likely the official homepage.

Consider the URL structure, domain name, title, and description to make your choice.
The official homepage is typically the primary website owned and operated by the company itself, not a news article, directory listing (unless it's a highly official business register), or social media page.

If none of the URLs seem to be the official homepage, or if you are unsure, respond with "None".
Only respond with the number of the best choice or "None".

Search Results:
{chr(10).join(formatted_results)}

Company Name: "{company_name}"
Which number corresponds to the most likely official homepage? Respond with the number only, or "None".
    """
    try:
        response = llm.invoke(prompt_text)
        selected_choice = response.content.strip().lower()

        if selected_choice == "none":
            return None

        if selected_choice.isdigit():
            selected_index = int(selected_choice) - 1
            if 0 <= selected_index < len(search_results):
                chosen_url = search_results[selected_index].get("url")
                print(f"LLM selected URL ({selected_choice}): {chosen_url} for company: {company_name}")
                return chosen_url
            else:
                print(f"LLM selected an out-of-bounds index: {selected_choice} for {company_name}", file=sys.stderr)
                return None
        else:
            for i, result in enumerate(search_results):
                if result.get("url") in selected_choice:
                    print(f"LLM selected URL by finding it in a non-numeric response: {result.get('url')} for {company_name}")
                    return result.get("url")
            print(f"LLM response for URL selection was not a valid number or 'None' for {company_name}: '{selected_choice}'", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error during LLM call for URL selection for {company_name}: {e}", file=sys.stderr)
        return None


def get_brave_search_candidates(company: str, api_key: str, count: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches potential candidate URLs from Brave Search API using the provided api_key.
    Returns a list of dictionaries, where each dictionary contains 'url', 'title', 'description'.
    """
    if not api_key:
        print("Error: BRAVE_API_KEY not provided for Brave Search.", file=sys.stderr)
        return []

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": f'"{company}" homepage official site',
        "count": count,
        "country": "ch",
        "search_lang": "de",
        "spellcheck": "false"
    }

    candidate_results = []
    try:
        print(f"Querying Brave Search for: '{params['q']}' with country '{params['country']}'")
        resp = httpx.get(SEARCH_URL, headers=headers, params=params, timeout=10.0)
        resp.raise_for_status()
        results_json = resp.json().get("web", {}).get("results", [])

        for r in results_json:
            url = r.get("url")
            title = r.get("title")
            description = r.get("description")

            if not url:
                continue
            parsed_url = urlparse(url)
            host = parsed_url.hostname or ""
            if not host or any(domain in host for domain in BLACKLIST):
                continue

            company_main_name_part = company.lower().split(" ")[0].replace(",", "").replace(".", "")
            company_name_no_spaces = company.lower().replace(" ", "").replace(".", "").replace(",", "")
            host_cleaned = host.lower()
            
            candidate_results.append({
                "url": url, "title": title, "description": description,
                "is_ch_domain": host.endswith(".ch"),
                "company_match_in_host": (company_main_name_part in host_cleaned) or \
                                         (company_name_no_spaces in host_cleaned) or \
                                         (host_cleaned.startswith(company_main_name_part)) or \
                                         (host_cleaned.startswith(company_name_no_spaces))
            })
        candidate_results.sort(key=lambda x: (not x["is_ch_domain"], not x["company_match_in_host"]))
        print(f"Found {len(candidate_results)} potential candidates for '{company}' from Brave Search.")
        return candidate_results
    except httpx.RequestError as e:
        print(f"Brave Search API request error for {company}: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"Brave Search API returned an error for {company}: {e.response.status_code} - {e.response.text}", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Brave Search API response was not valid JSON for {company}.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in get_brave_search_candidates for {company}: {e}", file=sys.stderr)
    return []


def get_wikidata_homepage(company: str) -> str | None:
    """Fetches company homepage from Wikidata."""
    params_search = {"action": "wbsearchentities", "format": "json", "language": "de", "uselang": "de", "type": "item", "search": company}
    try:
        r_search = httpx.get(WIKIDATA_SEARCH, params=params_search, timeout=5.0)
        r_search.raise_for_status()
        search_results = r_search.json().get("search", [])
        if not search_results: return None

        qid = None
        for res in search_results:
            res_label = res.get("label", "").lower()
            res_aliases = [alias.get("value", "").lower() for alias in res.get("aliases", []) if alias.get("value")]
            if company.lower() == res_label or company.lower() in res_aliases:
                qid = res.get("id")
                break
        if not qid: # Fallback matching logic
            for res in search_results:
                if company.lower() in res.get("label", "").lower(): qid = res.get("id"); break
        if not qid and search_results:
            first_with_desc = next((res.get("id") for res in search_results if res.get("description")), None)
            qid = first_with_desc or search_results[0].get("id")
        if not qid: return None

        params_claims = {"action": "wbgetclaims", "format": "json", "entity": qid, "property": "P856"}
        r_claims = httpx.get(WIKIDATA_ENTITY, params=params_claims, timeout=5.0)
        r_claims.raise_for_status()
        claims_data = r_claims.json().get("claims", {}).get("P856", [])
        if not claims_data: return None

        mainsnak = claims_data[0].get("mainsnak")
        if mainsnak and mainsnak.get("datavalue") and isinstance(mainsnak["datavalue"].get("value"), str):
            url = mainsnak["datavalue"]["value"]
            if url.startswith(("http://", "https://")):
                parsed_url = urlparse(url)
                if parsed_url.hostname and not any(domain in parsed_url.hostname for domain in BLACKLIST):
                    print(f"Wikidata found URL: {url} for {company} (QID: {qid})")
                    return url
    except httpx.RequestError as e:
        print(f"Wikidata API request error for {company}: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"Wikidata API returned an error for {company}: {e.response.status_code} - {e.response.text}", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Wikidata API response was not valid JSON for {company}.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in get_wikidata_homepage for {company}: {e}", file=sys.stderr)
    return None


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
        brave_candidates = get_brave_search_candidates(company_name, api_key=BRAVE_API_KEY_GLOBAL, count=5)
        if brave_candidates:
            if llm_url_selector:
                print(f"Found {len(brave_candidates)} Brave candidates for '{company_name}'. Asking LLM to select...")
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
  "Hauptsitz": "<text oder null>",
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


def console_main() -> None:
    """Entry point for the console script."""
    global BRAVE_API_KEY_GLOBAL, OPENAI_API_KEY_GLOBAL

    parser = argparse.ArgumentParser(
        description="Search for company information from a CSV using Brave/Wikidata/Agent and output to CSV."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file (columns: company_number,company_name)")
    parser.add_argument("output_csv", help="Path to the output CSV file for results")
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

    for i, row in enumerate(companies_to_process):
        if len(row) < 2:
            print(f"Skipping invalid row {i+2} in input CSV (not enough columns): {row}", file=sys.stderr)
            error_entry = ["INVALID_INPUT_ROW"] * (2 + len(EXPECTED_JSON_KEYS))
            if len(row) > 0: error_entry[0] = row[0]
            output_rows_buffer.append(error_entry)
            continue

        company_number, company_name = row[0].strip(), row[1].strip()
        if not company_name: # Basic check for empty company name
            print(f"Skipping row {i+2} due to empty company_name (company_number: {company_number}).", file=sys.stderr)
            error_entry = [company_number, "EMPTY_COMPANY_NAME"] + ["null"] * len(EXPECTED_JSON_KEYS)
            output_rows_buffer.append(error_entry)
            continue

        print(f"\n--- Processing company {i+1}/{total_companies}: '{company_name}' ({company_number}) ---")
        
        try:
            extracted_info_dict = asyncio.run(process_company_data(company_name, company_number))
            
            current_row_data = [company_number, company_name]
            for key in EXPECTED_JSON_KEYS:
                current_row_data.append(extracted_info_dict.get(key, "null")) # Ensure all keys are added
            output_rows_buffer.append(current_row_data)

        except Exception as e: # Catch-all for unexpected errors during a single company's processing
            print(f"CRITICAL UNHANDLED ERROR processing company '{company_name}': {e}", file=sys.stderr)
            error_entry = [company_number, company_name] + [f"UNHANDLED_ERROR: {str(e)[:100]}"] * len(EXPECTED_JSON_KEYS)
            output_rows_buffer.append(error_entry)
        
        # Incremental write to output CSV
        if (i + 1) % 5 == 0 or (i + 1) == total_companies: # Write every 5 companies or at the very end
            print(f"\nSaving progress: Writing {len(output_rows_buffer)-1} data entries to {args.output_csv}...")
            try:
                with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile: # 'w' to overwrite with full buffer
                    writer = csv.writer(outfile)
                    writer.writerows(output_rows_buffer)
                print(f"Successfully wrote to {args.output_csv}")
            except IOError as e:
                print(f"Error writing to output CSV {args.output_csv}: {e}. Data so far might be lost if script stops.", file=sys.stderr)
    
    print(f"\nProcessing complete. All collected data written to {args.output_csv}")


if __name__ == "__main__":
    console_main()