import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger  # Import the Logger
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI

# --- Imports from the Brave/Wikidata script ---
import httpx
import json
from urllib.parse import urlparse
from typing import List, Dict, Any # Added for type hinting

# Load environment variables from .env file
load_dotenv()

# --- Constants for Brave/Wikidata search ---
SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/w/api.php"
BLACKLIST = {"wikipedia.org", "facebook.com", "twitter.com", "linkedin.com", "pflegeheimvergleich.ch"} # Domains to ignore

# Global variables for API Keys
BRAVE_API_KEY = None
OPENAI_API_KEY = None

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
    # print(f"\n--- LLM Prompt for URL Selection ---\n{prompt_text}\n---------------------------------") # For debugging

    try:
        response = llm.invoke(prompt_text)
        selected_choice = response.content.strip().lower()
        # print(f"LLM raw response for URL selection: '{selected_choice}'") # For debugging

        if selected_choice == "none":
            return None

        if selected_choice.isdigit():
            selected_index = int(selected_choice) - 1
            if 0 <= selected_index < len(search_results):
                chosen_url = search_results[selected_index].get("url")
                print(f"LLM selected URL ({selected_choice}): {chosen_url} for company: {company_name}")
                return chosen_url
            else:
                print(f"LLM selected an out-of-bounds index: {selected_choice}", file=sys.stderr)
                return None
        else:
            # Fallback: If LLM returns something else, try to see if a URL is in its response.
            for i, result in enumerate(search_results):
                if result.get("url") in selected_choice:
                    print(f"LLM selected URL by finding it in a non-numeric response: {result.get('url')}")
                    return result.get("url")
            print(f"LLM response for URL selection was not a valid number or 'None': '{selected_choice}'", file=sys.stderr)
            return None

    except Exception as e:
        print(f"Error during LLM call for URL selection: {e}", file=sys.stderr)
        return None


def get_brave_search_candidates(company: str, count: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches potential candidate URLs from Brave Search API.
    Returns a list of dictionaries, where each dictionary contains 'url', 'title', 'description'.
    """
    global BRAVE_API_KEY
    if not BRAVE_API_KEY:
        print("Error: BRAVE_API_KEY not available for Brave Search.", file=sys.stderr)
        return []

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
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
        # print(f"Brave raw results: {json.dumps(results_json, indent=2)}") # For debugging

        for r in results_json:
            url = r.get("url")
            title = r.get("title")
            description = r.get("description") # Snippet/description

            if not url:
                continue

            parsed_url = urlparse(url)
            host = parsed_url.hostname or ""

            if not host:
                continue

            if any(domain in host for domain in BLACKLIST):
                # print(f"Skipping blacklisted URL: {url}") # For debugging
                continue

            company_main_name_part = company.lower().split(" ")[0].replace(",", "").replace(".", "")
            company_name_no_spaces = company.lower().replace(" ", "").replace(".", "").replace(",", "")
            host_cleaned = host.lower()
            title_cleaned = title.lower() if title else ""
            
            # Light pre-filter: more likely to be relevant if name in host or title
            # This is not a strict filter; the LLM makes the final call.
            pre_filter_match = (
                company_main_name_part in host_cleaned or
                company_name_no_spaces in host_cleaned or
                company_main_name_part in title_cleaned or
                company_name_no_spaces in title_cleaned or
                host_cleaned.startswith(company_main_name_part) or
                host_cleaned.startswith(company_name_no_spaces)
            )
            
            # For a very open approach to the LLM, you might remove the pre_filter_match condition
            # if pre_filter_match:
            candidate_results.append({
                "url": url,
                "title": title,
                "description": description,
                "is_ch_domain": host.endswith(".ch"),
                "company_match_in_host": (company_main_name_part in host_cleaned) or \
                                         (company_name_no_spaces in host_cleaned) or \
                                         (host_cleaned.startswith(company_main_name_part)) or \
                                         (host_cleaned.startswith(company_name_no_spaces))
            })
            # else:
                # print(f"Skipping URL due to weak pre-filter: {url} for company {company}") # For debugging

        candidate_results.sort(key=lambda x: (
            not x["is_ch_domain"],
            not x["company_match_in_host"]
        ))
        print(f"Found {len(candidate_results)} potential candidates for '{company}' from Brave Search.")
        return candidate_results

    except httpx.RequestError as e:
        print(f"Brave Search API request error: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"Brave Search API returned an error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
    except json.JSONDecodeError:
        print("Brave Search API response was not valid JSON.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in get_brave_search_candidates: {e}", file=sys.stderr)
    return []


def get_wikidata_homepage(company: str) -> str | None:
    """Fetches company homepage from Wikidata."""
    params_search = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "de",
        "uselang": "de", # For language of labels in response
        "type": "item",
        "search": company
    }
    try:
        # print(f"Querying Wikidata entities for: '{company}'") # For debugging
        r_search = httpx.get(WIKIDATA_SEARCH, params=params_search, timeout=5.0)
        r_search.raise_for_status()
        search_results_json = r_search.json()
        search_results = search_results_json.get("search", [])

        if not search_results:
            # print(f"Wikidata: No entity found for '{company}'") # For debugging
            return None

        qid = None
        # Try to find a good QID match
        for res in search_results:
            res_label = res.get("label", "").lower()
            # res_desc = res.get("description","").lower() # Could also check description
            res_aliases = [alias.get("value", "").lower() for alias in res.get("aliases", []) if alias.get("value")]
            # print(f"Wikidata candidate: QID={res.get('id')}, Label='{res_label}', Aliases={res_aliases}") # For debugging
            
            # Simple direct match in label or aliases
            if company.lower() == res_label or company.lower() in res_aliases:
                qid = res.get("id")
                # print(f"Wikidata: Selected QID {qid} for '{company}' due to direct match in label/alias.") # For debugging
                break
        
        # Fallback if no direct match, check for partial match or take the first if it has a description (more likely an org)
        if not qid:
            for res in search_results:
                res_label = res.get("label", "").lower()
                if company.lower() in res_label: # Company name is part of the label
                    qid = res.get("id")
                    # print(f"Wikidata: Selected QID {qid} for '{company}' due to partial match in label.") # For debugging
                    break
        
        if not qid and search_results: # Fallback to the first result if still no QID
            # Prefer results with a description, as they are more likely to be organizations
            first_with_desc = next((res.get("id") for res in search_results if res.get("description")), None)
            if first_with_desc:
                qid = first_with_desc
                # print(f"Wikidata: Fell back to first QID with description {qid} for '{company}'.") # For debugging
            else:
                qid = search_results[0].get("id") # Absolute fallback to the very first result
                # print(f"Wikidata: Fell back to very first QID {qid} for '{company}'.") # For debugging


        if not qid:
            # print(f"Wikidata: No suitable QID found for '{company}' after filtering.") # For debugging
            return None

        params_claims = {
            "action": "wbgetclaims",
            "format": "json",
            "entity": qid,
            "property": "P856"  # P856 is the property for "official website"
        }
        # print(f"Wikidata: Fetching P856 for QID {qid}") # For debugging
        r_claims = httpx.get(WIKIDATA_ENTITY, params=params_claims, timeout=5.0)
        r_claims.raise_for_status()
        claims_data = r_claims.json().get("claims", {}).get("P856", [])

        if not claims_data:
            # print(f"Wikidata: No P856 (official website) claim for QID {qid} ({company}).") # For debugging
            return None

        mainsnak = claims_data[0].get("mainsnak")
        if mainsnak and mainsnak.get("datavalue") and isinstance(mainsnak["datavalue"].get("value"), str):
            url = mainsnak["datavalue"]["value"]
            if url.startswith("http://") or url.startswith("https://"):
                parsed_url = urlparse(url)
                if parsed_url.hostname and not any(domain in parsed_url.hostname for domain in BLACKLIST):
                    print(f"Wikidata found URL: {url} for {company} (QID: {qid})")
                    return url
                # else:
                    # print(f"Wikidata URL {url} for {company} was blacklisted.") # For debugging
        # else:
            # print(f"Wikidata: P856 claim for QID {qid} ({company}) had unexpected structure.") # For debugging

    except httpx.RequestError as e:
        print(f"Wikidata API request error: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"Wikidata API returned an error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
    except json.JSONDecodeError:
        print("Wikidata API response was not valid JSON.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in get_wikidata_homepage for {company}: {e}", file=sys.stderr)
    return None


async def main(company_name):
    global BRAVE_API_KEY, OPENAI_API_KEY
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    llm_url_selector = None
    if OPENAI_API_KEY:
        try:
            llm_url_selector = ChatOpenAI(
                model="gpt-4.1-mini", # Or your preferred model like "gpt-3.5-turbo", "gpt-4-turbo"
                temperature=0,
                api_key=OPENAI_API_KEY
            )
            print("LLM for URL selection initialized.")
        except Exception as e:
            print(f"Error initializing LLM for URL selection: {e}. Proceeding without LLM URL selection.", file=sys.stderr)
            OPENAI_API_KEY = None # Ensure it's treated as unavailable
    else:
        print("Warning: OPENAI_API_KEY not set. LLM-based URL selection will be skipped.", file=sys.stderr)

    company_url = None
    source_of_url = "None"

    if BRAVE_API_KEY:
        print(f"\nAttempting to find URL for '{company_name}' using Brave Search...")
        brave_candidates = get_brave_search_candidates(company_name, count=5) # Get top 5 candidates

        if brave_candidates:
            if llm_url_selector:
                print(f"Found {len(brave_candidates)} candidates. Asking LLM to select the best...")
                company_url = select_best_url_with_llm(company_name, brave_candidates, llm_url_selector)
                if company_url:
                    source_of_url = "Brave Search + LLM"
                else:
                    print("LLM did not select a URL. Applying Brave heuristic fallback.")
            else: # LLM not available or failed to initialize
                print("LLM not available for URL selection. Applying Brave heuristic fallback.")

            # Fallback logic if LLM didn't find a URL or wasn't used
            if not company_url:
                # Using the pre-sorted brave_candidates list with heuristics
                ch_official_urls = [c['url'] for c in brave_candidates if c['is_ch_domain'] and c['company_match_in_host']]
                other_official_urls = [c['url'] for c in brave_candidates if not c['is_ch_domain'] and c['company_match_in_host']]
                any_non_blacklisted_urls = [c['url'] for c in brave_candidates] # Already pre-filtered by blacklist

                if ch_official_urls:
                    company_url = ch_official_urls[0]
                    print(f"Found .ch official candidate(s) from Brave (heuristic): {company_url}")
                elif other_official_urls:
                    company_url = other_official_urls[0]
                    print(f"Found other official candidate(s) from Brave (non-.ch heuristic): {company_url}")
                elif any_non_blacklisted_urls:
                    company_url = any_non_blacklisted_urls[0]
                    print(f"Found any non-blacklisted candidate from Brave (heuristic fallback): {company_url}")
                
                if company_url:
                    source_of_url = "Brave Search (heuristic fallback)"
                else:
                    print("Brave heuristic fallback also failed to find a URL.")
        else:
            print("No candidates found from Brave Search.")

    if not company_url:
        status_message = f"Previous method ({source_of_url})" if source_of_url != "None" else "Brave Search skipped or failed"
        print(f"{status_message} did not yield a URL for '{company_name}'. Trying Wikidata...")
        company_url = get_wikidata_homepage(company_name)
        if company_url:
            source_of_url = "Wikidata"

    root_url_for_prompt = "null"
    if company_url:
        print(f"\n==> Final URL identified for '{company_name}': {company_url} (Source: {source_of_url})")
        parsed_found_url = urlparse(company_url)
        root_url_for_prompt = f"{parsed_found_url.scheme}://{parsed_found_url.netloc}"
    else:
        print(f"\n==> Could not find URL for '{company_name}' using available methods.", file=sys.stderr)

    # --- MCP Agent part ---
    mcp_config_path = os.path.join(os.path.dirname(__file__), "startpage_mcp.json")
    if not os.path.exists(mcp_config_path):
        print(f"Error: MCP config file not found at {mcp_config_path}", file=sys.stderr)
        sys.exit(1)
    
    agent_llm = None
    if OPENAI_API_KEY: # Re-check as it might have been nullified if URL LLM failed
        try:
            agent_llm = ChatOpenAI(
                model="gpt-4.1-mini", # Or your agent's preferred model
                temperature=0,
                api_key=OPENAI_API_KEY # Use the global OPENAI_API_KEY
            )
            print("LLM for MCPAgent initialized.")
        except Exception as e:
            print(f"Error initializing LLM for MCPAgent: {e}. Agent cannot run.", file=sys.stderr)
            sys.exit(1) # Agent is critical, so exit
    else:
        print("Error: OPENAI_API_KEY is not available. MCPAgent cannot be initialized.", file=sys.stderr)
        sys.exit(1)

    client = MCPClient.from_config_file(mcp_config_path)
    agent = MCPAgent(llm=agent_llm, client=client, max_steps=30)

    prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.

Die offizielle Webseite für "{company_name}" wurde als "{root_url_for_prompt if company_url else 'nicht gefunden'}" identifiziert (Quelle: {source_of_url}).

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

    print(f"\nÜbermittelter Prompt an den Agenten (Auszug):")
    print(f"Die offizielle Webseite für \"{company_name}\" wurde als \"{root_url_for_prompt if company_url else 'nicht gefunden'}\" identifiziert (Quelle: {source_of_url}).")
    if company_url:
        print(f"1. Öffne diese URL: {root_url_for_prompt}")
    print("-" * 20)

    result = await agent.run(prompt, max_steps=30)
    print(f"\nResult from Agent:\n{result}")


def console_main() -> None:
    """Entry point for the console script."""
    parser = argparse.ArgumentParser(
        description="Search for company information using Brave/Wikidata (with LLM URL selection) and then an agent."
    )
    parser.add_argument("company", help="The company name to search for")
    args = parser.parse_args()

    # load_dotenv() is at the top of the script.
    # The main function handles API key checks.
    asyncio.run(main(args.company))


if __name__ == "__main__":
    console_main()