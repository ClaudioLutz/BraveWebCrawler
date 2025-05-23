import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger  # Import the Logger
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI

# --- Imports from the Brave/Wikidata script ---
import httpx
import json
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# --- Constants for Brave/Wikidata search ---
# API_KEY will be loaded from os.getenv in the main function
SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/w/api.php"
BLACKLIST = {"wikipedia.org", "facebook.com", "twitter.com", "linkedin.com"} # Domains to ignore

# Global variable for Brave API Key
BRAVE_API_KEY = None

# Enable mcp_use debug logging
Logger.set_debug(2)

def is_official_candidate(url: str, company: str) -> bool:
    """Checks if a URL is a potential official website."""
    host = urlparse(url).hostname or ""
    # Not blacklisted
    if any(domain in host for domain in BLACKLIST):
        return False
    # Prefer if company name is in the host
    return company.lower() in host.lower()

def get_brave_homepage(company: str, count: int = 10) -> str | None:
    """Fetches company homepage using Brave Search API, prioritizing .ch domains."""
    global BRAVE_API_KEY
    if not BRAVE_API_KEY:
        print("Error: BRAVE_API_KEY not available for Brave Search.", file=sys.stderr)
        return None

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    # Refined query parameters for better regional results
    params = {
        "q": f'"{company}" offizielle Webseite', # "Schweiz" for context
        "count": count,
        "country": "ch",        # Prioritize Swiss results
        "search_lang": "de"     # Prioritize German language results (common in CH)
    }

    try:
        resp = httpx.get(SEARCH_URL, headers=headers, params=params, timeout=10.0)
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])

        ch_official_urls = []
        other_official_urls = []
        any_non_blacklisted_urls = []

        for r in results:
            url = r.get("url", "")
            if not url:
                continue

            parsed_url = urlparse(url)
            host = parsed_url.hostname or ""

            if not host:  # Skip if no hostname
                continue

            # Check blacklist first
            if any(domain in host for domain in BLACKLIST):
                continue

            is_ch_domain = host.endswith(".ch")

            # Enhanced company name matching in host:
            # Takes the first word of the company name (e.g., "Migros" from "Migros Genossenschaft")
            # and a version with spaces removed for matching.
            company_main_name = company.lower().split(" ")[0].replace(",", "").replace(".", "")
            company_name_no_spaces = company.lower().replace(" ", "").replace(".", "") # also remove dots
            host_cleaned = host.lower()

            company_match_in_host = (company_main_name in host_cleaned) or \
                                    (company_name_no_spaces in host_cleaned) or \
                                    (host_cleaned.startswith(company_main_name)) or \
                                    (host_cleaned.startswith(company_name_no_spaces))


            added_to_priority_list = False
            if is_ch_domain and company_match_in_host:
                ch_official_urls.append(url)
                added_to_priority_list = True
            elif company_match_in_host:  # Not .ch, but company name in host
                other_official_urls.append(url)
                added_to_priority_list = True
            
            if not added_to_priority_list:
                # Add to this list only if not already in a higher-priority one and host exists
                any_non_blacklisted_urls.append(url)

        # Return based on priority
        if ch_official_urls:
            print(f"Found .ch official candidate(s) from Brave: {ch_official_urls[0]} (using this one)")
            return ch_official_urls[0]

        if other_official_urls:
            print(f"Found other official candidate(s) from Brave (non-.ch): {other_official_urls[0]} (using this one)")
            return other_official_urls[0]
        
        if any_non_blacklisted_urls:
            print(f"Found any non-blacklisted, non-official candidate from Brave as fallback: {any_non_blacklisted_urls[0]} (using this one)")
            return any_non_blacklisted_urls[0]

    except httpx.RequestError as e:
        print(f"Brave Search API request error: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"Brave Search API returned an error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
    except json.JSONDecodeError:
        print("Brave Search API response was not valid JSON.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in get_brave_homepage: {e}", file=sys.stderr)
    return None

def get_wikidata_homepage(company: str) -> str | None:
    """Fetches company homepage from Wikidata."""
    # 1) Find the entity
    params_search = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "de",
        "country": "ch",  
        "search": company
    }
    try:
        r_search = httpx.get(WIKIDATA_SEARCH, params=params_search, timeout=5.0)
        r_search.raise_for_status()
        search_results = r_search.json().get("search", [])
        if not search_results:
            return None
        
        qid = search_results[0].get("id")
        if not qid:
            return None
            
        # 2) Fetch P856 (official website property) for that entity
        params_claims = {
            "action": "wbgetclaims",
            "format": "json",
            "entity": qid,
            "property": "P856" # P856 is the property for "official website"
        }
        r_claims = httpx.get(WIKIDATA_ENTITY, params=params_claims, timeout=5.0)
        r_claims.raise_for_status()
        claims_data = r_claims.json().get("claims", {}).get("P856", [])
        
        if not claims_data:
            return None
        
        # Ensure the claim has the expected structure
        mainsnak = claims_data[0].get("mainsnak")
        if mainsnak and mainsnak.get("datavalue") and isinstance(mainsnak["datavalue"].get("value"), str):
            url = mainsnak["datavalue"]["value"]
            # Basic validation that it's a URL
            if url.startswith("http://") or url.startswith("https://"):
                 # Check against blacklist
                parsed_url = urlparse(url)
                if parsed_url.hostname and not any(domain in parsed_url.hostname for domain in BLACKLIST):
                    return url
        
    except httpx.RequestError as e:
        print(f"Wikidata API request error: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"Wikidata API returned an error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
    except json.JSONDecodeError:
        print("Wikidata API response was not valid JSON.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in get_wikidata_homepage: {e}", file=sys.stderr)
    return None

async def main(company_name):
    global BRAVE_API_KEY
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY") # Load Brave API Key

    if not BRAVE_API_KEY:
        print("Warning: BRAVE_API_KEY is not set. Brave Search will be skipped.", file=sys.stderr)

    company_url = None
    if BRAVE_API_KEY: # Only attempt Brave search if API key is present
        print(f"Attempting to find URL for '{company_name}' using Brave Search...")
        company_url = get_brave_homepage(company_name)

    if not company_url:
        status_message = "Brave Search did not find URL" if BRAVE_API_KEY else "Brave Search skipped"
        print(f"{status_message} for '{company_name}'. Trying Wikidata...")
        company_url = get_wikidata_homepage(company_name)

    root_url_for_prompt = "null"
    if company_url:
        print(f"Found URL: {company_url}")
        parsed_found_url = urlparse(company_url)
        root_url_for_prompt = f"{parsed_found_url.scheme}://{parsed_found_url.netloc}"
    else:
        print(f"Could not find URL for '{company_name}' using available methods.", file=sys.stderr)

    # Create MCPClient from config file
    mcp_config_path = os.path.join(os.path.dirname(__file__), "startpage_mcp.json")
    if not os.path.exists(mcp_config_path):
        print(f"Error: MCP config file not found at {mcp_config_path}", file=sys.stderr)
        sys.exit(1)
    client = MCPClient.from_config_file(mcp_config_path)
    
    # Initialize the OpenAI client
    # Ensure OPENAI_API_KEY is set in your .env file
    llm = ChatOpenAI(
        model="gpt-4.1-mini", # Keeping your specified model name
        temperature=0
    )
    
    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30) # max_steps might need adjustment
    
    prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.

Die offizielle Webseite für "{company_name}" wurde als "{root_url_for_prompt if company_url else 'nicht gefunden'}" identifiziert.

Wenn eine URL ({root_url_for_prompt}) vorhanden ist und nicht 'null' oder 'nicht gefunden' lautet:
1. Öffne diese URL: {root_url_for_prompt}
2. Durchsuche diese Seite und relevante Unterseiten (z. B. /about, /unternehmen, /impressum, /geschichte)
   und sammle die unten genannten Fakten.

Wenn KEINE URL gefunden wurde (d.h. als "{root_url_for_prompt}" angegeben ist) ODER Informationen auf der Webseite nicht auffindbar sind, gib für die entsprechenden Felder **null** zurück.

Fakten zu sammeln:
   • Aktueller CEO / Geschäftsführer
   • Gründer (Komma-getrennt bei mehreren)
   • Inhaber (Besitzer der Firma)
   • Aktuelle Mitarbeiterzahl (Zahl oder Bereich, z. B. "200-250")
   • Gründungsjahr (JJJJ)
   • Offizielle Website (die bereits ermittelte Root-URL: "{root_url_for_prompt}")
   • Was macht diese Firma besser als ihre Konkurrenz. (maximal 10 Wörter)
   • Addresse Hauptsitz
   • Firmenidentifikationsnummer (meistens im Impressum, z.B. CHE-XXX.XXX.XXX)
   • Haupt-Telefonnummer
   • Haupt-Emailadresse

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
  "Firmenidentifikationsnummer": "<CHE- oder null>",
  "HauptTelefonnummer": "<xxx xxx xx xx oder null>",
  "HauptEmailAdresse": "<xx@xx.xx oder null>"
}}
"""
    
    print(f"\nÜbermittelter Prompt an den Agenten (URL-Teil):")
    print(f"Die offizielle Webseite für \"{company_name}\" wurde als \"{root_url_for_prompt if company_url else 'nicht gefunden'}\" identifiziert.")
    if company_url:
        print(f"1. Öffne diese URL: {root_url_for_prompt}")
    print("-" * 20)

    result = await agent.run(prompt, max_steps=30) 
    
    print(f"\nResult: {result}")

def console_main() -> None:
    """Entry point for the ``brave-search`` console script."""
    parser = argparse.ArgumentParser(
        description="Search for company information using Brave/Wikidata and then an agent."
    )
    parser.add_argument("company", help="The company name to search for")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set. Please set it in your .env file or environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(main(args.company))


if __name__ == "__main__":
    console_main()
