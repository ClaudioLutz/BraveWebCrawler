import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger  # Import the Logger
from langchain_openai import ChatOpenAI
from urllib.parse import urlparse # Keep for root_url_for_prompt
# from langchain_google_genai import ChatGoogleGenerativeAI

# --- Imports from the Brave/Wikidata script ---
# import httpx # Moved to search_common
# import json # Moved to search_common
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
    BLACKLIST # Still used by get_wikidata_homepage logic if it were here, but it's in search_common
)

# Global variables for API Keys
# BRAVE_API_KEY = None # Will be loaded in main and passed as arg
# OPENAI_API_KEY = None # Will be loaded in main

# Enable mcp_use debug logging
Logger.set_debug(2)


async def main(company_name):
    # Load API keys from environment variables
    # These are local to the main function now, not global to the module
    brave_api_key_local = os.getenv("BRAVE_API_KEY")
    openai_api_key_local = os.getenv("OPENAI_API_KEY")

    llm_url_selector = None
    if openai_api_key_local:
        try:
            llm_url_selector = ChatOpenAI(
                model="gpt-4.1-mini", # Or your preferred model like "gpt-3.5-turbo", "gpt-4-turbo"
                temperature=0,
                api_key=openai_api_key_local
            )
            print("LLM for URL selection initialized.")
        except Exception as e:
            print(f"Error initializing LLM for URL selection: {e}. Proceeding without LLM URL selection.", file=sys.stderr)
            # openai_api_key_local = None # Ensure it's treated as unavailable if we need to check it later
    else:
        print("Warning: OPENAI_API_KEY not set. LLM-based URL selection will be skipped.", file=sys.stderr)

    company_url = None
    source_of_url = "None"

    if brave_api_key_local:
        print(f"\nAttempting to find URL for '{company_name}' using Brave Search...")
        # Pass brave_api_key_local to the function
        brave_candidates = get_brave_search_candidates(company_name, brave_api_key_local, count=5)

        if brave_candidates:
            if llm_url_selector:
                print(f"Found {len(brave_candidates)} candidates. Asking LLM to select the best...")
                # Pass llm_url_selector to the function
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
    if openai_api_key_local: # Re-check as it might have been nullified if URL LLM failed
        try:
            agent_llm = ChatOpenAI(
                model="gpt-4.1-mini", # Or your agent's preferred model
                temperature=0,
                api_key=openai_api_key_local # Use the local openai_api_key_local
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