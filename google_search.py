import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger  # Import the Logger
from langchain_openai import ChatOpenAI
from urllib.parse import urlparse # Keep for root_url_for_prompt

# Load environment variables from .env file
load_dotenv()

# --- Import functions and constants from search_common ---
from search_common import (
    select_best_url_with_llm,
    get_Google_Search_candidates,  # MODIFIED: Import Google search function
    get_wikidata_homepage,
    BLACKLIST 
)

# Enable mcp_use debug logging
Logger.set_debug(1)


async def main(company_name):
    # Load API keys from environment variables
    google_api_key_local = os.getenv("GOOGLE_API_KEY")     # MODIFIED: Load Google API Key
    google_cx_local = os.getenv("GOOGLE_CX")               # MODIFIED: Load Google CX
    openai_api_key_local = os.getenv("OPENAI_API_KEY")

    llm_url_selector = None
    if openai_api_key_local:
        try:
            llm_url_selector = ChatOpenAI(
                model="gpt-4.1-mini", 
                temperature=0,
                api_key=openai_api_key_local
            )
            print("LLM for URL selection initialized.")
        except Exception as e:
            print(f"Error initializing LLM for URL selection: {e}. Proceeding without LLM URL selection.", file=sys.stderr)
    else:
        print("Warning: OPENAI_API_KEY not set. LLM-based URL selection will be skipped.", file=sys.stderr)

    company_url = None
    source_of_url = "None"

    # --- MODIFIED SECTION: Use Google Search ---
    if google_api_key_local and google_cx_local:
        print(f"\nAttempting to find URL for '{company_name}' using Google Search...")
        # Pass google_api_key_local and google_cx_local to the function
        # Ensure get_Google Search_candidates is an async function if you await it
        google_candidates = await get_Google_Search_candidates(company_name, google_api_key_local, google_cx_local, count=5)

        if google_candidates:
            if llm_url_selector:
                print(f"Found {len(google_candidates)} candidates from Google. Asking LLM to select the best...")
                # Ensure select_best_url_with_llm expects candidates in the format provided by get_Google Search_candidates
                company_url = select_best_url_with_llm(company_name, google_candidates, llm_url_selector)
                if company_url:
                    source_of_url = "Google Search + LLM"
                else:
                    print("LLM did not select a URL from Google results. Applying heuristic fallback.")
            else: # LLM not available or failed to initialize
                print("LLM not available for URL selection. Applying Google heuristic fallback.")

            # Fallback logic if LLM didn't find a URL or wasn't used
            if not company_url:
                # Using the google_candidates list with heuristics
                # These keys ('is_ch_domain', 'company_match_in_host') must be populated by 
                # your get_Google Search_candidates function in search_common.py
                ch_official_urls = [c['url'] for c in google_candidates if c.get('is_ch_domain') and c.get('company_match_in_host')]
                other_official_urls = [c['url'] for c in google_candidates if not c.get('is_ch_domain') and c.get('company_match_in_host')]
                any_non_blacklisted_urls = [c['url'] for c in google_candidates] # Assumes blacklist is applied in get_Google Search_candidates

                if ch_official_urls:
                    company_url = ch_official_urls[0]
                    print(f"Found .ch official candidate(s) from Google (heuristic): {company_url}")
                elif other_official_urls:
                    company_url = other_official_urls[0]
                    print(f"Found other official candidate(s) from Google (non-.ch heuristic): {company_url}")
                elif any_non_blacklisted_urls:
                    company_url = any_non_blacklisted_urls[0]
                    print(f"Found any non-blacklisted candidate from Google (heuristic fallback): {company_url}")
                
                if company_url:
                    source_of_url = "Google Search (heuristic fallback)"
                else:
                    print("Google heuristic fallback also failed to find a URL.")
        else:
            print("No candidates found from Google Search.")
    else:
        print("Warning: GOOGLE_API_KEY or GOOGLE_CX not set. Skipping Google Search.", file=sys.stderr)
    # --- END OF MODIFIED SECTION ---

    if not company_url:
        status_message = f"Previous method ({source_of_url})" if source_of_url != "None" else "Google Search skipped or failed"
        print(f"{status_message} did not yield a URL for '{company_name}'. Trying Wikidata...")
        # If get_wikidata_homepage involves I/O, consider making it async too
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

    # --- MCP Agent part (remains the same) ---
    mcp_config_path = os.path.join(os.path.dirname(__file__), "sequential_mcp_config.json")
    if not os.path.exists(mcp_config_path):
        print(f"Error: MCP config file not found at {mcp_config_path}", file=sys.stderr)
        sys.exit(1)
    
    agent_llm = None
    if openai_api_key_local: 
        try:
            agent_llm = ChatOpenAI(
                model="gpt-4.1-mini", 
                temperature=0,
                api_key=openai_api_key_local
            )
            print("LLM for MCPAgent initialized.")
        except Exception as e:
            print(f"Error initializing LLM for MCPAgent: {e}. Agent cannot run.", file=sys.stderr)
            sys.exit(1) 
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
        description="Search for company information using Google/Wikidata (with LLM URL selection) and then an agent." # MODIFIED description
    )
    parser.add_argument("company", help="The company name to search for")
    args = parser.parse_args()

    asyncio.run(main(args.company))


if __name__ == "__main__":
    console_main()