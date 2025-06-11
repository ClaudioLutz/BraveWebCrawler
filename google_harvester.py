# ─── google_harvester.py ─────────────────────────────────────────────────────
import re
import asyncio
import httpx
import logging # Added logging
from datetime import datetime # Added datetime
from models import CompanyFacts

# Initialize Logger
logger = logging.getLogger(__name__)

# Regexes remain a good tool for well-defined patterns
CHE_ID_RX = re.compile(r"(CHE-\d{3}\.\d{3}\.\d{3})")
PHONE_RX  = re.compile(r"\+?\d[\d\s\-().]{7,}")
EMAIL_RX  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
FOUNDED_RX = re.compile(r"\b(18|19|20)\d{2}\b")

async def _google_search(query: str, api_key: str, cx: str) -> list[dict]:
    """Generic helper to call the Google Custom Search API."""
    url = "https://customsearch.googleapis.com/customsearch/v1"
    try:
        async with httpx.AsyncClient(timeout=15) as client: # Timeout is already set
            r = await client.get(url, params={"key": api_key, "cx": cx, "q": query})
            r.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
            # Attempt to parse JSON and get "items"
            response_json = r.json()
            return response_json.get("items", [])
    except httpx.HTTPStatusError as e:
        logger.error(f"Google Search API HTTPStatusError for query '{query}': {e.response.status_code} - {e.response.text}", exc_info=True)
        return []
    except httpx.RequestError as e: # Catches network errors, timeouts not leading to a response status code etc.
        logger.error(f"Google Search API RequestError for query '{query}': {e}", exc_info=True)
        return []
    except httpx.InvalidURL as e: # If the URL constructed is somehow invalid
        logger.error(f"Google Search API InvalidURL for query '{query}': {e}", exc_info=True)
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Google Search API JSONDecodeError for query '{query}'. Response text: {r.text[:200]}... Error: {e}", exc_info=True) # Log part of the response
        return []
    except Exception as e: # Catch-all for any other unexpected errors
        logger.error(f"Unexpected error in _google_search for query '{query}': {e}", exc_info=True)
        return []

async def get_facts_from_google(name: str, api_key: str, cx: str) -> CompanyFacts:
    """
    Gathers company facts from Google, prioritizing authoritative sources like Zefix.
    """
    facts = CompanyFacts()

    # --- Stage 1: High-trust search on Zefix for official data ---
    zefix_results = await _google_search(f'"{name}" site:zefix.ch', api_key, cx) # Corrected function call
    if zefix_results:
        zefix_snippet = zefix_results[0].get("snippet", "")
        
        # FIX: More robustly parse the Zefix snippet
        if che_match := CHE_ID_RX.search(zefix_snippet):
            facts.firmen_id = che_match.group(1)
        
        # FIX: Extract address specifically from snippet, not the generic page title
        # Look for a pattern like "Sitz: Address" or "Adresse: Address"
        address_match = re.search(r"(?i)(?:Sitz|Adresse):\s*(.*)", zefix_snippet)
        if address_match:
            facts.haupt_sitz = address_match.group(1).strip()
            logger.info(f"Extracted address for '{name}' from Zefix snippet: {facts.haupt_sitz}")
        else:
            logger.warning(f"Could not extract address (Sitz/Adresse) from Zefix snippet for '{name}'. Snippet: '{zefix_snippet[:100]}...'")
            # facts.haupt_sitz remains None or its default value


    # --- Stage 2: Parallel search for remaining, less-structured data ---
    fields_to_find = facts.missing_fields()
    
    # Define queries only for fields we still need
    queries = {
        "founded": f'"{name}" Gründungsjahr "founded in"',
        "haupt_tel": f'"{name}" Hauptsitz kontakt telefon OR "{name}" headquarters contact phone',
        # FIX: Use haupt_mail to match the model, not haupt_email
        "haupt_mail": f'"{name}" Hauptsitz kontakt email OR "{name}" headquarters contact email',
        "geschaeftsbericht": f'"{name}" filetype:pdf ("Geschäftsbericht" OR "annual report")',
    }
    
    tasks = []
    for field in fields_to_find:
        if query := queries.get(field):
            tasks.append(_search_single_fact(field, query, api_key, cx))

    if tasks:
        other_facts_list = await asyncio.gather(*tasks)
        
        # Merge the results from the parallel searches
        temp_facts = CompanyFacts()
        for field, value in other_facts_list:
            if value:
                setattr(temp_facts, field, value)
        facts.merge_with(temp_facts)

    return facts

async def _search_single_fact(field: str, query: str, api_key: str, cx: str) -> tuple[str, str | None]:
    """Runs a single Google query and extracts one fact."""
    items = await _google_search(query, api_key, cx) # Corrected function call
    if not items:
        return field, None

    if field == "geschaeftsbericht":
        # FIX: Check for the current or previous year in the link to avoid old reports
        current_year = datetime.now().year # Dynamically get current year
        for item in items:
            link = item.get("link", "").lower()
            if link.endswith(".pdf") and (str(current_year) in link or str(current_year - 1) in link):
                return field, item["link"]
        # Fallback to first PDF if no recent one is found
        pdf_link = next((i["link"] for i in items if i.get("link", "").lower().endswith(".pdf")), None)
        return field, pdf_link

    # FIX: Process snippets one by one to avoid cross-contamination
    for item in items[:3]:
        text = item.get("snippet", "")
        match field:
            case "founded":
                if m := FOUNDED_RX.search(text): return field, m.group(0)
            case "haupt_tel":
                if m := PHONE_RX.search(text): return field, m.group(0).strip()
            # FIX: Use haupt_mail to match the model
            case "haupt_mail": 
                if m := EMAIL_RX.search(text): return field, m.group(0).lower()
    
    return field, None
