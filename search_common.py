import httpx
import json
import sys
import re
import logging # Added logging
from urllib.parse import urlparse # urlunparse might be needed if you add more advanced normalization
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

# --- Initialize Logger ---
logger = logging.getLogger(__name__)

# --- Constants for Search APIs ---
# Brave
SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
# Wikidata
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/w/api.php"
# Google
Google_Search_API_URL = "https://www.googleapis.com/customsearch/v1" # ADDED for Google

BLACKLIST = {"wikipedia.org", "facebook.com", "twitter.com", "linkedin.com", "pflegeheimvergleich.ch", "moneyhouse.ch"} # Domains to ignore

def select_best_url_with_llm(company_name: str, search_results: List[Dict[str, Any]], llm: ChatOpenAI) -> str | None:
    """
    Uses a GPT model to select the most promising URL from a list of search results.
    Each item in search_results should be a dict with at least 'url', 'title', and 'description'.
    """
    if not search_results:
        logger.info(f"No search results provided for {company_name}, skipping LLM selection.")
        return None

    formatted_results = []
    for i, result in enumerate(search_results):
        formatted_results.append(
            f"{i+1}. URL: {result.get('url', 'N/A')}\n"
            f"   Title: {result.get('title', 'N/A')}\n"
            # Google calls snippet 'snippet', Brave calls it 'description'. Standardize to 'description' for the prompt.
            f"   Description: {result.get('description') or result.get('snippet', 'N/A')}"
        )

    prompt_text = f"""
You are an expert at identifying the official homepage of a company from a list of search engine results.
Given the company name "{company_name}" and the following search results, please select the number corresponding to the URL that is most likely the official homepage.

Consider the URL structure, domain name, title, and description to make your choice.
The official homepage is typically the primary website owned and operated by the company itself, not a news article, directory listing (unless it's a highly official business register), or social media page.
**If multiple good candidates exist, prioritize Swiss websites (e.g., those with a .ch domain or clearly indicating Swiss origin).**

If none of the URLs seem to be the official homepage, or if you are unsure, respond with "None".
Only respond with the number of the best choice or "None".

Search Results:
{chr(10).join(formatted_results)}

Company Name: "{company_name}"
Which number corresponds to the most likely official homepage? Respond with the number only, or "None".
    """
    logger.debug(f"LLM Prompt for URL selection for '{company_name}':\n{prompt_text}")

    try:
        response = llm.invoke(prompt_text)
        raw_response_content = response.content.strip()
        logger.info(f"LLM raw response for URL selection for '{company_name}': '{raw_response_content}'")

        selected_choice = raw_response_content.lower()

        if selected_choice == "none":
            logger.info(f"LLM indicated no suitable URL for '{company_name}'.")
            return None

        # Try to extract a number if the response is not purely numeric
        numeric_match = re.search(r'\d+', selected_choice)
        extracted_number_str = None

        if selected_choice.isdigit():
            extracted_number_str = selected_choice
        elif numeric_match:
            extracted_number_str = numeric_match.group(0)
            logger.info(f"Extracted number '{extracted_number_str}' from LLM response '{raw_response_content}' for '{company_name}'.")

        if extracted_number_str:
            try:
                selected_index = int(extracted_number_str) - 1
                if 0 <= selected_index < len(search_results):
                    chosen_url = search_results[selected_index].get("url")
                    logger.info(f"LLM selected URL ({extracted_number_str}): {chosen_url} for company: {company_name}")
                    return chosen_url
                else:
                    logger.warning(f"LLM selected an out-of-bounds index: {extracted_number_str} (parsed as {selected_index}) for '{company_name}'. Number of results: {len(search_results)}.")
                    return None
            except ValueError: # Should not happen if isdigit() or regex match was successful, but as a safeguard
                logger.warning(f"Could not convert extracted string '{extracted_number_str}' to int for '{company_name}'. Original response: '{raw_response_content}'", exc_info=True)
                return None
        else:
            # Fallback: If LLM returns something else non-numeric and without a clear number.
            # Original fallback was to check if URL is in response, this might be too broad.
            # For now, we will log that no valid number was found.
            # Consider if the old fallback `result.get("url") in selected_choice` is desired.
            # It could lead to unintended selections if the LLM is chatty.
            logger.warning(f"LLM response for '{company_name}' was not 'None', did not contain a digit, and was not directly a number: '{raw_response_content}'")
            return None

    except Exception as e:
        logger.error(f"Error during LLM call or processing for URL selection for '{company_name}': {e}", exc_info=True)
        return None

# --- Google Search Function ---
async def get_Google_Search_candidates(
    company_name: str,
    api_key: str,
    cx: str,
    count: int = 5
) -> List[Dict[str, Any]]:
    """
    Fetches potential candidate URLs from Google Custom Search API.
    Processes them to a format similar to get_brave_search_candidates.
    Returns a list of dictionaries, where each dictionary contains 'url', 'title', 'description' (from 'snippet'),
    'is_ch_domain', and 'company_match_in_host'.
    """
    if not api_key or not cx:
        logger.error("Google API Key or CX not provided. Skipping Google Search.")
        return []

    params = {
        "key": api_key,
        "cx": cx,
        "q": f'"{company_name}" offizielle homepage', # Using a similar query structure to Brave
        "num": min(count, 10),  # API allows up to 10 results per page
        "lr": "lang_de", # Restrict to German language pages
        "cr": "countryCH" # Restrict to Switzerland
    }
    # print(f"Querying Google Custom Search for: '{params['q']}' with country CH, lang DE")

    candidate_results = []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(Google_Search_API_URL, params=params, timeout=10.0)
            response.raise_for_status()
            search_data = response.json()

        if "items" in search_data:
            for item in search_data["items"]:
                url = item.get("link")
                title = item.get("title")
                # Google uses "snippet" for description
                description_or_snippet = item.get("snippet")

                if not url:
                    continue

                parsed_url = urlparse(url)
                host = parsed_url.hostname or ""

                if not host:
                    continue

                if any(domain_part in host for domain_part in BLACKLIST):
                    # print(f"Skipping blacklisted URL (Google): {url}")
                    continue

                # Replicate heuristic logic from Brave for consistency
                company_main_name_part = company_name.lower().split(" ")[0].replace(",", "").replace(".", "")
                company_name_no_spaces = company_name.lower().replace(" ", "").replace(".", "").replace(",", "")
                host_cleaned = host.lower()
                # title_cleaned = title.lower() if title else "" # Not used in Brave's final append, so mirror that

                is_ch = host.endswith(".ch")
                company_match = (company_main_name_part in host_cleaned) or \
                                (company_name_no_spaces in host_cleaned) or \
                                (host_cleaned.startswith(company_main_name_part)) or \
                                (host_cleaned.startswith(company_name_no_spaces))

                candidate_results.append({
                    "url": url,
                    "title": title,
                    "description": description_or_snippet, # Standardize to 'description' key for LLM prompt
                    "is_ch_domain": is_ch,
                    "company_match_in_host": company_match,
                    "raw_google_item": item # Optional: for debugging or more complex heuristics
                })

        # Sort candidates similar to Brave's logic for heuristic fallback consistency
        candidate_results.sort(key=lambda x: (
            not x["is_ch_domain"],
            not x["company_match_in_host"]
        ))
        logger.info(f"Found {len(candidate_results)} potential candidates for '{company_name}' from Google Search.")
        return candidate_results

    except httpx.RequestError as e:
        logger.error(f"Google Search API request error for '{company_name}': {e}", exc_info=True)
    except httpx.HTTPStatusError as e:
        logger.error(f"Google Search API returned an error for '{company_name}': {e.response.status_code} - {e.response.text}", exc_info=True)
    except json.JSONDecodeError as e: # Added 'e' to capture exception info
        logger.error(f"Google Search API response was not valid JSON for '{company_name}'. Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_Google_Search_candidates for '{company_name}': {e}", exc_info=True)
    return []


# --- Brave Search Function (kept for reference or potential future use) ---
def get_brave_search_candidates(company: str, brave_api_key: str, count: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches potential candidate URLs from Brave Search API.
    Returns a list of dictionaries, where each dictionary contains 'url', 'title', 'description'.
    """
    if not brave_api_key:
        logger.error("Brave API key not provided. Skipping Brave Search.")
        return []

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key
    }
    params = {
        "q": f'"{company}" offizielle homepage',
        "count": count,
        "country": "ch",
        "search_lang": "de",
        "spellcheck": "false"
    }

    candidate_results = []
    try:
        logger.info(f"Querying Brave Search for: '{params['q']}' with country '{params['country']}' for company '{company}'")
        # Note: For async operation, this should use an async client as well.
        # For now, keeping it synchronous as per original structure.
        # If your main script always awaits this, convert it to async.
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
            # title_cleaned = title.lower() if title else "" # Not used in final append, matching structure.
            
            # Removed 'pre_filter_match' as it wasn't directly used for filtering here,
            # the results are appended and then sorted. The match logic is in 'company_match_in_host'.

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

        candidate_results.sort(key=lambda x: (
            not x["is_ch_domain"],
            not x["company_match_in_host"]
        ))
        logger.info(f"Found {len(candidate_results)} potential candidates for '{company}' from Brave Search.")
        return candidate_results

    except httpx.RequestError as e:
        logger.error(f"Brave Search API request error for '{company}': {e}", exc_info=True)
    except httpx.HTTPStatusError as e:
        logger.error(f"Brave Search API returned an error for '{company}': {e.response.status_code} - {e.response.text}", exc_info=True)
    except json.JSONDecodeError as e: # Added 'e' to capture exception info
        logger.error(f"Brave Search API response was not valid JSON for '{company}'. Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_brave_search_candidates for '{company}': {e}", exc_info=True)
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
        logger.debug(f"Querying Wikidata entities for: '{company}'")
        # Note: For async operation, this should use an async client.
        r_search = httpx.get(WIKIDATA_SEARCH, params=params_search, timeout=5.0)
        r_search.raise_for_status()
        search_results_json = r_search.json()
        search_results = search_results_json.get("search", [])

        if not search_results:
            logger.info(f"Wikidata: No entity found for '{company}' in initial search.")
            return None

        qid = None
        # Try to find a good QID match
        for res in search_results:
            res_label = res.get("label", "").lower()
            # Wikidata aliases are a list of dicts, each with 'language' and 'value'
            res_aliases = [alias.get("value", "").lower() for alias_obj in res.get("aliases", []) for alias in alias_obj if isinstance(alias_obj, dict) and alias_obj.get("language") == "de" and alias_obj.get("value")]
            
            if company.lower() == res_label or company.lower() in res_aliases:
                qid = res.get("id")
                break
        
        if not qid: # Fallback: if company name is IN the label (broader match)
            for res in search_results:
                res_label = res.get("label", "").lower()
                if company.lower() in res_label: 
                    qid = res.get("id")
                    break
        
        if not qid and search_results: # Fallback: first result with a description, or just first result
            first_with_desc = next((res.get("id") for res in search_results if res.get("description")), None)
            if first_with_desc:
                qid = first_with_desc
            else:
                qid = search_results[0].get("id")


        if not qid:
            return None

        params_claims = {
            "action": "wbgetclaims",
            "format": "json",
            "entity": qid,
            "property": "P856"  # P856 is the property for "official website"
        }
        r_claims = httpx.get(WIKIDATA_ENTITY, params=params_claims, timeout=5.0)
        r_claims.raise_for_status()
        claims_data = r_claims.json().get("claims", {}).get("P856", [])

        if not claims_data:
            return None

        # Iterate through claims to find a non-deprecated, preferred rank or normal rank URL
        preferred_url = None
        normal_url = None

        for claim in claims_data:
            if claim.get("rank") == "deprecated":
                continue
            
            mainsnak = claim.get("mainsnak")
            if mainsnak and mainsnak.get("datavalue") and isinstance(mainsnak["datavalue"].get("value"), str):
                current_url = mainsnak["datavalue"]["value"]
                if current_url.startswith("http://") or current_url.startswith("https://"):
                    parsed_url = urlparse(current_url)
                    if parsed_url.hostname and not any(domain in parsed_url.hostname for domain in BLACKLIST):
                        if claim.get("rank") == "preferred":
                            preferred_url = current_url
                            break # Found preferred, use this
                        if not normal_url: # Store first normal rank URL
                             normal_url = current_url
        
        url_to_return = preferred_url or normal_url # Prioritize preferred

        if url_to_return:
            logger.info(f"Wikidata found URL: {url_to_return} for {company} (QID: {qid})")
            return url_to_return
        else:
            logger.info(f"Wikidata: No suitable official website URL found for {company} (QID: {qid}) after checking claims.")
            return None
            
    except httpx.RequestError as e:
        logger.error(f"Wikidata API request error for '{company}': {e}", exc_info=True)
    except httpx.HTTPStatusError as e:
        logger.error(f"Wikidata API returned an error for '{company}': {e.response.status_code} - {e.response.text}", exc_info=True)
    except json.JSONDecodeError as e: # Added 'e' to capture exception info
        logger.error(f"Wikidata API response was not valid JSON for '{company}'. Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_wikidata_homepage for '{company}': {e}", exc_info=True)
    return None

# This function is designed to be called by the main script before handing off to the agent.
# It's already async.
async def is_url_relevant_to_company(url: str, company_name: str, client: httpx.AsyncClient) -> bool:
    """
    Performs a pre-check to see if the URL seems relevant to the company name
    by fetching the page title and checking against the domain.
    """
    if not url or url == "null" or not url.startswith(('http://', 'https://')): # Added check for valid URL start
        # If no valid URL, it's "relevant" in the sense that we don't block processing based on this check.
        # The agent will then receive "null" or the invalid URL and should handle it.
        return True # Or False, depending on desired behavior for bad URLs. True lets agent handle it.

    try:
        logger.debug(f"Relevance pre-check: Fetching {url} for title for company '{company_name}'...")
        response = await client.get(url, follow_redirects=True, timeout=10.0)
        response.raise_for_status()
        html_content = response.text

        title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        page_title = title_match.group(1).strip() if title_match else ""

        normalized_company_name = company_name.lower().replace(" ag", "").replace(" gmbh", "").replace(" sa", "").replace(" sàrl", "").replace(".", "").replace(",", "").strip()
        normalized_page_title = page_title.lower()
        
        parsed_url_obj = urlparse(url)
        domain = parsed_url_obj.netloc.lower()

        company_name_parts = [part for part in normalized_company_name.split() if len(part) > 2]
        if not company_name_parts: # Handle cases like "AG" or very short names
                 company_name_parts = [normalized_company_name] if normalized_company_name else []


        title_match_found = any(part in normalized_page_title for part in company_name_parts)
        domain_for_check = domain.replace("www.", "")
        domain_match_found = any(part in domain_for_check for part in company_name_parts)
        
        # If title is specific and matches, it's relevant
        if page_title and not any(generic in normalized_page_title for generic in ["home", "startseite", "accueil", "benvenuto", "willkommen", "homepage", "website", "site officiel"]):
            if title_match_found:
                logger.debug(f"Relevance pre-check for '{company_name}' at '{url}': Title='{page_title}'. Seems relevant (specific title match).")
                return True
        
        # If domain matches, it's relevant
        if domain_match_found:
            logger.debug(f"Relevance pre-check for '{company_name}' at '{url}': Domain='{domain}'. Seems relevant (domain match).")
            return True
        
        # If title is generic but still matches, it's also relevant (e.g. "Company Name - Home")
        if title_match_found:
            logger.debug(f"Relevance pre-check for '{company_name}' at '{url}': Title='{page_title}'. Seems relevant (generic title match).")
            return True

        logger.info(f"Relevance pre-check for '{company_name}' at '{url}': Title='{page_title}', Domain='{domain}'. Potential mismatch.")
        return False

    except httpx.TimeoutException as e:
        logger.warning(f"Timeout fetching URL {url} for relevance pre-check for '{company_name}': {e}", exc_info=True)
        return True # Treat timeout as potentially relevant to let agent decide
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error {e.response.status_code} for URL {url} during relevance pre-check for '{company_name}'.", exc_info=True)
        if e.response.status_code in [403, 404, 500, 502, 503, 504]: # Definite issues
            return False
        return True # Other HTTP errors, let agent try
    except httpx.RequestError as e: 
        logger.warning(f"Request error fetching URL {url} for relevance pre-check for '{company_name}': {e}", exc_info=True)
        return False # Likely network issue or bad URL, consider not relevant
    except Exception as e: 
        logger.error(f"Unexpected error during relevance pre-check for {url} for '{company_name}': {e}", exc_info=True)
        return True # Unknown error, let agent try