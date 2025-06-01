import httpx
import json
import sys
import re
import logging # Added
from urllib.parse import urlparse
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception
)

logger = logging.getLogger(__name__) # Added

# --- Constants for Brave/Wikidata search ---
SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/w/api.php"
BLACKLIST = {"wikipedia.org", "facebook.com", "twitter.com", "linkedin.com", "pflegeheimvergleich.ch"}

# --- Tenacity Retry Configuration ---
def is_server_error(exception: BaseException) -> bool:
    return isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code >= 500

common_retry_params = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=2, max=10),
    "retry": retry_if_exception_type(httpx.RequestError) | \
             retry_if_exception_type(httpx.TimeoutException) | \
             retry_if_exception(is_server_error),
    "reraise": True
}

def select_best_url_with_llm(company_name: str, search_results: List[Dict[str, Any]], llm: ChatOpenAI) -> str | None:
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
**If multiple good candidates exist, prioritize Swiss websites (e.g., those with a .ch domain or clearly indicating Swiss origin).**

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
            logger.info(f"LLM for URL selection for '{company_name}' responded with 'None'.")
            return None

        if selected_choice.isdigit():
            selected_index = int(selected_choice) - 1
            if 0 <= selected_index < len(search_results):
                chosen_url = search_results[selected_index].get("url")
                logger.info(f"LLM selected URL ({selected_choice}): {chosen_url} for company: {company_name}")
                return chosen_url
            else:
                logger.warning(f"LLM for '{company_name}' selected an out-of-bounds index for URL: {selected_choice}")
                return None
        else:
            for i, result in enumerate(search_results): # Fallback for non-numeric response
                if result.get("url") in selected_choice:
                    logger.info(f"LLM for '{company_name}' selected URL by finding it in a non-numeric response: {result.get('url')}")
                    return result.get("url")
            logger.warning(f"LLM response for URL selection for '{company_name}' was not a valid number or 'None': '{selected_choice}'")
            return None

    except Exception as e:
        logger.error(f"Error during LLM call for URL selection for '{company_name}': {e}", exc_info=True)
        return None

@retry(**common_retry_params)
def get_brave_search_candidates(company: str, brave_api_key: str, count: int = 5) -> List[Dict[str, Any]]:
    if not brave_api_key:
        logger.error("Brave API key not provided. Cannot perform Brave Search.")
        return []

    headers = {"Accept": "application/json", "X-Subscription-Token": brave_api_key}
    params = {"q": f'"{company}" offizielle homepage', "count": count, "country": "ch", "search_lang": "de", "spellcheck": "false"}

    candidate_results = []
    try:
        logger.debug(f"Querying Brave Search for: '{params['q']}' with country '{params['country']}'")
        resp = httpx.get(SEARCH_URL, headers=headers, params=params, timeout=10.0)
        resp.raise_for_status()
        results_json = resp.json().get("web", {}).get("results", [])

        for r in results_json:
            url, title, description = r.get("url"), r.get("title"), r.get("description")
            if not url: continue

            parsed_url = urlparse(url)
            host = parsed_url.hostname or ""
            if not host or any(domain in host for domain in BLACKLIST):
                logger.debug(f"Skipping blacklisted or invalid host URL from Brave: {url}")
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
        logger.info(f"Found {len(candidate_results)} potential candidates for '{company}' from Brave Search.")
        return candidate_results
    except json.JSONDecodeError:
        logger.error("Brave Search API response was not valid JSON.", exc_info=True)
    except Exception as e:
        logger.error(f"Error in get_brave_search_candidates for '{company}' (after retries if any): {e}", exc_info=True)
    return []

@retry(**common_retry_params)
def get_wikidata_homepage(company: str) -> str | None:
    params_search = {
        "action": "wbsearchentities", "format": "json", "language": "de",
        "uselang": "de", "type": "item", "search": company
    }
    try:
        logger.debug(f"Querying Wikidata entities for: '{company}'")
        r_search = httpx.get(WIKIDATA_SEARCH, params=params_search, timeout=5.0)
        r_search.raise_for_status()
        search_results_json = r_search.json()
        search_results = search_results_json.get("search", [])
        if not search_results:
            logger.debug(f"Wikidata: No entity found for '{company}'")
            return None

        qid = None
        for res in search_results:
            if company.lower() == res.get("label", "").lower() or \
               company.lower() in [alias.get("value", "").lower() for alias in res.get("aliases", []) if alias.get("value")]:
                qid = res.get("id"); break
        if not qid:
            for res in search_results:
                if company.lower() in res.get("label", "").lower(): qid = res.get("id"); break
        if not qid and search_results: 
            qid = next((res.get("id") for res in search_results if res.get("description")), search_results[0].get("id"))

        if not qid:
            logger.debug(f"Wikidata: Could not determine QID for '{company}' from search results.")
            return None

        logger.debug(f"Wikidata: Determined QID {qid} for '{company}'. Fetching claims for P856.")
        params_claims = {"action": "wbgetclaims", "format": "json", "entity": qid, "property": "P856"}
        r_claims = httpx.get(WIKIDATA_ENTITY, params=params_claims, timeout=5.0)
        r_claims.raise_for_status()
        claims_data = r_claims.json().get("claims", {}).get("P856", [])
        if not claims_data:
            logger.debug(f"Wikidata: No P856 claims (official website) found for QID {qid} ({company}).")
            return None

        mainsnak = claims_data[0].get("mainsnak")
        if mainsnak and mainsnak.get("datavalue") and isinstance(mainsnak["datavalue"].get("value"), str):
            url = mainsnak["datavalue"]["value"]
            if (url.startswith("http://") or url.startswith("https://")):
                parsed_url = urlparse(url)
                if parsed_url.hostname and not any(domain in parsed_url.hostname for domain in BLACKLIST):
                    logger.info(f"Wikidata found URL: {url} for {company} (QID: {qid})")
                    return url
                else:
                    logger.debug(f"Wikidata URL for {company} ({url}) was blacklisted or invalid.")
    except json.JSONDecodeError:
        logger.error(f"Wikidata API response was not valid JSON for {company}.", exc_info=True)
    except Exception as e:
        logger.error(f"Error in get_wikidata_homepage for {company} (after retries if any): {e}", exc_info=True)
    return None

@retry(**common_retry_params)
async def is_url_relevant_to_company(url: str, company_name: str, client: httpx.AsyncClient) -> bool:
    if not url or url == "null": return True

    try:
        logger.debug(f"Relevance pre-check: Fetching {url} for title matching '{company_name}'.")
        response = await client.get(url, follow_redirects=True, timeout=10.0)
        response.raise_for_status()
        html_content = response.text

        title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        page_title = title_match.group(1).strip() if title_match else ""

        normalized_company_name = company_name.lower().replace(" ag", "").replace(" gmbh", "").replace(" sa", "").replace(" sàrl", "").replace(".", "").replace(",", "").strip()
        normalized_page_title = page_title.lower()
        domain = urlparse(url).netloc.lower()
        company_name_parts = [part for part in normalized_company_name.split() if len(part) > 2] or [normalized_company_name]

        title_match_found = any(part in normalized_page_title for part in company_name_parts)
        domain_match_found = any(part in domain.replace("www.", "") for part in company_name_parts)
        
        is_relevant_flag = False
        if page_title and not any(generic in normalized_page_title for generic in ["home", "startseite", "accueil", "benvenuto", "willkommen"]):
            if title_match_found: is_relevant_flag = True
        
        if not is_relevant_flag and domain_match_found: is_relevant_flag = True
        
        logger.info(f"Relevance pre-check for '{company_name}' at '{url}': Title='{page_title}', Domain='{domain}'. Relevant: {is_relevant_flag}")
        return is_relevant_flag

    except Exception as e:
        # This will be caught by tenacity and retried. If all retries fail, the error will be reraised.
        # The calling function (pre_screen_urls in core_processing) should handle it.
        # We log it here as it happens before reraise.
        logger.warning(f"Error during relevance pre-check for {url} (company: {company_name}, attempt pending/failed): {e}", exc_info=True)
        raise # Reraise to allow tenacity to handle retries or final error propagation
