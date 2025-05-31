import httpx
import json
import sys # Moved from the bottom
import re # Added for is_url_relevant_to_company
from urllib.parse import urlparse
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

# --- Constants for Brave/Wikidata search ---
SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/w/api.php"
BLACKLIST = {"wikipedia.org", "facebook.com", "twitter.com", "linkedin.com", "pflegeheimvergleich.ch"} # Domains to ignore

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
**If multiple good candidates exist, prioritize Swiss websites (e.g., those with a .ch domain or clearly indicating Swiss origin).**

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


def get_brave_search_candidates(company: str, brave_api_key: str, count: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches potential candidate URLs from Brave Search API.
    Returns a list of dictionaries, where each dictionary contains 'url', 'title', 'description'.
    """
    if not brave_api_key:
        print("Error: brave_api_key not provided for Brave Search.", file=sys.stderr)
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
            
            pre_filter_match = (
                company_main_name_part in host_cleaned or
                company_name_no_spaces in host_cleaned or
                company_main_name_part in title_cleaned or
                company_name_no_spaces in title_cleaned or
                host_cleaned.startswith(company_main_name_part) or
                host_cleaned.startswith(company_name_no_spaces)
            )
            
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
            res_aliases = [alias.get("value", "").lower() for alias in res.get("aliases", []) if alias.get("value")]
            
            if company.lower() == res_label or company.lower() in res_aliases:
                qid = res.get("id")
                break
        
        if not qid:
            for res in search_results:
                res_label = res.get("label", "").lower()
                if company.lower() in res_label: 
                    qid = res.get("id")
                    break
        
        if not qid and search_results: 
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

        mainsnak = claims_data[0].get("mainsnak")
        if mainsnak and mainsnak.get("datavalue") and isinstance(mainsnak["datavalue"].get("value"), str):
            url = mainsnak["datavalue"]["value"]
            if url.startswith("http://") or url.startswith("https://"):
                parsed_url = urlparse(url)
                if parsed_url.hostname and not any(domain in parsed_url.hostname for domain in BLACKLIST):
                    print(f"Wikidata found URL: {url} for {company} (QID: {qid})")
                    return url
    except httpx.RequestError as e:
        print(f"Wikidata API request error: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"Wikidata API returned an error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
    except json.JSONDecodeError:
        print("Wikidata API response was not valid JSON.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in get_wikidata_homepage for {company}: {e}", file=sys.stderr)
    return None

async def is_url_relevant_to_company(url: str, company_name: str, client: httpx.AsyncClient) -> bool:
    """
    Performs a pre-check to see if the URL seems relevant to the company name
    by fetching the page title and checking against the domain.
    """
    if not url or url == "null":
        # If no URL, it's "relevant" in the sense that we don't block processing based on this check.
        # The agent will then receive "null" and should handle it.
        return True

    try:
        # print(f"Pre-check: Fetching {url} for title...") # Keep print in main processor for clarity
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
             company_name_parts = [normalized_company_name]

        title_match_found = any(part in normalized_page_title for part in company_name_parts)
        # Check if company name parts are in the domain, excluding common TLDs or subdomains like 'www'
        # This is a simple check; more sophisticated checks might be needed for complex domain structures.
        domain_for_check = domain.replace("www.", "")
        domain_match_found = any(part in domain_for_check for part in company_name_parts)
        
        # Prioritize title match if title is specific
        if page_title and not any(generic in normalized_page_title for generic in ["home", "startseite", "accueil", "benvenuto", "willkommen"]):
            if title_match_found:
                # print(f"Relevance pre-check for '{company_name}' at '{url}': Title='{page_title}'. Seems relevant (title match).")
                return True
        
        # Domain match is also a strong indicator
        if domain_match_found:
            # print(f"Relevance pre-check for '{company_name}' at '{url}': Domain='{domain}'. Seems relevant (domain match).")
            return True
        
        # print(f"Relevance pre-check for '{company_name}' at '{url}': Title='{page_title}', Domain='{domain}'. Potential mismatch.")
        return False

    except httpx.TimeoutException:
        # print(f"Timeout fetching URL {url} for relevance pre-check.", file=sys.stderr)
        return False # Treat timeout as potential mismatch
    except httpx.RequestError: # Catching general request error
        # print(f"Error fetching URL {url} for relevance pre-check: {e}", file=sys.stderr)
        return False
    except httpx.HTTPStatusError: # Catching HTTP status errors
        # print(f"HTTP error {e.response.status_code} for URL {url} during relevance pre-check.", file=sys.stderr)
        return False
    except Exception: # Catching any other unexpected errors
        # print(f"Unexpected error during relevance pre-check for {url}: {e}", file=sys.stderr)
        return False
