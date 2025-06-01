import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx # For exception types
from langchain_openai import ChatOpenAI # For type hinting and mocking

# Assuming search_common.py is in the parent directory relative to tests/
# Adjust sys.path if necessary or ensure your project structure allows this import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from search_common import (
    get_brave_search_candidates,
    get_wikidata_homepage,
    is_url_relevant_to_company,
    select_best_url_with_llm,
    BLACKLIST # Import for use in tests if needed
)

class TestSearchCommon(unittest.TestCase):

    @patch('search_common.httpx.get')
    def test_get_brave_search_candidates_success(self, mock_httpx_get):
        # Configure mock response for Brave Search
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"url": "https://www.example.com/official", "title": "Example Company Official", "description": "Official site of Example Company."},
                    {"url": "https://www.example.ch/swiss", "title": "Example Company CH", "description": "Swiss site of Example Company."},
                    {"url": "https://www.news.com/example", "title": "Example News", "description": "News about Example Company."},
                    {"url": f"https://{list(BLACKLIST)[0]}/example", "title": "Blacklisted", "description": "A blacklisted domain."}
                ]
            }
        }
        mock_httpx_get.return_value = mock_response

        company_name = "Example Company"
        api_key = "test_brave_key"
        results = get_brave_search_candidates(company_name, api_key, count=4)

        self.assertEqual(len(results), 3) # Expecting 3 non-blacklisted results
        self.assertEqual(results[0]['url'], "https://www.example.ch/swiss") # .ch should be prioritized
        self.assertTrue(results[0]['is_ch_domain'])
        self.assertTrue(results[0]['company_match_in_host']) # example in example.ch
        self.assertEqual(results[1]['url'], "https://www.example.com/official")
        self.assertFalse(results[1]['is_ch_domain'])
        self.assertTrue(results[1]['company_match_in_host']) # example in example.com

        mock_httpx_get.assert_called_once() # Check that httpx.get was called

    @patch('search_common.httpx.get')
    def test_get_brave_search_candidates_api_error(self, mock_httpx_get):
        mock_response = MagicMock()
        mock_response.status_code = 500 # Simulate server error
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Server Error", request=MagicMock(), response=mock_response
        )
        mock_httpx_get.return_value = mock_response

        # We need to mock the tenacity retry decorator to avoid actual retries during test
        with patch('search_common.retry', lambda **kwargs: lambda f: f): # No-op retry
             results = get_brave_search_candidates("Test", "key")
             self.assertEqual(results, [])

    def test_get_brave_search_candidates_no_api_key(self):
        # Test behavior when API key is missing (should return empty list and log an error)
        # No need to mock httpx.get as it shouldn't be called
        with patch('search_common.logger.error') as mock_logger_error:
            results = get_brave_search_candidates("Example Company", None) # Pass None as API key
            self.assertEqual(results, [])
            mock_logger_error.assert_called_with("Brave API key not provided. Cannot perform Brave Search.")


    @patch('search_common.httpx.get')
    def test_get_wikidata_homepage_success(self, mock_httpx_get):
        # Simulate Wikidata entity search response
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "search": [{"id": "Q12345", "label": "Example AG", "description": "A company"}]
        }

        # Simulate Wikidata claims response
        mock_claims_response = MagicMock()
        mock_claims_response.status_code = 200
        mock_claims_response.json.return_value = {
            "claims": {
                "P856": [{ # P856 is "official website"
                    "mainsnak": {"datavalue": {"type": "string", "value": "https://www.example-wikidata.com"}}
                }]
            }
        }
        # Configure mock_httpx_get to return different responses based on call order or args
        mock_httpx_get.side_effect = [mock_search_response, mock_claims_response]

        url = get_wikidata_homepage("Example AG")
        self.assertEqual(url, "https://www.example-wikidata.com")
        self.assertEqual(mock_httpx_get.call_count, 2)

    @patch('search_common.httpx.get')
    def test_get_wikidata_homepage_no_p856(self, mock_httpx_get):
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {"search": [{"id": "Q12345", "label": "Example Company"}]}

        mock_claims_response = MagicMock()
        mock_claims_response.status_code = 200
        mock_claims_response.json.return_value = {"claims": {}} # No P856 claim

        mock_httpx_get.side_effect = [mock_search_response, mock_claims_response]
        url = get_wikidata_homepage("Example Company")
        self.assertIsNone(url)

    @patch('search_common.httpx.get')
    def test_get_wikidata_homepage_entity_not_found(self, mock_httpx_get):
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {"search": []} # No entity found
        mock_httpx_get.return_value = mock_search_response

        url = get_wikidata_homepage("NonExistentCompany")
        self.assertIsNone(url)
        mock_httpx_get.assert_called_once() # Only search call should happen


class TestAsyncSearchCommon(unittest.IsolatedAsyncioTestCase): # For async methods

    @patch('search_common.httpx.AsyncClient.get', new_callable=AsyncMock) # Mock async get
    async def test_is_url_relevant_to_company_relevant_title(self, mock_async_client_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<title>Example Corp Official Site</title>"
        mock_async_client_get.return_value = mock_response # mock_async_client_get is already an AsyncMock

        # Create a dummy AsyncClient instance, its .get method is already mocked by new_callable
        dummy_client = httpx.AsyncClient()

        relevant = await is_url_relevant_to_company("https://www.example.com", "Example Corp", dummy_client)
        self.assertTrue(relevant)
        mock_async_client_get.assert_awaited_once()

    @patch('search_common.httpx.AsyncClient.get', new_callable=AsyncMock)
    async def test_is_url_relevant_to_company_irrelevant_title(self, mock_async_client_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<title>News about Something Else</title>"
        mock_async_client_get.return_value = mock_response

        dummy_client = httpx.AsyncClient()
        relevant = await is_url_relevant_to_company("https://www.othersite.com", "Example Corp", dummy_client)
        self.assertFalse(relevant) # Expecting False due to title mismatch and domain mismatch

    @patch('search_common.httpx.AsyncClient.get', new_callable=AsyncMock)
    async def test_is_url_relevant_to_company_domain_match(self, mock_async_client_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Generic title, but domain will match
        mock_response.text = "<title>Welcome!</title>"
        mock_async_client_get.return_value = mock_response

        dummy_client = httpx.AsyncClient()
        # URL's domain contains "examplecorp"
        relevant = await is_url_relevant_to_company("https://www.examplecorp.com", "Example Corp", dummy_client)
        self.assertTrue(relevant)


    @patch('search_common.ChatOpenAI') # Mock the class
    def test_select_best_url_with_llm_selects_number(self, MockChatOpenAI):
        # Configure the mock instance and its invoke method
        mock_llm_instance = MockChatOpenAI.return_value
        mock_llm_response = MagicMock()
        mock_llm_response.content = "2" # LLM selects the second URL
        mock_llm_instance.invoke.return_value = mock_llm_response

        search_results = [
            {"url": "https://www.one.com", "title": "One", "description": "Site one"},
            {"url": "https://www.two.com", "title": "Two", "description": "Site two"},
        ]
        company_name = "Test Company"

        # Pass the mock_llm_instance to the function
        # The function expects an instance, so we use the return_value of the mocked class
        selected_url = select_best_url_with_llm(company_name, search_results, mock_llm_instance)
        self.assertEqual(selected_url, "https://www.two.com")
        mock_llm_instance.invoke.assert_called_once()

    @patch('search_common.ChatOpenAI')
    def test_select_best_url_with_llm_selects_none(self, MockChatOpenAI):
        mock_llm_instance = MockChatOpenAI.return_value
        mock_llm_response = MagicMock()
        mock_llm_response.content = "None"
        mock_llm_instance.invoke.return_value = mock_llm_response

        search_results = [{"url": "https://www.one.com", "title": "One", "description": "Site one"}]
        company_name = "Test Company"

        selected_url = select_best_url_with_llm(company_name, search_results, mock_llm_instance)
        self.assertIsNone(selected_url)

    @patch('search_common.ChatOpenAI')
    def test_select_best_url_with_llm_non_numeric_response(self, MockChatOpenAI):
        mock_llm_instance = MockChatOpenAI.return_value
        mock_llm_response = MagicMock()
        # LLM responds with text containing the URL instead of a number
        mock_llm_response.content = "I think https://www.thecorrectone.com is the best."
        mock_llm_instance.invoke.return_value = mock_llm_response

        search_results = [
            {"url": "https://www.one.com", "title": "One", "description": "Site one"},
            {"url": "https://www.thecorrectone.com", "title": "Correct", "description": "The right site"},
        ]
        company_name = "Test Company"

        selected_url = select_best_url_with_llm(company_name, search_results, mock_llm_instance)
        self.assertEqual(selected_url, "https://www.thecorrectone.com")


if __name__ == '__main__':
    unittest.main()
