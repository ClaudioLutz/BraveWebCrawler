# Technical Documentation - Brave Search Company Agent

## 1. Overview

This document provides detailed technical information about the Brave Search Company Agent. The agent's primary goal is to find official company websites using the Brave Search API (with Wikidata as a fallback) and then employ a Playwright-driven agent (via MCP - Model Context Protocol) to automatically navigate these websites and extract predefined business data.

The system has been refactored to centralize core processing logic, enhance robustness through API retries and structured logging, and improve usability with features like processing resumption.

## 2. Architecture

### 2.1. System Components & Flow

The overall process flow involves script execution, configuration loading, URL discovery, optional agent-based data extraction, and result output.

```mermaid
flowchart TD
    A[Start Script e.g., brave_processor.py] --> B(Load .env & Configs)
    B --> C{Input: Company List CSV}
    C --> D[Read Companies]
    D --> E{Resume? Check Output CSV}
    E -- Yes --> F[Load Processed Company Numbers & Existing Data]
    E -- No --> G[Initialize Fresh Output Buffer]
    F --> H[Filter Company List (Skip Processed)]
    G --> H
    H --> I{For each Company in (Filtered) List}
    I -- Start Processing --> J[core_processing.process_company_info]
    J --> K[Append Result to Output Buffer]
    K --> I
    I -- Loop End --> L[Write Output Buffer to CSV]
    L --> M[End]

    subgraph CoreProcessing [core_processing.process_company_info]
        direction LR
        CP1[Initialize LLMs & MCP Client] --> CP2[URL Discovery & Validation]
        CP2 -- URL Found & Valid --> CP3[MCPAgent Execution]
        CP2 -- No Valid URL --> CP4[Return Error/Status]
        CP3 -- Data Extracted --> CP4
        CP3 -- Error/Timeout --> CP4
        CP4 --> CP5[Cleanup MCP/Browser Resources]
    end

    J --> CoreProcessing

    subgraph URLDiscovery [URL Discovery & Validation in search_common.py]
      direction TB
      UD1[Brave Search API Call] --> UD2{URL Found?}
      UD2 -- No --> UD3[Wikidata API Call]
      UD2 -- Yes --> UD4[Pre-screen URL Relevance]
      UD3 --> UD4
      UD4 --> UD5[Return Best URL/Status]
    end
    CoreProcessing --> URLDiscovery
```

**Key Components:**

*   **Main Scripts (`brave_search.py`, `brave_processor.py`, `brave_parallel_processing.py`):** User-facing scripts to initiate processing for single companies or batches. They handle argument parsing, loading initial configurations, and invoking `core_processing.py`.
*   **`core_processing.py`:** The central module orchestrating the entire data extraction workflow for a single company via its `process_company_info` function. It manages LLM initialization, URL discovery (delegated to `search_common.py`), MCP client setup (static or dynamic), agent execution, and resource cleanup.
*   **`search_common.py`:** Contains utility functions for URL discovery (Brave Search, Wikidata) and URL relevance pre-screening. API calls here are enhanced with `tenacity` for retries.
*   **`logging_utils.py`:** Provides a `setup_logging` function to configure standardized logging (format, level, console/file output) across the application.
*   **Configuration Files:**
    *   `.env`: For API keys and environment-specific settings.
    *   `sequential_mcp_config.json`: Static MCP configuration for sequential processing.
    *   `parallel_mcp_launcher.json`: Template for dynamic MCP configuration in parallel processing.
*   **`tests/` directory:** Contains unit tests for various components (e.g., `test_search_common.py`).

### 2.2. Technology Stack

-   **Python 3.11+**: Main runtime environment.
-   **`httpx`**: Asynchronous HTTP client for API requests.
-   **Brave Search API**: Primary source for company website URLs.
-   **Wikidata API**: Fallback source for company website URLs.
-   **`mcp_sdk` (or `mcp_use`)**: MCP client library for Python (actual import may vary).
-   **`langchain` & `langchain_openai`**: LLM integration framework.
-   **OpenAI GPT Models**: For URL selection, relevance checking, and agent-based data extraction. Model names are configurable.
-   **Playwright & MCP Server**: For browser automation and web interaction.
-   **`tenacity`**: For implementing retry mechanisms on external API calls.
-   **`logging` (Python built-in)**: For structured application logging.
-   **`argparse`**: For command-line argument parsing.
-   **`csv`**: For reading and writing CSV files.
-   **`multiprocessing`**: For parallel execution in `brave_parallel_processing.py`.
-   **`pytest` & `pytest-asyncio`**: For running unit tests.

## 3. Code Architecture

### 3.1. `core_processing.py` - Central Workflow

The `process_company_info` async function is the heart of the application.

```python
async def process_company_info(
    company_name: str,
    company_number: str,
    config: dict, # Main configuration dictionary
    mcp_config_path_or_object: str | dict # For MCP client setup
) -> dict:
    # ...
```

**Parameters:**

*   `company_name` (str): Name of the company.
*   `company_number` (str): Identifier for the company.
*   `config` (dict): A dictionary containing various operational parameters:
    *   `OPENAI_API_KEY` (str): OpenAI API key.
    *   `BRAVE_API_KEY` (str, optional): Brave Search API key.
    *   `URL_LLM_MODEL` (str): Model name for URL selection/validation (e.g., "gpt-4.1-mini").
    *   `AGENT_LLM_MODEL` (str): Model name for the data extraction agent (e.g., "gpt-4.1-mini").
    *   `LLM_RELEVANCE_CHECK_MODEL` (str, optional): Specific model for URL relevance if different from `URL_LLM_MODEL`.
    *   `AGENT_MAX_STEPS` (int): Maximum steps for the MCPAgent.
    *   `AGENT_TIMEOUT` (int): Timeout in seconds for the MCPAgent's run.
*   `mcp_config_path_or_object` (str | dict):
    *   If `str`: Path to a static MCP configuration file (e.g., `sequential_mcp_config.json`).
    *   If `dict`: Parameters for dynamic MCP setup (used by parallel processing):
        *   `base_mcp_launcher_path` (str): Path to the base MCP launcher template (e.g., `parallel_mcp_launcher.json`).
        *   `headless` (bool): Whether to run the browser in headless mode.

**Returns:**

A dictionary containing:
*   `company_name`, `company_number`
*   `final_url`: The determined official URL (or "null").
*   `source_of_url`: How the URL was found (e.g., "Brave Search + LLM", "Wikidata").
*   `processing_status`: Overall status (e.g., "AGENT_OK", "NO_URL_FOUND", "AGENT_PROCESSING_TIMEOUT").
*   `error_message`: Any error message encountered (or `None`).
*   Keys from `EXPECTED_JSON_KEYS` with extracted data or "null".

**Workflow within `process_company_info`:**
1.  Initializes LLMs (for URL selection and agent) based on `config`.
2.  Calls `search_common.discover_urls` to find the company website.
3.  Calls `search_common.pre_screen_urls` to validate URL relevance.
4.  If a valid URL is found:
    *   Sets up `MCPClient` based on `mcp_config_path_or_object` (static or dynamic).
    *   Initializes `MCPAgent`.
    *   Generates and runs the agent prompt for data extraction.
    *   Parses the agent's result.
5.  Performs cleanup of MCP server and temporary browser profiles in a `finally` block.
6.  Returns the structured result dictionary.

### 3.2. `search_common.py` - URL Discovery & Utilities

This module provides functions for:
*   `get_brave_search_candidates()`: Fetches URLs from Brave Search API.
*   `get_wikidata_homepage()`: Fetches homepage URL from Wikidata.
*   `select_best_url_with_llm()`: Uses an LLM to pick the best URL from candidates.
*   `is_url_relevant_to_company()`: Asynchronously checks if a URL's content seems relevant to the company.
*   `discover_urls()`: Orchestrates the use of the above functions to find the best URL.
*   `pre_screen_urls()`: Uses `is_url_relevant_to_company` for a list of URLs.
    API calls within this module (`httpx.get`, `httpx.AsyncClient.get`) are decorated with `tenacity` for retries.

### 3.3. `logging_utils.py` - Logging Configuration

Provides `setup_logging(log_level, log_file)` to initialize Python's `logging` system with a consistent format and output handlers (console, optional file). This is called by the main executable scripts.

### 3.4. Main Executable Scripts

*   **`brave_search.py`**: CLI for single company processing. Uses static MCP config.
*   **`brave_processor.py`**: Processes a CSV of companies sequentially. Uses static MCP config. Implements resume functionality.
*   **`brave_parallel_processing.py`**: Processes a CSV of companies in parallel using `multiprocessing`. Uses dynamic MCP config. Implements resume functionality.

## 4. Configuration

### 4.1. Environment Variables (`.env` file)

*   `OPENAI_API_KEY` (required): Your OpenAI API key.
*   `BRAVE_API_KEY` (required for Brave Search): Your Brave Search API key.
*   `URL_LLM_MODEL` (optional): Model for URL selection/validation (e.g., "gpt-4.1-mini", "gpt-3.5-turbo"). Default: "gpt-4.1-mini".
*   `AGENT_LLM_MODEL` (optional): Model for the data extraction agent (e.g., "gpt-4.1-mini", "gpt-4-turbo"). Default: "gpt-4.1-mini".
*   `LLM_RELEVANCE_CHECK_MODEL` (optional): Specific model for URL relevance checks if different from `URL_LLM_MODEL`.
*   `LOG_LEVEL` (optional): Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: "INFO".
*   `LOG_FILE_BRAVE_SEARCH` (optional): Log file name for `brave_search.py`. Default: "brave_search.log".
*   `LOG_FILE_BRAVE_PROCESSOR` (optional): Log file name for `brave_processor.py`. Default: "brave_processor.log".
*   `LOG_FILE_BRAVE_PARALLEL` (optional): Log file name for `brave_parallel_processing.py`. Default: "brave_parallel_processing.log".
*   `HEADLESS_BROWSING` (optional, for `brave_parallel_processing.py`): "True" or "False". Controls if worker browsers run headless. Default: "True".
*   `AGENT_MAX_STEPS` (optional): Max steps for the agent. Default: 25-30.
*   `AGENT_TIMEOUT` (optional): Timeout for agent execution per company. Default: 60 seconds.

### 4.2. Command-Line Arguments

*   **`brave_search.py <company_name>`**:
    *   `company_name`: The name of the company to search.
*   **`brave_processor.py <input_csv> <output_csv>`**:
    *   `input_csv`: Required path to the input CSV file.
    *   `output_csv`: Required path for the output CSV file.
*   **`brave_parallel_processing.py <input_csv> <output_csv> [--workers N]`**:
    *   `input_csv`: Required path to the input CSV file.
    *   `output_csv`: Required path for the output CSV file.
    *   `--workers N` (optional): Number of worker processes. Defaults to CPU count.

### 4.3. MCP Configuration Files

*   **`sequential_mcp_config.json`**:
    *   Used by: `brave_search.py`, `brave_processor.py`, and `core_processing.py` when passed a string path.
    *   Defines how to launch the Playwright MCP server for single-threaded operation.
*   **`parallel_mcp_launcher.json`**:
    *   Used as a template by `core_processing.py` when `mcp_config_path_or_object` is a dictionary (typically for `brave_parallel_processing.py`).
    *   Contains placeholders that `core_processing.py` fills to point to dynamically generated, worker-specific Playwright configurations ensuring isolated browser sessions.

## 5. MCP (Model Context Protocol) Integration Details

The `core_processing.process_company_info` function handles MCP client setup:

1.  **Static Configuration (String Path):**
    *   If `mcp_config_path_or_object` is a string, it's treated as a path to a JSON file (e.g., `sequential_mcp_config.json`).
    *   `MCPClient.from_config_file()` is called with this path.
    *   This mode is used by `brave_search.py` and `brave_processor.py`.

2.  **Dynamic Configuration (Dictionary Object):**
    *   If `mcp_config_path_or_object` is a dictionary, it must contain parameters for dynamic setup:
        *   `base_mcp_launcher_path` (str): Path to a template launcher JSON (e.g., `parallel_mcp_launcher.json`).
        *   `headless` (bool): Controls headless mode for the browser.
    *   `core_processing.py` then:
        *   Creates a unique temporary directory for the browser user profile (`userDataDir`).
        *   Generates a `runtime-playwright-config.json` in this temp directory, specifying the `userDataDir` and `headless` mode. Window positioning arguments are added if not headless.
        *   Copies the `base_mcp_launcher_path` template to `runtime-mcp-launcher.json` within the temp directory.
        *   Modifies this `runtime-mcp-launcher.json` to point its Playwright server's `--config` argument to the generated `runtime-playwright-config.json`.
        *   `MCPClient.from_config_file()` is called with the path to this dynamic `runtime-mcp-launcher.json`.
    *   This mode is used by `brave_parallel_processing.py` to ensure each worker process has an isolated browser environment.
    *   The temporary directory is cleaned up in the `finally` block of `process_company_info`.

## 6. Error Handling and Robustness

Several mechanisms enhance the application's robustness:

*   **API Retries (`tenacity`):**
    *   Functions in `search_common.py` that make external HTTP calls (e.g., `get_brave_search_candidates`, `get_wikidata_homepage`, `is_url_relevant_to_company`) are decorated with `tenacity.@retry`.
    *   Retries occur for transient network errors, timeouts (`httpx.RequestError`, `httpx.TimeoutException`), and 5xx server errors (`httpx.HTTPStatusError`).
    *   The strategy involves up to 3 attempts with exponential backoff (1s multiplier, min 2s, max 10s).
    *   If all retries fail, the last exception is reraised and handled by the calling functions.
*   **Structured Logging:**
    *   Python's `logging` module is used throughout the application.
    *   Configuration is centralized via `logging_utils.setup_logging`.
    *   Logs include timestamps, logger name (module), log level, process ID, and the message.
    *   Output can go to console and/or a log file (e.g., `brave_processor.log`).
    *   Log level is configurable via the `LOG_LEVEL` environment variable.
    *   Detailed error information, including stack traces (via `logger.exception()` or `exc_info=True`), is recorded for easier debugging.
*   **Browser Cleanup:**
    *   The `finally` block in `core_processing.process_company_info` ensures cleanup attempts:
        1.  Graceful shutdown of the MCP client/server (`client_mcp.stop_servers()`). This call is wrapped in a `try-except` to prevent its failure from halting further cleanup.
        2.  Forceful termination of any lingering browser processes associated with the temporary profile directory using `kill_chrome_processes_using(tmp_profile_dir)`.
        3.  Removal of the temporary profile directory using `rmtree_with_retry(tmp_profile_dir)`, which itself has retry logic.
*   **Timeout for Agent:** The `AGENT_TIMEOUT` configuration for `core_processing.process_company_info` prevents individual company processing from hanging indefinitely.

## 7. Key Features

### 7.1. Resume From Previous Run

The batch processing scripts (`brave_processor.py` and `brave_parallel_processing.py`) support resuming from a previous incomplete run.

**Mechanism:**

1.  **Check Output File:** When a script starts, it checks if the specified output CSV file already exists.
2.  **Read Processed IDs:** If the file exists and its header is compatible with the expected format, the script reads all rows. It extracts the `company_number` from each row and stores these in a set of `processed_company_numbers`. The existing data is loaded into an in-memory buffer.
3.  **Filter Input:** As the script reads companies from the input CSV, it checks if a company's `company_number` is already in the `processed_company_numbers` set.
4.  **Skip or Process:**
    *   If the company number is found in the set, the company is skipped, and a log message is generated.
    *   If not found, the company is processed as usual.
5.  **Combine and Write:**
    *   New results (for companies processed in the current run) are appended to the in-memory buffer (which contains old data if resumed).
    *   The output CSV file is overwritten with the entire content of this buffer upon periodic saves (for `brave_processor.py`) and at the end of the script. This ensures the output file is always complete and consistent.

This feature is particularly useful for long-running batch jobs, allowing them to continue from where they left off without reprocessing already completed entries.

## 8. Testing

A basic unit testing strategy is in place, with tests located in the `tests/` directory.

*   **Frameworks:**
    *   `unittest`: Python's built-in unit testing framework.
    *   `unittest.mock`: For creating mock objects and patching.
    *   `pytest`: A popular third-party test runner (can run `unittest`-based tests).
    *   `pytest-asyncio`: For running asynchronous tests with `pytest`.
*   **Current Test Files:**
    *   `tests/__init__.py`: Makes the `tests` directory a package.
    *   `tests/test_search_common.py`: Contains unit tests for functions in `search_common.py`.
*   **What is Tested:**
    *   **`search_common.py` utilities:**
        *   `get_brave_search_candidates`: Mocking `httpx.get` to test successful API responses, error handling, blacklist filtering, and API key presence.
        *   `get_wikidata_homepage`: Mocking `httpx.get` for Wikidata API calls, testing different scenarios (entity found with/without website, entity not found).
        *   `is_url_relevant_to_company`: Asynchronous tests using `unittest.IsolatedAsyncioTestCase` (or `pytest-asyncio` compatible `async def` methods), mocking `httpx.AsyncClient.get` to test URL relevance logic based on page titles and domains.
        *   `select_best_url_with_llm`: Mocking the `ChatOpenAI` client and its `invoke` method to test various LLM response patterns.
*   **Running Tests:**
    Tests can typically be run using:
    ```bash
    python -m unittest discover tests
    # OR
    pytest
    ```
    (Ensure `pytest` and `pytest-asyncio` are installed if using `pytest`.)

## 9. File Structure (Updated)

```
BraveWebCrawler/
├── .env                     # Environment variables
├── .gitignore
├── brave_search.py          # CLI for single company
├── brave_processor.py       # Sequential batch processor
├── brave_parallel_processing.py # Parallel batch processor
├── core_processing.py       # Central processing logic <--- NEW
├── search_common.py         # URL discovery utilities
├── logging_utils.py         # Logging setup <--- NEW
├── sequential_mcp_config.json
├── parallel_mcp_launcher.json # Template for parallel runs
├── pyproject.toml
├── requirements.txt         # Dependencies
├── README.md                # User guide
├── DOCUMENTATION.md         # This file
├── input/                   # Example input CSV directory
│   └── example_input.csv
├── output/                  # Default directory for output CSVs
├── tests/                   # Unit tests <--- NEW
│   ├── __init__.py
│   └── test_search_common.py
└── venv312/                 # Python virtual environment
```

## 10. Playwright MCP Server Setup
(This section remains largely the same as provided in the prompt, with minor context adjustments if needed)
...

## 11. Troubleshooting Guide
(This section remains largely the same, with updates to refer to new log files and `LOG_LEVEL`)
...

## 12. Future Enhancements
...
## 13. Contributing
...
```
