```ascii
                 .--~~~~~~~~~~~~~~~~~~--.
                /                        \
               |                          |
               |                          |
               |                          |
               |    C  R  A  W  L  E  R   |
               |                          |
                \                        /
                 `--~~~~~~~~~~~~~~~~~~--'
          \   \   \   \      ||      /   /   /   /
           \   \   \   \     ||     /   /   /   /
            \   \   \   \    ||    /   /   /   /
             *------------*--::--*------------*
            /   /   /   /    ||    \   \   \   \
           /   /   /   /     ||     \   \   \   \
          /   /   /   /      ||      \   \   \   \
   <<==================================================>>
          \   \   \   \      ||      /   /   /   /
           \   \   \   \     ||     /   /   /   /
            \   \   \   \    ||    /   /   /   /
             *------------*--::--*------------*
            /   /   /   /    ||    \   \   \   \
           /   /   /   /     ||     \   \   \   \
          /   /   /   /      ||      \   \   \   \
                 .--~~~~~~~~~~~~~~~~~~--.
                     (  COMPANY-DATA  )         \   
                 `--~~~~~~~~~~~~~~~~~~--'
```
# Company Data Extraction Agent

An intelligent agent suite for finding official company websites and extracting key business data. It utilizes various search strategies and employs an AI-powered agent with Playwright (via MCP) for web navigation and data extraction.

## Overview

This project provides a collection of Python scripts to automate the process of:
1.  **Discovering Company Websites:** Using Brave Search API, Google Custom Search API (both with Wikidata fallback), or by tasking an AI agent to search on Startpage.com.
2.  **Extracting Data:** An AI agent, powered by OpenAI's GPT models and controlling a browser via Playwright (through the Model Context Protocol - MCP), navigates the identified websites to extract structured information.

The system is designed to be flexible, offering scripts for single company lookups, sequential batch processing, and parallel batch processing for enhanced speed.

## Features

-   🌐 **Multiple URL Discovery Strategies**:
    -   Brave Search API + Wikidata fallback
    -   Google Custom Search API + Wikidata fallback
    -   Agent-driven Startpage.com search
-   🤖 **AI-Powered Agent**: Uses OpenAI GPT models for intelligent URL selection (for API methods) and data extraction from web pages.
-   🎭 **Advanced Browser Automation**: Leverages Playwright through the Model Context Protocol (MCP) for robust and reliable web interaction. This allows the agent to understand and interact with web pages based on their structure (accessibility tree) rather than relying on visual screenshots.
-   ⚙️ **Processing Modes**:
    -   Single company CLI scripts.
    -   Sequential batch processing from CSV.
    -   Parallel batch processing from CSV for significantly faster execution, with isolated browser environments per task.
-   📋 **Structured JSON Output**: Extracted data is saved in a consistent format, including a detailed `processing_status` for each company.
-   🇨🇭 **Swiss Company Focus**: Optimized for Swiss companies, but adaptable.

## Prerequisites

-   **Python 3.11+**
-   **Node.js** (v18+ recommended for Playwright MCP)
-   **npm/npx** (comes with Node.js)

## Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd BraveWebCrawler
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv312  # Or your preferred venv name
# On Windows:
venv312\Scripts\activate
# On macOS/Linux:
source venv312/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
# If you need the CLI command 'brave-search' (for brave_search.py only):
# pip install -e .
```

### 4. Set Up Environment Variables
Create a `.env` file in the `BraveWebCrawler` directory:
```env
# OpenAI API Key (required for all agent operations)
OPENAI_API_KEY=your_openai_api_key_here

# Brave Search API Key (required for Brave Search based scripts)
BRAVE_API_KEY=your_brave_api_key_here

# Google Custom Search API Key & CX ID (required for Google Search based scripts)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CX=your_google_custom_search_engine_id # Note: This was previously GOOGLE_CX_ID in some older internal versions.
```

### 5. Install Playwright MCP Server and Browsers

**a. Install Playwright MCP Package:**
It's recommended to pin to version `0.0.26` for this project due to observed stability.
```bash
npm install -g @playwright/mcp@0.0.26
```
(You can omit `-g` for a project-local install if `npx` is configured to find local packages.)

**b. Install Playwright Browsers:**
Ensure your Python virtual environment is activated. Use PowerShell or cmd.exe on Windows for this step.
```bash
python -m playwright install
```
This downloads the necessary browser binaries (Chromium, Firefox, WebKit).

**c. PowerShell Execution Policy (Windows Only):**
If you encounter issues running scripts in PowerShell, you might need to adjust the execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Or for the current process only:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

## Usage

All scripts read input from the `input/` directory (automatically finding the newest CSV) and write output to the specified CSV file path.

### General Notes:
*   Activate your Python virtual environment before running any script.
*   Ensure the `.env` file is correctly set up with the necessary API keys for the chosen script.

### Script Examples:

**1. Brave Search Based Scripts:**
   *   **Single Company Lookup:**
      ```bash
      python BraveWebCrawler/brave_search.py "Name of Company AG"
      ```
   *   **Sequential Batch Processing:**
      ```bash
      python BraveWebCrawler/brave_processor.py path/to/your_output_brave_sequential.csv
      ```
   *   **Parallel Batch Processing:**
      ```bash
      python BraveWebCrawler/brave_parallel_processing.py path/to/your_output_brave_parallel.csv --workers 4
      ```
      (`--workers` is optional, defaults to CPU core count)

**2. Google Search Based Scripts:**
   *   **Single Company Lookup:**
      ```bash
      python BraveWebCrawler/google_search.py "Another Company GmbH"
      ```
   *   **Parallel Batch Processing:** (Sequential Google script not typically used, parallel is preferred)
      ```bash
      # Default (headless mode)
      python BraveWebCrawler/google_parallel_processing.py path/to/your_output_google_parallel.csv --workers 4
      # Headful mode
      python BraveWebCrawler/google_parallel_processing.py path/to/your_output_google_parallel.csv --workers 4 --headful
      ```
      (This script runs browsers in headless mode by default. Use `--headful` to see browser windows.)

**3. Startpage Agent Based Script:**
   *   **Parallel Batch Processing:**
      ```bash
      python BraveWebCrawler/startpage_parallel_processing.py path/to/your_output_startpage_parallel.csv --workers 4
      ```
      (This script tasks the agent to perform the search on Startpage.com directly. Typically runs non-headless.)

### Input CSV Format
For batch processing, create a CSV file (e.g., `input/companies.csv`) with at least these columns:
```csv
company_number,company_name
CH-020.3.000.001-0,Example AG
CH-020.3.000.002-8,Test Corp
```

### Output CSV
The output CSV will include the input columns plus the extracted data fields and a `processing_status` column.
Expected data fields: `official_website`, `founded`, `Hauptsitz`, `Firmenidentifikationsnummer`, `HauptTelefonnummer`, `HauptEmailAdresse`, `Geschäftsbericht`, `extracted_company_name`.

Example `processing_status` values for `google_parallel_processing.py`:
- `Google Search API (AGENT_OK)`: Successful extraction using a URL from Google Search.
- `Google Search_NO_CANDIDATES_FOUND`: Google Search found no URLs for the company.
- `Google Search_LLM_NO_URL_SELECTED`: Google Search found URLs, but the LLM didn't select one as suitable.
- `AGENT_TIMEOUT_WITH_GOOGLE_URL (Google Search API)`: Agent timed out while processing the Google-provided URL.
- `AGENT_URL_NOT_CONFIRMED_OR_DATA_NULL (Google Search API)`: Agent processed the URL but couldn't confirm it or found no data.
- `AGENT_JSON_ERROR_WITH_GOOGLE_URL (Google Search API)`: Agent returned malformed JSON.
- `Google Search_ERROR: ...`: An error occurred during the Google Search API call.
(Other scripts like `brave_parallel_processing.py` will have different source prefixes like `Brave Search + LLM (...)` or `Startpage Agent (...)`.)
Common general statuses include:
- `NO_URL_FOUND` (if no strategy yields a URL)
- `AGENT_PROCESSING_TIMEOUT` (general timeout)
- `TEMP_DIR_CREATION_ERROR`
- `POOL_EXECUTION_ERROR`

## How It Works (Simplified)

1.  **Load Company Data**: Scripts read company names (and numbers) from an input CSV.
2.  **URL Discovery**:
    *   **API Methods (Brave/Google)**: Query the API, use an LLM to select the best URL, optionally fallback to Wikidata. A quick relevance pre-check is done.
    *   **Startpage Method**: The agent is directly prompted to search on Startpage.com.
3.  **MCP Agent Tasking**: An AI agent is given a prompt and tools to control a Playwright browser.
    *   It navigates to the target URL (or finds one on Startpage).
    *   It attempts to extract predefined data points (founding year, address, etc.).
4.  **Results**: Extracted data and a status are saved to an output CSV.

Parallel scripts use `multiprocessing` and create isolated browser environments for each company to ensure stability and speed.

## Configuration Files

-   **`sequential_mcp_config.json`**: Used by single-threaded scripts and `brave_processor.py` to launch the Playwright MCP server.
    ```json
    {
      "mcpServers": {
        "playwright": {
          "command": "npx",
          "args": ["-y", "@playwright/mcp@0.0.26"]
        }
      }
    }
    ```
-   **`parallel_mcp_launcher.json`**: A template for parallel processing scripts. These scripts dynamically create worker-specific configurations based on this template, pointing to isolated browser profiles and setting headless/headed mode.
    ```json
    {
      "mcpServers": {
        "playwright": {
          "command": "npx",
          "args": [
            "-y", "@playwright/mcp@0.0.26",
            "--config", "<path_to_runtime_playwright_config.json>" // Placeholder
          ]
        }
      }
    }
    ```

## Troubleshooting

-   **API Key Errors**: Ensure `.env` is correctly placed in the `BraveWebCrawler` directory and has valid keys.
-   **Playwright/MCP Issues**:
    -   Confirm `npm install -g @playwright/mcp@0.0.26` was successful.
    -   Ensure `python -m playwright install` was run in the active venv.
    -   Check Node.js version (v18+ recommended).
    -   Look for errors in the console when the MCP server attempts to start.
-   **Permissions**: Especially on Windows for PowerShell or for creating temporary directories.
-   **Timeouts**: `AGENT_PROCESSING_TIMEOUT` in scripts might need adjustment for very slow websites.

For more detailed technical information, see `DOCUMENTATION.md`.

## License

This project is licensed under the MIT License.
