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
                         (  DATA  )         \   
                 `--~~~~~~~~~~~~~~~~~~--'
```
# Brave Search Company Agent

An intelligent company information extraction agent that uses Brave Search API and Wikidata as fallback to find official company websites, then uses MCP (Model Context Protocol) with Playwright to automatically navigate and extract detailed business data. The core processing logic is centralized in `core_processing.py`, featuring API retry mechanisms for robustness and structured logging for better traceability.

## Features

- 🔍 **Brave Search API Integration**: Primary search method for finding official company websites
- 📊 **Wikidata Fallback**: Secondary search using Wikidata when Brave Search fails
- 🌐 **Intelligent Website Navigation**: Automated browsing and data extraction from company websites
- 📋 **Structured JSON Output**: Consistent company data format with Swiss-specific fields
- 🤖 **AI-Powered Content Analysis**: Uses OpenAI GPT models for intelligent data extraction
- 🎭 **Browser Automation**: Playwright MCP server for reliable web scraping
- 🇨🇭 **Swiss Company Focus**: Optimized for Swiss companies with CHE numbers and local data
- ♻️ **Resume Capability**: Batch processing scripts can resume from previous runs.
- 🛡️ **Robust API Calls**: Retry mechanisms for external API calls using `tenacity`.
- 📝 **Structured Logging**: Detailed logging using Python's `logging` module.

## Prerequisites

Before running this script, ensure you have the following installed:

### System Requirements
- **Python 3.11+**
- **Node.js** (v16 or higher)
- **npm/npx** (comes with Node.js)

### PowerShell Execution Policy (Windows)
If you're on Windows, you need to allow PowerShell script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd BraveWebCrawler
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv312
# On Windows:
venv312\Scripts\activate
# On macOS/Linux:
source venv312/bin/activate
```

### 3. Install Python Dependencies
Install the project in editable mode so the `brave-search` command is available. The `requirements.txt` includes libraries like `tenacity` for retries and `pytest` for testing.
```bash
pip install -e .
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with your API keys and other configurations:
```env
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Brave Search API Key (required for primary search method)
BRAVE_API_KEY=your_brave_api_key_here

# Google API Key (optional, for Gemini models if integrated)
# GOOGLE_API_KEY=your_google_api_key_here

# LLM Model Configuration (optional, defaults will be used if not set)
URL_LLM_MODEL="gpt-4.1-mini"      # Model for URL selection and relevance checks
AGENT_LLM_MODEL="gpt-4.1-mini"    # Model for the main data extraction agent

# Logging Configuration (optional)
LOG_LEVEL="INFO"                  # Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# LOG_FILE_BRAVE_SEARCH="brave_search.log" # Specific log file for brave_search.py
# LOG_FILE_BRAVE_PROCESSOR="brave_processor.log"
# LOG_FILE_BRAVE_PARALLEL="brave_parallel_processing.log"
# HEADLESS_BROWSING="True" # For brave_parallel_processing.py, True or False
```
- `URL_LLM_MODEL`: Specifies the language model used for selecting the best URL from search results and for pre-screening URL relevance. Defaults to "gpt-4.1-mini" if not set.
- `AGENT_LLM_MODEL`: Specifies the language model used by the main agent for data extraction from websites. Defaults to "gpt-4.1-mini" if not set.
- `LOG_LEVEL`: Controls the verbosity of logs. Set to `DEBUG` for detailed diagnostic information.

### 5. Verify Node.js and npx
```bash
node --version  # Should show v16+ 
npx --version   # Should show version number
```

## Usage

### Basic Usage (Single Company)

```bash
# Search for Migros (Swiss retailer)
brave-search "Migros"

# Search for any company
brave-search "Nestlé"
```
This script uses `sequential_mcp_config.json`.

### Batch CSV Usage
For processing multiple companies from a CSV file:

**Sequential Processing:**
```bash
python brave_processor.py path/to/your_input.csv path/to/your_output.csv
```
Example:
```bash
python brave_processor.py input/my_companies.csv output/sequential_results.csv
```

**Parallel Processing:**
```bash
python brave_parallel_processing.py path/to/your_input.csv path/to/your_output.csv --workers 4
```
Example:
```bash
python brave_parallel_processing.py input/my_companies.csv output/parallel_results.csv --workers 4
```

The input CSV must have columns named `company_number` and `company_name` (a header row is expected).
The output CSV will include the original columns, the extracted data fields (as per `EXPECTED_JSON_KEYS` in `core_processing.py`), `final_url`, `source_of_url`, `processing_status`, and `error_message`.

**Resume Capability:** The batch processing scripts (`brave_processor.py` and `brave_parallel_processing.py`) can resume from a previous run if an output file exists at the specified location. Already processed companies (identified by `company_number` in the output file) will be skipped.

### Expected Output Fields
The script's CSV output includes: `company_number`, `company_name`, `final_url`, `source_of_url`, `official_website`, `founded`, `Hauptsitz`, `Firmenidentifikationsnummer`, `HauptTelefonnummer`, `HauptEmailAdresse`, `Geschäftsbericht`, `extracted_company_name`, `processing_status`, and `error_message`.

## Logging and Troubleshooting

The scripts now use Python's standard `logging` module for more detailed and structured output.
- **Log Files**: Each main script (`brave_search.py`, `brave_processor.py`, `brave_parallel_processing.py`) generates its own log file in the project root (e.g., `brave_search.log`). These files contain detailed operational logs, including debug messages, informational updates, warnings, and errors.
- **Log Level**: The verbosity of the logs can be controlled using the `LOG_LEVEL` environment variable (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`). The default is `INFO`.
- **Console Output**: Basic progress and critical errors are also shown on the console.

### Common Issues

#### 1. PowerShell Execution Policy Error
**Error:** `running scripts is disabled on this system`
**Solution:** 
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. Missing Dependencies
**Error:** `ModuleNotFoundError: No module named 'some_module'`
**Solution:** Ensure all dependencies are installed:
```bash
pip install -e .
# or pip install -r requirements.txt
```

#### 3. Node.js/npx Not Found
**Error:** `'npx' is not recognized as an internal or external command`
**Solution:** Install Node.js from https://nodejs.org/

#### 4. API Key Missing
**Error:** Authentication errors or warnings about missing keys.
**Solution:** Ensure `OPENAI_API_KEY` and `BRAVE_API_KEY` are correctly set in your `.env` file.

#### 5. MCP Connection Issues
**Error:** Connection timeouts or MCP server errors.
**Solution:** 
- Ensure npx is working: `npx --version`
- Check internet connection.
- Verify Playwright MCP server can be downloaded (see Playwright MCP Server Setup).
- Check the script-specific log file for detailed MCP error messages.

### Debug Mode for `mcp_use`
The `mcp_use` library has its own logger. To enable its debug logging (if troubleshooting MCP interactions):
```python
# In the relevant script, e.g., core_processing.py if mcp_sdk is a wrapper around mcp_use
# from mcp_use.logging import Logger
# Logger.set_debug(2) # Change to 2 for verbose MCP debugging
```
Note: This is separate from the main application logging configured via `logging_utils.py`.

## How It Works

1.  **Configuration**: Loads API keys and settings from environment variables. Logging is initialized.
2.  **Core Processing (`core_processing.py`)**: This centralized module handles the main workflow:
    *   **URL Discovery (`search_common.py`)**:
        *   Uses Brave Search API as the primary method.
        *   Falls back to Wikidata if Brave Search yields no results.
        *   External API calls (Brave, Wikidata, page content fetching) are made more robust with retry mechanisms (`tenacity`).
    *   **URL Pre-screening**: A quick check is performed on the discovered URL's content to assess relevance to the company.
    *   **MCP Agent Interaction**:
        *   If a relevant URL is found, an `MCPAgent` is initialized.
        *   The agent navigates the website using Playwright (managed by an MCP server).
        *   It uses an LLM (specified by `AGENT_LLM_MODEL`) to understand page content and extract predefined data fields.
    *   **Cleanup**: Ensures browser instances and temporary profiles are properly shut down and removed.
3.  **Output**: Results, including extracted data and processing status, are returned. For batch scripts, these are written to a CSV file. Structured logs provide detailed insight into each step.

## Supported Models

### OpenAI Models (Configurable via Env Vars)
- `URL_LLM_MODEL`: For URL selection/validation (default: "gpt-4.1-mini")
- `AGENT_LLM_MODEL`: For data extraction by the agent (default: "gpt-4.1-mini")
- Other compatible OpenAI models can be specified.

## File Structure
```
BraveWebCrawler/
├── .env                     # Environment variables (API keys, etc.)
├── brave_search.py          # CLI script for single company search
├── brave_processor.py       # Script for sequential batch processing
├── brave_parallel_processing.py # Script for parallel batch processing
├── core_processing.py       # Handles the core logic of company data processing
├── search_common.py         # Common URL discovery and pre-screening utilities
├── logging_utils.py         # Provides logging configuration
├── sequential_mcp_config.json # MCP config for single-threaded scripts
├── parallel_mcp_launcher.json # Template MCP config for parallel script
├── pyproject.toml           # Project metadata and package setup
├── requirements.txt         # Python package dependencies
├── README.md                # This file
├── DOCUMENTATION.md         # Detailed technical documentation
├── input/                     # Default directory for input CSV files
│   └── example_input.csv
├── output/                    # Default directory for output CSV files
└── tests/                     # Unit tests
    ├── __init__.py
    └── test_search_common.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (add unit tests if applicable)
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the Troubleshooting section and generated log files.
2. Review the documentation.
3. Open an issue on the repository.
```
