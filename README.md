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

An intelligent company information extraction agent that uses Brave Search API and Wikidata as fallback to find official company websites, then uses MCP (Model Context Protocol) with Playwright to automatically navigate and extract detailed business data.

## Features

- 🔍 **Brave Search API Integration**: Primary search method for finding official company websites
- 📊 **Wikidata Fallback**: Secondary search using Wikidata when Brave Search fails
- 🌐 **Intelligent Website Navigation**: Automated browsing and data extraction from company websites
- 📋 **Structured JSON Output**: Consistent company data format with Swiss-specific fields
- 🤖 **AI-Powered Content Analysis**: Uses OpenAI GPT models for intelligent data extraction
- 🎭 **Browser Automation**: Playwright MCP server for reliable web scraping
- 🇨🇭 **Swiss Company Focus**: Optimized for Swiss companies with CHE numbers and local data

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
Install the project in editable mode so the `brave-search` command is available:
```bash
pip install -e .
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with your API keys:
```env
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Brave Search API Key (required for primary search method)
BRAVE_API_KEY=your_brave_api_key_here

# Google API Key (optional, for Gemini models)
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Verify Node.js and npx
```bash
node --version  # Should show v16+ 
npx --version   # Should show version number
```

## Usage

### Basic Usage

- **PowerShell**  
```powershell
# (just once per session, to allow scripts)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
# activate venv
.\venv312\Scripts\Activate.ps1
#python brave_search "Permapack"
python company_processor.py output.csv
python company_parallel_processor.py output.csv --workers 4

```

### Examples
```bash
# Search for Migros (Swiss retailer)
brave-search "Migros"

# Search for any company
brave-search "Nestlé"
brave-search "Credit Suisse"
```

### Batch CSV Usage
For processing multiple companies at once, run:
```bash
python company_processor.py input.csv output.csv
```
```bash
python company_parallel_processor.py input.csv output.csv
```
The input CSV must have the columns `company_number` and `company_name` with a header row.
The output CSV will include the original columns, the extracted data fields, and a `processing_status` column indicating the outcome (e.g., URL source, "AGENT_OK", or specific error codes like "NO_URL_FOUND", "PRE_CHECK_URL_MISMATCH", "AGENT_PROCESSING_TIMEOUT").

### Expected Output
The script's CSV output includes the following company information fields, plus a `processing_status` column:
```
company_number,company_name,official_website,founded,Hauptsitz,Firmenidentifikationsnummer,HauptTelefonnummer,HauptEmailAdresse,Geschäftsbericht,processing_status
...
1234,Example AG,https://example.com,2000,"Example Street 1, 8000 Zurich",CHE-111.222.333,+41 44 555 6677,info@example.com,https://example.com/report.pdf,Brave Search + LLM (AGENT_OK)
5678,Another Corp,null,null,null,null,null,null,null,NO_URL_FOUND
9012,Problem Inc,https://problem.inc,null,null,null,null,null,null,AGENT_PROCESSING_TIMEOUT
```
*(Note: The `ceo`, `founder`, `owner`, `employees`, `better_then_the_rest` fields mentioned in a previous version of this README are not part of the current `EXPECTED_JSON_KEYS` and thus not extracted by default by `company_processor.py` or `company_parallel_processor.py`.)*

## Configuration Files

### sequential_mcp_config.json
Configures the MCP server connection for single-threaded scripts:
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

### parallel_mcp_launcher.json
Template MCP configuration for parallel processing. The actual configuration used at runtime is dynamically generated.
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@0.0.26", "--config", "<path_to_runtime_playwright_config.json>"]
    }
  }
}
```

### .env
Contains API keys and environment variables:
```env
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Playwright MCP Server Setup

The crawler relies on Playwright’s MCP server over stdio. Due to a temporary regression in v0.0.27, we pin to v0.0.26 until it’s fixed.

### 1. Pin the MCP version

In your `sequential_mcp_config.json`, change the MCP server entry to:

```jsonc
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@0.0.26"]
    }
  }
}
```

### 2. Install the MCP package

Run **once** (you can skip `-g` if you prefer local installs):

```bash
npm i -g @playwright/mcp@0.0.26
```
This downloads the MCP server package.

### 3. Install the Playwright browsers

> **Important:** The Playwright CLI can hang or misbehave in Git-Bash/MINGW64. Use PowerShell or cmd.exe for this step.

1. **Activate your virtual environment**  
   Make sure you’re inside your project folder (where `venv312` lives), then run _exactly one_ of these, depending on your shell:

   - **PowerShell**  
     ```powershell
     # (just once per session, to allow scripts)
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

     # activate venv
     .\venv312\Scripts\Activate.ps1
     ```
   - **Command Prompt (cmd.exe)**  
     ```cmd
     venv312\Scripts\activate.bat
     ```
   - **Git Bash / WSL / other POSIX-like shells**  
     ```bash
     source venv312/Scripts/activate
     ```

   After activation you should see `(venv312)` at the start of your prompt.

2. **Install the browsers**  
   Now that your venv is active in PowerShell or cmd.exe, run:
   ```bash
   python -m playwright install
   ```
   This will download the Chromium, Firefox, and WebKit binaries that Playwright needs.

3. **Verify the installation**
   ```bash
   playwright --version
   playwright install --help
   ```
   If those commands print help text or a version number, you’re good to go!

### 4. Verify the server is reachable

(Optional) You can test that the server will accept an LSP‐style initialize over stdio:

```bash
cat << 'EOF' > init.json
{"jsonrpc":"2.0","id":0,"method":"initialize","params":{
   "protocolVersion":"1.0.0",
   "capabilities":{},
   "clientInfo":{"name":"mcp_manual_test","version":"1.0.0"}
}}
EOF

length=$(wc -c < init.json)
printf 'Content-Length: %d\r\n\r\n' "$length"
cat init.json \
  | npx -y @playwright/mcp@0.0.26
```

You should get back a JSON‐RPC `result` message. If you see it, the server is healthy and your Python client will be able to complete the handshake.

---

Once that’s in place, your Python script’s `MCPAgent` will move past the “Initializing MCP session” hang and begin driving Playwright automatically.

## Troubleshooting

### Common Issues

#### 1. PowerShell Execution Policy Error
**Error:** `running scripts is disabled on this system`
**Solution:** 
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. Missing Dependencies
**Error:** `ModuleNotFoundError: No module named 'mcp_use'`
**Solution:** 
```bash
pip install mcp-use langchain-openai python-dotenv
```

#### 3. Node.js/npx Not Found
**Error:** `'npx' is not recognized as an internal or external command`
**Solution:** Install Node.js from https://nodejs.org/

#### 4. OpenAI API Key Missing
**Error:** `openai.AuthenticationError`
**Solution:** Add your OpenAI API key to the `.env` file

#### 5. Brave API Key Missing
**Error:** `Warning: BRAVE_API_KEY is not set. Brave Search will be skipped.`
**Solution:** Add your Brave Search API key to the `.env` file

#### 6. MCP Connection Issues
**Error:** Connection timeouts or MCP server errors
**Solution:** 
- Ensure npx is working: `npx --version`
- Check internet connection
- Verify Playwright MCP server can be downloaded

### Debug Mode
Enable debug logging by modifying the script:
```python
# Enable mcp_use debug logging
Logger.set_debug(2)  # Change to 2 for verbose debugging
```

## How It Works

1. **Initialization**: Loads environment variables and creates MCP client
2. **Primary Search**: Uses Brave Search API to find official company website.
3. **Fallback Search**: If Brave Search fails, queries Wikidata for official website.
4. **URL Pre-check**: Performs a quick relevance check on the found URL by examining its title and domain against the company name.
5. **Website Navigation**: Uses Playwright MCP server to open the found website (if relevant).
6. **Data Extraction**: Crawls relevant pages (about, contact, impressum, etc.) with a 35-second timeout for the agent's web interaction and data extraction task per company.
7. **AI Analysis**: Uses GPT to extract structured company information.
8. **Output**: Writes results to a CSV file, including a `processing_status` column detailing the outcome for each company.

## Supported Models

### OpenAI Models
- `gpt-4.1-mini` (default)
- `gpt-4o`
- `gpt-4`

### Google Models (commented out by default)
- `gemini-2.5-pro-preview-05-06`

## File Structure
```
BraveWebCrawler/
├── .env                     # Environment variables (API keys, etc.)
├── brave_search.py          # CLI script for single company search
├── company_processor.py     # Script for sequential batch processing
├── company_parallel_processor.py # Script for parallel batch processing
├── search_common.py         # Common URL discovery utilities
├── sequential_mcp_config.json       # MCP config for single-threaded scripts
├── parallel_mcp_launcher.json # Template MCP config for parallel script
├── pyproject.toml           # Project metadata and package setup
├── requirements.txt         # Python package dependencies
├── README.md                # This file
├── DOCUMENTATION.md         # Detailed technical documentation
└── ... (other generated files like __pycache__, .egg-info, venv312/)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on the repository
