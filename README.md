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
```bash
brave-search "Company Name"
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
The input CSV must have the columns `company_number` and `company_name` with a header row.

### Expected Output
The script returns a JSON object with the following company information:
```json
{
  "official_website": "https://company.com",
  "ceo": "John Doe",
  "founder": "Jane Smith, Bob Johnson",
  "owner": "Parent Company Ltd",
  "employees": "1000-1500",
  "founded": "1995",
  "better_then_the_rest": "innovative technology solutions",
  "Hauptsitz": "Bahnhofstrasse 1, 8001 Zürich",
  "Firmenidentifikationsnummer": "CHE-123.456.789",
  "HauptTelefonnummer": "+41 44 123 45 67",
  "HauptEmailAdresse": "info@company.com",
  "Geschäftsbericht": "https://company.com/annual-report.pdf"
}
```

## Configuration Files

### startpage_mcp.json
Configures the MCP server connection:
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
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

In your `startpage_mcp.json`, change the MCP server entry to:

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
2. **Primary Search**: Uses Brave Search API to find official company website
3. **Fallback Search**: If Brave Search fails, queries Wikidata for official website
4. **Website Navigation**: Uses Playwright MCP server to open the found website
5. **Data Extraction**: Crawls relevant pages (about, contact, impressum, etc.)
6. **AI Analysis**: Uses GPT to extract structured company information
7. **Output**: Returns formatted JSON with Swiss-specific company details

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
├── brave_search.py          # CLI script for single search
├── company_processor.py     # Batch processing from CSV
├── search_common.py         # Shared utilities for Brave/Wikidata search & LLM URL selection
├── startpage_mcp.json       # MCP server configuration
├── pyproject.toml           # Package setup
├── requirements.txt         # Dependencies
├── README.md
├── DOCUMENTATION.md
└── ...
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
