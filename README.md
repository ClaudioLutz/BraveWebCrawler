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
cd startpage-search-agent
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
```bash
pip install mcp-use langchain-openai python-dotenv httpx
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
python brave_search.py "Company Name"
```

### Examples
```bash
# Search for Migros (Swiss retailer)
python brave_search.py "Migros"

# Search for any company
python brave_search.py "Nestlé"
python brave_search.py "Credit Suisse"
```

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
  "Standorte": ["Zurich", "Geneva", "Basel"],
  "Firmenidentifikationsnummer": "CHE-123.456.789",
  "HauptTelefonnummer": "+41 44 123 45 67",
  "HauptEmailAdresse": "info@company.com"
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
brave-search-agent/
├── brave_search.py          # Main script
├── startpage_mcp.json       # MCP server configuration
├── .env                     # Environment variables (not tracked)
├── .gitignore              # Git ignore file
├── README.md               # This file
├── DOCUMENTATION.md        # Detailed technical documentation
└── venv312/               # Python virtual environment (not tracked)
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
