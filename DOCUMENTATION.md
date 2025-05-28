# Technical Documentation - Brave Search Company Agent

## Overview

This document provides detailed technical information about the Brave Search Company Agent, including architecture, implementation details, and advanced configuration options. The agent uses Brave Search API as the primary method to find official company websites, with Wikidata as a fallback, then employs MCP (Model Context Protocol) with Playwright for automated data extraction.

## Architecture

### System Components

```mermaid
graph TD
    A[brave_search.py] --> B[Brave Search API]
    A --> C[Wikidata API]
    A --> D[MCP Client]
    D --> E[Playwright MCP Server]
    E --> F[Browser Automation]
    A --> G[OpenAI LLM]
    A --> H[Environment Config]
    H --> I[.env file]
    H --> J[startpage_mcp.json]
    B --> K[Official Website URL]
    C --> K
    F --> L[Company Website]
    G --> M[JSON Output]
    K --> F
```

### Technology Stack

- **Python 3.11+**: Main runtime environment
- **httpx**: HTTP client for API requests
- **Brave Search API**: Primary search method for finding company websites
- **Wikidata API**: Fallback search method
- **mcp-use**: MCP client library for Python
- **LangChain**: LLM integration framework
- **OpenAI GPT**: Language model for content analysis
- **Playwright**: Browser automation via MCP server
- **Node.js/npx**: Runtime for Playwright MCP server

## Code Architecture

### Main Components

#### 1. Environment Setup
```python
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Enable debug logging
Logger.set_debug(2)
```

#### 2. MCP Client Configuration
```python
# Create MCPClient from config file
client = MCPClient.from_config_file(
    os.path.join(os.path.dirname(__file__), "startpage_mcp.json")
)
```

#### 3. LLM Initialization
```python
# Initialize the OpenAI client
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)
```

#### 4. Agent Creation
```python
# Create agent with the client
agent = MCPAgent(llm=llm, client=client, max_steps=30)
```

#### 5. Batch Processing
Use `company_processor.py` to process many companies from a CSV file:
```bash
python company_processor.py input.csv output.csv
```
The input file must have `company_number` and `company_name` columns.

#### 6. Shared Search Utilities (`search_common.py`)
This module consolidates common functionalities for finding company URLs, including:
- Querying the Brave Search API (`get_brave_search_candidates`).
- Querying the Wikidata API (`get_wikidata_homepage`).
- Using an LLM to select the best URL from candidates (`select_best_url_with_llm`).
These functions are parameterized to accept API keys and LLM instances directly.

## MCP (Model Context Protocol) Integration

### What is MCP?

MCP is a protocol that enables AI models to securely connect to external tools and data sources. In this project, it allows the Python script to control a browser through the Playwright MCP server.

### MCP Server Configuration

The `startpage_mcp.json` file configures the Playwright MCP server:

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

### MCP Communication Flow

1. **Initialization**: MCPClient connects to Playwright server via npx
2. **Tool Discovery**: Client discovers available browser automation tools
3. **Tool Execution**: Agent uses tools to navigate and extract data
4. **Data Return**: Results are passed back through MCP protocol

## Prompt Engineering

### Search Strategy

The agent uses a two-phase approach:

1. **URL Discovery Phase**: 
   - Primary: Brave Search API with query `"{company_name} offizielle Webseite Schweiz"`
   - Fallback: Wikidata API for official website property (P856)
   - Filtering: Excludes social media, job sites, and blacklisted domains

2. **Data Extraction Phase**: 
   - Uses found URL or proceeds without URL if none found
   - Employs German-language prompt for Swiss company focus
   - Extracts specific data points with null handling

### Data Extraction Fields

The prompt specifies exact fields to extract:

```python
prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.

Die offizielle Webseite für "{company_name}" wurde als "{root_url_for_prompt if company_url else 'nicht gefunden'}" identifiziert.

Wenn eine URL ({root_url_for_prompt}) vorhanden ist und nicht 'null' oder 'nicht gefunden' lautet:
1. Öffne diese URL: {root_url_for_prompt}
2. Durchsuche diese Seite und relevante Unterseiten (z. B. /about, /unternehmen, /impressum, /geschichte)
   und sammle die unten genannten Fakten.

Wenn KEINE URL gefunden wurde (d.h. als "{root_url_for_prompt}" angegeben ist) ODER Informationen auf der Webseite nicht auffindbar sind, gib für die entsprechenden Felder **null** zurück.

Fakten zu sammeln:
   • Aktueller CEO / Geschäftsführer
   • Gründer (Komma-getrennt bei mehreren)
   • Inhaber (Besitzer der Firma)
   • Aktuelle Mitarbeiterzahl (Zahl oder Bereich, z. B. "200-250")
   • Gründungsjahr (JJJJ)
   • Offizielle Website (die bereits ermittelte Root-URL: "{root_url_for_prompt}")
   • Was macht diese Firma besser als ihre Konkurrenz. (maximal 10 Wörter)
   • Hauptsitz (Adresse des Firmenhauptsitzes)
   • Firmenidentifikationsnummer (meistens im Impressum, z.B. CHE-XXX.XXX.XXX)
   • Haupt-Telefonnummer
   • Haupt-Emailadresse
   • Geschäftsbericht (URL zum PDF, falls vorhanden)
"""
```

### URL Discovery Functions

The core logic for URL discovery now resides in `search_common.py`. Functions such as `get_brave_search_candidates` and `select_best_url_with_llm` are designed to receive necessary API keys or LLM instances as parameters.

#### Brave Search Implementation
```python
def get_brave_search_candidates(company: str, brave_api_key: str, count: int = 5) -> List[Dict[str, Any]]:
    """Fetches company homepage using Brave Search API."""
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key
    }
    params = {"q": f'"{company}" homepage official site', "count": count, "country": "ch", "search_lang": "de"}
    # Implementation includes candidate filtering and blacklist checking
```

#### Wikidata Fallback Implementation (`get_wikidata_homepage`)
```python
def get_wikidata_homepage(company: str) -> str | None:
    """Fetches company homepage from Wikidata."""
    # 1. Search for entity by company name
    # 2. Fetch P856 (official website) property
    # 3. Validate and filter results
```

## Error Handling and Debugging

### Debug Levels

```python
Logger.set_debug(0)  # No debug output
Logger.set_debug(1)  # INFO level messages
Logger.set_debug(2)  # DEBUG level messages (full verbose)
```

### Common Error Scenarios

#### 1. Brave Search API Failures
- **Cause**: Invalid API key, rate limiting, or network issues
- **Detection**: HTTP status errors or request timeouts
- **Resolution**: Check API key, verify quota limits, ensure network connectivity

#### 2. Wikidata API Failures
- **Cause**: Network issues or API rate limiting
- **Detection**: HTTP errors or JSON parsing failures
- **Resolution**: Implement retry logic, check network connectivity

#### 3. MCP Connection Failures
- **Cause**: npx not available or execution policy issues
- **Detection**: Connection timeout errors
- **Resolution**: Verify npx installation and PowerShell execution policy

#### 4. LLM API Errors
- **Cause**: Invalid API key or rate limiting
- **Detection**: OpenAI authentication errors
- **Resolution**: Check API key and usage limits

#### 5. Browser Automation Failures
- **Cause**: Website changes, network issues, or anti-bot measures
- **Detection**: Playwright timeout or navigation errors
- **Resolution**: Adjust timeouts or update navigation logic

## Performance Considerations

### Execution Time
- **Typical runtime**: 30-60 seconds per company
- **Factors affecting speed**: Website complexity, network latency, LLM response time

### Resource Usage
- **Memory**: ~200-500MB during execution
- **Network**: Moderate bandwidth for page loading and API calls
- **CPU**: Low to moderate usage

### Optimization Strategies

1. **Reduce max_steps**: Lower from 30 to 15-20 for faster execution
2. **Adjust temperature**: Keep at 0 for consistent results
3. **Use faster models**: Consider gpt-3.5-turbo for speed vs accuracy trade-off

## Security Considerations

### API Key Management
- Store keys in `.env` file (never commit to version control)
- Application logic, particularly shared utility functions in `search_common.py`, is designed to receive API keys and configured LLM instances as parameters, promoting better encapsulation.
- Use environment-specific keys for development/production
- Implement key rotation policies

### Browser Security
- Playwright runs in isolated environment
- No persistent browser data stored
- Automatic cleanup after execution

### Data Privacy
- No company data is stored locally
- All processing happens in memory
- Consider data retention policies for logs

## Customization Options

### Changing Target Search Engine

To use a different search engine, modify the prompt:

```python
# For Google instead of startpage.com
prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.

1. Öffne google.com
2. Suche nach: "{company_name} offizielle Webseite Schweiz"
...
"""
```

### Adding New Data Fields

To extract additional information, modify the JSON structure in the prompt:

```python
{{
  "official_website": "<url oder null>",
  "ceo": "<name oder null>",
  "founder": "<name oder null>",
  "owner": "<name oder null>",
  "employees": "<zahl oder null>",
  "founded": "<jahr oder null>",
  "better_then_the_rest": "<kurze beschreibung>",
  "Hauptsitz": "<adresse oder null>",
  "Firmenidentifikationsnummer": "<CHE-... oder null>",
  "HauptTelefonnummer": "<telefon oder null>",
  "HauptEmailAdresse": "<email oder null>",
  "Geschäftsbericht": "<url oder null>"
}}
```

### Using Different LLM Models

#### OpenAI Models
```python
# GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0)

# GPT-4o (faster)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

#### Google Gemini Models
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-preview-05-06", 
    temperature=0
)
```

## Monitoring and Logging

### Log Levels
- **INFO**: Basic execution flow
- **DEBUG**: Detailed MCP communication
- **ERROR**: Failures and exceptions

### Log Analysis
Monitor logs for:
- Connection establishment times
- Tool execution success rates
- LLM response quality
- Error patterns

## Deployment Considerations

### Production Environment
- Use production-grade API keys
- Implement proper error handling
- Set up monitoring and alerting
- Consider rate limiting

### Scaling
- Implement queuing for multiple requests
- Use connection pooling for MCP clients
- Consider distributed execution for high volume

### Maintenance
- Regular dependency updates
- Monitor for website structure changes
- Update prompts based on success rates

## API Reference

### Main Function
```python
async def main(company_name: str) -> str
```
**Parameters:**
- `company_name`: Name of the company to search for

**Returns:**
- JSON string with company information

### Configuration Files

#### startpage_mcp.json
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "env": {
        "DISPLAY": ":1"  // Optional for headless environments
      }
    }
  }
}
```

#### .env
```env
OPENAI_API_KEY=sk-...
BRAVE_API_KEY=BSA...  # Required for primary search
GOOGLE_API_KEY=AIza...  # Optional
```

## Testing

### Unit Testing
```python
import pytest
import asyncio
from brave_search import main

@pytest.mark.asyncio
async def test_company_search():
    result = await main("Test Company")
    assert result is not None
    # Add more assertions
```

### Integration Testing
- Test with known companies
- Verify JSON structure
- Check for null handling

### Performance Testing
- Measure execution time
- Test with various company types
- Monitor resource usage

## Troubleshooting Guide

### Step-by-Step Debugging

1. **Verify Environment**
   ```bash
   python --version  # Should be 3.11+
   node --version    # Should be v16+
   npx --version     # Should show version
   ```

2. **Test MCP Connection**
   ```python
   from mcp_use import MCPClient
   client = MCPClient.from_config_file("startpage_mcp.json")
   # Should connect without errors
   ```

3. **Test LLM Connection**
   ```python
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-4.1-mini")
   response = llm.invoke("Hello")
   print(response.content)
   ```

4. **Enable Verbose Logging**
   ```python
   Logger.set_debug(2)
   ```

### Common Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Install missing packages with pip |
| `PowerShell execution policy` | Run `Set-ExecutionPolicy RemoteSigned` |
| `OpenAI authentication` | Check API key in .env file |
| `MCP connection timeout` | Verify npx and internet connection |
| `Browser automation fails` | Check website accessibility |

## Future Enhancements

### Planned Features
- Support for multiple search engines
- Result caching and storage
- Web interface for easier usage
- Multi-language support

### Completed Features
- Batch processing via `company_processor.py`

### Technical Improvements
- Async processing for better performance
- Better error recovery mechanisms
- Enhanced data validation
- Improved prompt engineering

## Contributing

### Development Setup
1. Fork the repository
2. Create virtual environment
3. Install development dependencies
4. Run tests before submitting changes

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Include docstrings for functions
- Write tests for new features

### Pull Request Process
1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request with description
