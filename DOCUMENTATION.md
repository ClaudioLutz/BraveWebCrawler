# Technical Documentation - Startpage Company Search Agent

## Overview

This document provides detailed technical information about the Startpage Company Search Agent, including architecture, implementation details, and advanced configuration options.

## Architecture

### System Components

```mermaid
graph TD
    A[startpage_search.py] --> B[MCP Client]
    B --> C[Playwright MCP Server]
    C --> D[Browser Automation]
    A --> E[OpenAI LLM]
    A --> F[Environment Config]
    F --> G[.env file]
    F --> H[startpage_mcp.json]
    D --> I[startpage.com]
    D --> J[Company Website]
    E --> K[JSON Output]
```

### Technology Stack

- **Python 3.11+**: Main runtime environment
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

The agent uses a sophisticated German-language prompt that instructs it to:

1. **Navigate to startpage.com**
2. **Perform targeted search**: `"{company_name} offizielle Webseite Schweiz"`
3. **Filter results**: Avoid social media, job sites, and PDFs
4. **Extract specific data points**: CEO, founder, employees, etc.

### Data Extraction Fields

The prompt specifies exact fields to extract:

```python
prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.

1. Öffne startpage.com  
2. Suche im Suchfeld nach: "{company_name} offizielle Webseite Schweiz"  
3. Öffne die offizielle Domain der Firma  
   (keine Social- oder Job-Sites, keine PDFs).  
4. Durchsuche Unterseiten (z. B. /about, /unternehmen, /impressum, /geschichte)
   und sammle diese Fakten:

   • Aktueller CEO / Geschäftsführer  
   • Gründer (Komma-getrennt bei mehreren)  
   • Inhaber (Besitzer der Firma)
   • Aktuelle Mitarbeiterzahl (Zahl oder Bereich, z. B. "200-250")  
   • Gründungsjahr (JJJJ)  
   • Offizielle Website (Root-URL ohne Pfad)
   • Was macht diese Firma besser als ihre Konkurenz. (maximal 10 Wörter)
   • Auflistung der Standorte
   • Firmenidentifikationsnummer (meistens im Impressum)
   • Haupt-Telefonnummer
   • Haupt-Emailadresse

Wenn ein Feld nicht auffindbar ist, gib **null** zurück.
"""
```

## Error Handling and Debugging

### Debug Levels

```python
Logger.set_debug(0)  # No debug output
Logger.set_debug(1)  # INFO level messages
Logger.set_debug(2)  # DEBUG level messages (full verbose)
```

### Common Error Scenarios

#### 1. MCP Connection Failures
- **Cause**: npx not available or execution policy issues
- **Detection**: Connection timeout errors
- **Resolution**: Verify npx installation and PowerShell execution policy

#### 2. LLM API Errors
- **Cause**: Invalid API key or rate limiting
- **Detection**: OpenAI authentication errors
- **Resolution**: Check API key and usage limits

#### 3. Browser Automation Failures
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
  "new_field": "<description>",  # Add new field here
  ...
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
GOOGLE_API_KEY=AIza...  # Optional
```

## Testing

### Unit Testing
```python
import pytest
import asyncio
from startpage_search import main

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
- Batch processing capabilities
- Result caching and storage
- Web interface for easier usage
- Multi-language support

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
