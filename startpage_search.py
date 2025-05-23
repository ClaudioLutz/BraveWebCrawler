import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import Logger  # Import the Logger
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Enable mcp_use debug logging
Logger.set_debug(2)

async def main(company_name):
    # Create MCPClient from config file
    client = MCPClient.from_config_file(os.path.join(os.path.dirname(__file__), "startpage_mcp.json"))
    
    # Initialize the OpenAI client
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    # Initialize the Google Gemini client
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-pro-preview-05-06", 
    #     temperature=0
    # )
    
    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)
    
    # Run the query
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

Antworte **ausschließlich** mit genau diesem JSON, ohne jeglichen Text davor
oder danach:

{{
  "official_website": "<url oder null>",
  "ceo": "<name oder null>",
  "founder": "<name(s) oder null>",
  "owner": "<name(s) oder null>",
  "employees": "<zahl/bereich oder null>",
  "founded": "<jahr oder null>",
  "better_then_the_rest": "<text>",
  "Standorte": "<[Liste]>",
  "Firmenidentifikationsnummer": "<CHE->",
  "HauptTelefonnummer": "xxx xxx xx xx",
  "HauptEmailAdresse": "xx@xx.xx",
}}
"""
    
    result = await agent.run(prompt, max_steps=30)
    
    print(f"\nResult: {result}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Search for company information on startpage.com')
    parser.add_argument('company', help='The company name to search for')
    args = parser.parse_args()
    
    # Run the main function with the company name
    asyncio.run(main(args.company))
