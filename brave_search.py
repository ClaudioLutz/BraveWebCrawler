import asyncio
import os
import sys
import argparse
import json
from pathlib import Path
import logging # Added

from dotenv import load_dotenv

import core_processing
from logging_utils import setup_logging # Added

logger = logging.getLogger(__name__) # Added

load_dotenv()

async def async_main(company_name: str):
    logger.info(f"Starting brave_search for company: '{company_name}'")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    brave_api_key = os.getenv("BRAVE_API_KEY")

    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables. This is critical for core_processing.")
        # core_processing will also error out, but good to log here too.
    
    if not brave_api_key:
        logger.warning("BRAVE_API_KEY not found in environment variables. URL discovery might be limited.")

    config = {
        "OPENAI_API_KEY": openai_api_key,
        "BRAVE_API_KEY": brave_api_key,
        "URL_LLM_MODEL": os.getenv("URL_LLM_MODEL", "gpt-4.1-mini"),
        "AGENT_LLM_MODEL": os.getenv("AGENT_LLM_MODEL", "gpt-4.1-mini"),
        "LLM_RELEVANCE_CHECK_MODEL": os.getenv("LLM_RELEVANCE_CHECK_MODEL", "gpt-3.5-turbo"),
        "AGENT_MAX_STEPS": int(os.getenv("AGENT_MAX_STEPS", 30)),
        "AGENT_TIMEOUT": int(os.getenv("AGENT_TIMEOUT", 60)),
    }
    logger.debug(f"Configuration for core_processing: {json.dumps(config, indent=2)}")

    sequential_mcp_config_filename = "sequential_mcp_config.json"
    sequential_mcp_path = Path(__file__).parent / sequential_mcp_config_filename

    if not sequential_mcp_path.exists():
        logger.warning(f"MCP config file '{sequential_mcp_path}' not found. Creating a dummy one.")
        dummy_mcp_config_content = {
            "mcpServers": {"playwright": {"executable": "echo", "args": ["dummy server for sequential_mcp_config"]}}
        }
        try:
            with open(sequential_mcp_path, "w") as f:
                json.dump(dummy_mcp_config_content, f, indent=2)
            logger.info(f"Dummy '{sequential_mcp_path}' created.")
        except IOError as e:
            logger.error(f"Error creating dummy MCP config: {e}. Execution might fail if MCP is required.", exc_info=True)

    logger.info(f"Processing company: {company_name} using MCP config: {sequential_mcp_path}")

    result_data = await core_processing.process_company_info(
        company_name=company_name,
        company_number="N/A_CLI",
        config=config,
        mcp_config_path_or_object=str(sequential_mcp_path)
    )

    logger.info("\n--- Processing Result ---")
    # Using logger.info for the JSON result, or print if preferred for CLI tool final output.
    # For consistency with logging, using logger.info.
    # The JSON string might be very long, so logging it might be verbose for files.
    # Consider logging a summary or specific fields if verbosity is an issue.
    try:
        result_json_str = json.dumps(result_data, indent=2, ensure_ascii=False)
        # Split large JSON into multiple log messages if necessary, or log selectively
        max_log_length = 1000 # Example max length per log message
        if len(result_json_str) > max_log_length:
            logger.info("Processing Result (first part):")
            for chunk_i in range(0, len(result_json_str), max_log_length):
                 logger.info(result_json_str[chunk_i:chunk_i+max_log_length])
        else:
             logger.info(result_json_str)
        # Alternatively, just print to stdout for CLI tools as the final output
        # print(json.dumps(result_data, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error serializing result_data to JSON: {e}", exc_info=True)
        logger.info(f"Raw result_data (may not be fully JSON serializable): {result_data}")


def console_main() -> None:
    # Setup logging as early as possible
    # Log file name can be configured here or passed as an argument
    log_file_name = os.getenv("LOG_FILE_BRAVE_SEARCH", "brave_search.log")
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logging(log_level=log_level, log_file=log_file_name)

    parser = argparse.ArgumentParser(
        description="Search for company information using the core processing module."
    )
    parser.add_argument("company", help="The company name to search for")
    args = parser.parse_args()

    try:
        asyncio.run(async_main(args.company))
    except Exception as e:
        logger.critical(f"An unexpected error occurred in brave_search: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown() # Flushes and closes all handlers

if __name__ == "__main__":
    console_main()
