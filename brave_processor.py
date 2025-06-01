import asyncio
import os
import sys
import argparse
import csv
import json
import time
from pathlib import Path
import logging

from dotenv import load_dotenv

import core_processing
from logging_utils import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()

PROCESSING_STATUS_COLUMN = "processing_status"

async def async_process_companies(input_csv_path_str: str, output_csv_path_str: str):
    input_csv_path = Path(input_csv_path_str)
    if not input_csv_path.is_file():
        logger.critical(f"Input CSV file not found: {input_csv_path}")
        sys.exit(1)

    logger.info(f"Using input CSV file: {input_csv_path}")
    output_csv_path = Path(output_csv_path_str)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    brave_api_key = os.getenv("BRAVE_API_KEY")

    if not openai_api_key:
        logger.critical("OPENAI_API_KEY is not set. Exiting.")
        sys.exit(1)
    if not brave_api_key: # Changed from warning to info as it's a valid operational mode
        logger.info("BRAVE_API_KEY not found. URL discovery will rely on other methods if available.")


    config = {
        "OPENAI_API_KEY": openai_api_key,
        "BRAVE_API_KEY": brave_api_key,
        "URL_LLM_MODEL": os.getenv("URL_LLM_MODEL", "gpt-4.1-mini"),
        "AGENT_LLM_MODEL": os.getenv("AGENT_LLM_MODEL", "gpt-4.1-mini"),
        "LLM_RELEVANCE_CHECK_MODEL": os.getenv("LLM_RELEVANCE_CHECK_MODEL", "gpt-3.5-turbo"),
        "AGENT_MAX_STEPS": int(os.getenv("AGENT_MAX_STEPS", 30)),
        "AGENT_TIMEOUT": int(os.getenv("AGENT_TIMEOUT", 60)),
    }
    logger.debug(f"Core processing configuration: {json.dumps(config, indent=2)}")

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

    # Define the expected header structure for the output CSV
    expected_header = ['company_number', 'company_name', 'final_url', 'source_of_url'] + \
                      core_processing.EXPECTED_JSON_KEYS + \
                      [PROCESSING_STATUS_COLUMN, 'error_message']
    
    processed_company_numbers = set()
    output_rows_buffer = []
    company_number_idx = 0 # Default to first column for company_number

    if output_csv_path.is_file():
        logger.info(f"Output file {output_csv_path} exists. Attempting to resume.")
        try:
            with open(output_csv_path, 'r', encoding='utf-8-sig', newline='') as outfile_read:
                reader = csv.reader(outfile_read)
                header_from_file = next(reader, None)
                if header_from_file:
                    # Compare with expected_header, or at least ensure 'company_number' exists
                    if header_from_file == expected_header:
                        logger.info("Existing output CSV header matches expected header. Resuming.")
                        output_rows_buffer.append(header_from_file)
                        try:
                            company_number_idx = header_from_file.index('company_number')
                        except ValueError: # Should not happen if headers match
                            logger.error("'company_number' not found in matched header? Using index 0. This is unexpected.")
                            company_number_idx = 0

                        for row in reader:
                            if row:
                                output_rows_buffer.append(row)
                                if len(row) > company_number_idx and row[company_number_idx]:
                                    processed_company_numbers.add(row[company_number_idx])
                        logger.info(f"Loaded {len(processed_company_numbers)} already processed company numbers.")
                    else:
                        logger.warning(f"Existing output CSV header does not match expected header. Starting fresh output. "
                                       f"Expected: {expected_header}, Found: {header_from_file}")
                        output_rows_buffer.append(expected_header) # Use the new expected header
                else: # File is empty or has no header
                    logger.info("Existing output CSV is empty or has no header. Starting with new header.")
                    output_rows_buffer.append(expected_header)
        except Exception as e:
            logger.error(f"Error reading existing output file {output_csv_path}: {e}. Starting with new header.", exc_info=True)
            output_rows_buffer = [expected_header] # Start fresh if read fails
    else:
        logger.info(f"Output file {output_csv_path} does not exist. Starting with new header.")
        output_rows_buffer = [expected_header]

    
    companies_to_process_from_input = []
    try:
        with open(input_csv_path, 'r', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            input_header = next(reader, None) # Read and store the input header
            if not input_header or 'company_name' not in input_header or 'company_number' not in input_header:
                logger.critical("Input CSV must contain 'company_name' and 'company_number' columns in the header.")
                sys.exit(1)

            # Determine indices from input header
            try:
                input_company_name_idx = input_header.index('company_name')
                input_company_number_idx = input_header.index('company_number')
            except ValueError:
                logger.critical("Could not find 'company_name' or 'company_number' in input CSV header.")
                sys.exit(1)

            for i, row_data in enumerate(reader):
                if len(row_data) <= max(input_company_name_idx, input_company_number_idx): # Check if row has enough columns
                    logger.warning(f"Skipping invalid row {i+2} in {input_csv_path}: not enough columns. Content: {row_data}")
                    # We don't add to output_rows_buffer here as it's an input file issue.
                    continue

                company_number = row_data[input_company_number_idx].strip()
                company_name = row_data[input_company_name_idx].strip()

                if not company_name:
                    logger.warning(f"Skipping row {i+2} in {input_csv_path} for company number '{company_number}': company name is empty.")
                    continue
                if not company_number: # Company number is crucial for resume functionality
                    logger.warning(f"Skipping row {i+2} in {input_csv_path} for company name '{company_name}': company number is empty.")
                    continue
                companies_to_process_from_input.append((company_name, company_number))
    except FileNotFoundError:
        logger.critical(f"Input CSV file not found at {input_csv_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error reading input CSV {input_csv_path}: {e}", exc_info=True)
        sys.exit(1)

    total_companies_in_input = len(companies_to_process_from_input)
    logger.info(f"Found {total_companies_in_input} valid companies to process from {input_csv_path}.")

    companies_actually_processed_this_run = 0

    for i, (company_name, company_number) in enumerate(companies_to_process_from_input):
        if company_number in processed_company_numbers:
            logger.info(f"Skipping already processed company {i+1}/{total_companies_in_input}: '{company_name}' ({company_number})")
            continue

        logger.info(f"--- Processing company {i+1}/{total_companies_in_input}: '{company_name}' ({company_number}) ---")
        companies_actually_processed_this_run += 1
        
        try:
            result_dict = await core_processing.process_company_info(
                company_name=company_name,
                company_number=company_number,
                config=config,
                mcp_config_path_or_object=str(sequential_mcp_path)
            )
        except Exception as e:
            logger.error(f"Critical error calling process_company_info for '{company_name}': {e}", exc_info=True)
            result_dict = {key: "null" for key in core_processing.EXPECTED_JSON_KEYS}
            result_dict["company_name"] = company_name
            result_dict["company_number"] = company_number
            result_dict["final_url"] = "null"
            result_dict["source_of_url"] = "null"
            result_dict[PROCESSING_STATUS_COLUMN] = "CORE_PROCESSING_CALL_ERROR"
            result_dict["error_message"] = str(e)

        current_row_data = [
            result_dict.get("company_number", company_number),
            result_dict.get("company_name", company_name),
            result_dict.get("final_url", "null"),
            result_dict.get("source_of_url", "null")
        ]
        for key in core_processing.EXPECTED_JSON_KEYS:
            current_row_data.append(result_dict.get(key, "null"))
        
        current_row_data.append(result_dict.get(PROCESSING_STATUS_COLUMN, "STATUS_KEY_MISSING"))
        current_row_data.append(result_dict.get("error_message", None))

        output_rows_buffer.append(current_row_data)
        processed_company_numbers.add(company_number) # Add to set after processing attempt
        
        # Save progress: Overwrite the file with the current buffer
        # The frequency of saving can be adjusted.
        if companies_actually_processed_this_run % 5 == 0 or (i + 1) == total_companies_in_input:
            logger.info(f"Saving progress to {output_csv_path_str} ({len(output_rows_buffer)-1} total rows)...")
            try:
                with open(output_csv_path_str, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(output_rows_buffer)
            except IOError as e:
                logger.error(f"Error writing to output CSV {output_csv_path_str}: {e}", exc_info=True)
    
    logger.info(f"Processing complete. Total unique companies in output: {len(output_rows_buffer)-1}. "
                f"Companies processed in this run: {companies_actually_processed_this_run}.")


def console_main() -> None:
    log_file_name = os.getenv("LOG_FILE_BRAVE_PROCESSOR", "brave_processor.log")
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logging(log_level=log_level, log_file=log_file_name)

    parser = argparse.ArgumentParser(description="Sequentially process company data from a CSV file with resume capability.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path to the output CSV file for results.")
    args = parser.parse_args()

    try:
        asyncio.run(async_process_companies(args.input_csv, args.output_csv))
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main processing loop: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    console_main()
