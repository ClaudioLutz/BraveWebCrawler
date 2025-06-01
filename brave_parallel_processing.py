import asyncio
import os
import sys
import argparse
import csv
import json
import multiprocessing
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

from dotenv import load_dotenv

import core_processing
from logging_utils import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()

PROCESSING_STATUS_COLUMN = "processing_status"

def run_company_processing_parallel_wrapper(
    company_name: str,
    company_number: str,
    shared_config_dict: Dict[str, Any],
    mcp_dynamic_params_dict: Dict[str, Any]
) -> Dict[str, Any]:
    worker_logger = logging.getLogger(f"{__name__}.worker.{os.getpid()}")
    worker_logger.info(f"Processing: {company_name} ({company_number})")
    try:
        result = asyncio.run(
            core_processing.process_company_info(
                company_name,
                company_number,
                config=shared_config_dict,
                mcp_config_path_or_object=mcp_dynamic_params_dict
            )
        )
        # Ensure company_name and company_number from input are in the result for consistency
        result['company_name'] = company_name
        result['company_number'] = company_number
        worker_logger.info(f"Finished: {company_name} ({company_number}), Status: {result.get(PROCESSING_STATUS_COLUMN)}")
        return result
    except Exception as e:
        worker_logger.error(f"Error processing {company_name} ({company_number}): {e}", exc_info=True)
        error_result = {key: "null" for key in core_processing.EXPECTED_JSON_KEYS}
        error_result["company_name"] = company_name
        error_result["company_number"] = company_number
        error_result["final_url"] = "null"
        error_result["source_of_url"] = "null"
        error_result[PROCESSING_STATUS_COLUMN] = f"POOL_WRAPPER_ERROR: {type(e).__name__}"
        error_result["error_message"] = str(e)
        return error_result

def console_main() -> None:
    log_file_name = os.getenv("LOG_FILE_BRAVE_PARALLEL", "brave_parallel_processing.log")
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logging(log_level=log_level, log_file=log_file_name)

    parser = argparse.ArgumentParser(description="Parallel company data processor with resume capability.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes.")
    args = parser.parse_args()

    input_csv_path_obj = Path(args.input_csv)
    if not input_csv_path_obj.is_file():
        logger.critical(f"Input CSV file not found: {args.input_csv}")
        sys.exit(1)
    logger.info(f"Using input CSV: {args.input_csv}")
    output_csv_path = Path(args.output_csv)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    brave_api_key = os.getenv("BRAVE_API_KEY")

    if not openai_api_key:
        logger.critical("OPENAI_API_KEY not set in environment variables.")
        sys.exit(1)
    if not brave_api_key: # Changed from warning to info
        logger.info("BRAVE_API_KEY not found. URL discovery will rely on other methods if available.")

    main_shared_config = {
        "OPENAI_API_KEY": openai_api_key,
        "BRAVE_API_KEY": brave_api_key,
        "URL_LLM_MODEL": os.getenv("URL_LLM_MODEL", "gpt-4.1-mini"),
        "AGENT_LLM_MODEL": os.getenv("AGENT_LLM_MODEL", "gpt-4.1-mini"),
        "LLM_RELEVANCE_CHECK_MODEL": os.getenv("LLM_RELEVANCE_CHECK_MODEL", "gpt-3.5-turbo"),
        "AGENT_MAX_STEPS": int(os.getenv("AGENT_MAX_STEPS", 25)),
        "AGENT_TIMEOUT": int(os.getenv("AGENT_TIMEOUT", 60)),
    }
    logger.debug(f"Main shared config for workers: {json.dumps(main_shared_config, indent=2)}")

    headless_browsing = os.getenv("HEADLESS_BROWSING", "True").lower() == "true"
    logger.info(f"Headless browsing mode for parallel workers: {headless_browsing}")

    parallel_mcp_launcher_filename = "parallel_mcp_launcher.json"
    parallel_mcp_launcher_path = Path(__file__).parent / parallel_mcp_launcher_filename

    if not parallel_mcp_launcher_path.exists():
        logger.warning(f"Base MCP launcher template '{parallel_mcp_launcher_path}' not found. Creating a dummy one.")
        dummy_launcher_content = {
            "mcpServers": {"playwright": {"executable": "echo", "args": ["dummy server for parallel_mcp_launcher"]}}
        }
        try:
            with open(parallel_mcp_launcher_path, "w") as f:
                json.dump(dummy_launcher_content, f, indent=2)
            logger.info(f"Dummy '{parallel_mcp_launcher_path}' created.")
        except IOError as e:
            logger.critical(f"Could not create dummy '{parallel_mcp_launcher_path}': {e}. This file is required for dynamic MCP setup.", exc_info=True)
            sys.exit(1)

    mcp_dynamic_config_params = {
        "base_mcp_launcher_path": str(parallel_mcp_launcher_path.resolve()),
        "headless": headless_browsing,
    }
    logger.debug(f"MCP dynamic config params for workers: {json.dumps(mcp_dynamic_config_params, indent=2)}")

    expected_header = ['company_number', 'company_name', 'final_url', 'source_of_url'] + \
                      core_processing.EXPECTED_JSON_KEYS + \
                      [PROCESSING_STATUS_COLUMN, 'error_message']

    processed_company_numbers = set()
    output_rows_buffer = [] # This will hold all rows to be written (old and new)
    company_number_idx = 0 # Default, will be updated if header is found

    if output_csv_path.is_file():
        logger.info(f"Output file {output_csv_path} exists. Attempting to resume.")
        try:
            with open(output_csv_path, 'r', encoding='utf-8-sig', newline='') as outfile_read:
                reader = csv.reader(outfile_read)
                header_from_file = next(reader, None)
                if header_from_file:
                    if header_from_file == expected_header:
                        logger.info("Existing output CSV header matches expected. Resuming.")
                        output_rows_buffer.append(header_from_file)
                        try:
                            company_number_idx = header_from_file.index('company_number')
                        except ValueError:
                            logger.error("'company_number' not in matched header? Using index 0.")
                            company_number_idx = 0

                        for row in reader:
                            if row:
                                output_rows_buffer.append(row)
                                if len(row) > company_number_idx and row[company_number_idx]:
                                    processed_company_numbers.add(row[company_number_idx])
                        logger.info(f"Loaded {len(processed_company_numbers)} already processed company numbers from existing output file.")
                    else:
                        logger.warning(f"Existing output CSV header does not match expected. Starting fresh output. "
                                       f"Expected: {expected_header}, Found: {header_from_file}")
                        output_rows_buffer.append(expected_header)
                else:
                    logger.info("Existing output CSV is empty or has no header. Starting with new header.")
                    output_rows_buffer.append(expected_header)
        except Exception as e:
            logger.error(f"Error reading existing output file {output_csv_path}: {e}. Starting with new header.", exc_info=True)
            output_rows_buffer = [expected_header]
    else:
        logger.info(f"Output file {output_csv_path} does not exist. Starting with new header.")
        output_rows_buffer = [expected_header]

    all_companies_from_input_tuples: List[Tuple[str, str]] = []
    try:
        with open(args.input_csv, 'r', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            input_header = next(reader, None)
            if not input_header or 'company_name' not in input_header or 'company_number' not in input_header:
                logger.critical("Input CSV must contain 'company_name' and 'company_number' columns in the header.")
                sys.exit(1)

            input_company_name_idx = input_header.index('company_name')
            input_company_number_idx = input_header.index('company_number')

            for i, row_data in enumerate(reader):
                if len(row_data) <= max(input_company_name_idx, input_company_number_idx):
                    logger.warning(f"Skipping invalid row {i+2} in {args.input_csv}: not enough columns. Content: {row_data}")
                    continue

                company_number = row_data[input_company_number_idx].strip()
                company_name = row_data[input_company_name_idx].strip()

                if not company_name:
                    logger.warning(f"Skipping row {i+2} in {args.input_csv} for company number '{company_number}': company name is empty.")
                    continue
                if not company_number:
                    logger.warning(f"Skipping row {i+2} in {args.input_csv} for company name '{company_name}': company number is empty (required for resume).")
                    continue
                all_companies_from_input_tuples.append((company_name, company_number))
    except FileNotFoundError:
        logger.critical(f"Input CSV file not found at {args.input_csv}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error reading input CSV {args.input_csv}: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Found {len(all_companies_from_input_tuples)} total companies in input file.")

    companies_for_pool_args_list = []
    for name, number in all_companies_from_input_tuples:
        if number in processed_company_numbers:
            logger.info(f"Skipping already processed company: {name} ({number})")
        else:
            companies_for_pool_args_list.append((name, number, main_shared_config, mcp_dynamic_config_params))

    total_companies_for_pool = len(companies_for_pool_args_list)
    logger.info(f"Number of companies to process in this run (after filtering duplicates): {total_companies_for_pool}")

    if total_companies_for_pool > 0:
        num_workers = min(args.workers, total_companies_for_pool, os.cpu_count())
        logger.info(f"Using {num_workers} worker processes.")
        
        start_time = time.time()
        pool = multiprocessing.Pool(processes=num_workers)
        try:
            newly_processed_results_list = pool.starmap(run_company_processing_parallel_wrapper, companies_for_pool_args_list)

            for result_dict in newly_processed_results_list:
                # Ensure company_name and company_number are present, using original from input if necessary
                # The wrapper should already add these back if they were lost.
                row_company_number = result_dict.get("company_number", "UNKNOWN_NUMBER_FROM_POOL")
                row_company_name = result_dict.get("company_name", "UNKNOWN_NAME_FROM_POOL")

                current_row_data = [
                    row_company_number,
                    row_company_name,
                    result_dict.get("final_url", "null"),
                    result_dict.get("source_of_url", "null")
                ]
                for key in core_processing.EXPECTED_JSON_KEYS:
                    current_row_data.append(result_dict.get(key, "null"))

                current_row_data.append(result_dict.get(PROCESSING_STATUS_COLUMN, "STATUS_KEY_MISSING"))
                current_row_data.append(result_dict.get("error_message", None))
                output_rows_buffer.append(current_row_data)

        except Exception as e_pool:
            logger.critical(f"Error during multiprocessing pool execution: {e_pool}", exc_info=True)
            # No specific error rows added here as individual task errors are handled by wrapper.
            # This would be for catastrophic pool failures.
        finally:
            logger.debug("Closing multiprocessing pool.")
            pool.close()
            pool.join()

        end_time = time.time()
        logger.info(f"Parallel processing of {total_companies_for_pool} companies finished in {end_time - start_time:.2f} seconds.")
    else:
        logger.info("No new companies to process in this run.")

    logger.info(f"Writing {len(output_rows_buffer)-1} total data entries to {args.output_csv}...")
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(output_rows_buffer)
        logger.info(f"Successfully wrote to {args.output_csv}")
    except IOError as e:
        logger.error(f"Error writing to output CSV {args.output_csv}: {e}", exc_info=True)
    
    logger.info("Processing complete.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        current_context = multiprocessing.get_start_method(allow_none=True)
        if current_context != 'spawn':
            logger.warning(f"Could not force multiprocessing start_method to 'spawn' (current: {current_context}): {e}. Proceeding...")
        else:
            pass # Already spawn

    console_main()
    logging.shutdown()
