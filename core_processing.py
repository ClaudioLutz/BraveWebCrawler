import asyncio
import json
import os
import httpx
from urllib.parse import urlparse
import tempfile
import shutil
import time
import psutil
from pathlib import Path
import sys
import logging

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from search_common import (
    discover_urls,
    pre_screen_urls,
)

logger = logging.getLogger(__name__)

try:
    from mcp_sdk import MCPAgent, MCPClient
except ImportError:
    logger.warning("mcp_sdk not found. Using placeholder MCPAgent and MCPClient.")
    class MCPAgent:
        def __init__(self, llm, client, max_steps=None): pass
        async def run(self, prompt, max_steps=None):
            await asyncio.sleep(0)
            return json.dumps({key: "null_placeholder" for key in EXPECTED_JSON_KEYS})

    class MCPClient:
        @staticmethod
        def from_config_file(config_file_path: str): return MCPClient()
        def __init__(self): pass
        async def stop_servers(self): pass


EXPECTED_JSON_KEYS = [
    "official_website", "founded", "Hauptsitz", "Firmenidentifikationsnummer",
    "HauptTelefonnummer", "HauptEmailAdresse", "Geschäftsbericht", "extracted_company_name"
]

def rmtree_with_retry(path_to_remove: Path, attempts: int = 3, delay_seconds: float = 0.5):
    for attempt in range(attempts):
        try:
            if path_to_remove.exists():
                shutil.rmtree(path_to_remove)
                logger.debug(f"Successfully removed directory: {path_to_remove} on attempt {attempt + 1}")
            else:
                logger.debug(f"Directory not found for removal (already deleted?): {path_to_remove}")
                return
            return
        except PermissionError as e:
            logger.warning(f"PermissionError removing {path_to_remove} on attempt {attempt + 1}/{attempts}: {e}")
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
            else:
                logger.error(f"Failed to remove directory {path_to_remove} after {attempts} attempts due to PermissionError.")
        except FileNotFoundError:
            logger.debug(f"Directory not found for removal during rmtree attempt (already deleted?): {path_to_remove}")
            return
        except Exception as e:
            logger.error(f"Unexpected error removing {path_to_remove} on attempt {attempt + 1}: {e}", exc_info=True)
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
            else:
                logger.error(f"Failed to remove directory {path_to_remove} due to unexpected error after {attempts} attempts.")

def kill_chrome_processes_using(path: Path):
    if not psutil:
        logger.warning("psutil not installed, skipping Chrome process cleanup.")
        return
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and ('chrome' in proc.info['name'].lower() or 'chromium' in proc.info['name'].lower()):
                    cmdline_args = proc.info['cmdline']
                    if cmdline_args and isinstance(cmdline_args, list):
                        cmdline_str = ' '.join(cmdline_args)
                        if str(path) in cmdline_str and "--user-data-dir" in cmdline_str:
                            logger.info(f"Terminating browser process {proc.info['pid']} using profile {path}")
                            proc.terminate()
                            try:
                                proc.wait(timeout=0.5)
                            except psutil.TimeoutExpired:
                                logger.warning(f"Browser process {proc.info['pid']} did not terminate gracefully, killing.")
                                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as e_inner:
                logger.debug(f"Minor error checking process {proc.info.get('pid', 'N/A')}: {e_inner}")
                continue
    except Exception as e_outer:
        logger.error(f"Error iterating processes for Chrome cleanup: {e_outer}", exc_info=True)


async def process_company_info(
    company_name: str,
    company_number: str,
    config: dict,
    mcp_config_path_or_object: str | dict,
):
    logger.info(f"Starting processing for company: '{company_name}' ({company_number})")
    result_data = {key: "null" for key in EXPECTED_JSON_KEYS}
    result_data["company_name"] = company_name
    result_data["company_number"] = company_number
    result_data["final_url"] = "null"
    result_data["source_of_url"] = "NO_URL_FOUND"
    result_data["processing_status"] = "PENDING_URL_SEARCH"
    result_data["error_message"] = None

    tmp_profile_dir = None
    client_mcp = None
    # dynamic_mcp_launcher_path_obj = None # Not strictly needed to be stored at this scope

    openai_api_key = config.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found in config.")
        result_data["error_message"] = "OpenAI API key not found in config."
        result_data["processing_status"] = "ERROR_CONFIG_OPENAI_KEY"
        return result_data

    llm_url_selector = None
    url_selector_llm_model_name = config.get("URL_LLM_MODEL", "gpt-4.1-mini")
    logger.debug(f"Using URL selector LLM model: {url_selector_llm_model_name}")

    try:
        llm_url_selector = ChatOpenAI(model=url_selector_llm_model_name, temperature=0, api_key=SecretStr(openai_api_key))
    except Exception as e:
        logger.error(f"Error initializing LLM for URL selection (model: {url_selector_llm_model_name}): {e}", exc_info=True)
        result_data["error_message"] = f"Error initializing LLM for URL selection: {e}"
        result_data["processing_status"] = "ERROR_LLM_URL_INIT"

    try:
        logger.info(f"Discovering URLs for '{company_name}'...")
        discovered_url, source_of_url = await discover_urls(
            company_name=company_name, config=config, llm=llm_url_selector, company_number=company_number
        )
        result_data["source_of_url"] = source_of_url
        logger.info(f"URL discovery for '{company_name}' completed. Source: {source_of_url}, URL: {discovered_url}")
    except Exception as e:
        logger.error(f"Error during URL discovery for '{company_name}': {e}", exc_info=True)
        result_data["error_message"] = f"Error during URL discovery: {e}"
        result_data["processing_status"] = "ERROR_URL_DISCOVERY"
        discovered_url = None

    if discovered_url:
        try:
            parsed_url = urlparse(discovered_url)
            result_data["final_url"] = f"{parsed_url.scheme}://{parsed_url.netloc}"
            result_data["processing_status"] = result_data["source_of_url"]
            logger.debug(f"Parsed URL for '{company_name}': {result_data['final_url']}")
        except Exception as e:
            logger.error(f"Error parsing discovered URL '{discovered_url}' for '{company_name}': {e}", exc_info=True)
            result_data["error_message"] = f"Error parsing discovered URL '{discovered_url}': {e}"
            result_data["processing_status"] = "ERROR_URL_PARSING"
            result_data["final_url"] = "null"
    else:
        logger.info(f"No URL discovered for '{company_name}'. Status: {result_data['source_of_url']}")
        result_data["processing_status"] = result_data["source_of_url"] if result_data["source_of_url"] else "NO_URL_FOUND"

    if result_data["final_url"] != "null" and "ERROR" not in result_data["processing_status"]:
        logger.info(f"Performing URL pre-screen for '{company_name}' on URL: {result_data['final_url']}")
        # httpx.AsyncClient should be managed by the caller of pre_screen_urls or within pre_screen_urls itself
        # For now, assuming pre_screen_urls handles its client or one is passed appropriately by its direct caller.
        # If pre_screen_urls is called from here and needs an AsyncClient, it should be created here.
        # The previous step's code `async with httpx.AsyncClient() as http_client:` was correct.
        async with httpx.AsyncClient() as http_client_for_prescreen:
            try:
                screened_results = await pre_screen_urls(
                    urls_with_sources=[(result_data["final_url"], result_data["source_of_url"])],
                    company_name=company_name, llm=llm_url_selector, http_client=http_client_for_prescreen, config=config
                )
                if screened_results:
                    _, _, is_relevant = screened_results[0]
                    if is_relevant:
                        result_data["processing_status"] = f"{result_data['source_of_url']} (RELEVANT)"
                        logger.info(f"URL {result_data['final_url']} deemed relevant for '{company_name}'.")
                    else:
                        result_data["processing_status"] = "PRE_CHECK_URL_MISMATCH"
                        mismatch_err = f"URL '{result_data['final_url']}' deemed not relevant by pre-check for '{company_name}'."
                        logger.warning(mismatch_err)
                        result_data["error_message"] = f"{result_data['error_message']}; {mismatch_err}" if result_data["error_message"] else mismatch_err
                else:
                    logger.warning(f"URL pre-screen for '{company_name}' on {result_data['final_url']} returned no results.")
                    result_data["processing_status"] = "ERROR_PRE_SCREEN_UNEXPECTED_RESULT"
                    result_data["final_url"] = "null"
            except Exception as e:
                logger.error(f"Error during URL pre-check for '{company_name}' on {result_data['final_url']}: {e}", exc_info=True)
                result_data["error_message"] = f"Error during URL pre-check: {e}"
                result_data["processing_status"] = "ERROR_PRE_SCREEN"

    if result_data["final_url"] == "null" or \
       "ERROR" in result_data["processing_status"] or \
       "MISMATCH" in result_data["processing_status"] or \
       "NO_URL_FOUND" in result_data["processing_status"]:
        logger.info(f"Skipping agent processing for '{company_name}' due to URL status: {result_data['processing_status']}.")
        if result_data["final_url"] == "null" and "ERROR" not in result_data["processing_status"]:
             result_data["processing_status"] = "NO_URL_FOUND_SKIPPING_AGENT"
        if "official_website" not in result_data: result_data["official_website"] = "null" # Ensure key exists
        return result_data

    logger.info(f"Proceeding to agent processing for '{company_name}' with URL: {result_data['final_url']}")
    result_data["processing_status"] = f"{result_data['source_of_url']} (PENDING_AGENT)" # Base status on URL source
    result_data["official_website"] = result_data["final_url"] # Agent will use this as starting point

    try:
        agent_executor_llm_model_name = config.get("AGENT_LLM_MODEL", "gpt-4.1-mini")
        logger.debug(f"Using agent executor LLM model: {agent_executor_llm_model_name} for '{company_name}'")
        agent_llm = ChatOpenAI(model=agent_executor_llm_model_name, temperature=0, api_key=SecretStr(openai_api_key))

        logger.debug(f"Setting up MCPClient for '{company_name}'. Config type: {'path' if isinstance(mcp_config_path_or_object, str) else 'dict'}")
        if isinstance(mcp_config_path_or_object, str):
            if not Path(mcp_config_path_or_object).exists():
                raise FileNotFoundError(f"MCP config file not found: {mcp_config_path_or_object}")
            client_mcp = MCPClient.from_config_file(mcp_config_path_or_object)
        elif isinstance(mcp_config_path_or_object, dict):
            # Sanitize company_name for directory path
            safe_company_name = "".join(c if c.isalnum() else "_" for c in company_name)
            tmp_profile_dir_name = f"mcp_profile_{os.getpid()}_{safe_company_name}_{time.time_ns()}"
            tmp_profile_dir = Path(tempfile.gettempdir()) / tmp_profile_dir_name
            tmp_profile_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created temporary profile directory for '{company_name}': {tmp_profile_dir}")

            playwright_config_content = {
                "browser": {
                    "userDataDir": str(tmp_profile_dir.resolve()),
                    "launchOptions": {
                        "headless": mcp_config_path_or_object.get("headless", True),
                        "args": ["--disable-breakpad", "--disable-extensions", "--no-sandbox", "--disable-gpu"] # Added more common args
                    }}}
            if not playwright_config_content["browser"]["launchOptions"]["headless"]:
                pid_hash = os.getpid()
                x_pos = (pid_hash % 4) * 500
                y_pos = ((pid_hash // 4) % 3) * 300
                playwright_config_content["browser"]["launchOptions"]["args"].extend([
                    f"--window-position={x_pos},{y_pos}", f"--window-size=800,600"
                ])
            logger.debug(f"Playwright config for '{company_name}': {playwright_config_content}")

            runtime_playwright_config_path = tmp_profile_dir / "runtime-playwright-config.json"
            with open(runtime_playwright_config_path, 'w') as f: json.dump(playwright_config_content, f, indent=2)

            base_mcp_launcher_ref = mcp_config_path_or_object.get("base_mcp_launcher_path", "parallel_mcp_launcher.json")
            base_mcp_launcher_path_obj = Path(base_mcp_launcher_ref)
            if not base_mcp_launcher_path_obj.is_absolute(): # If relative, assume it's relative to core_processing.py's dir
                 base_mcp_launcher_path_obj = Path(__file__).parent / base_mcp_launcher_ref
            if not base_mcp_launcher_path_obj.exists():
                raise FileNotFoundError(f"Base MCP launcher template not found: {base_mcp_launcher_ref} (resolved to {base_mcp_launcher_path_obj})")

            with open(base_mcp_launcher_path_obj, 'r') as f: mcp_launcher_template = json.load(f)

            args_list = mcp_launcher_template.get("mcpServers", {}).get("playwright", {}).get("args", [])
            try:
                config_arg_index = args_list.index("--config")
                if config_arg_index + 1 < len(args_list): args_list[config_arg_index + 1] = str(runtime_playwright_config_path.resolve())
                else: args_list.extend(["--config", str(runtime_playwright_config_path.resolve())])
            except ValueError: args_list.extend(["--config", str(runtime_playwright_config_path.resolve())])

            if "mcpServers" not in mcp_launcher_template: mcp_launcher_template["mcpServers"] = {}
            if "playwright" not in mcp_launcher_template["mcpServers"]: mcp_launcher_template["mcpServers"]["playwright"] = {}
            mcp_launcher_template["mcpServers"]["playwright"]["args"] = args_list

            dynamic_mcp_launcher_path = tmp_profile_dir / "runtime-mcp-launcher.json" # Renamed for clarity
            with open(dynamic_mcp_launcher_path, 'w') as f: json.dump(mcp_launcher_template, f, indent=2)
            logger.debug(f"Dynamic MCP launcher for '{company_name}' created at: {dynamic_mcp_launcher_path}")

            client_mcp = MCPClient.from_config_file(str(dynamic_mcp_launcher_path.resolve()))
        else:
            raise TypeError("mcp_config_path_or_object must be a string (filepath) or a dictionary (dynamic config).")

        logger.info(f"MCPClient initialized for '{company_name}'. Initializing MCPAgent...")
        agent = MCPAgent(llm=agent_llm, client=client_mcp, max_steps=config.get("AGENT_MAX_STEPS", 25))

        prompt = f"""
Du bist ein Web-Agent mit Playwright-Werkzeugen.
Die initial vermutete Webseite für "{company_name}" ist "{result_data["final_url"]}" (Quelle: {result_data["source_of_url"]}).

Deine Aufgabe ist es, die folgenden Fakten über "{company_name}" zu finden und zu extrahieren.

1.  **Überprüfung und Navigation:**
    a. Öffne die initial vermutete URL: {result_data["final_url"]}
    b. Überprüfe sorgfältig, ob diese Webseite tatsächlich die offizielle Webseite von "{company_name}" ist.
    c. **Wenn die Seite NICHT die korrekte Webseite ist:**
        i. Navigiere zu www.startpage.com.
        ii. Suche nach "{company_name}".
        iii. Analysiere die Suchergebnisse und identifiziere die wahrscheinlichste offizielle Webseite. Navigiere zu dieser Seite.
        iv. Wenn du nach der Startpage-Suche immer noch keine passende Seite findest oder die Navigation fehlschlägt, verwende die ursprüngliche URL {result_data["final_url"]} als Basis für die Faktensuche oder setze Felder auf "null". Die "official_website" im JSON sollte dann {result_data["final_url"]} sein.
    d. **Wenn die Seite die korrekte Webseite ist (oder du nach der Startpage-Suche auf der korrekten Seite gelandet bist):**
        i. Die URL, auf der du dich befindest und von der du Daten extrahierst, ist die "official_website".
        ii. Durchsuche diese Seite und relevante Unterseiten und sammle die unten genannten Fakten.

2.  **Fakten zu sammeln:** {', '.join(EXPECTED_JSON_KEYS)}

Antworte **ausschließlich** mit genau diesem JSON, ohne jeglichen Text davor oder danach.
Der Wert für "official_website" MUSS die URL sein, von der du die Informationen letztendlich gesammelt hast.
Wenn eine Information nicht gefunden werden kann, verwende "null" als Wert.
{{
  "official_website": "<URL der Webseite, von der die Daten tatsächlich gesammelt wurden>",
  "founded": "<jahr JJJJ oder null>",
  "Hauptsitz": "<vollständige Adresse oder null>",
  "Firmenidentifikationsnummer": "<ID oder null>",
  "HauptTelefonnummer": "<nummer oder null>",
  "HauptEmailAdresse": "<email oder null>",
  "Geschäftsbericht" : "<url/PDF-Link oder null>",
  "extracted_company_name": "<reiner Text des auf Webseite gefundenen Firmennamens, keine URL, oder null>"
}}"""
        logger.debug(f"Agent prompt for '{company_name}' prepared. Initial URL: {result_data['final_url']}")

        agent_timeout = config.get("AGENT_TIMEOUT", 60)
        logger.info(f"Running agent for '{company_name}' with timeout {agent_timeout}s.")
        agent_task = agent.run(prompt, max_steps=config.get("AGENT_MAX_STEPS", 20))
        agent_result_str = await asyncio.wait_for(agent_task, timeout=agent_timeout)
        logger.info(f"Agent for '{company_name}' finished.")

        agent_json_result = json.loads(agent_result_str)
        logger.debug(f"Agent result for '{company_name}': {agent_json_result}")
        for key in EXPECTED_JSON_KEYS:
            result_data[key] = agent_json_result.get(key, "null")

        if "official_website" in agent_json_result and agent_json_result["official_website"] != "null":
            if result_data["final_url"] != agent_json_result["official_website"]:
                logger.info(f"Agent for '{company_name}' updated official_website from {result_data['final_url']} to {agent_json_result['official_website']}.")
            result_data["final_url"] = agent_json_result["official_website"]

        result_data["processing_status"] = f"{result_data['source_of_url']} (AGENT_OK)"

    except asyncio.TimeoutError:
        logger.error(f"Agent processing for '{company_name}' timed out after {agent_timeout} seconds.", exc_info=False) # No need for stack trace on timeout
        result_data["error_message"] = f"Agent processing timed out after {agent_timeout} seconds."
        result_data["processing_status"] = "AGENT_PROCESSING_TIMEOUT"
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON result from agent for '{company_name}'. Result string snippet: '{agent_result_str[:200]}...'", exc_info=True)
        result_data["error_message"] = "Failed to decode JSON result from agent."
        result_data["processing_status"] = "AGENT_JSON_DECODE_ERROR"
    except FileNotFoundError as e_fnf:
        logger.error(f"MCP configuration file not found for '{company_name}': {e_fnf}", exc_info=True)
        result_data["error_message"] = str(e_fnf)
        result_data["processing_status"] = "ERROR_MCP_CONFIG_FILE"
    except TypeError as e_type:
        logger.error(f"MCP configuration type error for '{company_name}': {e_type}", exc_info=True)
        result_data["error_message"] = str(e_type)
        result_data["processing_status"] = "ERROR_MCP_CONFIG_TYPE"
    except Exception as e_agent: # Catch-all for other agent execution errors
        logger.error(f"Agent execution error for '{company_name}': {e_agent}", exc_info=True)
        result_data["error_message"] = f"Agent execution error: {str(e_agent)}"
        result_data["processing_status"] = f"AGENT_EXECUTION_ERROR: {type(e_agent).__name__}"

    finally:
        if client_mcp:
            logger.debug(f"Attempting to stop/close MCPClient for '{company_name}'.")
            try:
                if hasattr(client_mcp, 'stop_servers') and callable(client_mcp.stop_servers):
                    await client_mcp.stop_servers()
                    logger.debug(f"MCPClient.stop_servers() called for '{company_name}'.")
                elif hasattr(client_mcp, 'close') and callable(client_mcp.close):
                    await client_mcp.close()
                    logger.debug(f"MCPClient.close() called for '{company_name}'.")
            except Exception as e_stop_client:
                logger.error(f"Error stopping/closing MCPClient for '{company_name}': {e_stop_client}", exc_info=True)
        else:
            logger.debug(f"No active MCPClient to stop for '{company_name}'.")

        if tmp_profile_dir: # Only attempt cleanup if tmp_profile_dir was created
            logger.info(f"Starting cleanup of temporary profile directory for '{company_name}': {tmp_profile_dir}")
            try:
                kill_chrome_processes_using(tmp_profile_dir)
                # Short delay to allow OS to release file locks after process termination
                await asyncio.sleep(0.5) # Use asyncio.sleep in async function
                rmtree_with_retry(tmp_profile_dir)
                logger.info(f"Successfully cleaned up temporary profile directory for '{company_name}': {tmp_profile_dir}")
            except Exception as e_cleanup:
                logger.error(f"Error during cleanup of temporary profile directory for '{company_name}' ({tmp_profile_dir}): {e_cleanup}", exc_info=True)
        else:
            logger.debug(f"No temporary profile directory to clean up for '{company_name}'.")

    logger.info(f"Finished processing for '{company_name}'. Final status: {result_data['processing_status']}")
    return result_data


if __name__ == "__main__":
    async def main_test_run():
        logger.info("Starting main_test_run for core_processing.py")
        dummy_mcp_config_content = {"mcpServers": {"playwright": {"executable": "echo", "args": []}}}
        dummy_sequential_config_path = Path("dummy_sequential_mcp_config.json")
        with open(dummy_sequential_config_path, "w") as f: json.dump(dummy_mcp_config_content, f)

        dummy_base_launcher_content = {"mcpServers": {"playwright": {"executable": "echo", "args": ["--config", "dummy"]}}}
        dummy_parallel_launcher_path = Path("dummy_parallel_mcp_launcher.json") # Used as base for dynamic
        # Ensure this dummy parallel launcher is in the same directory as core_processing.py for the test path to find it.
        # Or adjust path in dynamic_mcp_params if it's elsewhere.
        # For the test, let's assume it's created in current working dir if core_processing.py is run directly.
        # Path(__file__).parent should make it relative to core_processing.py itself.

        with open(Path(__file__).parent / "dummy_parallel_mcp_launcher.json", "w") as f: json.dump(dummy_base_launcher_content, f)


        test_config_global = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY"),
            "URL_LLM_MODEL": os.getenv("URL_LLM_MODEL", "gpt-3.5-turbo"),
            "AGENT_LLM_MODEL": os.getenv("AGENT_LLM_MODEL", "gpt-4-turbo"),
            "LLM_RELEVANCE_CHECK_MODEL": os.getenv("LLM_RELEVANCE_CHECK_MODEL", "gpt-3.5-turbo"),
            "AGENT_TIMEOUT": 10, # Shortened for testing
            "AGENT_MAX_STEPS": 3, # Shortened for testing
        }

        if not test_config_global["OPENAI_API_KEY"]:
            logger.error("Skipping core_processing.py main_test_run: OPENAI_API_KEY not set.")
            if dummy_sequential_config_path.exists(): dummy_sequential_config_path.unlink(missing_ok=True)
            if (Path(__file__).parent / "dummy_parallel_mcp_launcher.json").exists(): (Path(__file__).parent / "dummy_parallel_mcp_launcher.json").unlink(missing_ok=True)
            return

        logger.info(f"--- Test Run using Config: --- \n{json.dumps(test_config_global, indent=2)}")

        logger.info("\n--- Test Case 1: Known Company (Static MCP Config) ---")
        result1 = await process_company_info(
            company_name="NVIDIA", # Simpler name for dir
            company_number="NVDA123",
            config=test_config_global,
            mcp_config_path_or_object=str(dummy_sequential_config_path)
        )
        logger.info(f"Test Case 1 Result: \n{json.dumps(result1, indent=2)}")

        logger.info("\n--- Test Case 2: Known Company (Dynamic MCP Config) ---")
        dynamic_mcp_params = {
             # This path will be resolved relative to core_processing.py if not absolute
            "base_mcp_launcher_path": "dummy_parallel_mcp_launcher.json",
            "headless": True
        }
        result2 = await process_company_info(
            company_name="Microsoft", # Simpler name
            company_number="MSFT456",
            config=test_config_global,
            mcp_config_path_or_object=dynamic_mcp_params
        )
        logger.info(f"Test Case 2 Result: \n{json.dumps(result2, indent=2)}")

        if dummy_sequential_config_path.exists(): dummy_sequential_config_path.unlink(missing_ok=True)
        if (Path(__file__).parent / "dummy_parallel_mcp_launcher.json").exists(): (Path(__file__).parent / "dummy_parallel_mcp_launcher.json").unlink(missing_ok=True)
        logger.info("Finished main_test_run for core_processing.py")

    if os.getenv("RUN_CORE_PROCESSING_MAIN_TEST_RUN"):
        try:
            from logging_utils import setup_logging
            # Using a more specific log file for this test run.
            setup_logging(log_level=logging.DEBUG, log_file="core_processing_self_test.log")
        except ImportError:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s', stream=sys.stdout)
            logger.warning("logging_utils not found, using basicConfig for test run.")

        asyncio.run(main_test_run())
    else:
        if not logging.getLogger().hasHandlers():
             logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s', stream=sys.stdout)
        logger.info(f"Skipping {Path(__file__).name} main_test_run(). Set RUN_CORE_PROCESSING_MAIN_TEST_RUN=1 to run.")
    pass
