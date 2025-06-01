import logging
import sys

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configures basic logging for the application.

    Args:
        log_level (int): The minimum logging level to output.
        log_file (str, optional): Path to a file to save logs.
                                 If None, logs are only sent to console.
    """
    # Get the root logger. Configuring the root logger will affect all loggers
    # unless they are specifically configured otherwise.
    # For more complex applications, you might configure specific named loggers.
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Remove any existing handlers to avoid duplicate logs if this function is called multiple times
    # or if other libraries (like mcp_use) might have configured the root logger.
    # Note: This can be aggressive. A more targeted approach might be needed in complex scenarios.
    # For this project, assuming we want to control the primary log output here.
    if logger.hasHandlers():
        for handler in logger.handlers[:]: # Iterate over a copy of the list
            logger.removeHandler(handler)
            handler.close() # Close handler before removing

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for info/debug, sys.stderr for error by default in StreamHandler
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level) # Console handler should also respect the log_level
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level) # File handler should also respect the log_level
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging for {log_file}: {e}")

    # BasicConfig is a one-time setup. If called multiple times, subsequent calls have no effect.
    # We are doing manual setup above, so basicConfig is not strictly needed here,
    # and can sometimes interfere if other libs call it.
    # logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s', handlers=handlers)

    logger.info("Logging setup complete.")

if __name__ == '__main__':
    # Example usage:
    setup_logging(log_level=logging.DEBUG, log_file="example_app.log")

    # Get a logger for a specific module/component
    test_logger = logging.getLogger("my_app_test") # or logging.getLogger(__name__) in a module

    test_logger.debug("This is a debug message.")
    test_logger.info("This is an info message.")
    test_logger.warning("This is a warning message.")
    test_logger.error("This is an error message.")

    try:
        x = 1 / 0
    except ZeroDivisionError:
        test_logger.exception("A ZeroDivisionError occurred!")

    print("Check 'example_app.log' and console for output.")
