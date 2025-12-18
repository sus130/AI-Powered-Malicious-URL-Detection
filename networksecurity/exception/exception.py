import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys = None):
        """
        Receives an error_message and error_details parameter from the system.
        """
        self.error_message = error_message
        
        try:
            # Get traceback safely
            if error_details:
                _, _, exc_tb = error_details.exc_info()
                if exc_tb is not None:
                    self.lineno = exc_tb.tb_lineno
                    self.file_name = exc_tb.tb_frame.f_code.co_filename
                else:
                    self.lineno = "unknown"
                    self.file_name = "unknown"
            else:
                self.lineno = "unknown"
                self.file_name = "unknown"
        except Exception:
            self.lineno = "unknown"
            self.file_name = "unknown"

    def __str__(self):
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )
