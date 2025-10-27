import sys

def error_message_detail(err, err_detail:sys):
    _, _ , exc_tb = err_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    err_message = "Error occurred in Python Script name: [{0}], line number: [{1}], error message: [{2}]".format(
        file_name,
        line_number,
        str(err)
    )
    return err_message

class Custom_Exception(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)
        
    def __str__(self):
        return self.error_message

from logger_config import logger

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logger.info("Divided by zero")
        raise Custom_Exception(e, sys)