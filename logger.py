import os

class Logger:
    """
    日志类
    """

    def __init__(self):
        self.__log_file_path = "log.txt"
        self.__level = {}

    def set_log_file_path(self, log_file_path: str):
        """
        设置日志文件路径
        Args:
            log_file_path: 日志文件路径
        """
        self.__log_file_path = log_file_path
    
    def write_to_console(self, msg):
        """
        输出日志到控制台
        """
        print(msg)
    
    def init(self):
        """
        初始化日志功能
        """
        # 如果文件不存在 创建文件
        if not os.path.exists(self.__log_file_path):
            file = open(self.__log_file_path, 'w')
            file.close()
        self.logger = open(self.__log_file_path)

    def close_log_file(self):
        """
        关闭日志
        """
        self.logger.close()