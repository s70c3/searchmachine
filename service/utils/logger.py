from datetime import datetime as dt


class LoggerBase:
    def __init__(self, filepath, duplicate_to_stdout):
        self.filepath = filepath
        self.duplicate_to_stdout = duplicate_to_stdout
    
    def _write_msg(self, msg):
        with open(self.filepath, 'a') as log:
            log.write(msg+'\n')
            if self.duplicate_to_stdout:
                print(msg)
    
    def info(self, msg):
        msg = '[%s][%4s]%s' % (self._time(), 'INFO', msg)
        self._write_msg(msg)
   
    def error(self, msg):
        msg = '[%s][%4s]%s' % (self._time(), 'ERR', msg)
        self._write_msg(msg)

    def _time(self):
        return dt.now().strftime('%d/%m/%y %H:%M:%S')

    
    
class LoggerYellot(LoggerBase):
    def __init__(self, filepath, duplicate_to_stdout=True):
        super(LoggerYellot, self).__init__(filepath, duplicate_to_stdout)

    def info(self, method, msg, params=None):
        msg = '[%s] %s %s' % (method, msg, params or '')
        super().info(msg)
        
    def error(self, code, method, msg, params=None):
        msg = '[%d][%s] %s %s' % (code, method, msg, params or '')
        super().error(msg)