from datetime import datetime as dt


class LoggerBase:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def _write_msg(self, msg):
        with open(self.filepath, 'a') as log:
            log.write(msg+'\n')
    
    def info(self, msg):
        msg = '[%s][%4s]%s' % (self._time(), 'INFO', msg)
        self._write_msg(msg)
   
    def error(self, msg):
        msg = '[%s][%4s]%s' % (self._time(), 'ERR', msg)
        self._write_msg(msg)

    def _time(self):
        return dt.now().strftime('%d/%m/%y %H:%M:%S')

    
    
class LoggerYellot(LoggerBase):
    def __init__(self, filepath):
        super(LoggerYellot, self).__init__(filepath)

    def info(self, method, msg, params=None):
        msg = '[%s] %s %s' % (method, msg, params or '')
        super().info(msg)
        
    def error(self, code, method, msg, params=None):
        msg = '[%d][%s] %s %s' % (code, method, msg, params or '')
        super().error(msg)