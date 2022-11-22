


class StatusCodeErr(Exception):
    def __init__(self,msg,code,url):
        self.code = code
        self.url = url
        return super().__init__(msg)

class UnexpectedReturnErr(Exception):
    def __init__(self,msg):
        return super().__init__(msg)