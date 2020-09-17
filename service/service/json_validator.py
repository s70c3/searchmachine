class Validator:
    def __init__(self, schema):
        '''
        Validates json with a given type schema
        @param  schema     dict of fields. The end elements should be types
        '''
        self.schema = schema

    def _validate(self, pattern, item):
        errs = []
        for key in pattern.keys():
            if key not in item.keys():
                errs.append({'no_param': '%s key not in params' % key})
                continue
            elif type(pattern[key]) != type(item[key]):
                errs.append(
                    {'wrong_param': '%s key has wrong param type. Type %s expected' % (key, type(pattern[key]))})
                continue

            if type(pattern[key]) == dict:
                deep_errs = self._validate(pattern[key], item[key])
                if len(deep_errs) != 0:
                    errs.append(deep_errs)
        return errs

    def validate(self, jsn):
        '''
        Validates given json with schema
        @param  jsn    dict with data
        @return bool, errors
        '''
        errs = self._validate(self.schema, jsn)
        return errs