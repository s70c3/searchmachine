from .detail import Detail


class PackingParametersBase:
    def __init__(self, params):
        '''
        Common functionality of parameters parsing for packings
        @param params: json with params for packing
        '''
        self.errors = []
        self.warnings = []
        self.params = params

        self.details = []
        self.visualize = self._get_or_default(params, 'render_packing_maps', False)

        # parse material data
        if self._check_field(self.params, 'material'):
            self.material = self.params['material']
            if self._check_field(self.params['material'], 'width', 'material -> width') and \
                    self._check_field(self.params['material'], 'height', 'material -> height'):
                self.material_width = self.params['material']['width']
                self.material_height = self.params['material']['height']

    def _check_field(self, obj, field_name, extra_filed_name=None):
        if field_name not in obj.keys():
            if extra_filed_name:
                field_name = extra_filed_name
            self.errors.append({'no_param': 'param %s not in request data' % field_name})
            return False
        return True

    def _get_or_default(self, obj, key, default_value, extra_field_name=None):
        if key not in obj.keys():
            value = default_value
            if extra_field_name:
                key = extra_field_name
            self.warnings.append(
                {'no_param': 'param %s not in request data. Default value %s used' % (key, str(default_value))})
        else:
            value = obj[key]
        return value

    def all_details_fits_material(self):
        for idx, detail in enumerate(self.details):
            if detail.w > self.material_width or detail.h > self.material_height:
                msg = 'detail %d  size (%d, %d) doesnt fit to material size (%d, %d)' % (idx,
                                                                                         detail.w,
                                                                                         detail.h,
                                                                                         self.material_width,
                                                                                         self.material_height)
                self.errors.append({'wrong_param': msg})
        return len(self.errors) == 0


class RectPackingParameters(PackingParametersBase):
    def __init__(self, params):
        '''
        Parsing parameters for rect packing. Quantity, width and height are neccessary, dxfs are optionals
        @param params: json with params for packing
        '''
        super(RectPackingParameters, self).__init__(params)

        # parse details data
        if self._check_field(self.params, 'details'):
            items = self.params['details']
            for idx, item in enumerate(items):
                if self._check_field(item, 'width', 'details -> #%d -> width' % idx) and \
                        self._check_field(item, 'height', 'details -> #%d -> height' % idx) and \
                        self._check_field(item, 'quantity', 'details -> #%d -> quantity' % idx):
                    w, h = int(item['width']), int(item['height'])
                    q = int(item['quantity'])
                    dxf = self._get_or_default(item, 'dxf', None, 'detail -> #%d -> dxf' % idx)
                    dxf = None
                    detail = Detail(w, h, q, idx, dxf)
                    self.details.append(detail)
                else:
                    # broken details data doesnt fix here, just passing. Developers
                    # should inspect incidents with logs in self.errors and their data
                    pass


class DxfPackingParameters(PackingParametersBase):
    def __init__(self, params):
        '''
        Parsing parameters for polygonal packing. Quantity and dxfs are neccessary
        @param params: json with params for packing
        '''
        super(DxfPackingParameters, self).__init__(params)

        # parse details data
        if self._check_field(self.params, 'details'):
            items = self.params['details']
            for idx, item in enumerate(items):
                if self._check_field(item, 'quantity', 'details -> #%d -> quantity' % idx) and \
                   self._check_field(item, 'dxf', 'detail -> #%d -> dxf' % idx):
                    q = int(item['quantity'])
                    w = self._get_or_default(item, 'width', -1, 'details -> #%d -> width' % idx)
                    h = self._get_or_default(item, 'height', 1, 'details -> #%d -> height' % idx)
                    dxf = item['dxf']
                    detail = Detail(w, h, q, idx, dxf)
                    
                    dxf_load_errors = detail.get_loading_errors()
                    if len(dxf_load_errors):
                        self.warnings.append({'dxf_load_error': 'detail -> #%d -> dxf. Rect with sides %d %d used as default dxf. Errors: %s' % (idx, w, h, str(dxf_load_errors))})
                    
                    self.details.append(detail)
                else:
                    # broken details data doesnt fix here, just passing. Developers
                    # should inspect incidents with logs in self.errors and their data
                    pass

    def all_details_fits_material(self):
        for idx, detail in enumerate(self.details):
            
            w, h = detail.get_dxf_size()
            if w > self.material_width or h > self.material_height:
                
                self.details[idx].rotate()
                w, h = self.details[idx].get_size()
                if w > self.material_width or h > self.material_height:
                    msg = 'detail %d  size (%d, %d) doesnt fit to material size (%d, %d)' % (idx,
                                                                                             w, h,
                                                                                             self.material_width,
                                                                                             self.material_height)
                    self.errors.append({'wrong_param': msg})
        return len(self.errors) == 0