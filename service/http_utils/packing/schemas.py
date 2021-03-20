from webargs import fields

class RectSchema(fields.ma.Schema):

    detail_schema = {'width': fields.Number(required=True),
                     'height': fields.Number(required=True),
                     'quantity': fields.Int(required=True),
                     'dxf': fields.Str(required=False)}

    material = {'width': fields.Number(required=True),
                'height': fields.Number(required=True)}

    details = fields.List(fields.Nested(detail_schema))
    material = fields.Nested(material)
    render_packing_maps = fields.Bool(required=False, default=True)


class DxfSchema(fields.ma.Schema):

    detail_schema = {'quantity': fields.Int(required=True),
                     'dxf': fields.Str(required=True)}

    material = {'width': fields.Number(required=True),
                'height': fields.Number(required=True)}

    details = fields.List(fields.Nested(detail_schema))
    material = fields.Nested(material)
    render_packing_maps = fields.Bool(required=False, default=True)
    iterations = fields.Int(required=True)
    rotations = fields.Int(required=True)