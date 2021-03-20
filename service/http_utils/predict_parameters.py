from tornado.web import RequestHandler
from service.models.predict_thickness.model import ThicknessModel
from service.models.text_recog.model import LinearSizesModel
from service.models.nomeclature_recognition.model import NomenclatureModel
from .validation import PDFValidator


class PredictParamsBySchemaHandler(RequestHandler):
    # TODO: add webargs
    def post(self):
        # validate pdf
        pdf_validator = PDFValidator()
        parse_errors = pdf_validator.get_parse_errors(self)
        if len(parse_errors):
            self.write({'parse_error': 'Cant decode pdf. Maybe its not a pdf file or broken pdf'})
            return
        img = pdf_validator.get_image()
        given_material = self.get_argument('material', None)

        pred_mass, pred_material = NomenclatureModel.predict(img)
        params = {'mass': pred_mass,
                  'material': pred_material,
                  'material_thickness_by_img': ThicknessModel.predict(pred_material),
                  'meterial_thickness_by_given_material': ThicknessModel.predict(given_material)}

        return self.write(params)
