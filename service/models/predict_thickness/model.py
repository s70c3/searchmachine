import re
from typing import Optional

from typeguard import check_argument_types
from service.models.base import BaseModel


class ThicknessModel(BaseModel):
    """Fetches material thickness from the given material string"""

    @staticmethod
    def _preprocess(material):
        # Delete all external symbols
        allowed_symbs = '-xх,. '
        is_allowed = lambda c: c.isdigit() or c in allowed_symbs
        material = material.lower()

        filtred_material = ''
        prev_added_c = None
        for i, c in enumerate(material[1:], start=1):
            if is_allowed(c):
                if prev_added_c is None or c.isdigit():
                    filtred_material += c
                elif c in allowed_symbs and c != prev_added_c:
                    filtred_material += c
                else:
                    continue
                prev_added_c = c
        return filtred_material

    @staticmethod
    def _get_best_candidate(candidate_sizes_string):
        # Select substring that looks the most closely to sizes
        candidates = candidate_sizes_string.split()
        for s in candidates:
            if ',' in s:
                return s
        for s in candidates:
            if re.search(r"\d{1,2}(x|х)\d{1,5}(x|х)\d{1,5}", s):
                return s
        for s in candidates:
            if re.search(r"\d{1,2}(x|х)\d{1,5}", s):
                return s
        return None

    @staticmethod
    def _fetch_thickness(candidate_str) -> float or None:
        # Fetch thickness string from candidate string
        if candidate_str is None:
            return None

        s = candidate_str.strip('- ').replace(',', '.')
        s = s.replace('x', '@').replace('х', '@').replace('-', '@')
        if '@' in s:
            s = s[:s.index('@')]
        try:
            if float(s) > 100:
                return None
            else:
                return float(s)
        except:
            # If doesnt cast or is too large for thickness
            pass
        return None

    @staticmethod
    def predict(material_string: Optional[str]) -> float or None:
        assert check_argument_types()

        if material_string is None:
            return None
        candidate_str = ThicknessModel._preprocess(material_string)
        candidate = ThicknessModel._get_best_candidate(candidate_str)
        thickness = ThicknessModel._fetch_thickness(candidate)
        return thickness
