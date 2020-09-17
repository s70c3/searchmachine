from math import trunc
import numpy as np
import pandas as pd
from collections import Counter


def fast_hist(arr, bins):
    histed = [0 for x in range(bins)]
    if len(arr) == 0:
        return histed
    if isinstance(arr, tuple) and all([len(e)==0 for e in arr]):
        return histed

    mx = max(arr)
    mn = min(arr)
    step = (mx-mn)/bins

    if mx == mn:
        norm = 1 / (mx-mn + 1)
    else:
        norm = 1 / (mx-mn)
    nx = bins

    for elem in arr:
        try:
            ix = trunc((elem - mn) * norm * nx)
            if ix == bins:
                ix -=1
            histed[ix] += 1
        except:
            pass#rint('-->','el', elem, 'min max', mn, mx)

    return np.array(histed)




       
def raw_data_to_features(df):
    materials = df['Материал'].copy()

    material_categories = set()
    material_categories.add('too_rare')
    material_freqs = Counter()

    for ix, cat in materials.apply(lambda s: s.split()[0].lower()).iteritems():
        material_categories.add(cat)
        material_freqs[cat] += 1

    # set -> dict
    material_categories = {cat: i for i, cat in enumerate(list(material_categories))}
    print(material_freqs)
    
    
    
    def filter_correct_dims(dims):
        dims_ = []
        for d in dims:
            try:
                dims_.append(float(d))
            except ValueError:
                pass
        return dims_


    df = df[df['Размер'].apply(lambda s: len(filter_correct_dims(s.lower().split('х'))) == 3)]

    mul = lambda arr: arr[0] * mul(arr[1:]) if len(arr) > 1 else arr[0]
    calc_vol = lambda params: mul([float(x) for x in filter_correct_dims(params.lower().split('х'))])
    calc_dims = lambda s: np.array([float(x) for x in filter_correct_dims(s.lower().split('х'))])
    def get_material(s):
        mat = s.split()[0].lower()
        if material_freqs[mat] < 70:
            mat = 'too_rare'
        return mat

    def get_price_category(price, price_levels=5):
        if price < 10:
            return 0
        elif price < 200:
            return 1
        elif price < 2000:
            return 2
        elif price < 4000:
            return 3
        else:
            return 4


    ndf = pd.DataFrame()
    # raw features
    ndf['size1'] = df['Размер'].apply(calc_dims).apply(lambda v: sorted(v)[0])
    ndf['size2'] = df['Размер'].apply(calc_dims).apply(lambda v: sorted(v)[1])
    ndf['size3'] = df['Размер'].apply(calc_dims).apply(lambda v: sorted(v)[2])
    ndf['volume'] = (list(map(calc_vol, df['Размер'])))
    ndf['mass'] = (df['Масса заготовки'].astype(float))

    # mass features
    ndf['userful_mass_perc'] = (df['Масса ДЕС'] / df['Масса заготовки']).astype(float)
    ndf['sqr_trash_mass'] = np.square((df['Масса заготовки'] - df['Масса ДЕС']).astype(float))
    ndf['log_mass'] = np.log1p(ndf.mass)
    ndf['sqrt_mass'] = np.sqrt(ndf.mass)

    # volume features
    ndf['log_volume'] = np.log1p(ndf.volume)

    # other features
    ndf['log_density']  = np.log(1000*ndf['mass'] / ndf['volume'])
    ndf['material_category'] = df['Материал'].apply(lambda mat: get_material(mat))
    ndf['price_category'] = df['Цена'].apply(get_price_category)

    ndf['log_price'] = np.log(df['Цена'].astype(float))


    # Drop outliers
    ndf = ndf[ndf.volume > 10]

    # Drop unneccessary
    # ndf = ndf.drop(['size%d'% i for i in [1]],axis=1)
    # ndf = ndf.drop('mass volume'.split(), axis=1)
    # ndf = ndf[ (ndf.log_price > 2)]



    # indexed = ndf.copy()
    # indexed['detail_name'] = df['Номенклатура']
    # indexed.to_csv('featured_details_mape014.csv', index=False)

    return ndf