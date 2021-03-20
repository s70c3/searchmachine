import os
import cairosvg

FILE_FOLDER = 'models/packing/files/'

def _compress_pngs(archive_name):
    '''Creates archive from png files in folder. Deletes png files'''
    os.system(f'cd {FILE_FOLDER}; zip {archive_name} *.png')
    os.system(f'rm {FILE_FOLDER}*.png')


def _get_archive_name_prefix(archive_name):
    '''Returns the greatest number plus one of archives named as given and stored in folder'''
    prefixes = list(filter(lambda f: f.endswith(archive_name), os.listdir(FILE_FOLDER)))
    prefixes = list(map(lambda f: int(f.split('_')[0]), prefixes))
    prefix = max([0] + prefixes) + 1
    return prefix


def save_rendered_packings(maps, name_suffix=''):
    '''
    Packs given files to archive and deletes them
    @param  maps           list of PIL Images. Names of files to pack. Files must be in FILE_FOLDER
    @param name_suffix     added to the end of archive name
    @return name of archive inside FILE_FOLDER
    '''
    
    # select the greatest prefix in files dir or 0 if no files in directory
    archive_name = '_packings'
    if len(name_suffix):
        archive_name += '_'+name_suffix
    archive_name += '.zip'

    # get the greatest count number of saved archives
    prefix = _get_archive_name_prefix(archive_name)
    archive_name = str(prefix) + archive_name

    for i, img in enumerate(maps):
        img.save(FILE_FOLDER+'{}_{}.png'.format(prefix, i))

    _compress_pngs(archive_name)
    return f'files/{archive_name}'


def save_svgs(elements):
    '''
    Saves batch of xml.dom.minidom objects as png and compresses to archive
    @param elements   list of xml.dom.minidom objects - packmaps with details
    '''

    def get_packmap_size(packmap):
        is_rect = lambda elem: 'Element: rect at' in str(elem)
        elems = list(filter(is_rect, packmap.childNodes))
        rect = elems[0]
        width = float(rect.getAttribute('width'))
        height = float(rect.getAttribute('height'))
        return width, height

    svg_file_pattern = '''<?xml version="1.0" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
    "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <svg width="{w}" height="{h}" version="1.1"
        xmlns="http://www.w3.org/2000/svg">{svg}</svg>'''

    for i, elem in enumerate(elements):
        w,h = get_packmap_size(elem)
        source_code = svg_file_pattern.format(w=w, h=h, svg=elem.toxml())

        # convert svg to png
        file = FILE_FOLDER + str(i) + '.svg'
        with open(file, 'w') as f:
            f.write(source_code)
        cairosvg.svg2png(url=file, write_to=file.replace('svg', 'png'))
    os.system(f'rm {FILE_FOLDER}*.svg')

    archive_name = '_packings_nest.zip'
    prefix = _get_archive_name_prefix(archive_name)
    archive_name = str(prefix) + archive_name

    _compress_pngs(archive_name)
    return f'files/{archive_name}'

def rm_svgs():
    os.system(f'rm {FILE_FOLDER}*.svg')