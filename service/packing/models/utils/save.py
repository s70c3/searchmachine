import os

FILE_FOLDER = 'packing/models/files/'

def save_rendered_packings(maps):
    '''
    Packs given files to archive and deletes them
    @param  maps   list of str. Names of files to pack. Files must be in FILE_FOLDER
    @return name of archive inside FILE_FOLDER
    '''
    
    # select the greatest prefix in files dir or 0 if no files in directory
    prefixes = list(filter(lambda f: f.endswith('_packings.zip'), os.listdir(FILE_FOLDER)))
    prefixes = list(map(lambda f: int(f.split('_')[0]), prefixes))
    prefix = max([0] + prefixes) + 1
    
    for i, img in enumerate(maps):
        img.save(FILE_FOLDER+'{}_{}.png'.format(prefix, i))
    os.system('cd {folder}; zip {prefix}_packings.zip {prefix}_*.png'.format(folder=FILE_FOLDER, prefix=prefix))
    os.system('rm {}{}_*.png'.format(FILE_FOLDER, prefix))
    return FILE_FOLDER + str(prefix) + '_packings.zip'