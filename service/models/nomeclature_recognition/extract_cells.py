from . import extract_table

def get_cell_imgs(img):
    t_vh = extract_table.remove_text(img)
    consts = extract_table.ImageConstants(img)
    
    lower_img_bbox = extract_table.get_lower_image(t_vh, consts)
    lower_img = extract_table.get_subimg(t_vh, lower_img_bbox)
    
    table_bbox = extract_table.get_table(lower_img, consts)
    table_img = extract_table.get_subimg(lower_img, table_bbox)
    combined_bbox = extract_table.combine_bboxes(mbbox=lower_img_bbox, rbbox=table_bbox)
    
    cell_extractor = extract_table.CellExtractor(table_img, consts)

    mass_bbox = cell_extractor.get_mass_cell()
    mass_header_bbox = cell_extractor.get_mass_header_cell()
    name_bbox, detail_bbox, material_bbox = cell_extractor.get_material_name_detail_cells()
    bboxes = [mass_bbox, mass_header_bbox, name_bbox, detail_bbox, material_bbox]

    combined_bboxes = [extract_table.combine_bboxes(combined_bbox, b) for b in bboxes]
    subimgs = [extract_table.get_subimg(img, b) for b in combined_bboxes]
    mass_subimg, mass_header_subimg, name_subimg, detail_subimg, material_subimg = subimgs
    return mass_subimg, mass_header_subimg, name_subimg, detail_subimg, material_subimg