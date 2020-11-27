import argparse
import cv2
from pandas import np

from service.contour_extraction.contour_extraction import get_contour, crop_conturs, load_image_from_path, \
    get_img_with_padding

def main(path):

    images = crop_conturs(load_image_from_path(path))
    for i in images:
        i = get_img_with_padding(i)
        mask = np.ones(i.shape, dtype=np.uint8)
        contour = get_contour(i)
        cv2.drawContours(mask, contour, -1, (255, 255, 255), 2)
        cv2.imshow(get_contour(mask))
        print(contour)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract size')
    parser.add_argument('--img', help='Path to img file')
    args = parser.parse_args()
    main(args.img)