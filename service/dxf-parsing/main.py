import argparse
import cv2
import dxf_parsing as dxf


def main(dxf_name):
    print(dxf.get_size_from_dxf(dxf_name))
    print(dxf.get_contour_from_dxf(dxf_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse dxf')
    parser.add_argument('--dxf-file', help='Path to dxf file')
    args = parser.parse_args()
    main(args.dxf_file)