import cv2
import click

from cv_qr_finder import CVQRPatternFinder
from qr_finder import QRPatternFinder


@click.command()
@click.option("--image-path", type=click.Path(exists=True), required=True, help="Path to image with QR-Codes")
@click.option("--result-path", type=click.Path(), required=True, help="Path to resulting image")
@click.option("--algorithm", type=str, required=True, help="Algorithm to use for detection")
def main(image_path: click.Path, result_path: click.Path, algorithm: str):
    image_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if algorithm == "own":
        pattern_finder = QRPatternFinder(image_orig)
    elif algorithm == "cv2":
        pattern_finder = CVQRPatternFinder(image_orig)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} is not available. Available algorithms: ['own', 'cv2']")

    detection_points = pattern_finder.find_qr_patterns()
    img = cv2.cvtColor(image_orig, cv2.COLOR_GRAY2RGB)

    for x, y in detection_points:
        img = cv2.circle(img, (int(x), int(y)), radius=10, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(result_path, img)


if __name__ == "__main__":
    main()