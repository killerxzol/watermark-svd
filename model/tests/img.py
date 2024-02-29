import numpy as np
from PIL import Image
from model import Watermark


def read(path):
    text = ""
    with open(path, mode="r") as file:
        for line in file:
            text += line.strip("\n")
    return text


def main(img_path="../../.data/img/3.gif", txt_path="../../.data/txt/1.txt"):
    image = np.array(Image.open(img_path))
    text = read(txt_path)

    watermark = Watermark(
        block_size=4,
        intercept=1,
        redundancy=5,
        max_iter=11
    )

    watermark_image = watermark.encode(image.copy(), text)
    Watermark.save_image(watermark_image, "../../.data/output_img/")

    decoded_text = watermark.decode(watermark_image)

    print(f"\nWatermark params: [block_size={watermark.block_size}; "
          f"intercept={watermark.intercept}; "
          f"redundancy={watermark.redundancy}; "
          f"max_iter={watermark.max_iter}].")
    print(f"\nFrobenius norm: {Watermark.image_norm(image - watermark_image)}")
    print(f"\nDecoded text: {decoded_text}")


if __name__ == "__main__":
    main()
