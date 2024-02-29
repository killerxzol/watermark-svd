import time
import numpy as np
from math import floor
from PIL import Image


class Watermark:

    def __init__(
            self,
            block_size,
            intercept,
            redundancy,
            max_iter
    ):
        self.block_size = block_size
        self.intercept = intercept
        self.redundancy = redundancy
        self.max_iter = max_iter

    def encode(self, image, text):
        return self._encode(image, text)

    def _encode(self, image, text):
        row_capacity = floor(image.shape[0] / self.block_size)
        col_capacity = floor(image.shape[1] / self.block_size)
        bit_capacity = floor(
            (self.block_size - self.intercept - 1) * (self.block_size - self.intercept) * 0.5
        )
        max_bit = floor(row_capacity * col_capacity * bit_capacity / self.redundancy)
        max_char = floor(max_bit / 7)

        print(f"\nEncoding | {max_char} (out of {len(text)}) characters will be encoded.")
        text = text if len(text) < max_char else text[:max_char]
        self.binary_text = [
            bit for bit in self._encode_text(text) for _ in range(self.redundancy)
        ]

        for _ in range(self.max_iter):
            break_out_flag = False
            binary_text = self.binary_text
            for j_block in range(col_capacity):
                if break_out_flag: break
                for i_block in range(row_capacity):
                    if not binary_text:
                        break_out_flag = True
                        break
                    elif len(binary_text) <= bit_capacity:
                        block_text = binary_text
                        while len(block_text) < bit_capacity:
                            block_text.append(1)
                        binary_text = None
                    else:
                        block_text = binary_text[:bit_capacity]
                        binary_text = binary_text[bit_capacity:]

                    block = image[
                            (self.block_size * i_block):(self.block_size * (i_block + 1)),
                            (self.block_size * j_block):(self.block_size * (j_block + 1)),
                    ]
                    u, s, vt = np.linalg.svd(block, full_matrices=True)
                    for k in range(self.block_size):
                        if u[0, k] < 0:
                            u[:, k] = -u[:, k]
                            vt[k, :] = -vt[k, :]
                    u, s = self._encode_block(u, s, block_text)
                    block = np.round(np.dot(u * s, vt)).clip(0, 255)
                    image[
                        (self.block_size * i_block):(self.block_size * (i_block + 1)),
                        (self.block_size * j_block):(self.block_size * (j_block + 1)),
                    ] = block
        return image

    def _encode_block(self, u, s, text):
        avg = (s[1] + s[-1]) / (self.block_size - 2)
        for i in range(2, self.block_size - 1):
            s[i] = s[1] - (i - 1) * avg
        text_index = 0
        for j in range(self.intercept, self.block_size):
            for i in range(1, self.block_size - j):
                u[i, j] = abs(u[i, j]) + text[text_index]
                text_index += 1
            u = self._normalize_block(u, j)
            norm = np.sqrt(np.dot(u[:, j], u[:, j]))
            if norm != 0:
                u[:, j] = u[:, j] / norm
        return u, s

    def _normalize_block(self, block, column):
        coef, solution = np.zeros((column, column)), np.zeros(column)
        for i in range(column):
            for j in range(column):
                coef[i, j] = block[j + self.block_size - column, i]
            solution[i] = -np.dot(
                block[0:(self.block_size - column), column], block[0:(self.block_size - column), i]
            )
        block[(self.block_size - column):self.block_size, column] = np.linalg.lstsq(coef, solution, rcond=None)[0]
        return block

    def decode(self, image):
        return self._decode(image)

    def _decode(self, image):
        binary_text = []
        row_capacity = floor(image.shape[0] / self.block_size)
        col_capacity = floor(image.shape[1] / self.block_size)
        for j_block in range(col_capacity):
            for i_block in range(row_capacity):
                block = image[
                        (self.block_size * i_block):(self.block_size * (i_block + 1)),
                        (self.block_size * j_block):(self.block_size * (j_block + 1)),
                ]
                binary_text.extend(self._decode_block(block))
        return self._decode_text(binary_text, self.redundancy)

    def _decode_block(self, block):
        binary_text = []
        u, _, _ = np.linalg.svd(block, full_matrices=True)
        for j in range(self.block_size):
            if u[0, j] < 0:
                for i in range(self.block_size):
                    u[i, j] = -u[i, j]
        for j in range(self.intercept, self.block_size):
            for i in range(1, self.block_size - j):
                binary_text.append(-1 if u[i, j] < 0 else 1)
        return binary_text

    @staticmethod
    def image_norm(image):
        return np.linalg.norm(image)

    @staticmethod
    def _encode_text(text):
        binary_text = list("".join(format(ord(char), "07b") for char in text))
        binary_text = [-1 if bit == "0" else 1 for bit in binary_text]
        return binary_text

    @staticmethod
    def _decode_text(binary_text, redundancy=1):
        raw_text = []
        for _ in range(len(binary_text) % redundancy):
            binary_text.append(0)
        for i in range(0, len(binary_text), redundancy):
            raw_text.append("0" if sum(binary_text[i:(i + redundancy)]) < 0 else "1")
        for _ in range(len(raw_text) % 7):
            raw_text.append("0")
        text = ""
        for i in range(0, len(raw_text), 7):
            text += chr(int("".join(raw_text[i:(i + 7)]), 2))
        return text

    @staticmethod
    def save_image(image, path):
        img = Image.fromarray(image)
        img.save(path + time.asctime().replace(':', '-').replace(' ', '_') + '.bmp')
