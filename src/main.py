import struct
import numpy as np

def readIdxInt(file):
    data = file.read(4)
    return struct.unpack(">i", data)[0]

def readIdx(path):
    with open(path, "rb") as f:
        mn = readIdxInt(f)
        if mn != 2051:
            print("File did not contain expected magic number")
            return None

        numEntries = readIdxInt(f)
        imgWidth = readIdxInt(f)
        imgHeight = readIdxInt(f)

        if imgWidth <= 0 or imgHeight <= 0 or numEntries <= 0:
            return None

        vectorDim = imgWidth * imgHeight
        result = np.empty((numEntries, vectorDim))
        print(result.shape)

        for e in xrange(numEntries):
            for i in xrange(vectorDim):
                pixel = f.read(1)
                result[e, i] = struct.unpack("B", pixel)[0] / 255.0

        print(numEntries)


a = [1, 2, 3]
print("hello world")
print(a)

readIdx("data/test_images.idx3")
