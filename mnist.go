// Copyright 2013 Petar Maymounkov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package GoMNIST

import (
	"compress/gzip"
	"encoding/binary"
	"image"
	"image/color"
	"io"
	"os"
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width      = 28
	Height     = 28
)

// Image holds the pixel intensities of an image.
// 255 is foreground (black), 0 is background (white).
type RawImage []byte

type MatrixImg [][]float64

func (img RawImage) ColorModel() color.Model {
	return color.GrayModel
}

func (img RawImage) Bounds() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{0, 0},
		Max: image.Point{Width, Height},
	}
}

func (img RawImage) At(x, y int) color.Color {
	return color.Gray{img[y*Width+x]}
}

// ReadImageFile opens the named image file (training or test), parses it and
// returns all images in order.
func ReadImageFile(name string) (rows, cols int, allFinalFmtImgs [][][][]float32, err error) {
	f, err := os.Open(name)
	if err != nil {
		return 0, 0, nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return 0, 0, nil, err
	}
	rVals, colVals, returnedImgFileContents, anyErr := readImageFile(z)
	// fmt.Println("Image file contents: ", returnedImgFileContents)
	return rVals, colVals, returnedImgFileContents, anyErr
}

func readImageFile(r io.Reader) (rows, cols int, allFinalFmtImgs [][][][]float32, err error) {
	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)

	// var newMatrix = [1][28][28][1]float32{}

	// fmt.Println("New Matrix: ")
	// fmt.Println(newMatrix)

	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return 0, 0, nil, err
	}
	if magic != imageMagic {
		return 0, 0, nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return 0, 0, nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return 0, 0, nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return 0, 0, nil, err
	}
	imgs := make([]RawImage, n)
	// finalFmtImgs := [32][32]uint8{}
	// newFmtImgs := [28][28]uint8{}

	allFinalFmtImgs = [][][][]float32{{}, {}, {}, {}, {}}

	m := int(nrow * ncol)
	// fmt.Println("Values of nrow and ncol & n are : ", nrow, ncol, n)
	// fmt.Println("Value of m is: ", m)
	for i := 0; i < int(5); i++ {
		imgs[i] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		// fmt.Println("Value of m_ & image inside loop: ", m_, imgs[i])

		currentImageBytes := imgs[i]
		currentImageFloatVals := make([][]float32, m)
		var newFmtImg [][][]float32

		// imgBytesFloat := make([]float64, m)
		// for ctr, element := range currentImageBytes {
		// 	imgBytesFloat[ctr] = float64(element)
		// }
		// newFmtImgs := mat.NewDense(int(nrow), int(ncol), imgBytesFloat)
		// fmt.Println("New Matrix created is: ", newFmtImgs)
		// newRows, newCols := newFmtImgs.Dims()

		// // Should be 28 x 28
		// fmt.Println("Dimensions of the new matrix are : ", newRows, newCols)

		// Manually create a 28 x 28 nested slice
		for idx := 0; idx < len(currentImageBytes); idx++ {
			floatNum := float32(currentImageBytes[idx]) // can divide by 255 here if reqd by model
			currentImageFloatVals[idx] = []float32{floatNum}
		}

		for i := 0; i+28 <= len(currentImageFloatVals); i += 28 {
			newFmtImg = append(newFmtImg, currentImageFloatVals[i:i+28])
		}

		// fmt.Println("New 28 x 28 nested slice is :  ", newFmtImg)
		// fmt.Println("Length of nested slice: ", len(newFmtImg))

		// Now we pad it to make it 32 x 32
		// Add 2 elems at the start and end of each slice (0s)
		for idx, _ := range newFmtImg {
			zeroPadArray := [][]float32{{0}, {0}}
			newFmtImg[idx] = append(append(zeroPadArray, newFmtImg[idx]...), zeroPadArray...)
		}

		// fmt.Println("Should have 32 elemns in each slice: ", newFmtImg)

		// Add 2 rows at the start and end
		// emptyRows := [2][32]uint8{}
		emptyRows := [][][]float32{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}}
		withPaddingImg := append(append(emptyRows, newFmtImg...), emptyRows...)

		// append(append(emptyRows, newFmtImg...), emptyRows...)

		// fmt.Println("New 32 x 32 with padding Image is : ", withPaddingImg)
		// fmt.Println("Length of padded img is: ", len(withPaddingImg))

		finalFmtImg := withPaddingImg

		// fmt.Println("New 32 x 32 with padding Image is : ", finalFmtImg)
		allFinalFmtImgs[i] = finalFmtImg

		if err != nil {
			return 0, 0, nil, err
		}
		if m_ != int(m) {
			return 0, 0, nil, os.ErrInvalid
		}
	}
	// fmt.Println("Final Fmt Img: ", allFinalFmtImgs)
	return int(nrow), int(ncol), allFinalFmtImgs, nil
}

func separateInitialElems(bigSlice []uint8) (firstRowElems []uint8, remainingElems []uint8) {
	return bigSlice[:28], bigSlice[28:]
}

// Label is a digit label in 0 to 9
type Label uint8

// ReadLabelFile opens the named label file (training or test), parses it and
// returns all labels in order.
func ReadLabelFile(name string) (labels []Label, err error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	return readLabelFile(z)
}

func readLabelFile(r io.Reader) (labels []Label, err error) {
	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

// Construct matrix with the contents of the mnist data file
// print out matrix

// We have (ncol * nrow) bytes to read from the buffer.
// Collect (ncol * nrow) elems (bytes) and then loop over it, reset counter after every row.
// Read each row, ncol times and form the new matrix structure.

// [[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]]

// [[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 84 185 159 151 60 36 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 222 254 254 254 254 241 198 198 198 198 198 198 198 198 170 52 0 0 0 0 0 0] [0 0 0 0 0 0 67 114 72 114 163 227 254 225 254 254 254 250 229 254 254 140 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 17 66 14 67 67 67 59 21 236 254 106 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 83 253 209 18 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 22 233 255 83 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 129 254 238 44 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 59 249 254 62 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 133 254 187 5 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 205 248 58 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 126 254 182 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 75 251 240 57 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 19 221 254 166 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 3 203 254 219 35 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 38 254 254 77 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 31 224 254 115 1 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 133 254 254 52 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 61 242 254 254 52 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 121 254 254 219 40 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 121 254 207 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 116 125 171 255 255 150 93 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 169 253 253 253 253 253 253 218 30 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 169 253 253 253 213 142 176 253 253 122 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 52 250 253 210 32 12 0 6 206 253 140 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 77 251 210 25 0 0 0 122 248 253 65 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 31 18 0 0 0 0 209 253 253 65 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 117 247 253 198 10 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 76 247 253 231 63 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 128 253 253 144 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 176 246 253 159 12 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 25 234 253 233 35 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 198 253 253 141 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 78 248 253 189 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 19 200 253 253 141 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 134 253 253 173 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 248 253 253 25 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 248 253 253 43 20 20 20 20 5 0 5 20 20 37 150 150 150 147 10 0] [0 0 0 0 0 0 0 0 248 253 253 253 253 253 253 253 168 143 166 253 253 253 253 253 253 253 123 0] [0 0 0 0 0 0 0 0 174 253 253 253 253 253 253 253 253 253 253 253 249 247 247 169 117 117 57 0] [0 0 0 0 0 0 0 0 0 118 123 123 123 166 253 253 253 155 123 123 41 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 38 254 109 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 87 252 82 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 135 241 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 45 244 150 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 84 254 63 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 202 223 11 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 254 216 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 95 254 195 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 140 254 77 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 57 237 205 8 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 124 255 165 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 171 254 81 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 24 232 215 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 120 254 159 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 151 254 142 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 228 254 66 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 61 251 254 66 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 141 254 205 3 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 10 215 254 121 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 5 198 176 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 11 150 253 202 31 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 37 251 251 253 107 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 21 197 251 251 253 107 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 110 190 251 251 251 253 169 109 62 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 253 251 251 251 251 253 251 251 220 51 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 182 255 253 253 253 253 234 222 253 253 253 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 63 221 253 251 251 251 147 77 62 128 251 251 105 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 32 231 251 253 251 220 137 10 0 0 31 230 251 243 113 5 0 0 0 0 0] [0 0 0 0 0 0 0 37 251 251 253 188 20 0 0 0 0 0 109 251 253 251 35 0 0 0 0 0] [0 0 0 0 0 0 0 37 251 251 201 30 0 0 0 0 0 0 31 200 253 251 35 0 0 0 0 0] [0 0 0 0 0 0 0 37 253 253 0 0 0 0 0 0 0 0 32 202 255 253 164 0 0 0 0 0] [0 0 0 0 0 0 0 140 251 251 0 0 0 0 0 0 0 0 109 251 253 251 35 0 0 0 0 0] [0 0 0 0 0 0 0 217 251 251 0 0 0 0 0 0 21 63 231 251 253 230 30 0 0 0 0 0] [0 0 0 0 0 0 0 217 251 251 0 0 0 0 0 0 144 251 251 251 221 61 0 0 0 0 0 0] [0 0 0 0 0 0 0 217 251 251 0 0 0 0 0 182 221 251 251 251 180 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 218 253 253 73 73 228 253 253 255 253 253 253 253 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 113 251 251 253 251 251 251 251 253 251 251 251 147 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 31 230 251 253 251 251 251 251 253 230 189 35 10 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 62 142 253 251 251 251 251 253 107 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 72 174 251 173 71 72 30 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 50 224 0 0 0 0 0 0 0 70 29 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 121 231 0 0 0 0 0 0 0 148 168 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 4 195 231 0 0 0 0 0 0 0 96 210 11 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 69 252 134 0 0 0 0 0 0 0 114 252 21 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 45 236 217 12 0 0 0 0 0 0 0 192 252 21 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 168 247 53 0 0 0 0 0 0 0 18 255 253 21 0 0 0 0 0 0] [0 0 0 0 0 0 0 84 242 211 0 0 0 0 0 0 0 0 141 253 189 5 0 0 0 0 0 0] [0 0 0 0 0 0 0 169 252 106 0 0 0 0 0 0 0 32 232 250 66 0 0 0 0 0 0 0] [0 0 0 0 0 0 15 225 252 0 0 0 0 0 0 0 0 134 252 211 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 22 252 164 0 0 0 0 0 0 0 0 169 252 167 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 9 204 209 18 0 0 0 0 0 0 22 253 253 107 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 169 252 199 85 85 85 85 129 164 195 252 252 106 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 41 170 245 252 252 252 252 232 231 251 252 252 9 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 49 84 84 84 84 0 0 161 252 252 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 252 252 45 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 253 253 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 252 252 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 135 252 244 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 232 236 111 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 179 66 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]]

// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

// 0.011764706
// .01176470588
