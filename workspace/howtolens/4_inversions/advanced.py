

# To perform a linear inversion, the source-pixels / image-pixels are set up as a matrix, called a 'mapping matrix'.
# Mapper's posses this matrix as an attribute.
print(mapper.mapping_matrix.shape)

# As you can see, the matrix is of size (image-pixels x source-pixels). Every element where there is a source-pixel /
# image-pixel mapping, the matrix has an entry of '1'. All other elements are '0'.
#
# Lets look at the seventh column of the mapping matrix, which corresponds to the seventh source-pixel, which if you
# look at the images above, you'll note contains 3 image-pixels.
print('First 100 image-pixels in 7th Column of Mapping Matrix')
print(mapper.mapping_matrix[100,6])

# As expected, we see mostly zeros, but three non-zero entries of '1'. These are the image-pixels in source-pixel 7,
# and if you count the indexes of these pixels you'll find they are image pixels 35, 36 and 37.
print('Non-zero Image-Pixel mappings to Source-Pixel 7')
print(mapper.mapping_matrix[35:40, 6])