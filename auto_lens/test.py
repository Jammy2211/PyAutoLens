import image

slacs = image.Image.from_fits(filename='slacs_5_post.fits', hdu=1, pixel_scale=0.05)

mask = slacs.circle_mask(radius_arc=2.0)

print(mask)