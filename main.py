import registration_util as util


# Paths to images
fixed_path = 'image_data/3_2_t1.tif'
moving_path = 'image_data/3_2_t1_d.tif'

# Call cpselect
X_fixed, X_moving = util.cpselect(fixed_path, moving_path)
   
