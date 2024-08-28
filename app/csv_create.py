import os
import pandas as pd
def create_image_mask_csv(image_root_dir, mask_root_dir, output_csv):
    """
    Create a CSV file that maps each image to its corresponding mask, considering the directory structure.

    :param image_root_dir: Root directory containing the images, organized by city.
    :param mask_root_dir: Root directory containing the masks, organized by city.
    :param output_csv: Path to the output CSV file.
    """
    
    data = []

    # Walk through each city directory in the image root directory
    for city_name in os.listdir(image_root_dir):
        city_image_dir = os.path.join(image_root_dir, city_name)
        city_mask_dir = os.path.join(mask_root_dir, city_name)

        if not os.path.isdir(city_image_dir):
            continue
        
        # Ensure the city has a corresponding mask directory
        if not os.path.exists(city_mask_dir):
            print(f"Warning: No mask directory found for city {city_name}. Skipping.")
            continue

        # Loop through each image in the city's image directory
        for image_filename in os.listdir(city_image_dir):
            if not image_filename.endswith('leftImg8bit.png'):
                continue

            image_path = os.path.join(city_image_dir, image_filename)
            image_prefix = image_filename.replace('leftImg8bit.png', '')

            # Construct the corresponding mask filename
            mask_filename = image_prefix + 'gtFine_labelTrainIds.png'
            mask_path = os.path.join(city_mask_dir, mask_filename)

            # Check if the mask file exists
            if not os.path.exists(mask_path):
                print(f"Warning: No matching mask found for image {image_filename} in city {city_name}.")
                continue

            # Append the image and mask paths to the data list
            data.append({'image': image_path, 'mask': mask_path})

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


