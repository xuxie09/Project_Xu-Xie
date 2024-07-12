import os
import glob
import SimpleITK as sitk

def load_image(file_path):
    """
    :param file_path: file path of image
    :return: ITK data
    """
    _, file_extension = os.path.splitext(file_path)
    
    #  NIfTI 
    if file_extension in ['.nii', '.nii.gz']:
        image = sitk.ReadImage(file_path)
    
    #  DICOM 
    elif file_extension.lower() in ['.dcm', '.dicom']:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(file_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    
    # Others（e.g .hdr and .img pairs）
    elif file_extension.lower() in ['.hdr', '.img']:
        image = sitk.ReadImage(file_path)
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return image

def convert_to_nifti(image, output_file_path):
    """
    transform images format to NIfTI format
    :param image: ITK images
    :param output_file_path: the filepath of output NIFT images 
    """
    sitk.WriteImage(image, output_file_path)
    

def process_directory(directory_path, output_directory):
    """
    transform all images in a folder, and transfer them to NIFT format
    :param directory_path: directory path includes all images
    :param output_directory: output directory path of transformed NIFT format images
    """
    # ensure out directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # find all files in directory path
    all_files = glob.glob(os.path.join(directory_path, '*'))
    
    for file_path in all_files:
        try:
            # load image
            image = load_image(file_path)
            
            _, file_extension = os.path.splitext(file_path)
            
            # transfer images to NIFT format
            if file_extension.lower() not in ['.nii', '.nii.gz']:
                base_name = os.path.basename(os.path.splitext(file_path)[0])
                output_file_path = os.path.join(output_directory, base_name + ".nii")
                convert_to_nifti(image, output_file_path)

        
        except ValueError as e:
            print(e)
            continue


current = os.getcwd()
GT_path = (current+"/"+"dataset"+"/"+"GT"+"/")    
process_directory(GT_path, GT_path)

Org_path = (current+"/"+"dataset"+"/"+"Org"+"/")   
process_directory(Org_path, Org_path)
