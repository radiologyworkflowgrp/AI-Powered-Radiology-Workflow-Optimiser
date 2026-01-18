"""
DICOM to Image Converter
Converts DICOM files to PNG images for DuoFormer processing
"""

import pydicom
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

def convert_dicom_to_image(dicom_path_or_bytes):
    """
    Convert DICOM file to PNG image
    
    Args:
        dicom_path_or_bytes: Path to DICOM file or bytes
        
    Returns:
        tuple: (image_bytes, metadata_dict)
    """
    try:
        # Load DICOM
        if isinstance(dicom_path_or_bytes, (str, bytes)):
            if isinstance(dicom_path_or_bytes, str):
                ds = pydicom.dcmread(dicom_path_or_bytes)
            else:
                ds = pydicom.dcmread(io.BytesIO(dicom_path_or_bytes))
        else:
            raise ValueError("Input must be file path or bytes")
        
        # Extract pixel array
        pixel_array = ds.pixel_array
        
        # Normalize to 0-255 range
        pixel_array = pixel_array.astype(float)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        pixel_array = (pixel_array * 255).astype(np.uint8)
        
        # Apply windowing if available
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, list) else float(ds.WindowCenter[0])
            width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, list) else float(ds.WindowWidth[0])
            
            # Apply window
            lower = center - width / 2
            upper = center + width / 2
            pixel_array = np.clip(pixel_array, lower, upper)
            pixel_array = ((pixel_array - lower) / (upper - lower) * 255).astype(np.uint8)
        
        # Convert to RGB if grayscale
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=-1)
        
        # Create PIL Image
        image = Image.fromarray(pixel_array)
        
        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Extract metadata
        metadata = {
            'patient_name': str(ds.PatientName) if hasattr(ds, 'PatientName') else 'Unknown',
            'patient_id': str(ds.PatientID) if hasattr(ds, 'PatientID') else 'Unknown',
            'study_date': str(ds.StudyDate) if hasattr(ds, 'StudyDate') else 'Unknown',
            'modality': str(ds.Modality) if hasattr(ds, 'Modality') else 'Unknown',
            'study_description': str(ds.StudyDescription) if hasattr(ds, 'StudyDescription') else '',
            'series_description': str(ds.SeriesDescription) if hasattr(ds, 'SeriesDescription') else '',
            'image_size': f"{pixel_array.shape[1]}x{pixel_array.shape[0]}"
        }
        
        logger.info(f"Successfully converted DICOM: {metadata['patient_id']}")
        
        return img_bytes, metadata
        
    except Exception as e:
        logger.error(f"DICOM conversion failed: {e}", exc_info=True)
        raise ValueError(f"Failed to convert DICOM file: {str(e)}")


def validate_dicom(file_bytes):
    """
    Validate if file is a valid DICOM
    
    Args:
        file_bytes: File bytes to validate
        
    Returns:
        bool: True if valid DICOM
    """
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        return hasattr(ds, 'pixel_array')
    except Exception:
        return False
