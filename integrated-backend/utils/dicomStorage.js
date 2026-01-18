const dicomParser = require('dicom-parser');

/**
 * Extract metadata from DICOM file buffer
 * @param {Buffer} dicomBuffer - Binary DICOM file data
 * @returns {Object} - Extracted metadata
 */
function extractDICOMMetadata(dicomBuffer) {
    try {
        const dataSet = dicomParser.parseDicom(dicomBuffer);

        const metadata = {
            patientName: getString(dataSet, 'x00100010'),
            patientID: getString(dataSet, 'x00100020'),
            studyDate: getString(dataSet, 'x00080020'),
            studyTime: getString(dataSet, 'x00080030'),
            modality: getString(dataSet, 'x00080060'),
            studyDescription: getString(dataSet, 'x00081030'),
            seriesDescription: getString(dataSet, 'x0008103e'),
            institutionName: getString(dataSet, 'x00080080'),
            manufacturer: getString(dataSet, 'x00080070'),
            rows: getNumber(dataSet, 'x00280010'),
            columns: getNumber(dataSet, 'x00280011'),
            bitsAllocated: getNumber(dataSet, 'x00280100'),
            bitsStored: getNumber(dataSet, 'x00280101'),
            pixelSpacing: getString(dataSet, 'x00280030'),
            sliceThickness: getString(dataSet, 'x00180050'),
            kvp: getString(dataSet, 'x00180060'),
            exposureTime: getString(dataSet, 'x00181150'),
            xRayTubeCurrent: getString(dataSet, 'x00181151')
        };

        // Remove undefined values
        Object.keys(metadata).forEach(key => {
            if (metadata[key] === undefined || metadata[key] === null) {
                delete metadata[key];
            }
        });

        return metadata;
    } catch (error) {
        console.error('Error parsing DICOM metadata:', error);
        return {};
    }
}

/**
 * Helper function to get string from DICOM dataset
 */
function getString(dataSet, tag) {
    try {
        const element = dataSet.elements[tag];
        if (element) {
            return dataSet.string(tag);
        }
    } catch (error) {
        return undefined;
    }
    return undefined;
}

/**
 * Helper function to get number from DICOM dataset
 */
function getNumber(dataSet, tag) {
    try {
        const element = dataSet.elements[tag];
        if (element) {
            return dataSet.uint16(tag);
        }
    } catch (error) {
        return undefined;
    }
    return undefined;
}

/**
 * Generate thumbnail from DICOM image
 * Note: This is a simplified version. For production, you may want to use
 * specialized DICOM rendering libraries like cornerstone.js
 * @param {Buffer} dicomBuffer - Binary DICOM file data
 * @param {number} maxSize - Maximum dimension for thumbnail (default: 200)
 * @returns {Promise<Buffer>} - Thumbnail image buffer
 */
async function generateThumbnail(dicomBuffer, maxSize = 200) {
    try {
        // For now, return null as DICOM to image conversion requires more complex processing
        // In production, you would use libraries like cornerstone or dcmjs to render DICOM to image
        // then use sharp to create thumbnail
        return null;
    } catch (error) {
        console.error('Error generating thumbnail:', error);
        return null;
    }
}

/**
 * Validate DICOM file
 * @param {Buffer} buffer - File buffer
 * @returns {boolean} - True if valid DICOM file
 */
function isValidDICOM(buffer) {
    try {
        // DICOM files should have 'DICM' at bytes 128-131
        if (buffer.length < 132) {
            return false;
        }

        const dicmString = buffer.toString('ascii', 128, 132);
        return dicmString === 'DICM';
    } catch (error) {
        return false;
    }
}

/**
 * Get file size in a human-readable format
 * @param {number} bytes - Size in bytes
 * @returns {string} - Formatted size string
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

module.exports = {
    extractDICOMMetadata,
    generateThumbnail,
    isValidDICOM,
    formatFileSize
};
