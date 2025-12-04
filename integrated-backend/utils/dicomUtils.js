const dicomParser = require('dicom-parser');
const fs = require('fs');

/**
 * DICOM Utilities
 * Helper functions for parsing and handling DICOM files
 */

/**
 * Parse DICOM file and extract metadata
 * @param {string} filePath - Path to DICOM file
 * @returns {object} Extracted DICOM metadata
 */
async function parseDicomFile(filePath) {
    try {
        // Read the DICOM file
        const dicomFileAsBuffer = fs.readFileSync(filePath);
        const byteArray = new Uint8Array(dicomFileAsBuffer);

        // Parse the DICOM file
        const dataSet = dicomParser.parseDicom(byteArray);

        // Extract common DICOM tags
        const metadata = {
            studyInstanceUID: getString(dataSet, 'x0020000d'),
            seriesInstanceUID: getString(dataSet, 'x0020000e'),
            sopInstanceUID: getString(dataSet, 'x00080018'),
            modality: getString(dataSet, 'x00080060'),
            studyDate: getString(dataSet, 'x00080020'),
            studyTime: getString(dataSet, 'x00080030'),
            institutionName: getString(dataSet, 'x00080080'),
            manufacturer: getString(dataSet, 'x00080070'),
            rows: getNumber(dataSet, 'x00280010'),
            columns: getNumber(dataSet, 'x00280011'),
            numberOfFrames: getNumber(dataSet, 'x00280008'),
            patientAge: getString(dataSet, 'x00101010'),
            patientSex: getString(dataSet, 'x00100040'),
            patientName: getString(dataSet, 'x00100010'),
            patientID: getString(dataSet, 'x00100020'),
            bodyPartExamined: getString(dataSet, 'x00180015'),
            studyDescription: getString(dataSet, 'x00081030'),
            seriesDescription: getString(dataSet, 'x0008103e')
        };

        return metadata;
    } catch (error) {
        console.error('Error parsing DICOM file:', error);
        throw new Error(`Failed to parse DICOM file: ${error.message}`);
    }
}

/**
 * Get string value from DICOM dataset
 * @param {object} dataSet - DICOM dataset
 * @param {string} tag - DICOM tag
 * @returns {string|null} String value or null
 */
function getString(dataSet, tag) {
    try {
        const element = dataSet.elements[tag];
        if (element) {
            return dataSet.string(tag);
        }
        return null;
    } catch (error) {
        return null;
    }
}

/**
 * Get number value from DICOM dataset
 * @param {object} dataSet - DICOM dataset
 * @param {string} tag - DICOM tag
 * @returns {number|null} Number value or null
 */
function getNumber(dataSet, tag) {
    try {
        const element = dataSet.elements[tag];
        if (element) {
            const value = dataSet.string(tag);
            return parseInt(value, 10);
        }
        return null;
    } catch (error) {
        return null;
    }
}

/**
 * Validate DICOM file
 * @param {Buffer} fileBuffer - File buffer
 * @returns {boolean} True if valid DICOM file
 */
function validateDicomFile(fileBuffer) {
    try {
        // Check for DICOM magic number at offset 128
        // DICOM files should have 'DICM' at bytes 128-131
        if (fileBuffer.length < 132) {
            return false;
        }

        const magicNumber = fileBuffer.toString('ascii', 128, 132);
        return magicNumber === 'DICM';
    } catch (error) {
        return false;
    }
}

/**
 * Generate unique scan ID
 * @returns {string} Unique scan ID
 */
function generateScanId() {
    const timestamp = Date.now();
    const random = Math.floor(Math.random() * 10000);
    return `SCAN-${timestamp}-${random}`;
}

/**
 * Get DICOM file extension
 * @param {string} filename - Original filename
 * @returns {string} File extension
 */
function getDicomExtension(filename) {
    const ext = filename.toLowerCase().split('.').pop();
    // Common DICOM extensions
    if (['dcm', 'dicom', 'dic'].includes(ext)) {
        return ext;
    }
    return 'dcm'; // Default to .dcm
}

/**
 * Sanitize filename for storage
 * @param {string} filename - Original filename
 * @returns {string} Sanitized filename
 */
function sanitizeFilename(filename) {
    // Remove any path traversal attempts
    const basename = filename.replace(/^.*[\\\/]/, '');
    // Remove special characters except dots, dashes, and underscores
    return basename.replace(/[^a-zA-Z0-9._-]/g, '_');
}

module.exports = {
    parseDicomFile,
    validateDicomFile,
    generateScanId,
    getDicomExtension,
    sanitizeFilename,
    getString,
    getNumber
};
