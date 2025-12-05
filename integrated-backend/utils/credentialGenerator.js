const crypto = require('crypto');

/**
 * Generate random secure password
 * @param {number} length - Length of password (default 12)
 * @returns {string} Random password
 */
function generatePassword(length = 12) {
    const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
    let password = '';
    const randomBytes = crypto.randomBytes(length);

    for (let i = 0; i < length; i++) {
        password += charset[randomBytes[i] % charset.length];
    }

    return password;
}

/**
 * Generate email from patient name
 * @param {string} name - Patient name
 * @param {string} id - Patient ID
 * @returns {string} Generated email
 */
function generateEmail(name, id) {
    // Remove spaces and special characters, convert to lowercase
    const cleanName = name.toLowerCase().replace(/[^a-z0-9]/g, '');

    // Use first 8 chars of name + last 4 chars of ID
    const namePart = cleanName.substring(0, 8);
    const idPart = id.substring(id.length - 4);

    return `${namePart}${idPart}@patient.hospital.com`;
}

/**
 * Generate credentials for a new patient
 * @param {string} name - Patient name
 * @param {string} id - Patient ID
 * @returns {object} Credentials object with email and password
 */
function generateCredentials(name, id) {
    const email = generateEmail(name, id);
    const password = generatePassword(12);

    return {
        email,
        password,
        message: 'Please save these credentials securely. The password cannot be recovered.'
    };
}

module.exports = {
    generatePassword,
    generateEmail,
    generateCredentials
};
