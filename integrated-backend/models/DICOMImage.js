const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
    const DICOMImage = sequelize.define('DICOMImage', {
        id: {
            type: DataTypes.UUID,
            defaultValue: DataTypes.UUIDV4,
            primaryKey: true
        },
        patient_id: {
            type: DataTypes.STRING,
            allowNull: false
            // References removed for MongoDB compatibility
        },
        scan_id: {
            type: DataTypes.STRING,
            allowNull: false,
            unique: true
        },
        modality: {
            type: DataTypes.STRING(50),
            allowNull: false,
            comment: 'CT, MRI, X-RAY, etc.'
        },
        study_date: {
            type: DataTypes.DATE
        },
        series_description: {
            type: DataTypes.STRING
        },
        image_data: {
            type: DataTypes.BLOB('long'),
            allowNull: false,
            comment: 'Binary DICOM file data'
        },
        metadata: {
            type: DataTypes.JSONB,
            defaultValue: {},
            comment: 'DICOM tags and metadata'
        },
        file_size: {
            type: DataTypes.INTEGER,
            comment: 'Size in bytes'
        },
        thumbnail: {
            type: DataTypes.BLOB,
            comment: 'Small preview image'
        }
    }, {
        tableName: 'dicom_images',
        indexes: [
            { fields: ['patient_id'] },
            { fields: ['scan_id'] },
            { fields: ['modality'] },
            { fields: ['study_date'] }
        ]
    });

    return DICOMImage;
};
