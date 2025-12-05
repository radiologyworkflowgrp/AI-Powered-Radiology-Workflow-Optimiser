const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
    const RadiologyResult = sequelize.define('RadiologyResult', {
        id: {
            type: DataTypes.UUID,
            defaultValue: DataTypes.UUIDV4,
            primaryKey: true
        },
        patientId: {
            type: DataTypes.UUID,
            allowNull: false,
            references: {
                model: 'patients',
                key: 'id'
            }
        },
        patientName: {
            type: DataTypes.STRING,
            allowNull: false
        },
        testType: {
            type: DataTypes.STRING,
            allowNull: false
        },
        result: {
            type: DataTypes.TEXT
        },
        notes: {
            type: DataTypes.TEXT
        },
        imageUrl: {
            type: DataTypes.STRING
        },
        dicom_image_id: {
            type: DataTypes.UUID,
            references: {
                model: 'dicom_images',
                key: 'id'
            }
        }
    }, {
        tableName: 'radiology_results',
        indexes: [
            { fields: ['patientId'] }
        ]
    });

    return RadiologyResult;
};
