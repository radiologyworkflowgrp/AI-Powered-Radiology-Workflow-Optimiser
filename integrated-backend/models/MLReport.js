const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
    const MLReport = sequelize.define('MLReport', {
        id: {
            type: DataTypes.INTEGER,
            autoIncrement: true,
            primaryKey: true
        },
        patient_id: {
            type: DataTypes.STRING,
            allowNull: false
            // References removed for MongoDB compatibility
        },
        patient_name: {
            type: DataTypes.STRING,
            allowNull: false
        },
        report_type: {
            type: DataTypes.STRING(100),
            allowNull: false
        },
        ml_model: {
            type: DataTypes.STRING,
            allowNull: false
        },
        confidence_score: {
            type: DataTypes.FLOAT,
            validate: {
                min: 0,
                max: 1
            }
        },
        findings: {
            type: DataTypes.TEXT
        },
        impression: {
            type: DataTypes.TEXT
        },
        recommendation: {
            type: DataTypes.TEXT
        },
        image_url: {
            type: DataTypes.STRING(500)
        },
        report_status: {
            type: DataTypes.ENUM('pending', 'processing', 'completed', 'failed'),
            defaultValue: 'pending'
        },
        report_data: {
            type: DataTypes.JSONB
        },
        pdf_path: {
            type: DataTypes.STRING(500)
        },
        doctor_id: {
            type: DataTypes.STRING
            // References removed for MongoDB compatibility
        },
        reviewed_by: {
            type: DataTypes.STRING
        }
    }, {
        tableName: 'ml_reports',
        indexes: [
            { fields: ['patient_id'] },
            { fields: ['report_status'] },
            { fields: ['created_at'] }
        ]
    });

    return MLReport;
};
