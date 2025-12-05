const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
    const Prescription = sequelize.define('Prescription', {
        id: {
            type: DataTypes.UUID,
            defaultValue: DataTypes.UUIDV4,
            primaryKey: true
        },
        patientName: {
            type: DataTypes.STRING,
            allowNull: false
        },
        medication: {
            type: DataTypes.STRING,
            allowNull: false
        },
        dosage: {
            type: DataTypes.STRING,
            allowNull: false
        },
        frequency: {
            type: DataTypes.STRING,
            allowNull: false
        },
        duration: {
            type: DataTypes.STRING
        },
        status: {
            type: DataTypes.STRING,
            defaultValue: 'active'
        },
        refillCount: {
            type: DataTypes.INTEGER,
            defaultValue: 0
        }
    }, {
        tableName: 'prescriptions'
    });

    return Prescription;
};
