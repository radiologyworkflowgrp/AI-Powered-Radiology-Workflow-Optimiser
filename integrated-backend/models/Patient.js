const { DataTypes } = require('sequelize');
const bcrypt = require('bcryptjs');

module.exports = (sequelize) => {
    const Patient = sequelize.define('Patient', {
        id: {
            type: DataTypes.UUID,
            defaultValue: DataTypes.UUIDV4,
            primaryKey: true
        },
        name: {
            type: DataTypes.STRING,
            allowNull: false
        },
        email: {
            type: DataTypes.STRING,
            allowNull: true,
            unique: false,
            validate: {
                isEmail: true
            }
        },
        // Password removed - stored in MongoDB User collection for auth
        age: {
            type: DataTypes.INTEGER,
            validate: {
                min: 0,
                max: 150
            }
        },
        gender: {
            type: DataTypes.STRING
        },
        contact: {
            type: DataTypes.STRING
        },
        address: {
            type: DataTypes.TEXT
        },
        medical_history: {
            type: DataTypes.TEXT
        },
        symptoms: {
            type: DataTypes.ARRAY(DataTypes.STRING),
            defaultValue: []
        },
        vitals: {
            type: DataTypes.JSONB,
            defaultValue: {}
        },
        priority: {
            type: DataTypes.STRING,
            defaultValue: 'normal'
        },
        assignedDoctor: {
            type: DataTypes.JSONB,
            defaultValue: null
        },
        profileCompleted: {
            type: DataTypes.BOOLEAN,
            defaultValue: false
        }
    }, {
        tableName: 'patients',
        indexes: [
            { fields: ['email'] },
            { fields: ['priority'] }
        ],
        hooks: {
            // No hooks needed for password hashing here anymore
        }
    });

    // Password comparison is now handled by User model in MongoDB

    return Patient;
};
