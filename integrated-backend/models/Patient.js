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
            type: DataTypes.TEXT,
            allowNull: true
        },
        symptoms: {
            type: DataTypes.ARRAY(DataTypes.STRING),
            defaultValue: []
        },
        blood_type: {
            type: DataTypes.STRING(10),
            allowNull: true
        },
        height: {
            type: DataTypes.INTEGER,
            allowNull: true
        },
        weight: {
            type: DataTypes.INTEGER,
            allowNull: true
        },
        allergies: {
            type: DataTypes.TEXT,
            allowNull: true
        },
        vitals: {
            type: DataTypes.JSONB,
            defaultValue: {}
        },
        priority: {
            type: DataTypes.INTEGER,
            defaultValue: 2, // Medium priority
            validate: {
                min: 1,
                max: 3,
                isInt: true
            },
            comment: '1=High, 2=Medium, 3=Low'
        },
        assignedDoctor: {
            type: DataTypes.JSONB,
            defaultValue: null
        },
        status: {
            type: DataTypes.STRING,
            defaultValue: 'Admitted',
            validate: {
                isIn: [['Admitted', 'Doctor Appointment', 'Completed']]
            }
        },
        checkedByDoctor: {
            type: DataTypes.BOOLEAN,
            defaultValue: false
        },
        profileCompleted: {
            type: DataTypes.BOOLEAN,
            defaultValue: false
        }
    }, {
        tableName: 'patients',
        timestamps: true,
        underscored: true, // This maps createdAt -> created_at
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
