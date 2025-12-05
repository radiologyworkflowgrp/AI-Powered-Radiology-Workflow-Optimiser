const { DataTypes } = require('sequelize');
const bcrypt = require('bcryptjs');

module.exports = (sequelize) => {
    const Doctor = sequelize.define('Doctor', {
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
            allowNull: false,
            unique: true,
            validate: {
                isEmail: true
            }
        },
        password: {
            type: DataTypes.STRING,
            allowNull: false
        },
        specialty: {
            type: DataTypes.STRING,
            defaultValue: 'General Medicine'
        },
        availability: {
            type: DataTypes.STRING,
            defaultValue: 'Available'
        },
        profileCompleted: {
            type: DataTypes.BOOLEAN,
            defaultValue: false
        }
    }, {
        tableName: 'doctors',
        indexes: [
            { fields: ['email'] },
            { fields: ['availability'] },
            { fields: ['specialty'] }
        ],
        hooks: {
            beforeCreate: async (doctor) => {
                if (doctor.password) {
                    const salt = await bcrypt.genSalt(10);
                    doctor.password = await bcrypt.hash(doctor.password, salt);
                }
            },
            beforeUpdate: async (doctor) => {
                if (doctor.changed('password')) {
                    const salt = await bcrypt.genSalt(10);
                    doctor.password = await bcrypt.hash(doctor.password, salt);
                }
            }
        }
    });

    // Instance method to compare password
    Doctor.prototype.comparePassword = async function (candidatePassword) {
        return await bcrypt.compare(candidatePassword, this.password);
    };

    return Doctor;
};
