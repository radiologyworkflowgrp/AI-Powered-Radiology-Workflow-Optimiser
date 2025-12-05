const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
    const LoginActivity = sequelize.define('LoginActivity', {
        id: {
            type: DataTypes.UUID,
            defaultValue: DataTypes.UUIDV4,
            primaryKey: true
        },
        userId: {
            type: DataTypes.UUID,
            allowNull: false
        },
        role: {
            type: DataTypes.STRING,
            allowNull: false
        },
        email: {
            type: DataTypes.STRING,
            allowNull: false
        },
        ipAddress: {
            type: DataTypes.STRING
        },
        userAgent: {
            type: DataTypes.TEXT
        }
    }, {
        tableName: 'login_activities',
        indexes: [
            { fields: ['userId'] },
            { fields: ['role'] },
            { fields: ['created_at'] }
        ]
    });

    return LoginActivity;
};
