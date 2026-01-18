const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
    const LoginActivity = sequelize.define('LoginActivity', {
        id: {
            type: DataTypes.UUID,
            defaultValue: DataTypes.UUIDV4,
            primaryKey: true
        },
        role: {
            type: DataTypes.STRING,
            allowNull: false
        },
        email: {
            type: DataTypes.STRING,
            allowNull: false
        }
    }, {
        tableName: 'login_activities',
        indexes: [
            { fields: ['role'] },
            { fields: ['created_at'] }
        ],
        timestamps: true,
        underscored: true,
        createdAt: 'created_at',
        updatedAt: 'updated_at'
    });

    return LoginActivity;
};
