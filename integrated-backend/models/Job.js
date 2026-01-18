const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
    const Job = sequelize.define('Job', {
        id: {
            type: DataTypes.UUID,
            defaultValue: DataTypes.UUIDV4,
            primaryKey: true
        },
        type: {
            type: DataTypes.STRING,
            allowNull: false
        },
        status: {
            type: DataTypes.STRING,
            defaultValue: 'pending'
        },
        data: {
            type: DataTypes.JSONB,
            defaultValue: {}
        },
        result: {
            type: DataTypes.JSONB,
            defaultValue: {}
        }
    }, {
        tableName: 'jobs',
        indexes: [
            { fields: ['type'] },
            { fields: ['status'] }
        ]
    });

    return Job;
};
