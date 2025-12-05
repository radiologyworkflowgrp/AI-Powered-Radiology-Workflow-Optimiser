const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
    const ActivityLog = sequelize.define('ActivityLog', {
        id: {
            type: DataTypes.UUID,
            defaultValue: DataTypes.UUIDV4,
            primaryKey: true
        },
        action: {
            type: DataTypes.STRING,
            allowNull: false
        },
        description: {
            type: DataTypes.TEXT,
            allowNull: false
        },
        entityType: {
            type: DataTypes.STRING
        },
        entityId: {
            type: DataTypes.STRING
        },
        metadata: {
            type: DataTypes.JSONB,
            defaultValue: {}
        }
    }, {
        tableName: 'activity_logs',
        indexes: [
            { fields: ['action'] },
            { fields: ['entityType'] },
            { fields: ['created_at'] }
        ]
    });

    return ActivityLog;
};
