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
            type: DataTypes.STRING,
            field: 'entity_type'
        },
        entityId: {
            type: DataTypes.STRING,
            field: 'entity_id'
        },
        metadata: {
            type: DataTypes.JSONB,
            defaultValue: {}
        }
    }, {
        tableName: 'activity_logs',
        indexes: [
            { fields: ['action'] },
            { fields: ['entity_type'] },
            { fields: ['created_at'] }
        ],
        timestamps: true,
        underscored: true,
        createdAt: 'created_at',
        updatedAt: 'updated_at'
    });

    return ActivityLog;
};
