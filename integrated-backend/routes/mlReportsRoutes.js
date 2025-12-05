const express = require('express');
const router = express.Router();
const { MLReport } = require('../models');
const { body, param, query, validationResult } = require('express-validator');
const { Op } = require('sequelize');

// Middleware to handle validation errors
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      message: 'Validation failed',
      errors: errors.array()
    });
  }
  next();
};

// POST /api/ml-reports - Create new ML report
router.post('/',
  [
    body('patient_id').notEmpty().withMessage('Patient ID is required'),
    body('patient_name').notEmpty().withMessage('Patient name is required'),
    body('report_type').notEmpty().withMessage('Report type is required'),
    body('ml_model').notEmpty().withMessage('ML model name is required'),
    body('confidence_score').optional().isFloat({ min: 0, max: 1 }).withMessage('Confidence score must be between 0 and 1'),
    body('findings').optional().isString().withMessage('Findings must be text'),
    body('impression').optional().isString().withMessage('Impression must be text'),
    body('recommendation').optional().isString().withMessage('Recommendation must be text'),
    body('image_url').optional().isString().withMessage('Image URL must be string'),
    body('doctor_id').optional().isString().withMessage('Doctor ID must be string')
  ],
  handleValidationErrors,
  async (req, res) => {
    try {
      const report = await MLReport.create(req.body);

      res.status(201).json({
        message: 'ML report created successfully',
        report: report.toJSON()
      });
    } catch (error) {
      console.error('Error creating ML report:', error);
      res.status(500).json({
        message: 'Failed to create ML report',
        error: error.message
      });
    }
  }
);

// GET /api/ml-reports - Get all ML reports with pagination
router.get('/',
  [
    query('limit').optional().isInt({ min: 1, max: 100 }).withMessage('Limit must be between 1 and 100'),
    query('offset').optional().isInt({ min: 0 }).withMessage('Offset must be non-negative'),
    query('status').optional().isIn(['pending', 'processing', 'completed', 'failed']).withMessage('Invalid status')
  ],
  handleValidationErrors,
  async (req, res) => {
    try {
      const { limit = 50, offset = 0, status } = req.query;

      const where = status ? { report_status: status } : {};

      const reports = await MLReport.findAll({
        where,
        limit: parseInt(limit),
        offset: parseInt(offset),
        order: [['created_at', 'DESC']]
      });

      res.json({
        message: 'ML reports retrieved successfully',
        reports: reports.map(r => r.toJSON()),
        total: reports.length
      });
    } catch (error) {
      console.error('Error fetching ML reports:', error);
      res.status(500).json({
        message: 'Failed to fetch ML reports',
        error: error.message
      });
    }
  }
);

// GET /api/ml-reports/:id - Get ML report by ID
router.get('/:id',
  [
    param('id').isInt().withMessage('Report ID must be an integer')
  ],
  handleValidationErrors,
  async (req, res) => {
    try {
      const report = await MLReport.findByPk(req.params.id);

      if (!report) {
        return res.status(404).json({
          message: 'ML report not found'
        });
      }

      res.json({
        message: 'ML report retrieved successfully',
        report: report.toJSON()
      });
    } catch (error) {
      console.error('Error fetching ML report:', error);
      res.status(500).json({
        message: 'Failed to fetch ML report',
        error: error.message
      });
    }
  }
);

// GET /api/ml-reports/patient/:patientId - Get ML reports by patient ID
router.get('/patient/:patientId',
  [
    param('patientId').notEmpty().withMessage('Patient ID is required')
  ],
  handleValidationErrors,
  async (req, res) => {
    try {
      const reports = await MLReport.findAll({
        where: { patient_id: req.params.patientId },
        order: [['created_at', 'DESC']]
      });

      res.json({
        message: 'Patient ML reports retrieved successfully',
        reports: reports.map(r => r.toJSON()),
        total: reports.length
      });
    } catch (error) {
      console.error('Error fetching patient ML reports:', error);
      res.status(500).json({
        message: 'Failed to fetch patient ML reports',
        error: error.message
      });
    }
  }
);

// PUT /api/ml-reports/:id - Update ML report
router.put('/:id',
  [
    param('id').isInt().withMessage('Report ID must be an integer'),
    body('findings').optional().isString().withMessage('Findings must be text'),
    body('impression').optional().isString().withMessage('Impression must be text'),
    body('recommendation').optional().isString().withMessage('Recommendation must be text'),
    body('confidence_score').optional().isFloat({ min: 0, max: 1 }).withMessage('Confidence score must be between 0 and 1'),
    body('image_url').optional().isString().withMessage('Image URL must be string')
  ],
  handleValidationErrors,
  async (req, res) => {
    try {
      const report = await MLReport.findByPk(req.params.id);

      if (!report) {
        return res.status(404).json({
          message: 'ML report not found'
        });
      }

      await report.update(req.body);

      res.json({
        message: 'ML report updated successfully',
        report: report.toJSON()
      });
    } catch (error) {
      console.error('Error updating ML report:', error);
      res.status(500).json({
        message: 'Failed to update ML report',
        error: error.message
      });
    }
  }
);

// PUT /api/ml-reports/:id/status - Update report status
router.put('/:id/status',
  [
    param('id').isInt().withMessage('Report ID must be an integer'),
    body('status').isIn(['pending', 'processing', 'completed', 'failed']).withMessage('Invalid status'),
    body('reviewed_by').optional().isString().withMessage('Reviewed by must be string')
  ],
  handleValidationErrors,
  async (req, res) => {
    try {
      const { status, reviewed_by } = req.body;

      const report = await MLReport.findByPk(req.params.id);

      if (!report) {
        return res.status(404).json({
          message: 'ML report not found'
        });
      }

      await report.update({
        report_status: status,
        reviewed_by: reviewed_by || report.reviewed_by
      });

      res.json({
        message: 'ML report status updated successfully',
        report: report.toJSON()
      });
    } catch (error) {
      console.error('Error updating ML report status:', error);
      res.status(500).json({
        message: 'Failed to update ML report status',
        error: error.message
      });
    }
  }
);

// DELETE /api/ml-reports/:id - Delete ML report
router.delete('/:id',
  [
    param('id').isInt().withMessage('Report ID must be an integer')
  ],
  handleValidationErrors,
  async (req, res) => {
    try {
      const report = await MLReport.findByPk(req.params.id);

      if (!report) {
        return res.status(404).json({
          message: 'ML report not found'
        });
      }

      await report.destroy();

      res.json({
        message: 'ML report deleted successfully'
      });
    } catch (error) {
      console.error('Error deleting ML report:', error);
      res.status(500).json({
        message: 'Failed to delete ML report',
        error: error.message
      });
    }
  }
);

// GET /api/ml-reports/stats - Get ML report statistics
router.get('/stats', async (req, res) => {
  try {
    const [total, pending, processing, completed, failed] = await Promise.all([
      MLReport.count(),
      MLReport.count({ where: { report_status: 'pending' } }),
      MLReport.count({ where: { report_status: 'processing' } }),
      MLReport.count({ where: { report_status: 'completed' } }),
      MLReport.count({ where: { report_status: 'failed' } })
    ]);

    const stats = {
      total,
      by_status: {
        pending,
        processing,
        completed,
        failed
      }
    };

    res.json({
      message: 'ML report statistics retrieved successfully',
      stats
    });
  } catch (error) {
    console.error('Error fetching ML report statistics:', error);
    res.status(500).json({
      message: 'Failed to fetch ML report statistics',
      error: error.message
    });
  }
});

module.exports = router;
