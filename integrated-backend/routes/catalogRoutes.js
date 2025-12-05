const express = require('express');
const router = express.Router();
const Job = require('../mongoSchemas/Job');
const redisClient = require('../redisClient');
const rabbitmq = require('../rabbitmq');
const metrics = require('../metrics');
const logger = require('../logger');
const { v4: uuidv4 } = require('uuid');
const mongoose = require('mongoose');

// Legacy metrics storage (kept for backward compatibility)
let legacyMetrics = {
  enqueued: 0,
  failed: 0,
  requeued: 0,
  completed: 0
};

// POST /enqueue - Receive a single task and queue it
router.post('/enqueue', async (req, res) => {
  try {
    const { payload, metadata = {} } = req.body;

    if (!payload) {
      return res.status(400).json({
        success: false,
        message: 'Payload is required'
      });
    }

    // Check for idempotency key
    if (metadata.idempotency_key) {
      const existingJob = await Job.findOne({ 'metadata.idempotency_key': metadata.idempotency_key });
      if (existingJob) {
        return res.status(200).json({
          success: true,
          message: 'Job already exists',
          job_id: existingJob.job_id,
          status: existingJob.status
        });
      }
    }

    const job = new Job({
      payload,
      metadata: {
        priority: metadata.priority || 0,
        idempotency_key: metadata.idempotency_key,
        routing_key: metadata.routing_key,
        origin: metadata.origin || 'api'
      }
    });

    await job.save();

    // Update metrics
    legacyMetrics.enqueued++;

    // Store in Redis for immediate processing
    await redisClient.lPush('job_queue', JSON.stringify({
      job_id: job.job_id,
      payload: job.payload,
      metadata: job.metadata
    }));

    res.status(201).json({
      success: true,
      job_id: job.job_id,
      status: 'queued'
    });
  } catch (error) {
    console.error('Error enqueuing job:', error);
    legacyMetrics.failed++;
    res.status(503).json({
      success: false,
      message: 'Failed to enqueue job',
      error: error.message
    });
  }
});

// POST /enqueue/batch - Send multiple tasks in one request
router.post('/enqueue/batch', async (req, res) => {
  try {
    const { jobs } = req.body;

    if (!Array.isArray(jobs) || jobs.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Jobs array is required'
      });
    }

    const results = [];
    const errors = [];

    for (const jobData of jobs) {
      try {
        const { payload, metadata = {} } = jobData;

        if (!payload) {
          errors.push({
            index: results.length,
            error: 'Payload is required'
          });
          continue;
        }

        // Check for idempotency key
        if (metadata.idempotency_key) {
          const existingJob = await Job.findOne({ 'metadata.idempotency_key': metadata.idempotency_key });
          if (existingJob) {
            results.push({
              success: true,
              job_id: existingJob.job_id,
              status: existingJob.status,
              message: 'Job already exists'
            });
            continue;
          }
        }

        const job = new Job({
          payload,
          metadata: {
            priority: metadata.priority || 0,
            idempotency_key: metadata.idempotency_key,
            routing_key: metadata.routing_key,
            origin: metadata.origin || 'api'
          }
        });

        await job.save();
        legacyMetrics.enqueued++;

        // Store in Redis for immediate processing
        await redisClient.lPush('job_queue', JSON.stringify({
          job_id: job.job_id,
          payload: job.payload,
          metadata: job.metadata
        }));

        results.push({
          success: true,
          job_id: job.job_id,
          status: 'queued'
        });
      } catch (error) {
        legacyMetrics.failed++;
        errors.push({
          index: results.length,
          error: error.message
        });
      }
    }

    res.status(207).json({
      success: true,
      results,
      errors,
      total: jobs.length,
      successful: results.length,
      failed: errors.length
    });
  } catch (error) {
    console.error('Error in batch enqueue:', error);
    res.status(503).json({
      success: false,
      message: 'Failed to process batch enqueue',
      error: error.message
    });
  }
});

// GET /jobs/{job_id} - Retrieve status/metadata of a specific job
router.get('/jobs/:job_id', async (req, res) => {
  try {
    const { job_id } = req.params;

    const job = await Job.findOne({ job_id });

    if (!job) {
      return res.status(404).json({
        success: false,
        message: 'Job not found'
      });
    }

    res.json({
      success: true,
      job: {
        job_id: job.job_id,
        status: job.status,
        payload: job.payload,
        metadata: job.metadata,
        enqueued_at: job.enqueued_at,
        processed_at: job.processed_at,
        completed_at: job.completed_at,
        error: job.error,
        retry_count: job.retry_count
      }
    });
  } catch (error) {
    console.error('Error fetching job:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch job',
      error: error.message
    });
  }
});

// GET /health - App health check
router.get('/health', async (req, res) => {
  try {
    const checks = {
      mongodb: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
      redis: redisClient.isOpen ? 'connected' : 'disconnected',
      rabbitmq: rabbitmq.isConnected ? 'connected' : 'disconnected',
      queue_length: 0,
      rabbitmq_queues: {}
    };

    // Get queue length from Redis
    try {
      checks.queue_length = await redisClient.lLen('job_queue');
      metrics.setRedisQueueDepth(checks.queue_length);
    } catch (error) {
      checks.redis = 'error';
    }

    // Get RabbitMQ queue stats
    try {
      if (rabbitmq.isConnected) {
        const queueStats = await rabbitmq.getAllQueueStats();
        checks.rabbitmq_queues = queueStats;

        // Update metrics for each queue
        for (const [queueName, stats] of Object.entries(queueStats)) {
          metrics.setRabbitMQQueueDepth(queueName, stats.messageCount);
        }
      }
    } catch (error) {
      logger.warn('Failed to get RabbitMQ queue stats:', error.message);
      checks.rabbitmq = 'error';
    }

    const allHealthy = checks.mongodb === 'connected' && checks.redis === 'connected';

    if (allHealthy) {
      res.json({
        status: 'healthy',
        checks,
        timestamp: new Date().toISOString()
      });
    } else {
      res.status(503).json({
        status: 'unhealthy',
        checks,
        timestamp: new Date().toISOString()
      });
    }
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// GET /metrics - Enhanced Prometheus metrics endpoint
router.get('/metrics', async (req, res) => {
  try {
    // Update database connection metrics
    metrics.setDatabaseConnection('mongodb', mongoose.connection.readyState === 1);
    metrics.setDatabaseConnection('redis', redisClient.isOpen);
    metrics.setDatabaseConnection('rabbitmq', rabbitmq.isConnected);

    // Update queue depth metrics
    try {
      const redisQueueLength = await redisClient.lLen('job_queue');
      metrics.setRedisQueueDepth(redisQueueLength);
    } catch (error) {
      logger.warn('Failed to get Redis queue length:', error.message);
    }

    // Update RabbitMQ queue metrics
    try {
      if (rabbitmq.isConnected) {
        const queueStats = await rabbitmq.getAllQueueStats();
        for (const [queueName, stats] of Object.entries(queueStats)) {
          metrics.setRabbitMQQueueDepth(queueName, stats.messageCount);
        }
      }
    } catch (error) {
      logger.warn('Failed to get RabbitMQ queue stats:', error.message);
    }

    // Get and return metrics
    const metricsOutput = await metrics.getMetrics();
    res.set('Content-Type', metrics.getContentType());
    res.send(metricsOutput);
  } catch (error) {
    logger.error('Error generating metrics:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate metrics',
      error: error.message
    });
  }
});

// Admin endpoints

// POST /admin/requeue/{job_id} - Requeue from DLQ or mark retry
router.post('/admin/requeue/:job_id', async (req, res) => {
  try {
    const { job_id } = req.params;

    const job = await Job.findOne({ job_id });

    if (!job) {
      return res.status(404).json({
        success: false,
        message: 'Job not found'
      });
    }

    // Reset job status and increment retry count
    job.status = 'queued';
    job.retry_count += 1;
    job.error = undefined;
    job.processed_at = undefined;
    job.completed_at = undefined;

    await job.save();

    // Re-add to queue
    await redisClient.lPush('job_queue', JSON.stringify({
      job_id: job.job_id,
      payload: job.payload,
      metadata: job.metadata
    }));

    metrics.requeued++;

    res.json({
      success: true,
      message: 'Job requeued successfully',
      job_id: job.job_id,
      retry_count: job.retry_count
    });
  } catch (error) {
    console.error('Error requeuing job:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to requeue job',
      error: error.message
    });
  }
});

// POST /admin/publish-test - Publish a test message
router.post('/admin/publish-test', async (req, res) => {
  try {
    const { routing_key, payload } = req.body;

    const testPayload = payload || {
      message: 'Test message',
      timestamp: new Date().toISOString(),
      routing_key: routing_key || 'test.default'
    };

    const job = new Job({
      payload: testPayload,
      metadata: {
        priority: 0,
        routing_key: routing_key || 'test.default',
        origin: 'admin_test'
      }
    });

    await job.save();
    metrics.enqueued++;

    // Add to queue
    await redisClient.lPush('job_queue', JSON.stringify({
      job_id: job.job_id,
      payload: job.payload,
      metadata: job.metadata
    }));

    res.json({
      success: true,
      message: 'Test message published successfully',
      job_id: job.job_id
    });
  } catch (error) {
    console.error('Error publishing test message:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to publish test message',
      error: error.message
    });
  }
});

// DELETE /jobs/{job_id} - Cancel job (best-effort)
router.delete('/jobs/:job_id', async (req, res) => {
  try {
    const { job_id } = req.params;

    const job = await Job.findOne({ job_id });

    if (!job) {
      return res.status(404).json({
        success: false,
        message: 'Job not found'
      });
    }

    // Only cancel if job is not already processed
    if (job.status === 'processing' || job.status === 'completed') {
      return res.status(400).json({
        success: false,
        message: `Cannot cancel job in ${job.status} status`
      });
    }

    // Mark as cancelled
    job.status = 'cancelled';
    job.completed_at = new Date();
    await job.save();

    res.json({
      success: true,
      message: 'Job cancelled successfully',
      job_id: job.job_id
    });
  } catch (error) {
    console.error('Error cancelling job:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to cancel job',
      error: error.message
    });
  }
});

// POST /webhook - Receive third-party webhook and enqueue message
router.post('/webhook', async (req, res) => {
  try {
    const webhookData = req.body;

    // Extract relevant webhook data
    const payload = {
      webhook_data: webhookData,
      received_at: new Date().toISOString(),
      headers: req.headers
    };

    const metadata = {
      priority: 1, // Webhooks get slightly higher priority
      origin: 'webhook',
      routing_key: 'webhook.default'
    };

    const job = new Job({
      payload,
      metadata
    });

    await job.save();
    metrics.enqueued++;

    // Add to queue
    await redisClient.lPush('job_queue', JSON.stringify({
      job_id: job.job_id,
      payload: job.payload,
      metadata: job.metadata
    }));

    res.status(201).json({
      success: true,
      job_id: job.job_id,
      status: 'queued'
    });
  } catch (error) {
    console.error('Error processing webhook:', error);
    metrics.failed++;
    res.status(503).json({
      success: false,
      message: 'Failed to process webhook',
      error: error.message
    });
  }
});

module.exports = router;
