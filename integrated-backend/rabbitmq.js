const amqp = require('amqplib');

/**
 * RabbitMQ Connection Manager
 * Handles connection, channel management, and queue setup for ML model integration
 */

class RabbitMQManager {
  constructor() {
    this.connection = null;
    this.channel = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 5000; // 5 seconds

    // Configuration from environment variables
    this.config = {
      url: process.env.RABBITMQ_URL || 'amqp://127.0.0.1:5672',
      exchange: process.env.RABBITMQ_EXCHANGE || 'hospital_exchange',
      dlx: process.env.RABBITMQ_DLX || 'hospital_dlx',
      queues: {
        priority: 'priority_queue',
        waitlist: 'waitlist_queue',
        report: 'report_queue',
        pdf: 'pdf_queue'
      },
      routingKeys: {
        prioritizer: 'ml.prioritizer',
        reportGenerator: 'ml.report_generator',
        pdfGenerator: 'pdf.generator'
      }
    };
  }

  /**
   * Initialize RabbitMQ connection and setup
   */
  async connect() {
    try {
      console.log('[RabbitMQ] Connecting to:', this.config.url);

      // Create connection
      this.connection = await amqp.connect(this.config.url);
      this.isConnected = true;
      this.reconnectAttempts = 0;

      console.log('[RabbitMQ] ✓ Connected successfully');

      // Create channel
      this.channel = await this.connection.createChannel();
      console.log('[RabbitMQ] ✓ Channel created');

      // Setup error handlers
      this.connection.on('error', (err) => {
        console.error('[RabbitMQ] Connection error:', err.message);
        this.isConnected = false;
      });

      this.connection.on('close', () => {
        console.warn('[RabbitMQ] Connection closed');
        this.isConnected = false;
        this.handleReconnect();
      });

      // Setup exchanges and queues
      await this.setupInfrastructure();

      return true;
    } catch (error) {
      console.error('[RabbitMQ] Connection failed:', error.message);
      this.isConnected = false;
      this.handleReconnect();
      return false;
    }
  }

  /**
   * Setup exchanges, queues, and bindings
   */
  async setupInfrastructure() {
    try {
      // Declare main topic exchange
      await this.channel.assertExchange(this.config.exchange, 'topic', {
        durable: true
      });
      console.log(`[RabbitMQ] ✓ Exchange '${this.config.exchange}' declared`);

      // Declare dead letter exchange
      await this.channel.assertExchange(this.config.dlx, 'topic', {
        durable: true
      });
      console.log(`[RabbitMQ] ✓ Dead letter exchange '${this.config.dlx}' declared`);

      // Declare queues with dead letter exchange
      const queueOptions = {
        durable: true,
        arguments: {
          'x-dead-letter-exchange': this.config.dlx,
          'x-message-ttl': 3600000 // 1 hour TTL
        }
      };

      // Priority Queue - receives new patient data
      await this.channel.assertQueue(this.config.queues.priority, queueOptions);
      await this.channel.bindQueue(
        this.config.queues.priority,
        this.config.exchange,
        this.config.routingKeys.prioritizer
      );
      console.log(`[RabbitMQ] ✓ Queue '${this.config.queues.priority}' created and bound`);

      // Waitlist Queue - patients waiting for scanning
      await this.channel.assertQueue(this.config.queues.waitlist, queueOptions);
      console.log(`[RabbitMQ] ✓ Queue '${this.config.queues.waitlist}' created`);

      // Report Queue - receives scan results for ML processing
      await this.channel.assertQueue(this.config.queues.report, queueOptions);
      await this.channel.bindQueue(
        this.config.queues.report,
        this.config.exchange,
        this.config.routingKeys.reportGenerator
      );
      console.log(`[RabbitMQ] ✓ Queue '${this.config.queues.report}' created and bound`);

      // PDF Queue - final report generation
      await this.channel.assertQueue(this.config.queues.pdf, queueOptions);
      await this.channel.bindQueue(
        this.config.queues.pdf,
        this.config.exchange,
        this.config.routingKeys.pdfGenerator
      );
      console.log(`[RabbitMQ] ✓ Queue '${this.config.queues.pdf}' created and bound`);

      // Dead letter queue
      await this.channel.assertQueue('dead_letter_queue', { durable: true });
      await this.channel.bindQueue('dead_letter_queue', this.config.dlx, '#');
      console.log('[RabbitMQ] ✓ Dead letter queue created');

      console.log('[RabbitMQ] ✓ Infrastructure setup complete');
    } catch (error) {
      console.error('[RabbitMQ] Infrastructure setup failed:', error.message);
      throw error;
    }
  }

  /**
   * Publish message to exchange
   * @param {string} routingKey - Routing key for message
   * @param {object} message - Message payload
   * @param {object} options - Additional options
   */
  async publish(routingKey, message, options = {}) {
    try {
      if (!this.isConnected || !this.channel) {
        throw new Error('RabbitMQ not connected');
      }

      const messageBuffer = Buffer.from(JSON.stringify(message));

      const publishOptions = {
        persistent: true,
        contentType: 'application/json',
        timestamp: Date.now(),
        ...options
      };

      const published = this.channel.publish(
        this.config.exchange,
        routingKey,
        messageBuffer,
        publishOptions
      );

      if (published) {
        console.log(`[RabbitMQ] ✓ Message published to '${routingKey}'`);
        return true;
      } else {
        console.warn(`[RabbitMQ] ⚠ Message buffered for '${routingKey}'`);
        return false;
      }
    } catch (error) {
      console.error('[RabbitMQ] Publish error:', error.message);
      throw error;
    }
  }

  /**
   * Consume messages from queue
   * @param {string} queueName - Queue to consume from
   * @param {function} callback - Message handler function
   */
  async consume(queueName, callback) {
    try {
      if (!this.isConnected || !this.channel) {
        throw new Error('RabbitMQ not connected');
      }

      await this.channel.consume(queueName, async (msg) => {
        if (msg) {
          try {
            const content = JSON.parse(msg.content.toString());
            console.log(`[RabbitMQ] ← Message received from '${queueName}'`);

            // Call the callback with message content
            await callback(content, msg);

            // Acknowledge message
            this.channel.ack(msg);
            console.log(`[RabbitMQ] ✓ Message acknowledged`);
          } catch (error) {
            console.error('[RabbitMQ] Message processing error:', error.message);

            // Reject and requeue if processing fails
            this.channel.nack(msg, false, false); // Don't requeue, send to DLX
          }
        }
      }, {
        noAck: false // Manual acknowledgment
      });

      console.log(`[RabbitMQ] ✓ Consumer started for queue '${queueName}'`);
    } catch (error) {
      console.error('[RabbitMQ] Consume error:', error.message);
      throw error;
    }
  }

  /**
   * Get queue statistics
   * @param {string} queueName - Queue name
   */
  async getQueueStats(queueName) {
    try {
      if (!this.isConnected || !this.channel) {
        return { messageCount: 0, consumerCount: 0 };
      }

      const queueInfo = await this.channel.checkQueue(queueName);
      return {
        messageCount: queueInfo.messageCount,
        consumerCount: queueInfo.consumerCount
      };
    } catch (error) {
      console.error(`[RabbitMQ] Error getting stats for '${queueName}':`, error.message);
      return { messageCount: 0, consumerCount: 0 };
    }
  }

  /**
   * Get all queue statistics
   */
  async getAllQueueStats() {
    const stats = {};
    for (const [key, queueName] of Object.entries(this.config.queues)) {
      stats[queueName] = await this.getQueueStats(queueName);
    }
    return stats;
  }

  /**
   * Handle reconnection logic
   */
  async handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[RabbitMQ] Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(`[RabbitMQ] Reconnecting... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);
  }

  /**
   * Close connection gracefully
   */
  async close() {
    try {
      if (this.channel) {
        await this.channel.close();
        console.log('[RabbitMQ] Channel closed');
      }

      if (this.connection) {
        await this.connection.close();
        console.log('[RabbitMQ] Connection closed');
      }

      this.isConnected = false;
    } catch (error) {
      console.error('[RabbitMQ] Error closing connection:', error.message);
    }
  }

  /**
   * Health check
   */
  getHealthStatus() {
    return {
      connected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
      config: {
        url: this.config.url.replace(/\/\/.*@/, '//***@'), // Hide credentials
        exchange: this.config.exchange
      }
    };
  }
}

// Create singleton instance
const rabbitmqManager = new RabbitMQManager();

module.exports = rabbitmqManager;
