-- Create database if it doesn't exist
SELECT 'CREATE DATABASE radiology_hospital'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'radiology_hospital')\gexec

\c radiology_hospital

-- Create user if it doesn't exist
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'radiology_user') THEN

      CREATE ROLE radiology_user LOGIN PASSWORD 'SecurePassword123!';
   ELSE
      ALTER ROLE radiology_user WITH PASSWORD 'SecurePassword123!';
   END IF;
END
$do$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE radiology_hospital TO radiology_user;
ALTER DATABASE radiology_hospital OWNER TO radiology_user;
GRANT ALL ON SCHEMA public TO radiology_user;
