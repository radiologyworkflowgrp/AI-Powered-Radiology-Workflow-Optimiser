-- Fix priority column type conversion
-- This will drop existing priority values and recreate as INTEGER

-- Drop the existing priority column
ALTER TABLE patients DROP COLUMN IF EXISTS priority;

-- Add priority column as INTEGER with proper constraints
ALTER TABLE patients ADD COLUMN priority INTEGER DEFAULT 2 CHECK (priority >= 1 AND priority <= 3);

-- Add comment
COMMENT ON COLUMN patients.priority IS '1=High, 2=Medium, 3=Low';

-- Set all existing patients to default priority
UPDATE patients SET priority = 2 WHERE priority IS NULL;
