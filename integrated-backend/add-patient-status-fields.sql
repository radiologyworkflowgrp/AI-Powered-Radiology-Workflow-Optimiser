-- Add status and checkedByDoctor columns to patients table
-- Run this to add the new workflow fields

ALTER TABLE patients ADD COLUMN IF NOT EXISTS status VARCHAR(255) DEFAULT 'Admitted' 
  CHECK (status IN ('Admitted', 'Doctor Appointment', 'Completed'));

ALTER TABLE patients ADD COLUMN IF NOT EXISTS "checkedByDoctor" BOOLEAN DEFAULT false;

-- Set all existing patients to 'Admitted' status
UPDATE patients SET status = 'Admitted' WHERE status IS NULL;
UPDATE patients SET "checkedByDoctor" = false WHERE "checkedByDoctor" IS NULL;

-- Add comments for clarity
COMMENT ON COLUMN patients.status IS 'Admitted | Doctor Appointment | Completed';
COMMENT ON COLUMN patients."checkedByDoctor" IS 'Whether doctor has checked this patient';
