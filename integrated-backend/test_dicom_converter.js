// Test script to check DICOM conversion
const { spawn } = require('child_process');
const path = require('path');

console.log('Testing DICOM converter...');

const testPath = 'c:\\Users\\saite\\Downloads\\cloudfinal\\integrated-backend\\prioritization-ml';

const python = spawn('python', [
    '-c',
    `
import sys
sys.path.append('${testPath.replace(/\\/g, '\\\\')}')
print("Python path:", sys.path)
try:
    from dicom_converter import convert_dicom_to_image
    print("Successfully imported dicom_converter")
except Exception as e:
    print("Import error:", str(e))
    import traceback
    traceback.print_exc()
`
]);

python.stdout.on('data', (data) => {
    console.log('STDOUT:', data.toString());
});

python.stderr.on('data', (data) => {
    console.error('STDERR:', data.toString());
});

python.on('close', (code) => {
    console.log('Exit code:', code);
});
