import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DicomUpload = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [patients, setPatients] = useState([]);
    const [selectedPatient, setSelectedPatient] = useState('');
    const [scanType, setScanType] = useState('CT');
    const [notes, setNotes] = useState('');
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadResult, setUploadResult] = useState(null);
    const [error, setError] = useState(null);

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3002';

    // Fetch patients on component mount
    useEffect(() => {
        fetchPatients();
    }, []);

    const fetchPatients = async () => {
        try {
            const response = await axios.get(`${API_URL}/api/patients`);
            setPatients(response.data);
        } catch (err) {
            console.error('Error fetching patients:', err);
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            // Accept any file - DICOM files often don't have standard extensions
            // Backend will validate the actual DICOM format
            setSelectedFile(file);
            setError(null);
        }
    };

    const handlePatientChange = (e) => {
        setSelectedPatient(e.target.value);
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a DICOM file');
            return;
        }

        if (!selectedPatient) {
            setError('Please select a patient');
            return;
        }

        const patient = patients.find(p => p._id === selectedPatient);
        if (!patient) {
            setError('Invalid patient selected');
            return;
        }

        setUploading(true);
        setError(null);
        setUploadResult(null);
        setUploadProgress(0);

        try {
            const formData = new FormData();
            formData.append('dicomFile', selectedFile);
            formData.append('patientId', patient._id);
            formData.append('patientName', patient.name);
            formData.append('patientEmail', patient.email || '');
            formData.append('scanType', scanType);
            formData.append('notes', notes);
            formData.append('deviceType', 'mobile');
            formData.append('deviceId', navigator.userAgent);
            formData.append('deviceModel', 'Web Browser');

            const response = await axios.post(`${API_URL}/api/dicom/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setUploadProgress(percentCompleted);
                },
            });

            setUploadResult(response.data);
            setSelectedFile(null);
            setNotes('');

            // Reset file input
            const fileInput = document.getElementById('dicom-file-input');
            if (fileInput) fileInput.value = '';

        } catch (err) {
            console.error('Upload error:', err);
            setError(err.response?.data?.message || 'Failed to upload DICOM file');
        } finally {
            setUploading(false);
            setUploadProgress(0);
        }
    };

    return (
        <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
            <h2 className="text-3xl font-bold mb-6 text-gray-800">📱 Mobile CT Scanner Upload</h2>

            {/* Patient Selection */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Patient *
                </label>
                <select
                    value={selectedPatient}
                    onChange={handlePatientChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={uploading}
                >
                    <option value="">-- Select a patient --</option>
                    {patients.map(patient => (
                        <option key={patient._id} value={patient._id}>
                            {patient.name} ({patient.email})
                        </option>
                    ))}
                </select>
            </div>

            {/* Scan Type */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Scan Type *
                </label>
                <select
                    value={scanType}
                    onChange={(e) => setScanType(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={uploading}
                >
                    <option value="CT">CT Scan</option>
                    <option value="MRI">MRI</option>
                    <option value="X-RAY">X-Ray</option>
                    <option value="ULTRASOUND">Ultrasound</option>
                    <option value="PET">PET Scan</option>
                    <option value="OTHER">Other</option>
                </select>
            </div>

            {/* File Upload */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    DICOM File *
                </label>
                <input
                    id="dicom-file-input"
                    type="file"
                    onChange={handleFileChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={uploading}
                />
                {selectedFile && (
                    <p className="mt-2 text-sm text-gray-600">
                        Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                )}
            </div>

            {/* Notes */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Notes (Optional)
                </label>
                <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    rows={3}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Add any additional notes about this scan..."
                    disabled={uploading}
                />
            </div>

            {/* Upload Progress */}
            {uploading && (
                <div className="mb-6">
                    <div className="w-full bg-gray-200 rounded-full h-4">
                        <div
                            className="bg-blue-600 h-4 rounded-full transition-all duration-300"
                            style={{ width: `${uploadProgress}%` }}
                        />
                    </div>
                    <p className="text-center mt-2 text-sm text-gray-600">
                        Uploading... {uploadProgress}%
                    </p>
                </div>
            )}

            {/* Error Message */}
            {error && (
                <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
                    <p className="font-semibold">Error:</p>
                    <p>{error}</p>
                </div>
            )}

            {/* Success Message */}
            {uploadResult && uploadResult.success && (
                <div className="mb-6 p-4 bg-green-100 border border-green-400 text-green-700 rounded-lg">
                    <p className="font-semibold mb-2">✅ Upload Successful!</p>
                    <div className="text-sm space-y-1">
                        <p><strong>Scan ID:</strong> {uploadResult.scan.scanId}</p>
                        <p><strong>Patient:</strong> {uploadResult.scan.patientName}</p>
                        <p><strong>Type:</strong> {uploadResult.scan.scanType}</p>
                        <p><strong>Status:</strong> {uploadResult.scan.status}</p>
                        <p className="mt-2 text-xs text-green-600">
                            The scan has been queued for processing and will be analyzed by the ML system.
                        </p>
                    </div>
                </div>
            )}

            {/* Upload Button */}
            <button
                onClick={handleUpload}
                disabled={uploading || !selectedFile || !selectedPatient}
                className={`w-full py-3 px-6 rounded-lg font-semibold text-white transition ${uploading || !selectedFile || !selectedPatient
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                    }`}
            >
                {uploading ? 'Uploading...' : '📤 Upload DICOM File'}
            </button>

            {/* Info */}
            <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                    <strong>ℹ️ Information:</strong> After uploading, the DICOM file will be automatically queued for processing.
                    The ML system will analyze the scan and generate a report. You can view the results in the Radiology Results section.
                </p>
            </div>
        </div>
    );
};

export default DicomUpload;
