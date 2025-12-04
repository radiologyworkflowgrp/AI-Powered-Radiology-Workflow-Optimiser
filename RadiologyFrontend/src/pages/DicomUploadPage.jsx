import { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useUser } from "@/hooks/use-user";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const DicomUploadPage = () => {
    const { user } = useUser();
    const navigate = useNavigate();
    const [selectedFile, setSelectedFile] = useState(null);
    const [patients, setPatients] = useState([]);
    const [selectedPatient, setSelectedPatient] = useState("");
    const [scanType, setScanType] = useState("X-RAY");
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadResult, setUploadResult] = useState(null);
    const [error, setError] = useState(null);

    const API_URL = import.meta.env.VITE_API_URL || "http://localhost:3002";

    // Check if user is admin or doctor
    useEffect(() => {
        if (user && user.role !== "admin" && user.role !== "doctor") {
            navigate("/");
        }
    }, [user, navigate]);

    useEffect(() => {
        fetchPatients();
    }, []);

    const fetchPatients = async () => {
        try {
            const response = await axios.get(`${API_URL}/api/patients`);
            setPatients(response.data);
        } catch (err) {
            console.error("Error fetching patients:", err);
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedFile(file);
            setError(null);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError("Please select a DICOM file");
            return;
        }

        if (!selectedPatient) {
            setError("Please select a patient");
            return;
        }

        const patient = patients.find((p) => p._id === selectedPatient);
        if (!patient) {
            setError("Invalid patient selected");
            return;
        }

        setUploading(true);
        setError(null);
        setUploadResult(null);
        setUploadProgress(0);

        try {
            const formData = new FormData();
            formData.append("dicomFile", selectedFile);
            formData.append("patientId", patient._id);
            formData.append("patientName", patient.name);
            formData.append("patientEmail", patient.email || "");
            formData.append("scanType", scanType);
            formData.append("deviceType", "mobile"); // Changed from "web" to valid enum value
            formData.append("deviceId", "doctor-portal");
            formData.append("deviceModel", "Web Browser");

            const response = await axios.post(`${API_URL}/api/dicom/upload`, formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setUploadProgress(percentCompleted);
                },
            });

            setUploadResult(response.data);
            setSelectedFile(null);
            setSelectedPatient("");

            // Reset file input
            const fileInput = document.getElementById("dicom-file-input");
            if (fileInput) fileInput.value = "";
        } catch (err) {
            console.error("Upload error:", err);
            setError(err.response?.data?.message || "Failed to upload DICOM file");
        } finally {
            setUploading(false);
            setUploadProgress(0);
        }
    };

    if (!user || (user.role !== "admin" && user.role !== "doctor")) {
        return null;
    }

    return (
        <DashboardLayout title="DICOM Upload">
            <div className="max-w-2xl mx-auto">
                <Card>
                    <CardHeader>
                        <CardTitle>Upload DICOM File</CardTitle>
                        <CardDescription>
                            Upload DICOM files for automated analysis through DuoFormer AI model with ground truth filtering
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {/* Patient Selection */}
                        <div>
                            <label className="block text-sm font-medium mb-2">Select Patient *</label>
                            <select
                                value={selectedPatient}
                                onChange={(e) => setSelectedPatient(e.target.value)}
                                className="w-full px-3 py-2 border rounded-md"
                                disabled={uploading}
                            >
                                <option value="">-- Select a patient --</option>
                                {patients.map((patient) => (
                                    <option key={patient._id} value={patient._id}>
                                        {patient.name} ({patient.email})
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Scan Type */}
                        <div>
                            <label className="block text-sm font-medium mb-2">Scan Type *</label>
                            <select
                                value={scanType}
                                onChange={(e) => setScanType(e.target.value)}
                                className="w-full px-3 py-2 border rounded-md"
                                disabled={uploading}
                            >
                                <option value="X-RAY">Chest X-Ray (DuoFormer)</option>
                                <option value="CT">CT Scan</option>
                                <option value="MRI">MRI</option>
                                <option value="ULTRASOUND">Ultrasound</option>
                                <option value="PET">PET Scan</option>
                                <option value="OTHER">Other</option>
                            </select>
                        </div>

                        {/* File Upload */}
                        <div>
                            <label className="block text-sm font-medium mb-2">DICOM File *</label>
                            <input
                                id="dicom-file-input"
                                type="file"
                                onChange={handleFileChange}
                                className="w-full px-3 py-2 border rounded-md"
                                disabled={uploading}
                            />
                            {selectedFile && (
                                <p className="mt-2 text-sm text-muted-foreground">
                                    Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                                </p>
                            )}
                        </div>

                        {/* Upload Progress */}
                        {uploading && (
                            <div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-blue-600 h-2 rounded-full transition-all"
                                        style={{ width: `${uploadProgress}%` }}
                                    />
                                </div>
                                <p className="text-center mt-2 text-sm">Uploading... {uploadProgress}%</p>
                            </div>
                        )}

                        {/* Error Message */}
                        {error && (
                            <Alert variant="destructive">
                                <AlertDescription>{error}</AlertDescription>
                            </Alert>
                        )}

                        {/* Success Message */}
                        {uploadResult && uploadResult.success && (
                            <Alert>
                                <AlertDescription>
                                    <p className="font-semibold mb-2">✅ Upload Successful!</p>
                                    <p className="text-sm">Scan ID: {uploadResult.scan.scanId}</p>
                                    <p className="text-sm">Patient: {uploadResult.scan.patientName}</p>
                                    <p className="text-sm">Status: {uploadResult.scan.status}</p>
                                    <p className="text-sm mt-2 text-blue-600">
                                        🤖 Processing through DuoFormer AI with ground truth filtering...
                                    </p>
                                    <p className="text-sm text-gray-600">
                                        Results will appear in the radiology results section
                                    </p>
                                </AlertDescription>
                            </Alert>
                        )}

                        {/* Upload Button */}
                        <Button
                            onClick={handleUpload}
                            disabled={uploading || !selectedFile || !selectedPatient}
                            className="w-full"
                        >
                            {uploading ? "Uploading..." : "Upload DICOM File"}
                        </Button>
                    </CardContent>
                </Card>
            </div>
        </DashboardLayout>
    );
};

export default DicomUploadPage;
