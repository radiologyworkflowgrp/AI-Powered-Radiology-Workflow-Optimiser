import { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { addPatient, getPatients, updatePatient } from "@/services/api";
import { toast } from "sonner";
import { Textarea } from "@/components/ui/textarea";

const PatientInfoPortal = () => {
    const [patient, setPatient] = useState({
        name: "",
        age: "",
        gender: "",
        guardian_name: "",
        guardian_phone: "",
        allergies: "",
        blood_type: "",
        weight: "",
        height: "",
        medical_history: "",
        symptoms: "",
        status: "Admitted",
        date: new Date().toISOString().slice(0, 10),
    });
    const [pdfFile, setPdfFile] = useState(null);

    // Update panel state
    const [searchQuery, setSearchQuery] = useState("");
    const [allPatients, setAllPatients] = useState([]);
    const [filteredPatients, setFilteredPatients] = useState([]);
    const [selectedPatient, setSelectedPatient] = useState(null);
    const [updateFormData, setUpdateFormData] = useState({
        name: "",
        age: "",
        gender: "",
        guardian_name: "",
        guardian_phone: "",
        allergies: "",
        blood_type: "",
        weight: "",
        height: "",
        medical_history: "",
        symptoms: "",
        status: "",
        date: "",
    });

    // Fetch all patients on mount
    useEffect(() => {
        fetchPatients();
    }, []);

    const fetchPatients = async () => {
        try {
            const patients = await getPatients();
            setAllPatients(patients);
        } catch (error) {
            console.error("Failed to fetch patients:", error);
            toast.error("Failed to load patients");
        }
    };

    const handleChange = (e) => {
        const { id, value } = e.target;
        setPatient((prev) => ({ ...prev, [id]: value }));
    };

    const handleFileChange = (e) => {
        setPdfFile(e.target.files[0]);
    };

    const handleSelectChange = (id, value) => {
        setPatient((prev) => ({ ...prev, [id]: value }));
    };

    const handleAdmit = async () => {
        try {
            const formData = new FormData();
            for (const key in patient) {
                formData.append(key, patient[key]);
            }
            if (pdfFile) {
                formData.append('file', pdfFile);
            }

            const response = await addPatient(formData);
            console.log("Patient admitted:", response);
            toast.success("Patient admitted successfully!");
            // Clear form
            setPatient({
                name: "",
                age: "",
                gender: "",
                guardian_name: "",
                guardian_phone: "",
                allergies: "",
                medical_history: "",
                symptoms: "",
                weight: "",
                height: "",
                blood_type: "",
                status: "Admitted",
                date: new Date().toISOString().slice(0, 10),
            });
            setPdfFile(null);
            // Refresh patient list
            fetchPatients();
        } catch (error) {
            console.error("Failed to admit patient:", error);
            toast.error("Failed to admit patient. Please try again.");
        }
    };

    // Update panel handlers
    const handleSearch = () => {
        if (!searchQuery.trim()) {
            setFilteredPatients([]);
            return;
        }

        const query = searchQuery.toLowerCase();
        const results = allPatients.filter(p =>
            p.name.toLowerCase().includes(query) ||
            p._id.toLowerCase().includes(query)
        );
        setFilteredPatients(results);
    };

    const handleSelectPatient = (patient) => {
        setSelectedPatient(patient);
        setUpdateFormData({
            name: patient.name || "",
            age: patient.age || "",
            gender: patient.gender || "",
            guardian_name: patient.guardian_name || "",
            guardian_phone: patient.guardian_phone || "",
            allergies: patient.allergies || "",
            blood_type: patient.blood_type || "",
            weight: patient.weight || "",
            height: patient.height || "",
            medical_history: patient.medical_history || "",
            symptoms: patient.symptoms || "",
            status: patient.status || "",
            date: patient.date || "",
        });
        setFilteredPatients([]);
        setSearchQuery("");
    };

    const handleUpdateChange = (e) => {
        const { id, value } = e.target;
        setUpdateFormData((prev) => ({ ...prev, [id]: value }));
    };

    const handleUpdateSelectChange = (id, value) => {
        setUpdateFormData((prev) => ({ ...prev, [id]: value }));
    };

    const handleUpdate = async () => {
        if (!selectedPatient) {
            toast.error("Please select a patient first");
            return;
        }

        try {
            const response = await updatePatient(selectedPatient._id, updateFormData);
            console.log("Patient updated:", response);
            toast.success("Patient updated successfully!");
            // Clear form
            setSelectedPatient(null);
            setUpdateFormData({
                name: "",
                age: "",
                gender: "",
                guardian_name: "",
                guardian_phone: "",
                allergies: "",
                blood_type: "",
                weight: "",
                height: "",
                medical_history: "",
                symptoms: "",
                status: "",
                date: "",
            });
            // Refresh patient list
            fetchPatients();
        } catch (error) {
            console.error("Failed to update patient:", error);
            toast.error("Failed to update patient. Please try again.");
        }
    };

    return (
        <DashboardLayout title="Patient Admission Portal">
            <Card>
                <CardHeader>
                    <CardTitle>Admit New Patient</CardTitle>
                    <CardDescription>Fill out the form below to admit a new patient.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="name">Full Name</Label>
                        <Input id="name" value={patient.name} onChange={handleChange} />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="age">Age</Label>
                            <Input id="age" type="number" value={patient.age} onChange={handleChange} />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="gender">Gender</Label>
                            <Select onValueChange={(value) => handleSelectChange("gender", value)} value={patient.gender}>
                                <SelectTrigger>
                                    <SelectValue placeholder="Select gender" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="Male">Male</SelectItem>
                                    <SelectItem value="Female">Female</SelectItem>
                                    <SelectItem value="Other">Other</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="guardian_name">Guardian Name</Label>
                            <Input id="guardian_name" value={patient.guardian_name} onChange={handleChange} />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="guardian_phone">Guardian Phone</Label>
                            <Input id="guardian_phone" value={patient.guardian_phone} onChange={handleChange} />
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="allergies">Allergies</Label>
                            <Input id="allergies" value={patient.allergies} onChange={handleChange} />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="blood_type">Blood Type</Label>
                            <Select onValueChange={(value) => handleSelectChange("blood_type", value)} value={patient.blood_type}>
                                <SelectTrigger>
                                    <SelectValue placeholder="Select blood type" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="A+">A+</SelectItem>
                                    <SelectItem value="A-">A-</SelectItem>
                                    <SelectItem value="B+">B+</SelectItem>
                                    <SelectItem value="B-">B-</SelectItem>
                                    <SelectItem value="AB+">AB+</SelectItem>
                                    <SelectItem value="AB-">AB-</SelectItem>
                                    <SelectItem value="O+">O+</SelectItem>
                                    <SelectItem value="O-">O-</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="weight">Weight (kg)</Label>
                            <Input id="weight" type="number" value={patient.weight} onChange={handleChange} />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="height">Height (cm)</Label>
                            <Input id="height" type="number" value={patient.height} onChange={handleChange} />
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="date">Admission Date</Label>
                            <Input id="date" type="date" value={patient.date} onChange={handleChange} />
                        </div>
                    </div>
                    <div>
                        <div className="space-y-2">
                            <Label htmlFor="symptoms">Symptoms</Label>
                            <Textarea
                                id="symptoms"
                                placeholder="Enter patient symptoms here..."
                                className="min-h-[100px]"
                                value={patient.symptoms}
                                onChange={handleChange}
                            />
                        </div>
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="pdfFile">Medical History (PDF)</Label>
                        <Input id="pdfFile" type="file" accept=".pdf" onChange={handleFileChange} />
                    </div>
                    <Button onClick={handleAdmit}>Admit Patient</Button>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Update an Existing Patient</CardTitle>
                    <CardDescription>Search for a patient and update their information</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="searchQuery">Search Patient</Label>
                        <div className="flex gap-2">
                            <Input
                                id="searchQuery"
                                placeholder="Enter patient name or ID..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                            />
                            <Button onClick={handleSearch}>Search</Button>
                        </div>
                        {filteredPatients.length > 0 && (
                            <div className="border rounded-md mt-2 max-h-40 overflow-y-auto">
                                {filteredPatients.map((p) => (
                                    <div
                                        key={p._id}
                                        className="p-2 hover:bg-gray-100 cursor-pointer border-b last:border-b-0"
                                        onClick={() => handleSelectPatient(p)}
                                    >
                                        <div className="font-medium">{p.name}</div>
                                        <div className="text-sm text-gray-500">ID: {p._id}</div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {selectedPatient && (
                        <>
                            <div className="p-3 bg-blue-50 rounded-md">
                                <p className="text-sm font-medium">Selected Patient: {selectedPatient.name}</p>
                                <p className="text-xs text-gray-600">ID: {selectedPatient._id}</p>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="name">Full Name</Label>
                                <Input id="name" value={updateFormData.name} onChange={handleUpdateChange} />
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="age">Age</Label>
                                    <Input id="age" type="number" value={updateFormData.age} onChange={handleUpdateChange} />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="gender">Gender</Label>
                                    <Select onValueChange={(value) => handleUpdateSelectChange("gender", value)} value={updateFormData.gender}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select gender" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="Male">Male</SelectItem>
                                            <SelectItem value="Female">Female</SelectItem>
                                            <SelectItem value="Other">Other</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="guardian_name">Guardian Name</Label>
                                    <Input id="guardian_name" value={updateFormData.guardian_name} onChange={handleUpdateChange} />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="guardian_phone">Guardian Phone</Label>
                                    <Input id="guardian_phone" value={updateFormData.guardian_phone} onChange={handleUpdateChange} />
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="allergies">Allergies</Label>
                                    <Input id="allergies" value={updateFormData.allergies} onChange={handleUpdateChange} />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="blood_type">Blood Type</Label>
                                    <Select onValueChange={(value) => handleUpdateSelectChange("blood_type", value)} value={updateFormData.blood_type}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select blood type" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="A+">A+</SelectItem>
                                            <SelectItem value="A-">A-</SelectItem>
                                            <SelectItem value="B+">B+</SelectItem>
                                            <SelectItem value="B-">B-</SelectItem>
                                            <SelectItem value="AB+">AB+</SelectItem>
                                            <SelectItem value="AB-">AB-</SelectItem>
                                            <SelectItem value="O+">O+</SelectItem>
                                            <SelectItem value="O-">O-</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="weight">Weight (kg)</Label>
                                    <Input id="weight" type="number" value={updateFormData.weight} onChange={handleUpdateChange} />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="height">Height (cm)</Label>
                                    <Input id="height" type="number" value={updateFormData.height} onChange={handleUpdateChange} />
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="status">Status</Label>
                                    <Select onValueChange={(value) => handleUpdateSelectChange("status", value)} value={updateFormData.status}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select status" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="Admitted">Admitted</SelectItem>
                                            <SelectItem value="In Treatment">In Treatment</SelectItem>
                                            <SelectItem value="Discharged">Discharged</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="date">Date</Label>
                                    <Input id="date" type="date" value={updateFormData.date} onChange={handleUpdateChange} />
                                </div>
                            </div>
                            <div>
                                <div className="space-y-2">
                                    <Label htmlFor="symptoms">Symptoms</Label>
                                    <Textarea
                                        id="symptoms"
                                        placeholder="Enter patient symptoms here..."
                                        className="min-h-[100px]"
                                        value={updateFormData.symptoms}
                                        onChange={handleUpdateChange}
                                    />
                                </div>
                            </div>
                            <Button onClick={handleUpdate}>Update Patient</Button>
                        </>
                    )}
                </CardContent>
            </Card>
        </DashboardLayout>
    );
};

export default PatientInfoPortal;
