import { useEffect, useState } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useNavigate } from "react-router-dom";
import { ChevronUp, ChevronDown } from "lucide-react";
import { getPatients, addPatientQuick, getPatientById, getDoctors, updatePatient, getDoctorById, updateDoctor, getLogs } from "../../services/api";

const AdminDashboard = () => {
  const navigate = useNavigate();

  const [Patients, setPatients] = useState([]);
  const [Doctors, setDoctors] = useState([]);
  const [logs, setLogs] = useState([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [quickAdmitResult, setQuickAdmitResult] = useState(null);
  const [fetchByIdResult, setFetchByIdResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editedStatus, setEditedStatus] = useState('');
  const [editedDate, setEditedDate] = useState('');
  const [fetchDoctorByIdResult, setFetchDoctorByIdResult] = useState(null);
  const [editedDoctorAvailability, setEditedDoctorAvailability] = useState('');

  const fetchPatients = async () => {
    try {
      setLoading(true);
      const data = await getPatients();
      setPatients(data);
      console.log("PATIENTS FROM BACKEND:", data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchDoctors = async () => {
    try {
      const data = await getDoctors();
      setDoctors(data);
    } catch (err) {
      console.error(err.message);
    }
  };

  const fetchLogs = async () => {
    try {
      setLogsLoading(true);
      const response = await getLogs(100, 'all');
      if (response.success) {
        setLogs(response.logs);
      }
    } catch (err) {
      console.error("Error fetching logs:", err);
    } finally {
      setLogsLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
    fetchDoctors();
    fetchLogs();
  }, []);

  useEffect(() => {
    if (fetchByIdResult) {
      setEditedStatus(fetchByIdResult.status || '');
      setEditedDate(fetchByIdResult.date || (fetchByIdResult.createdAt && new Date(fetchByIdResult.createdAt).toISOString().slice(0, 10)) || '');
    }
  }, [fetchByIdResult]);

  useEffect(() => {
    if (fetchDoctorByIdResult) {
      setEditedDoctorAvailability(fetchDoctorByIdResult.availability || '');
    }
  }, [fetchDoctorByIdResult]);

  const handleQuickAdmit = async () => {
    const newPatient = {
      name: document.getElementById("patient_name").value,
      age: document.getElementById("patient_age").value,
      gender: document.getElementById("patient_gender").value,
      guardian_name: document.getElementById("patient_guardian_name").value,
      guardian_phone: document.getElementById("patient_guardian_phone").value,
      allergies: document.getElementById("patient_allergies").value,
      blood_type: document.getElementById("patient_blood_type").value,
      symptoms: document.getElementById("patient_symptoms").value,
      id: Math.floor(Math.random() * 1000000),
      type: "Admission",
      date: new Date().toISOString().split("T")[0],
      radiologist: "N/A",
      status: "Admitted",
    };
    try {
      const response = await addPatientQuick(newPatient);
      if (response && response.patient) {
        setQuickAdmitResult(response.patient);
      }
      fetchPatients();
    } catch (e) {
      console.error("Failed to add patient:", e);
    }
  };

  const handleFetchById = async () => {
    const id = document.getElementById("fetch_id_input").value;
    if (!id) return;
    try {
      const patient = await getPatientById(id);
      console.log("FETCHED PATIENT:", patient);
      setFetchByIdResult(patient);
    } catch (e) {
      console.error("Fetch by ID error:", e);
      setFetchByIdResult(null);
    }
  };

  const handleUpdatePatient = async () => {
    if (!fetchByIdResult) return;
    try {
      const updatedData = {
        status: editedStatus,
        date: editedDate,
      };
      const updatedPatient = await updatePatient(fetchByIdResult._id, updatedData);
      setFetchByIdResult(updatedPatient);
      fetchPatients();
    } catch (e) {
      console.error("Update patient error:", e);
    }
  };

  const handleFetchDoctorById = async () => {
    const id = document.getElementById("fetch_doctor_id_input").value;
    if (!id) return;
    try {
      const doctor = await getDoctorById(id);
      setFetchDoctorByIdResult(doctor);
    } catch (e) {
      console.error("Fetch by ID error:", e);
      setFetchDoctorByIdResult(null);
    }
  };

  const handleUpdateDoctor = async () => {
    if (!fetchDoctorByIdResult) return;
    try {
      const updatedData = {
        availability: editedDoctorAvailability,
      };
      const updatedDoctor = await updateDoctor(fetchDoctorByIdResult._id, updatedData);
      setFetchDoctorByIdResult(updatedDoctor);
      fetchDoctors();
    } catch (e) {
      console.error("Update doctor error:", e);
    }
  };

  const updatePriority = (id, change) => {
    setPatients(
      Patients.map((patient) =>
        patient.id === id || patient._id === id
          ? { ...patient, priority: Math.max(1, Math.min(5, patient.priority + change)) }
          : patient
      )
    );
  };

  const getPriorityColor = (priority) => {
    if (priority <= 1) return "destructive";
    if (priority <= 2) return "default";
    return "secondary";
  };

  if (loading) {
    return (
      <DashboardLayout title="Admin Dashboard">
        <p>Loading results...</p>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout title="Admin Dashboard">
        <p className="text-red-500">Error: {error}</p>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout title="Admin Dashboard">
      <div className="grid gap-6 md:grid-cols-2 md:grid-rows-2">
        {/* Quick Admit Card */}
        <Card>
          <CardHeader>
            <CardTitle>Admit Patient</CardTitle>
            <CardDescription>Register new patients</CardDescription>
          </CardHeader>
          <CardContent>
            <input
              id="patient_name"
              type="text"
              placeholder="Patient Name"
              className="mb-4 w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <div className="relative mb-4 flex gap-4 grid grid-cols-2">
              <input
                id="patient_age"
                type="number"
                min="0"
                max="120"
                placeholder="Patient Age"
                className="mb-4 w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <select
                id="patient_gender"
                className="mb-4 w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="">Select Gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
            <select
              id="patient_allergies"
              className="mb-4 w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="Any Allergies?">Possible Allergies?</option>
              <option value="None">None</option>
              <option value="Peanuts">Peanuts</option>
              <option value="Gluten">Gluten</option>
              <option value="Dairy">Dairy</option>
              <option value="Seafood">Seafood</option>
              <option value="Penicillin">Penicillin</option>
              <option value="Other">Other</option>
            </select>
            <select
              id="patient_blood_type"
              className="mb-4 w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="">Select Blood Type</option>
              <option value="A+">A+</option>
              <option value="A-">A-</option>
              <option value="B+">B+</option>
              <option value="B-">B-</option>
              <option value="AB+">AB+</option>
              <option value="AB-">AB-</option>
              <option value="O+">O+</option>
              <option value="O-">O-</option>
            </select>
            <input
              id="patient_guardian_name"
              type="text"
              placeholder="Guardian Name"
              className="mb-4 w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <input
              id="patient_guardian_phone"
              type="text"
              placeholder="Guardian Phone Number"
              className="mb-4 w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <textarea
              id="patient_symptoms"
              placeholder="Describe patient symptoms (e.g., chest pain, difficulty breathing, fever...)"
              rows="3"
              className="mb-4 w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <p className="mb-4 text-sm text-muted-foreground">
              Priority will be automatically calculated by our ML model based on symptoms.
            </p>
            <Button className="w-full" variant="secondary" onClick={handleQuickAdmit}>
              Quick admission
            </Button>
            {quickAdmitResult && (
              <div className="mt-4 p-4 border rounded bg-muted">
                <h3 className="font-medium mb-2">New Patient Added</h3>
                <p><strong>Name:</strong> {quickAdmitResult.name}</p>
                <p><strong>Gender:</strong> {quickAdmitResult.gender}</p>
                <p><strong>Age:</strong> {quickAdmitResult.age}</p>
                <p><strong>Guardian Name:</strong> {quickAdmitResult.guardian_name}</p>
                <p><strong>Guardian Phone:</strong> {quickAdmitResult.guardian_phone}</p>
                <p><strong>Allergies:</strong> {quickAdmitResult.allergies}</p>
                <p><strong>Blood Type:</strong> {quickAdmitResult.blood_type}</p>
                <p><strong>Date:</strong> {quickAdmitResult.date}</p>
              </div>
            )}
            <div className="mt-4">
              <input
                id="fetch_id_input"
                type="text"
                placeholder="Enter Patient ID"
                className="mr-2 w-1/2 rounded border border-input bg-background px-2 py-1 focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <Button variant="secondary" onClick={handleFetchById}>
                Fetch Patient
              </Button>
            </div>
            {fetchByIdResult && (
              <div className="mt-4 p-4 border rounded bg-muted">
                <h3 className="font-medium mb-2">Fetched Patient</h3>
                {console.log("Rendering fetched patient:", fetchByIdResult)}
                <p><strong>Name:</strong> {fetchByIdResult.name}</p>
                <p><strong>ID:</strong> {fetchByIdResult._id}</p>
                <p><strong>Status:</strong> {fetchByIdResult.status || 'N/A'}</p>
                <p><strong>Date:</strong> {fetchByIdResult.date || (fetchByIdResult.createdAt && new Date(fetchByIdResult.createdAt).toLocaleDateString()) || 'N/A'}</p>
                <p><strong>Guardian Name:</strong> {fetchByIdResult.guardian_name}</p>
                <p><strong>Guardian Phone:</strong> {fetchByIdResult.guardian_phone}</p>
                <p><strong>Allergies:</strong> {fetchByIdResult.allergies}</p>
                <p><strong>Blood Type:</strong> {fetchByIdResult.blood_type}</p>
                <p><strong>Priority:</strong> {fetchByIdResult.priority}</p>

                <div className="mt-4">
                  <label htmlFor="edit_status" className="block text-sm font-medium">Edit Status</label>
                  <select
                    id="edit_status"
                    value={editedStatus}
                    onChange={(e) => setEditedStatus(e.target.value)}
                    className="mt-1 block w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                  >
                    <option>Admitted</option>
                    <option>Under Observation</option>
                    <option>Discharged</option>
                  </select>
                </div>

                <div className="mt-4">
                  <label htmlFor="edit_date" className="block text-sm font-medium">Edit Date</label>
                  <div className="flex items-center">
                    <input
                      type="date"
                      id="edit_date"
                      value={editedDate}
                      onChange={(e) => setEditedDate(e.target.value)}
                      className="mt-1 block w-full rounded-md border-input bg-background shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                    />
                    <Button variant="secondary" className="ml-2" onClick={() => setEditedDate(new Date().toISOString().slice(0, 10))}>Today</Button>
                  </div>
                </div>

                <Button className="mt-4 w-full" onClick={handleUpdatePatient}>Update Patient</Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Doctors Card */}
        <Card>
          <CardHeader>
            <CardTitle>Doctors Available Today</CardTitle>
            <CardDescription>Manage doctor assignments</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Doctors.map((doctor) => (
                <div key={doctor._id} className="flex items-center justify-between border-b pb-4 last:border-0">
                  <div>
                    <p className="font-medium">{doctor.name}</p>
                    <p className="text-sm text-muted-foreground">{doctor.specialty}</p>
                  </div>
                  <Badge>{doctor.availability}</Badge>
                </div>
              ))}
            </div>
            <Button className="mt-4 w-full" onClick={() => navigate("/admin/doctors")}>Schedule Doctors</Button>

            <div className="mt-4">
              <input
                id="fetch_doctor_id_input"
                type="text"
                placeholder="Enter Doctor ID"
                className="mr-2 w-1/2 rounded border border-input bg-background px-2 py-1 focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <Button variant="secondary" onClick={handleFetchDoctorById}>
                Fetch Doctor
              </Button>
            </div>
            {fetchDoctorByIdResult && (
              <div className="mt-4 p-4 border rounded bg-muted">
                <h3 className="font-medium mb-2">Fetched Doctor</h3>
                <p><strong>Name:</strong> {fetchDoctorByIdResult.name}</p>
                <p><strong>Specialty:</strong> {fetchDoctorByIdResult.specialty}</p>
                <div className="mt-4">
                  <label htmlFor="edit_availability" className="block text-sm font-medium">Edit Availability</label>
                  <select
                    id="edit_availability"
                    value={editedDoctorAvailability}
                    onChange={(e) => setEditedDoctorAvailability(e.target.value)}
                    className="mt-1 block w-full rounded border border-input bg-background px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                  >
                    <option>Available</option>
                    <option>Unavailable</option>
                    <option>On-call</option>
                  </select>
                </div>
                <Button className="mt-4 w-full" onClick={handleUpdateDoctor}>Update Doctor</Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Patient Status Card */}
        <Card>
          <CardHeader>
            <CardTitle>Patient Status</CardTitle>
            <CardDescription>Overview of patient visits</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Patients.map((patient) => (
                <div key={patient._id || patient.id} className="flex items-center justify-between border-b pb-4 last:border-0">
                  <div className="flex-1">
                    <p className="font-medium">{patient.name}</p>
                    <p className="text-sm text-muted-foreground">Last Visit: {patient.lastVisit}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={getPriorityColor(patient.priority)}>Priority {patient.priority}</Badge>
                    <div className="flex flex-col gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={() => updatePriority(patient._id || patient.id, -1)}
                      >
                        <ChevronUp className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={() => updatePriority(patient._id || patient.id, 1)}
                      >
                        <ChevronDown className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <Button className="mt-4 w-full" onClick={() => navigate("/admin/patient-status")}>View All Patients</Button>
          </CardContent>
        </Card>

        {/* Hospital Log Card */}
        <Card>
          <CardHeader>
            <CardTitle>Hospital Log</CardTitle>
            <CardDescription>Patient information log</CardDescription>
          </CardHeader>
          <CardContent>
            <table className="mt-4 w-full table-auto border-collapse text-left">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Age</th>
                  <th>Gender</th>
                  <th>Status</th>
                  <th>Date</th>
                  <th>Priority</th>
                  <th>Assigned Doctor</th>
                </tr>
              </thead>
              <tbody className="divide-y border-t border-b">
                {Patients.map((p) => (
                  <tr key={p._id || p.id}>
                    <td className="border px-2 py-1">{p.name}</td>
                    <td className="border px-2 py-1">{p.age}</td>
                    <td className="border px-2 py-1">{p.gender}</td>
                    <td className="border px-2 py-1">{p.status || 'N/A'}</td>
                    <td className="border px-2 py-1">{p.date || (p.createdAt && new Date(p.createdAt).toLocaleDateString()) || 'N/A'}</td>
                    <td className="border px-2 py-1">{p.priority}</td>
                    <td className="border px-2 py-1">{p.assignedDoctor ? p.assignedDoctor.name : 'Unassigned'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
};

export default AdminDashboard;
