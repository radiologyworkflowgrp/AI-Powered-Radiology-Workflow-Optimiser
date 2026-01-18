import { useEffect, useState } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { getPatients } from "@/services/api";
import axios from "axios";
import PatientInfoModal from "@/components/PatientInfoModal";

const PatientStatus = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientCredentials, setPatientCredentials] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [loadingCredentials, setLoadingCredentials] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:3002";

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        setLoading(true);
        const data = await getPatients();
        setPatients(data);
        setFilteredPatients(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchPatients();
  }, []);

  useEffect(() => {
    const results = patients.filter(patient =>
      patient.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredPatients(results);
  }, [searchTerm, patients]);

  const handleViewDetails = async (patient) => {
    setSelectedPatient(patient);
    setIsModalOpen(true);
    setLoadingCredentials(true);

    try {
      // Fetch user credentials from MongoDB
      const response = await axios.get(`${API_URL}/api/users?email=${encodeURIComponent(patient.email)}`);
      if (response.data && response.data.length > 0) {
        setPatientCredentials(response.data[0]);
      } else {
        setPatientCredentials(null);
      }
    } catch (err) {
      console.error("Error fetching credentials:", err);
      setPatientCredentials(null);
    } finally {
      setLoadingCredentials(false);
    }
  };

  const getPriorityColor = (priority) => {
    if (priority === 1) return "destructive";
    if (priority === 2) return "default";
    return "secondary";
  };

  const getPriorityLabel = (priority) => {
    if (priority === 1) return "High";
    if (priority === 2) return "Medium";
    return "Low";
  };

  if (loading) {
    return (
      <DashboardLayout title="Patient Status">
        <p>Loading patients...</p>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout title="Patient Status">
        <p className="text-red-500">Error: {error}</p>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout title="Patient Status">
      <div className="flex flex-col gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Patient Search</CardTitle>
            <CardDescription>Search for patients by name.</CardDescription>
          </CardHeader>
          <CardContent>
            <Input
              type="text"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </CardContent>
        </Card>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {filteredPatients.map((patient) => (
            <Card key={patient.id || patient._id}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle>{patient.name}</CardTitle>
                    <CardDescription className="text-xs mt-1">
                      {patient.email}
                    </CardDescription>
                  </div>
                  <Badge variant={getPriorityColor(patient.priority)}>
                    {getPriorityLabel(patient.priority)}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-sm"><strong>Age:</strong> {patient.age}</p>
                <p className="text-sm"><strong>Gender:</strong> {patient.gender}</p>
                <p className="text-sm"><strong>Status:</strong> {patient.status || 'Admitted'}</p>
                {patient.assignedDoctor && (
                  <p className="text-sm"><strong>Doctor:</strong> {patient.assignedDoctor.name}</p>
                )}
                <p className="text-sm"><strong>Admitted:</strong> {patient.createdAt ? new Date(patient.createdAt).toLocaleDateString() : 'N/A'}</p>
                <Button
                  onClick={() => handleViewDetails(patient)}
                  variant="outline"
                  className="w-full mt-3"
                  size="sm"
                >
                  View Details
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Patient Details Modal */}
      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader className="flex flex-row items-center justify-between">
            <div>
              <DialogTitle>{selectedPatient?.name} - Patient Details</DialogTitle>
              <DialogDescription>Complete patient information and login credentials</DialogDescription>
            </div>
            <Button
              onClick={() => {
                setIsModalOpen(false);
                setIsEditModalOpen(true);
              }}
              variant="outline"
            >
              Edit Patient
            </Button>
          </DialogHeader>

          {selectedPatient && (
            <div className="space-y-4">
              {/* Login Credentials Section */}
              <Card className="bg-blue-50 border-blue-200">
                <CardHeader>
                  <CardTitle className="text-lg">🔐 Login Credentials</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {loadingCredentials ? (
                    <p className="text-sm text-muted-foreground">Loading credentials...</p>
                  ) : patientCredentials ? (
                    <>
                      <div className="bg-white p-3 rounded border">
                        <p className="text-sm font-medium mb-1">Email (Username)</p>
                        <p className="text-base font-mono">{patientCredentials.email}</p>
                      </div>
                      <div className="bg-white p-3 rounded border">
                        <p className="text-sm font-medium mb-1">Password</p>
                        <p className="text-base font-mono text-red-600">
                          {patientCredentials.plainPassword || "Password not stored (generated at creation)"}
                        </p>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        ⚠️ Share these credentials securely with the patient
                      </p>
                    </>
                  ) : (
                    <p className="text-sm text-muted-foreground">No credentials found</p>
                  )}
                </CardContent>
              </Card>

              {/* Patient Information */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Patient Information</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Name</p>
                    <p className="text-base">{selectedPatient.name}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Email</p>
                    <p className="text-base">{selectedPatient.email}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Age</p>
                    <p className="text-base">{selectedPatient.age}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Gender</p>
                    <p className="text-base">{selectedPatient.gender}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Priority</p>
                    <Badge variant={getPriorityColor(selectedPatient.priority)}>
                      {getPriorityLabel(selectedPatient.priority)}
                    </Badge>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Status</p>
                    <p className="text-base">{selectedPatient.status || 'Admitted'}</p>
                  </div>
                  {selectedPatient.contact && (
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Contact</p>
                      <p className="text-base">{selectedPatient.contact}</p>
                    </div>
                  )}
                  {selectedPatient.assignedDoctor && (
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Assigned Doctor</p>
                      <p className="text-base">{selectedPatient.assignedDoctor.name}</p>
                    </div>
                  )}
                  <div className="col-span-2">
                    <p className="text-sm font-medium text-muted-foreground">Admission Date</p>
                    <p className="text-base">
                      {selectedPatient.createdAt ? new Date(selectedPatient.createdAt).toLocaleString() : 'N/A'}
                    </p>
                  </div>
                  {selectedPatient.medical_history && (
                    <div className="col-span-2">
                      <p className="text-sm font-medium text-muted-foreground">Medical History</p>
                      <p className="text-base">{selectedPatient.medical_history}</p>
                    </div>
                  )}
                  <div className="col-span-2 mt-2 pt-2 border-t">
                    <p className="text-sm font-medium text-muted-foreground mb-2">Medical Profile</p>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div><span className="font-medium">Blood:</span> {selectedPatient.blood_type || 'N/A'}</div>
                      <div><span className="font-medium">Height:</span> {selectedPatient.height ? `${selectedPatient.height}cm` : 'N/A'}</div>
                      <div><span className="font-medium">Weight:</span> {selectedPatient.weight ? `${selectedPatient.weight}kg` : 'N/A'}</div>
                      <div><span className="font-medium">Allergies:</span> {selectedPatient.allergies || 'None'}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <PatientInfoModal
        isOpen={isEditModalOpen}
        onOpenChange={setIsEditModalOpen}
        initialPatient={selectedPatient}
        onSuccess={() => {
          setIsEditModalOpen(false);
          // Refresh logic if needed - currently PatientStatus fetches on mount
          window.location.reload(); // Simple refresh to show updates
        }}
      />
    </DashboardLayout>
  );
};

export default PatientStatus;