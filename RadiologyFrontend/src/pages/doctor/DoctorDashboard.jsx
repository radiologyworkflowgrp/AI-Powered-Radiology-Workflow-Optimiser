import { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useUser } from "@/hooks/use-user";
import { getPatients } from "@/services/api";
import PatientDetailsModal from "@/components/PatientDetailsModal";

const DoctorDashboard = () => {
  const { user } = useUser();
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const allPatients = await getPatients();
        // Backend already filters patients by doctor, so just use the response directly
        console.log('Fetched patients for doctor:', allPatients.length, allPatients);
        setPatients(allPatients);
      } catch (error) {
        console.error("Error fetching patients:", error);
      }
    };
    fetchPatients();
  }, [user]);

  const getPriorityColor = (priority) => {
    if (priority <= 1) return "destructive";
    if (priority <= 2) return "default";
    return "secondary";
  };

  const handleViewPatient = (patient) => {
    setSelectedPatient(patient);
    setIsModalOpen(true);
  };

  const handleCheckPatient = async (patient, checked) => {
    try {
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:3002";
      await fetch(`${API_URL}/api/patients/${patient.id}/check`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checked })
      });

      // Update local state
      setPatients(patients.map(p =>
        p.id === patient.id
          ? { ...p, checkedByDoctor: checked, status: checked ? 'Completed' : 'Doctor Appointment' }
          : p
      ));
    } catch (error) {
      console.error("Error checking patient:", error);
    }
  };

  return (
    <DashboardLayout title="Doctor Dashboard">
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Patient List</CardTitle>
            <CardDescription>View and manage your assigned patients</CardDescription>
          </CardHeader>
          <CardContent>
            {patients.length === 0 ? (
              <p className="text-muted-foreground text-center py-4">No patients assigned yet</p>
            ) : (
              <div className="space-y-4">
                {patients.map((patient) => (
                  <div key={patient._id || patient.id} className="flex items-center justify-between border-b pb-4 last:border-0">
                    <div>
                      <p className="font-medium">{patient.name}</p>
                      <p className="text-sm text-muted-foreground">
                        Age: {patient.age} | Gender: {patient.gender}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Admitted: {patient.createdAt ? new Date(patient.createdAt).toLocaleDateString() : 'N/A'}
                      </p>
                      {patient.status && (
                        <p className="text-sm text-muted-foreground">
                          Status: <span className="font-medium">{patient.status}</span>
                        </p>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant={getPriorityColor(patient.priority)}>
                        Priority {patient.priority}
                      </Badge>
                      <Button variant="outline" size="sm" onClick={() => handleViewPatient(patient)}>
                        View
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Patient Statistics</CardTitle>
            <CardDescription>Overview of your assigned patients</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b pb-2">
                <span className="text-sm font-medium">Total Patients</span>
                <span className="text-2xl font-bold">{patients.length}</span>
              </div>
              <div className="flex items-center justify-between border-b pb-2">
                <span className="text-sm font-medium">Patients Unchecked</span>
                <span className="text-2xl font-bold text-orange-600">
                  {patients.filter(p => !p.checkedByDoctor && p.status !== 'Completed').length}
                </span>
              </div>
              <div className="flex items-center justify-between border-b pb-2">
                <span className="text-sm font-medium">High Priority</span>
                <span className="text-2xl font-bold text-destructive">
                  {patients.filter(p => p.priority === 1).length}
                </span>
              </div>
              <div className="flex items-center justify-between border-b pb-2">
                <span className="text-sm font-medium">Medium Priority</span>
                <span className="text-2xl font-bold text-default">
                  {patients.filter(p => p.priority === 2).length}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Low Priority</span>
                <span className="text-2xl font-bold text-secondary">
                  {patients.filter(p => p.priority === 3).length}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <PatientDetailsModal
        patient={selectedPatient}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </DashboardLayout>
  );
};

export default DoctorDashboard;
