import { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useUser } from "@/hooks/use-user";
import { getPatients, getPatientRecentReports } from "@/services/api";
import PatientInfoModal from "@/components/PatientInfoModal";

const PatientDashboard = () => {
  const { user } = useUser();
  const [showModal, setShowModal] = useState(false);
  const [currentPatient, setCurrentPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recentReports, setRecentReports] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(false);

  const fetchPatientData = async () => {
    if (!user) return;
    try {
      setLoading(true);
      const patients = await getPatients();
      // Find patient by email (since user context has email)
      const foundPatient = patients.find(p => p.email === user.email);

      if (foundPatient) {
        setCurrentPatient(foundPatient);
        // Fetch recent reports for this patient
        await fetchRecentReports(foundPatient._id || foundPatient.id);
        // Show modal if profile not completed
        if (!foundPatient.profileCompleted) {
          setShowModal(true);
        }
      } else {
        // No patient record found, show modal to create profile
        setShowModal(true);
      }
    } catch (error) {
      console.error("Failed to fetch patient data:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchRecentReports = async (patientId) => {
    try {
      setReportsLoading(true);
      const reports = await getPatientRecentReports(patientId, 3); // Get 3 most recent
      setRecentReports(reports);
    } catch (error) {
      console.error("Failed to fetch recent reports:", error);
    } finally {
      setReportsLoading(false);
    }
  };

  useEffect(() => {
    fetchPatientData();
  }, [user]);

  const handleModalSuccess = () => {
    setShowModal(false);
    fetchPatientData();
  };

  if (loading) {
    return (
      <DashboardLayout title="Patient Dashboard">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-muted-foreground">Loading your information...</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout title="Patient Dashboard">
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Medical Information</CardTitle>
            <CardDescription>
              {currentPatient && currentPatient.medical_history ? (
                <a
                  href={`http://localhost:3002/${currentPatient.medical_history.replace(/\\/g, '/')}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  View Medical History PDF
                </a>
              ) : (
                "Your personal medical data"
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {currentPatient && currentPatient.profileCompleted ? (
              <>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Blood Type</p>
                  <p className="text-2xl font-bold text-primary">{currentPatient.blood_type || "N/A"}</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Height</p>
                  <p className="text-2xl font-bold text-primary">{currentPatient.height ? `${currentPatient.height} cm` : "N/A"}</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Weight</p>
                  <p className="text-2xl font-bold text-primary">{currentPatient.weight ? `${currentPatient.weight} kg` : "N/A"}</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Allergies</p>
                  <p className="text-sm text-muted-foreground">{currentPatient.allergies || "None"}</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Assigned Doctor</p>
                  <p className="text-xl font-bold text-primary">{currentPatient.assignedDoctor ? currentPatient.assignedDoctor.name : "Not Assigned"}</p>
                </div>
              </>
            ) : (
              <div className="text-center py-4">
                <p className="text-muted-foreground mb-4">No medical record found.</p>
                <Button onClick={() => setShowModal(true)}>Complete Profile</Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Upcoming Appointments</CardTitle>
            <CardDescription>Your scheduled appointments</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {currentPatient && currentPatient.date ? (
                <div className="border-b pb-4 last:border-0">
                  <p className="font-medium">Follow-up / Admission</p>
                  <p className="text-sm text-muted-foreground">{currentPatient.date}</p>
                </div>
              ) : (
                <p className="text-muted-foreground">No upcoming appointments.</p>
              )}
              <Button className="w-full mt-4">Schedule New Appointment</Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Current Medications</CardTitle>
            <CardDescription>Your active medications</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              {currentPatient && currentPatient.medications ? currentPatient.medications : "No medications recorded."}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Reports</CardTitle>
            <CardDescription>Your latest radiology and ML-generated reports</CardDescription>
          </CardHeader>
          <CardContent>
            {reportsLoading ? (
              <div className="text-center py-4">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mx-auto"></div>
                <p className="text-sm text-muted-foreground mt-2">Loading reports...</p>
              </div>
            ) : recentReports.length === 0 ? (
              <div className="text-center py-4">
                <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <p className="text-sm text-muted-foreground">No reports available</p>
                <p className="text-xs text-gray-400">Your reports will appear here once they're ready</p>
              </div>
            ) : (
              <div className="space-y-3">
                {recentReports.map((report) => (
                  <div key={report.id || report._id} className="border rounded-lg p-3">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-medium text-sm">
                          {report.type === 'ml_generated' ? report.report_type : 'Radiology Report'}
                        </h4>
                        <p className="text-xs text-muted-foreground">
                          {report.created_at ? new Date(report.created_at).toLocaleDateString() : 'Recent'}
                        </p>
                      </div>
                      <Badge variant={report.status_display === 'Completed' ? 'default' : 'secondary'} className="text-xs">
                        {report.status_display}
                      </Badge>
                    </div>
                    {report.status_description && (
                      <p className="text-xs text-gray-600 mb-2">{report.status_description}</p>
                    )}
                    {report.type === 'ml_generated' && report.confidence_score && (
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>Confidence: {(report.confidence_score * 100).toFixed(1)}%</span>
                        {report.ml_model && <span>• {report.ml_model}</span>}
                      </div>
                    )}
                    <div className="flex gap-2 mt-2">
                      <Button variant="outline" size="sm" className="text-xs">
                        View Details
                      </Button>
                    </div>
                  </div>
                ))}
                <Button variant="outline" size="sm" className="w-full mt-2">
                  View All Reports
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Common tasks</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button className="w-full" variant="outline">View Test Results</Button>
            <Button className="w-full" variant="outline">Message Doctor</Button>
          </CardContent>
        </Card>
      </div>

      <PatientInfoModal
        isOpen={showModal}
        onOpenChange={setShowModal}
        onSuccess={handleModalSuccess}
        hideUpdateTab={true}
        patientId={user?.id || user?.userId}
      />
    </DashboardLayout>
  );
};

export default PatientDashboard;
