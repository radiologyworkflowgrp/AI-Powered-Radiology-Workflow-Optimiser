import { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getPrescriptions, requestRefill } from "@/services/api";
import { toast } from "sonner";

const Prescriptions = () => {
  const [prescriptions, setPrescriptions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPrescriptions();
  }, []);

  const fetchPrescriptions = async () => {
    try {
      setLoading(true);
      const data = await getPrescriptions();
      setPrescriptions(data);
    } catch (error) {
      console.error("Failed to fetch prescriptions:", error);
      toast.error("Failed to load prescriptions");
    } finally {
      setLoading(false);
    }
  };

  const handleRequestRefill = async (prescriptionId) => {
    try {
      await requestRefill(prescriptionId);
      toast.success("Refill request submitted successfully!");
      // Refresh the prescriptions list
      fetchPrescriptions();
    } catch (error) {
      console.error("Failed to request refill:", error);
      toast.error("Failed to request refill. Please try again.");
    }
  };

  const getStatusVariant = (status) => {
    switch (status) {
      case "approved":
        return "default";
      case "pending":
        return "secondary";
      case "rejected":
        return "destructive";
      default:
        return "outline";
    }
  };

  if (loading) {
    return (
      <DashboardLayout title="Prescriptions">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-muted-foreground">Loading prescriptions...</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout title="Prescriptions">
      <div className="grid gap-6">
        {prescriptions.length === 0 ? (
          <Card>
            <CardContent className="pt-6">
              <p className="text-center text-muted-foreground">No prescriptions found.</p>
            </CardContent>
          </Card>
        ) : (
          prescriptions.map((prescription) => (
            <Card key={prescription._id}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle>{prescription.medicineName}</CardTitle>
                    <CardDescription>
                      {prescription.prescribedBy ? `Prescribed by ${prescription.prescribedBy}` : "Prescription"}
                    </CardDescription>
                  </div>
                  <Badge variant={getStatusVariant(prescription.status)}>
                    {prescription.status.charAt(0).toUpperCase() + prescription.status.slice(1)}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium">Patient Name</p>
                    <p className="text-sm text-muted-foreground">{prescription.patientName}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Date Prescribed</p>
                    <p className="text-sm text-muted-foreground">
                      {new Date(prescription.datePrescribed).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <div className="flex items-center justify-between pt-4 border-t">
                  <span className="text-sm text-muted-foreground">
                    Refill Count: {prescription.refillCount}
                  </span>
                  {prescription.status === "approved" && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleRequestRefill(prescription._id)}
                    >
                      Request Refill
                    </Button>
                  )}
                  {prescription.status === "pending" && (
                    <span className="text-sm text-amber-600 font-medium">
                      Refill request pending approval
                    </span>
                  )}
                  {prescription.status === "rejected" && (
                    <span className="text-sm text-red-600 font-medium">
                      Refill request rejected
                    </span>
                  )}
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </DashboardLayout>
  );
};

export default Prescriptions;
