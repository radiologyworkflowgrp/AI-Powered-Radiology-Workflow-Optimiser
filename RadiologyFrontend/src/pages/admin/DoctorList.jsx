import { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getDoctors } from "@/services/api";

const DoctorList = () => {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDoctors = async () => {
      try {
        setLoading(true);
        const data = await getDoctors();
        setDoctors(data);
      } catch (error) {
        console.error("Error fetching doctors:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchDoctors();
  }, []);

  const getStatusColor = (availability) => {
    if (availability === "Available") return "success";
    if (availability === "On-call") return "default";
    return "destructive";
  };

  return (
    <DashboardLayout title="Doctors List">
      <Card>
        <CardHeader>
          <CardTitle>All Doctors</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-muted-foreground text-center py-4">Loading doctors...</p>
          ) : doctors.length === 0 ? (
            <p className="text-muted-foreground text-center py-4">No doctors found</p>
          ) : (
            <div className="space-y-4">
              {doctors.map((doctor) => (
                <div key={doctor._id} className="flex items-center justify-between border-b pb-4 last:border-0">
                  <div>
                    <p className="font-medium">{doctor.name}</p>
                    <p className="text-sm text-muted-foreground">{doctor.specialty}</p>
                    <p className="text-xs text-muted-foreground">{doctor.email}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={getStatusColor(doctor.availability)}>{doctor.availability}</Badge>
                    <Button variant="outline" size="sm">View Details</Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </DashboardLayout>
  );
};

export default DoctorList;