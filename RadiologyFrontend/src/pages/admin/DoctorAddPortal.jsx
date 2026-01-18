import { useState } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { addDoctor } from "@/services/api";
import { toast } from "sonner";

const DoctorAddPortal = () => {
  const [doctor, setDoctor] = useState({
    name: "",
    specialty: "",
    availability: "Available",
  });

  const handleChange = (e) => {
    const { id, value } = e.target;
    setDoctor((prev) => ({ ...prev, [id]: value }));
  };

  const handleAddDoctor = async () => {
    try {
      const response = await addDoctor(doctor);
      console.log("Doctor added:", response);
      toast.success("Doctor added successfully!");
      // Clear form
      setDoctor({
        name: "",
        specialty: "",
        availability: "Available",
      });
    } catch (error) {
      console.error("Failed to add doctor:", error);
      toast.error("Failed to add doctor. Please try again.");
    }
  };

  return (
    <DashboardLayout title="Doctor Addition Portal">
      <Card>
        <CardHeader>
          <CardTitle>Add New Doctor</CardTitle>
          <CardDescription>Fill out the form below to add a new doctor.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Full Name</Label>
            <Input id="name" value={doctor.name} onChange={handleChange} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="specialty">Specialty</Label>
            <Input id="specialty" value={doctor.specialty} onChange={handleChange} />
          </div>
          <Button onClick={handleAddDoctor}>Add Doctor</Button>
        </CardContent>
      </Card>
    </DashboardLayout>
  );
};

export default DoctorAddPortal;
