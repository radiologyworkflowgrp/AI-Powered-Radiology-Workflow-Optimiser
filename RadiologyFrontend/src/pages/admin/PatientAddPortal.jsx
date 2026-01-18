import { useState } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { addPatient } from "@/services/api";
import { toast } from "sonner";

const PatientAddPortal = () => {
  const [patient, setPatient] = useState({
    name: "",
    age: "",
    gender: "",
    guardian_name: "",
    guardian_phone: "",
    allergies: "",
    blood_type: "",
    symptoms: "",
    status: "Admitted",
    date: new Date().toISOString().slice(0, 10),
  });

  const handleChange = (e) => {
    const { id, value } = e.target;
    setPatient((prev) => ({ ...prev, [id]: value }));
  };

  const handleSelectChange = (id, value) => {
    setPatient((prev) => ({ ...prev, [id]: value }));
  };

  const handleAdmit = async () => {
    try {
      const response = await addPatient(patient);
      console.log("Patient admitted:", response);
      toast.success("Patient admitted successfully! Priority will be calculated by ML model.");
      // Clear form
      setPatient({
        name: "",
        age: "",
        gender: "",
        guardian_name: "",
        guardian_phone: "",
        allergies: "",
        blood_type: "",
        symptoms: "",
        status: "Admitted",
        date: new Date().toISOString().slice(0, 10),
      });
    } catch (error) {
      console.error("Failed to admit patient:", error);
      toast.error("Failed to admit patient. Please try again.");
    }
  };

  return (
    <DashboardLayout title="Patient Admission Portal">
      <Card>
        <CardHeader>
          <CardTitle>Admit New Patient</CardTitle>
          <CardDescription>Fill out the form below to admit a new patient. Priority will be automatically calculated based on symptoms.</CardDescription>
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
          <div className="space-y-2">
            <Label htmlFor="symptoms">Symptoms</Label>
            <Textarea
              id="symptoms"
              value={patient.symptoms}
              onChange={handleChange}
              placeholder="Describe patient symptoms (e.g., chest pain, difficulty breathing, fever...)"
              rows={4}
            />
            <p className="text-sm text-muted-foreground">
              Priority will be automatically calculated by our ML model based on the symptoms provided.
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="date">Admission Date</Label>
            <Input id="date" type="date" value={patient.date} onChange={handleChange} />
          </div>
          <Button onClick={handleAdmit}>Admit Patient</Button>
        </CardContent>
      </Card>
    </DashboardLayout>
  );
};

export default PatientAddPortal;